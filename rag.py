"""RAG layer (Phase 5): embed + dedup + retrieve crypto news.

The embedder is pluggable so the system runs everywhere:
  * MiniLMEmbedder — sentence-transformers all-MiniLM-L6-v2 (384-d, real
    semantics). Selected automatically when the package imports (the Oracle
    target, where torch-CPU aarch64 wheels are available).
  * HashingEmbedder — dependency-light hashing trick (pure numpy, no torch).
    Default when sentence-transformers isn't installed, so RAG runs and tests
    under the numpy<2 pin that the vendored pandas_ta requires.

Vectors + items live in the same SQLite DB as everything else (via Store). A
brute-force numpy cosine is plenty for a few hundred-thousand headlines;
sqlite-vec can replace the search behind this same interface later.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from persistence import Store, get_store

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class Embedder:
    dim: int = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # -> (n, dim), L2-normalized
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Hashing-trick bag-of-words, L2-normalized. Deterministic, no model load."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class Model2VecEmbedder(Embedder):
    """Static-embedding model (enhancement C4): real semantic vectors WITHOUT
    torch — model2vec 0.8.2 installs cleanly against the numpy<2 pin (verified
    2026-07). ~30MB one-time download (potion-base-8M), then pure numpy
    inference. Select with RAG_EMBEDDER=model2vec."""

    def __init__(self, model: str = "minishlab/potion-base-8M"):
        from model2vec import StaticModel
        self._m = StaticModel.from_pretrained(model)
        probe = np.asarray(self._m.encode(["probe"]))
        self.dim = int(probe.shape[1])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        v = np.asarray(self._m.encode(list(texts)), dtype=np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n


class MiniLMEmbedder(Embedder):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model)
        self.dim = self._m.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        v = self._m.encode(list(texts), normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


def get_embedder() -> Embedder:
    pref = os.getenv("RAG_EMBEDDER", "auto").lower()
    if pref == "model2vec":
        return Model2VecEmbedder()   # explicit opt-in: raise loudly on failure
    if pref in ("auto", "minilm"):
        try:
            return MiniLMEmbedder()
        except Exception:
            if pref == "minilm":
                raise
    return HashingEmbedder()


def _to_bytes(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


class RagIndex:
    def __init__(self, store: Optional[Store] = None, embedder: Optional[Embedder] = None,
                 dedup_threshold: float = 0.9):
        self.store = store or get_store()
        self.embedder = embedder or get_embedder()
        self.dedup_threshold = dedup_threshold

    def ingest(self, items: List[Dict[str, Any]], dedup_window_ts: Optional[float] = None) -> Dict[str, int]:
        """Embed + store new items, skipping ids we already have and near-duplicates
        (cosine >= dedup_threshold against the recent corpus and within the batch)."""
        stats = {"added": 0, "deduped": 0, "skipped": 0}
        existing = [_from_bytes(b) for _id, b in self.store.news_embeddings(since_ts=dedup_window_ts)]
        mat = np.vstack(existing) if existing else None
        for it in items:
            iid = it["id"]
            if self.store.has_news_item(iid):
                stats["skipped"] += 1
                continue
            text = (it.get("title", "") + ". " + (it.get("body") or "")).strip()
            vec = self.embedder.embed([text])[0]
            if mat is not None and len(mat) and float((mat @ vec).max()) >= self.dedup_threshold:
                stats["deduped"] += 1
                continue
            self.store.add_news_item(
                item_id=iid, source=it.get("source"), title=it.get("title"), body=it.get("body", ""),
                url=it.get("url"), published_ts=it.get("published_ts"), assets=it.get("assets", []),
                embedding=_to_bytes(vec))
            mat = vec[None, :] if mat is None else np.vstack([mat, vec])
            stats["added"] += 1
        return stats

    def headlines_for(self, asset: str, k: int = 5, since_ts: Optional[float] = None) -> List[str]:
        """Recent headlines tagged with this asset (for prompt grounding)."""
        return [r["title"] for r in self.store.recent_news_for_asset(asset, since_ts=since_ts, limit=k)]

    def query(self, text: str, k: int = 5, since_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Semantic top-k over the corpus."""
        rows = self.store.news_embeddings(since_ts=since_ts)
        if not rows:
            return []
        ids = [r[0] for r in rows]
        mat = np.vstack([_from_bytes(r[1]) for r in rows])
        sims = mat @ self.embedder.embed([text])[0]
        order = np.argsort(-sims)[:k]
        return [{**self.store.get_news_item(ids[i]), "score": float(sims[i])} for i in order]
