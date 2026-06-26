"""Phase 5 verification: RAG embed/dedup/retrieve, ingestion tagging, and the
news agent grounding its prompt in retrieved headlines.

Runs on the dependency-light HashingEmbedder (no torch needed), so it exercises
the full pipeline under the numpy<2 pin.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import pandas_ta  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))


def test_hashing_embedder_similarity():
    from rag import HashingEmbedder
    e = HashingEmbedder(dim=256)
    v = e.embed(["bitcoin price rises sharply",
                 "bitcoin price climbs sharply",
                 "ethereum staking rewards update"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)     # normalized
    sim_close = float(v[0] @ v[1])
    sim_far = float(v[0] @ v[2])
    assert sim_close > sim_far                                         # semantics-ish
    assert np.allclose(e.embed(["bitcoin price rises sharply"])[0], v[0])  # deterministic


def test_tag_assets():
    from ingestion import tag_assets
    assert "BTC" in tag_assets("Bitcoin surges past 70k")
    assert "SOL" in tag_assets("$SOL pumps 20% today")
    assert "ETH" in tag_assets("ETHUSDT breaks resistance")
    assert "AAVE" in tag_assets("AAVE lending volume hits record")
    assert "OP" not in tag_assets("The op-ed was great")             # no false short-ticker hit


def test_rag_ingest_dedup_and_headlines(tmp_path):
    from persistence import Store
    from rag import RagIndex, HashingEmbedder
    store = Store(str(tmp_path / "r.db"))
    idx = RagIndex(store, embedder=HashingEmbedder(), dedup_threshold=0.9)
    items = [
        {"id": "a", "title": "Bitcoin hits new high", "body": "", "assets": ["BTC"],
         "source": "x", "url": "u1", "published_ts": 1.0},
        {"id": "b", "title": "Bitcoin hits new high", "body": "", "assets": ["BTC"],   # dup text
         "source": "x", "url": "u2", "published_ts": 2.0},
        {"id": "c", "title": "Ethereum upgrade ships", "body": "", "assets": ["ETH"],
         "source": "x", "url": "u3", "published_ts": 3.0},
    ]
    stats = idx.ingest(items)
    assert stats == {"added": 2, "deduped": 1, "skipped": 0}
    assert idx.headlines_for("BTC") == ["Bitcoin hits new high"]
    assert idx.headlines_for("ETH") == ["Ethereum upgrade ships"]

    # re-ingest: a,c already stored -> skipped; b still a near-dup -> deduped
    stats2 = idx.ingest(items)
    assert stats2 == {"added": 0, "deduped": 1, "skipped": 2}

    res = idx.query("bitcoin price", k=1)
    assert res and res[0]["title"] == "Bitcoin hits new high"
    store.close()


def test_ingest_all_with_fake_feed(tmp_path):
    from persistence import Store
    from rag import RagIndex, HashingEmbedder
    from ingestion import ingest_all

    store = Store(str(tmp_path / "i.db"))
    idx = RagIndex(store, embedder=HashingEmbedder())

    def fake_rss(url):
        return [{"title": "Solana network upgrade live", "link": "http://x/1", "summary": "$SOL update"},
                {"title": "Bitcoin ETF sees inflows", "link": "http://x/2", "summary": "Bitcoin demand"}]

    stats = ingest_all(idx, rss_sources=[("fake", "http://x")], rss_fetcher=fake_rss)
    assert stats["added"] == 2
    assert idx.headlines_for("SOL") == ["Solana network upgrade live"]
    assert idx.headlines_for("BTC") == ["Bitcoin ETF sees inflows"]
    store.close()


def test_news_agent_grounds_prompt_in_headlines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    from agents import llm_client
    from agents.news_agent import NewsAgent

    class CapturingLLM:
        def __init__(self):
            self.prompts = []

        def chat_json(self, prompt):
            self.prompts.append(prompt)
            if "panic-worthy" in prompt:
                return {"has_panic": False, "sentiment": "Neutral", "confidence": 0.5, "top_headlines": []}
            return {"pair": "NA", "sentiment": "Bullish", "confidence": 0.7, "top_headlines": []}

    cap = CapturingLLM()
    llm_client.set_client(cap)

    ag = NewsAgent()
    ag.scan_pair("BTCUSDT", headlines=["Bitcoin ETF approved", "BTC rallies to new high"])
    assert any("Bitcoin ETF approved" in p for p in cap.prompts)        # headline grounded into prompt
