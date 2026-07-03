"""Durable SQLite state for BitReinforceX (Phase 2b).

Replaces the scattered, unbounded JSON/JSONL files + in-memory session dict with
one SQLite database (WAL mode). It is the backbone for:
  * the auto-labelling grader (Phase 3) — every prediction is recorded WITH each
    agent's RL replay payload, so a prediction is graded against ITS OWN data,
    and
  * durable Telegram sessions (Phase 4) — feedback buttons survive a restart.

Concurrency
-----------
The runtime fans 48 pairs out across a thread pool (asyncio.to_thread). SQLite
connection objects are not safe to share across threads, so every operation runs
under one process-wide lock on a single ``check_same_thread=False`` connection.
Writes are tiny and fast, so serialising them is plenty; WAL keeps reads cheap
and gives crash-safe durability.

Note: the agents keep their bandit POLICY json files (learned state) as-is; this
DB stores predictions/outcomes/rewards/sessions, not the policies. Reward
application is already serial (grader loop + Telegram callbacks both run in the
single asyncio loop), which removes the old policy-file write race (#10).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.getenv("BITREINFORCEX_DB", "logs/bitreinforcex.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  TEXT PRIMARY KEY,
    cycle_id            TEXT,
    pair                TEXT NOT NULL,
    tf                  TEXT NOT NULL,
    created_ts          REAL NOT NULL,
    candle_close_ts     REAL,
    entry_price         REAL,
    horizon_k           INTEGER,
    grade_due_ts        REAL,
    final_action        TEXT,
    final_confidence    REAL,
    final_score         REAL,
    emitted             INTEGER DEFAULT 0,
    -- per-agent RL replay payloads (the fix for the feature-race) --
    news_action         TEXT,
    news_action_idx     INTEGER,
    news_feats          TEXT,      -- JSON list
    news_conf           REAL,
    research_action     TEXT,
    research_action_idx INTEGER,
    research_feats      TEXT,      -- JSON list
    research_conf       REAL,
    indicator_action    TEXT,
    indicator_conf      REAL,
    indicator_blend     TEXT,      -- JSON object
    brain_weights       TEXT,      -- JSON object
    label_source        TEXT DEFAULT 'pending',   -- pending | auto | manual
    graded              INTEGER DEFAULT 0,
    session_id          TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_due     ON predictions(grade_due_ts, graded);
CREATE INDEX IF NOT EXISTS idx_pred_pair_tf ON predictions(pair, tf, created_ts);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id   TEXT PRIMARY KEY REFERENCES predictions(id),
    realized_return REAL,
    realized_label  TEXT,
    threshold       REAL,
    horizon_k       INTEGER,
    graded_ts       REAL,
    source          TEXT
);

CREATE TABLE IF NOT EXISTS rewards (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id    TEXT REFERENCES predictions(id),
    agent            TEXT,
    predicted_action TEXT,
    reward           REAL,
    applied_ts       REAL,
    source           TEXT          -- auto | manual | correction
);
CREATE INDEX IF NOT EXISTS idx_rewards_pred ON rewards(prediction_id);

CREATE TABLE IF NOT EXISTS sessions (
    id               TEXT PRIMARY KEY,
    pair             TEXT,
    tf               TEXT,
    prediction_id    TEXT REFERENCES predictions(id),
    created_ts       REAL,
    active           INTEGER DEFAULT 1,
    customer_chat_id INTEGER,
    customer_msg_id  INTEGER,
    dev_chat_id      INTEGER,
    dev_msg_id       INTEGER,
    true_outcome     TEXT,
    superseded_by    TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(active);
CREATE INDEX IF NOT EXISTS idx_sessions_pairtf ON sessions(pair, tf, active);

CREATE TABLE IF NOT EXISTS macro_snapshots (
    ts             REAL PRIMARY KEY,
    btc_dominance  REAL,
    fear_greed     REAL
);

CREATE TABLE IF NOT EXISTS news_items (
    id               TEXT PRIMARY KEY,    -- stable hash of url/title
    source           TEXT,
    title            TEXT,
    body             TEXT,
    url              TEXT,
    published_ts     REAL,
    ingested_ts      REAL,
    assets           TEXT,                -- JSON list of base tickers
    embedding        BLOB,                -- float32 vector bytes
    sentiment_cached TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_pub ON news_items(published_ts);
"""

# Columns on `predictions` that hold JSON and must be (de)serialised.
_PRED_JSON_COLS = ("news_feats", "research_feats", "indicator_blend", "brain_weights",
                   "regime_feats", "deriv_feats")

# Accuracy-upgrade columns, added by the idempotent migration below. Additive
# ALTER TABLE only — legacy rows keep NULLs and grade via the legacy path.
_MIGRATION_COLS = {
    "predictions": [
        ("regime", "TEXT"),
        ("regime_feats", "TEXT"),          # JSON: adx/chop/vol_pct/atr/vol_ok/...
        ("atr", "REAL"),
        ("tp_price", "REAL"),
        ("sl_price", "REAL"),
        ("trigger_source", "TEXT"),        # gate reason at emit time
        ("deriv_action", "TEXT"),
        ("deriv_action_idx", "INTEGER"),
        ("deriv_feats", "TEXT"),           # JSON list
        ("deriv_conf", "REAL"),
        ("meta_p", "REAL"),                # meta-label p(correct), shadow
        ("calibrated_conf", "REAL"),
        ("final_action_v2", "TEXT"),       # deadzone-v2 shadow action (A7)
    ],
    "outcomes": [
        ("label_tb", "TEXT"),              # tp | sl | timeout
        ("barrier_hit_idx", "INTEGER"),
        ("exit_price", "REAL"),
    ],
}


def _dumps(v: Any) -> Optional[str]:
    return None if v is None else json.dumps(v)


def _loads(v: Any) -> Any:
    if v is None or v == "":
        return None
    try:
        return json.loads(v)
    except (TypeError, json.JSONDecodeError):
        return v


class Store:
    """Thread-safe SQLite store. One connection, one lock, WAL."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.executescript(_SCHEMA)
            self._migrate()
            self.conn.commit()

    def _migrate(self) -> None:
        """Idempotent additive migration: diff PRAGMA table_info against
        _MIGRATION_COLS and ALTER TABLE ADD COLUMN whatever is missing.
        Caller holds the lock (init path)."""
        for table, cols in _MIGRATION_COLS.items():
            existing = {r["name"] for r in
                        self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, ctype in cols:
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ctype}")

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #
    def record_prediction(
        self,
        decision: Dict[str, Any],
        *,
        cycle_id: Optional[str] = None,
        candle_close_ts: Optional[float] = None,
        entry_price: Optional[float] = None,
        horizon_k: Optional[int] = None,
        grade_due_ts: Optional[float] = None,
        emitted: bool = False,
        session_id: Optional[str] = None,
        created_ts: Optional[float] = None,
        prediction_id: Optional[str] = None,
        regime: Optional[str] = None,
        regime_feats: Optional[Dict[str, Any]] = None,
        atr: Optional[float] = None,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        trigger_source: Optional[str] = None,
        meta_p: Optional[float] = None,
        calibrated_conf: Optional[float] = None,
    ) -> str:
        """Insert one prediction, snapshotting each child agent's RL replay data.

        ``decision`` is a brain.decide() output. Timing fields (entry price,
        candle close, horizon, grade-due time) are supplied by the orchestrator
        which knows the schedule. Returns the new prediction id.
        """
        pid = prediction_id or uuid.uuid4().hex
        agents = decision.get("agents") or {}

        def _raw(name: str) -> Dict[str, Any]:
            return (agents.get(name) or {}).get("raw") or {}

        news_raw, research_raw, ind_raw = _raw("news"), _raw("research"), _raw("indicator")
        news_rl = news_raw.get("rl") or {}
        research_rl = research_raw.get("rl") or {}
        indicator_blend = ((ind_raw.get("details") or {}).get("blend"))
        final = decision.get("final") or {}

        # Regime rides inside the indicator details unless passed explicitly.
        ind_details = ind_raw.get("details") or {}
        regime = regime or ind_details.get("regime")
        regime_feats = regime_feats or ind_details.get("regime_feats")
        if atr is None and isinstance(regime_feats, dict):
            atr = regime_feats.get("atr")

        # Derivatives voter snapshot (Phase 4; harmless NULLs before that).
        deriv_raw = _raw("derivatives")
        deriv_rl = deriv_raw.get("rl") or {}

        row = {
            "id": pid,
            "cycle_id": cycle_id,
            "pair": decision.get("chartName") or decision.get("pair"),
            "tf": decision.get("timeframe"),
            "created_ts": created_ts if created_ts is not None else time.time(),
            "candle_close_ts": candle_close_ts,
            "entry_price": entry_price,
            "horizon_k": horizon_k,
            "grade_due_ts": grade_due_ts,
            "final_action": final.get("action"),
            "final_confidence": final.get("confidence"),
            "final_score": final.get("score"),
            "final_action_v2": final.get("action_v2"),
            "emitted": 1 if emitted else 0,
            "news_action": news_raw.get("action"),
            "news_action_idx": news_rl.get("action_idx"),
            "news_feats": _dumps(news_rl.get("features")),
            "news_conf": (agents.get("news") or {}).get("confidence"),
            "research_action": research_raw.get("action"),
            "research_action_idx": research_rl.get("action_idx"),
            "research_feats": _dumps(research_rl.get("feats")),
            "research_conf": (agents.get("research") or {}).get("confidence"),
            "indicator_action": ind_raw.get("action"),
            "indicator_conf": (agents.get("indicator") or {}).get("confidence"),
            "indicator_blend": _dumps(indicator_blend),
            "brain_weights": _dumps((decision.get("policy") or {}).get("weights")),
            "label_source": "pending",
            "graded": 0,
            "session_id": session_id,
            "regime": regime,
            "regime_feats": _dumps(regime_feats),
            "atr": atr,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "trigger_source": trigger_source,
            "deriv_action": deriv_raw.get("action"),
            "deriv_action_idx": deriv_rl.get("action_idx"),
            "deriv_feats": _dumps(deriv_rl.get("feats")),
            "deriv_conf": (agents.get("derivatives") or {}).get("confidence"),
            "meta_p": meta_p,
            "calibrated_conf": calibrated_conf,
        }
        cols = ", ".join(row.keys())
        ph = ", ".join(["?"] * len(row))
        with self._lock:
            self.conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({ph})", tuple(row.values()))
            self.conn.commit()
        return pid

    def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            r = cur.fetchone()
        return self._pred_row(r)

    def get_due_predictions(self, now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Ungraded, still-pending predictions whose grade-due time has passed."""
        now_ts = time.time() if now_ts is None else now_ts
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM predictions "
                "WHERE graded = 0 AND label_source = 'pending' AND grade_due_ts IS NOT NULL "
                "AND grade_due_ts <= ? ORDER BY grade_due_ts ASC",
                (now_ts,),
            )
            rows = cur.fetchall()
        return [self._pred_row(r) for r in rows]

    def mark_graded(self, prediction_id: str, label_source: str = "auto") -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE predictions SET graded = 1, label_source = ? WHERE id = ?",
                (label_source, prediction_id),
            )
            self.conn.commit()

    def claim_grading(self, prediction_id: str, label_source: str) -> bool:
        """Atomically claim a still-pending prediction for grading (A8).

        The auto-grader (worker thread) and a Telegram manual callback (event
        loop) can race on the same pending row; both would read 'pending' and
        double-apply rewards. This compare-and-swap makes exactly one caller
        the grader — the loser must re-read and take the correction path.
        """
        with self._lock:
            cur = self.conn.execute(
                "UPDATE predictions SET graded = 1, label_source = ? "
                "WHERE id = ? AND label_source = 'pending'",
                (label_source, prediction_id))
            self.conn.commit()
            return cur.rowcount == 1

    def _pred_row(self, r: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if r is None:
            return None
        d = dict(r)
        for c in _PRED_JSON_COLS:
            d[c] = _loads(d.get(c))
        return d

    # ------------------------------------------------------------------ #
    # Outcomes & rewards
    # ------------------------------------------------------------------ #
    def record_outcome(self, prediction_id: str, realized_return: Optional[float],
                       realized_label: str, threshold: float, horizon_k: int,
                       source: str = "auto", graded_ts: Optional[float] = None,
                       label_tb: Optional[str] = None,
                       barrier_hit_idx: Optional[int] = None,
                       exit_price: Optional[float] = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO outcomes "
                "(prediction_id, realized_return, realized_label, threshold, horizon_k, graded_ts, source, "
                "label_tb, barrier_hit_idx, exit_price) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (prediction_id, realized_return, realized_label, threshold, horizon_k,
                 graded_ts if graded_ts is not None else time.time(), source,
                 label_tb, barrier_hit_idx, exit_price),
            )
            self.conn.commit()

    def record_reward(self, prediction_id: str, agent: str, predicted_action: Optional[str],
                      reward: float, source: str = "auto", applied_ts: Optional[float] = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO rewards (prediction_id, agent, predicted_action, reward, applied_ts, source) "
                "VALUES (?,?,?,?,?,?)",
                (prediction_id, agent, predicted_action, float(reward),
                 applied_ts if applied_ts is not None else time.time(), source),
            )
            self.conn.commit()

    def rewards_for(self, prediction_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM rewards WHERE prediction_id = ?", (prediction_id,))
            return [dict(r) for r in cur.fetchall()]

    def get_outcome(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM outcomes WHERE prediction_id = ?", (prediction_id,))
            r = cur.fetchone()
        return dict(r) if r else None

    def training_rows(self, min_graded_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Graded predictions joined with their outcomes, oldest first — the
        nightly meta-label/calibration training set. JSON columns parsed."""
        q = ("SELECT p.*, o.realized_return, o.realized_label, o.label_tb, "
             "o.barrier_hit_idx, o.exit_price, o.graded_ts, o.source AS outcome_source "
             "FROM predictions p JOIN outcomes o ON o.prediction_id = p.id "
             "WHERE p.graded = 1")
        args: tuple = ()
        if min_graded_ts is not None:
            q += " AND o.graded_ts >= ?"
            args = (min_graded_ts,)
        q += " ORDER BY p.created_ts ASC"
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for c in _PRED_JSON_COLS:
                d[c] = _loads(d.get(c))
            out.append(d)
        return out

    # ------------------------------------------------------------------ #
    # Sessions (Telegram feedback lifecycle)
    # ------------------------------------------------------------------ #
    def create_session(self, *, pair: str, tf: str, prediction_id: Optional[str] = None,
                       customer_chat_id: Optional[int] = None, customer_msg_id: Optional[int] = None,
                       dev_chat_id: Optional[int] = None, dev_msg_id: Optional[int] = None,
                       session_id: Optional[str] = None, created_ts: Optional[float] = None) -> str:
        sid = session_id or uuid.uuid4().hex
        with self._lock:
            self.conn.execute(
                "INSERT INTO sessions (id, pair, tf, prediction_id, created_ts, active, "
                "customer_chat_id, customer_msg_id, dev_chat_id, dev_msg_id) "
                "VALUES (?,?,?,?,?,1,?,?,?,?)",
                (sid, pair, tf, prediction_id, created_ts if created_ts is not None else time.time(),
                 customer_chat_id, customer_msg_id, dev_chat_id, dev_msg_id),
            )
            self.conn.commit()
        return sid

    def link_session_prediction(self, session_id: str, prediction_id: str) -> None:
        """Back-fill sessions.prediction_id once the prediction row exists.

        The broadcaster creates the session BEFORE cycle records the prediction
        (the keyboard needs a session id), so the link can only be written here.
        Without it, manual REWARD buttons resolve to unknown_prediction.
        """
        with self._lock:
            self.conn.execute("UPDATE sessions SET prediction_id = ? WHERE id = ?",
                              (prediction_id, session_id))
            self.conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            r = cur.fetchone()
        return dict(r) if r else None

    def get_active_session(self, pair: str, tf: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM sessions WHERE pair = ? AND tf = ? AND active = 1 "
                "ORDER BY created_ts DESC LIMIT 1", (pair, tf))
            r = cur.fetchone()
        return dict(r) if r else None

    def supersede_active(self, pair: str, tf: str, new_session_id: str) -> Optional[str]:
        """Deactivate every active session for (pair, tf) EXCEPT new_session_id.

        Robust to call order (works whether the new session is created before or
        after this call) and to multiple stale actives. Returns the most recent
        superseded session id, or None if there was nothing to supersede.
        """
        with self._lock:
            cur = self.conn.execute(
                "SELECT id FROM sessions WHERE pair = ? AND tf = ? AND active = 1 AND id != ? "
                "ORDER BY created_ts DESC", (pair, tf, new_session_id))
            prev_ids = [r["id"] for r in cur.fetchall()]
            if prev_ids:
                self.conn.execute(
                    "UPDATE sessions SET active = 0, superseded_by = ? "
                    "WHERE pair = ? AND tf = ? AND active = 1 AND id != ?",
                    (new_session_id, pair, tf, new_session_id))
                self.conn.commit()
        return prev_ids[0] if prev_ids else None

    def deactivate_session(self, session_id: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE sessions SET active = 0 WHERE id = ?", (session_id,))
            self.conn.commit()

    def set_session_true_outcome(self, session_id: str, true_outcome: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE sessions SET true_outcome = ? WHERE id = ?",
                              (true_outcome, session_id))
            self.conn.commit()

    def gc_sessions(self, cutoff_ts: float) -> List[Dict[str, Any]]:
        """Deactivate active sessions older than cutoff_ts; return what was closed."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM sessions WHERE active = 1 AND created_ts < ?", (cutoff_ts,))
            stale = [dict(r) for r in cur.fetchall()]
            self.conn.execute(
                "UPDATE sessions SET active = 0 WHERE active = 1 AND created_ts < ?", (cutoff_ts,))
            self.conn.commit()
        return stale

    # ------------------------------------------------------------------ #
    # Macro snapshots (BTC dominance / fear&greed history for ROC features)
    # ------------------------------------------------------------------ #
    def add_macro_snapshot(self, ts: float, btc_dominance: Optional[float],
                           fear_greed: Optional[float]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO macro_snapshots (ts, btc_dominance, fear_greed) "
                "VALUES (?,?,?)", (float(ts), btc_dominance, fear_greed))
            self.conn.commit()

    def macro_snapshot_before(self, ts: float) -> Optional[Dict[str, Any]]:
        """Latest snapshot at or before ``ts`` (for ~24h rate-of-change)."""
        with self._lock:
            r = self.conn.execute(
                "SELECT * FROM macro_snapshots WHERE ts <= ? ORDER BY ts DESC LIMIT 1",
                (float(ts),)).fetchone()
        return dict(r) if r else None

    # ------------------------------------------------------------------ #
    # News items (RAG corpus)
    # ------------------------------------------------------------------ #
    def has_news_item(self, item_id: str) -> bool:
        with self._lock:
            return self.conn.execute("SELECT 1 FROM news_items WHERE id = ?", (item_id,)).fetchone() is not None

    def add_news_item(self, *, item_id: str, source: str, title: str, body: str, url: str,
                      published_ts: Optional[float], assets: List[str], embedding: Optional[bytes] = None,
                      sentiment_cached: Optional[str] = None, ingested_ts: Optional[float] = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO news_items "
                "(id, source, title, body, url, published_ts, ingested_ts, assets, embedding, sentiment_cached) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (item_id, source, title, body, url, published_ts,
                 ingested_ts if ingested_ts is not None else time.time(),
                 _dumps(assets), embedding, sentiment_cached))
            self.conn.commit()

    def news_embeddings(self, since_ts: Optional[float] = None):
        """[(id, embedding_bytes), ...] for items that have an embedding."""
        q = "SELECT id, embedding FROM news_items WHERE embedding IS NOT NULL"
        args: tuple = ()
        if since_ts is not None:
            q += " AND published_ts >= ?"
            args = (since_ts,)
        with self._lock:
            return [(r["id"], r["embedding"]) for r in self.conn.execute(q, args).fetchall()]

    def get_news_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self.conn.execute("SELECT * FROM news_items WHERE id = ?", (item_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["assets"] = _loads(d.get("assets"))
        return d

    def recent_news(self, since_ts: Optional[float] = None,
                    limit: int = 10) -> List[Dict[str, Any]]:
        """Most recent headlines market-wide (any asset) — grounds the shared
        overall scan (correctness v3, A4)."""
        q = "SELECT * FROM news_items"
        args: list = []
        if since_ts is not None:
            q += " WHERE published_ts >= ?"
            args.append(since_ts)
        q += " ORDER BY published_ts DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(q, tuple(args)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["assets"] = _loads(d.get("assets"))
            out.append(d)
        return out

    def recent_news_for_asset(self, asset: str, since_ts: Optional[float] = None,
                              limit: int = 5) -> List[Dict[str, Any]]:
        like = f'%"{asset.upper()}"%'
        q = "SELECT * FROM news_items WHERE assets LIKE ?"
        args: list = [like]
        if since_ts is not None:
            q += " AND published_ts >= ?"
            args.append(since_ts)
        q += " ORDER BY published_ts DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(q, tuple(args)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["assets"] = _loads(d.get("assets"))
            out.append(d)
        return out

    def set_sentiment_cached(self, item_id: str, sentiment: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE news_items SET sentiment_cached = ? WHERE id = ?",
                              (sentiment, item_id))
            self.conn.commit()

    def cached_sentiment_for_asset(self, asset: str) -> Optional[str]:
        like = f'%"{asset.upper()}"%'
        with self._lock:
            r = self.conn.execute(
                "SELECT sentiment_cached FROM news_items WHERE assets LIKE ? AND sentiment_cached IS NOT NULL "
                "ORDER BY published_ts DESC LIMIT 1", (like,)).fetchone()
        return r["sentiment_cached"] if r else None


# --------------------------------------------------------------------------- #
# Process-wide default store (tests inject their own with set_store()).
# --------------------------------------------------------------------------- #
_STORE: Optional[Store] = None
_STORE_LOCK = threading.Lock()


def get_store() -> Store:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = Store()
        return _STORE


def set_store(store: Optional[Store]) -> None:
    global _STORE
    with _STORE_LOCK:
        _STORE = store
