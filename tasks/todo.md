# BitReinforceX v2 — Accuracy Upgrade (plan: ~/.claude/plans/my-input-is-there-linked-dusk.md)

Branch: `feature/accuracy-upgrade-v2`. Rule: 22 existing tests stay green before every commit.

STATUS 2026-07-03 ~04:50: Phases 1-5 code committed (25aebba, 6e2bde6, 690e6b9, 6b99a4b,
4d8250f fix, perf commit, README), 107 tests green. Forced-cycle integration VERIFIED
(deriv vote moves final conf; regime/barriers/meta fields persisted; found+fixed
cycle passing legacy 3-agent tuple).
decide() optimized 487->258ms (supertrend numpy port parity-locked, NWE W-cache, alpha_trend arrays).
Baseline RESCOPED to 12 representative pairs (48-pair exact = ~25h; groups pool across pairs).
12-pair baseline RUNNING (logs/backtest/baseline-run.log), ETA ~2.5h.

NEXT (in order):
1. Baseline report archived -> logs/backtest/baseline/
2. A/B: GATE_V2_ENABLED=true env + --gate v2 --label gate-v2 --baseline .../baseline/report.json
   (env flag REQUIRED so decide() regime-conditions the trigger set)
3. Ship check: nwe/trend groups per tf x regime — precision up, CI-separated, 1h volume 0.5-1.5x
4. Deck: subagent writes docs/accuracy-upgrade.html (aesthetic ref docs/system-design.html),
   then scripts/build_deck_data.py injects report JSONs between /*__DECK_DATA_START__*/ markers
5. Full pytest + merge feature/accuracy-upgrade-v2 -> main + push

Smoke evidence: 1h NWE TB-precision 33% (62tp/125sl) on 3 majors = band-walk failure, quantified.
Caveat: backtest conf_over_80 path overfires on 4h/1d (indicator-only conf proxy vs live
brain conf) — ship decisions use NWE/trend trigger groups; caveat printed in every report.

## Phase 1 — Backtest harness + baseline
- [ ] `config.py` (flags + thresholds, env-driven)
- [ ] `grading/barriers.py` (barrier_prices, triple_barrier) — TDD
- [ ] Vectorize NWE repaint branch in `agents/custom_indicators.py` (+ `_nwe_repaint_reference`, allclose parity test)
- [ ] `agents/indicator_agent.py`: `log=True` kwarg on decide()
- [ ] `backtest/data.py` (ccxt pagination → CSV cache `data/history/`)
- [ ] `backtest/engine.py` (per-bar window replay via production decide + gate)
- [ ] `backtest/metrics.py` (summarize/compare, binomial CI)
- [ ] `backtest/sweep.py` (grid + walk-forward 70/30)
- [ ] `backtest/report.py` (json + md)
- [ ] `scripts/run_backtest.py` CLI
- [ ] tests: test_barriers.py, test_backtest_harness.py
- [ ] Smoke run 3 pairs → full baseline run 48 pairs archived `logs/backtest/baseline/`
- [ ] Commit Phase 1

## Phase 2 — Regime + trend triggers + Gate v2 + migration
- [ ] `agents/regime_agent.py` (classify_regime, hysteresis) — TDD
- [ ] Trend triggers in custom_indicators (supertrend_flip/donchian_breakout/squeeze_release)
- [ ] indicator_agent regime integration (flag-gated)
- [ ] signals.py `should_emit_signal_v2` truth table
- [ ] persistence `_migrate()` all new columns
- [ ] cycle.py persist regime/atr/tp/sl/trigger
- [ ] telegram `/regime`
- [ ] tests + backtest A/B vs baseline (ship criteria)
- [ ] Commit Phase 2

## Phase 3 — Triple-barrier grading
- [ ] grader.py TB path + reward_for_v2 + outcomes TB columns
- [ ] brain dump TB line
- [ ] tests/test_grader_tb.py (incl. legacy-row regression + manual override)
- [ ] Commit Phase 3

## Phase 4 — Derivatives + macro + ingestion
- [ ] utils/derivatives_fetcher.py (binanceusdm, TTL cache, has_futures)
- [ ] agents/derivatives_agent.py (8-dim feats, DerivativesRL bandit)
- [ ] brain 4th voter (AGENT_NAMES, score 1.5)
- [ ] utils/macro_fetcher.py (F&G + dominance) → market_context fields
- [ ] ingestion: drop CryptoPanic, +5 RSS feeds, wire hourly ingest_all
- [ ] grader derivatives rewards; telegram /derivs + deriv_note
- [ ] tests + rate-limit audit
- [ ] Commit Phase 4

## Phase 5 — Meta-labeling + calibration (nightly)
- [ ] jobs/features.py shared builder
- [ ] jobs/nightly.py (meta model, isotonic JSON knots, dev summary)
- [ ] cycle stamps meta_p/calibrated_conf; telegram line
- [ ] persistence.training_rows(); requirements sklearn+joblib; preflight
- [ ] scripts/run_training.py; tests (incl. train/serve skew test)
- [ ] Commit Phase 5

## Phase 6 — HTML evidence deck
- [ ] scripts/build_deck_data.py
- [ ] docs/accuracy-upgrade.html (self-contained, evidence from report JSONs)
- [ ] Commit Phase 6

## Wrap-up
- [ ] Full pytest suite green; end-to-end forced cycle check
- [ ] Merge feature branch → main, push
- [ ] Review section below

## Review

### v2 accuracy upgrade (2026-07-03, merged 734ec4c)
Shipped: backtest harness, regime-gated gate v2 (+2 evidence amendments), TB grading,
DerivativesAgent, nightly meta/calibration, evidence deck. 111 tests.

### v3 correctness + agent enhancements (2026-07-04, branch feature/correctness-v3)
11 audited defects fixed (5 critical): UTC schedule alignment, closed-candle
discipline, NWE event mode, news RAG wiring + hallucination guard, dominance
logic revival, brain deadzone-v2 shadow, grading claim race, hygiene batch.
Enhancements: FRED/stooq SPX-DXY trends, money-flow v2, ecosystem auto-refresh,
event-typed news (RL 5->10 logit-preserving migration), tiered headlines,
model2vec embedder option, RSI/OBV divergences, EB direct confidences,
D3 redundancy report, Lorentzian experiment.
Evidence: regression guard reproduced amended baseline (max drift 0.14pp);
NWE event mode SHIPPED (-25-36% duplicate 1h emissions, precision +0.8-1.1pp);
divergences HELD (deltas ~0, fire too rarely); EB conf self-arms on live data.
Universe: preflight freshness check found LRC dead 93d + MKR dead 291d
(46-pair universe silently) -> LTC/TRX in. 167 tests green.

## 2026-07-04 — deck-audit fixes (found by dev-deck verification agent)
- [x] sessions.prediction_id back-filled after record_prediction (cycle.py) —
      manual REWARD buttons were silent unknown_prediction no-ops since Phase 4
- [x] NewsAgent ctor NewsRL() (was pinned n_features=5; 10-dim features were
      truncated, 5->10 policy migration never ran live)
- [x] tests/conftest.py: autouse isolation of ALL logs/ artifact paths
- [x] 169 tests green; logs/ clean after full suite
