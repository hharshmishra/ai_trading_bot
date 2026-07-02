# BitReinforceX v2 — Accuracy Upgrade (plan: ~/.claude/plans/my-input-is-there-linked-dusk.md)

Branch: `feature/accuracy-upgrade-v2`. Rule: 22 existing tests stay green before every commit.

STATUS 2026-07-03: Phases 1-5 code committed (25aebba, 6e2bde6, 690e6b9, 6b99a4b), 107 tests green.
Baseline 48-pair backtest RUNNING in background (logs/backtest/baseline-run.log).
Remaining: baseline archive -> gate v2 A/B -> deck -> merge.
Smoke evidence: 1h NWE TB-precision 33% (62tp/125sl) on 3 majors = the band-walk failure, quantified.
Caveat noted: backtest conf_over_80 path overfires on 4h/1d (indicator-only conf proxy vs live brain conf) — ship decisions use NWE/trend trigger groups.

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
(fill at end)
