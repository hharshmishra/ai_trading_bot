# v3.4 confluence-vote evidence (2026-07-05)

Six candidate type-2 votes, each A/B'd against a shared baseline on the exact
production replay path. **Promotion rule** (documented in README): pooled
tb_precision delta > 0 **and** two-proportion z significant at 95% **and**
pooled expectancy_r not degraded.

**Setup** — 12-pair evidence set (BTC ETH SOL BNB XRP ADA DOGE AVAX LINK NEAR
ARB GALA), 1h/4h/1d, 2024-07-01 → 2026-07-04 (pinned end ⇒ identical bars for
every run), gate v2, window 500. Baseline: 28,349 graded emissions,
tb_precision 14.14%, expectancy +0.019R. Reports: `logs/backtest/v34-*/`.

## Verdicts

| vote | n | tb_prec | Δprec | z | expR | ΔexpR | sig groups ± | verdict |
|---|---|---|---|---|---|---|---|---|
| rsi30 | 28,330 | 14.14% | +0.01pp | 0.02 | +0.0193 | +0.0002 | 0 / 0 | ❌ stay off |
| mfi | 28,348 | 14.15% | +0.01pp | 0.05 | +0.0191 | +0.0000 | 0 / 0 | ❌ stay off |
| cci | 28,522 | 14.03% | −0.11pp | −0.38 | +0.0183 | −0.0008 | 0 / 0 | ❌ stay off |
| vwap | 27,936 | 14.23% | +0.09pp | 0.31 | +0.0230 | +0.0039 | 0 / 0 | ❌ stay off |
| fib | 28,100 | 14.16% | +0.02pp | 0.08 | +0.0195 | +0.0004 | 0 / 0 | ❌ stay off |
| ichimoku | 28,533 | 14.12% | −0.02pp | −0.07 | +0.0196 | +0.0005 | 0 / 0 | ❌ stay off |

## Reading

- **No candidate cleared the bar.** Every pooled delta is inside noise; no
  `tf|regime|reason` group moved significantly in either direction.
- **Fib golden pocket**: Δ +0.02pp, z = 0.08 — matches the published evidence
  (bounce probability at fib levels ≈ non-fib levels). It stays what the code
  makes it: an optional confluence vote, never a trigger.
- **VWAP** is the only candidate with a visibly positive expectancy delta
  (+0.0039R) — still not significant. Worth re-testing after more emissions
  accumulate or with a crossing (event) variant.
- **Why the needle barely moves**: one extra ±1 vote inside a 7-rule tally in
  one of two blended heads changes the final action on only ~0.1–2% of bars
  (see n deltas) — the A/B mostly compares identical decisions. The per-rule
  credibility learner (`T2_RULE_LEARNING`) is the mechanism that can still
  extract value live: it reweights each rule on actual graded outcomes
  instead of a fixed ±1.

## Decision

All six stay behind `T2_EXTRA_VOTES` (default off). `T2_RULE_LEARNING` is the
recommended go-live enablement instead — it lets the live reward stream decide
per-rule weights, which is finer-grained than any all-on/all-off vote flag.
Re-run this suite after go-live data accrues (`logs/backtest/v34-suite.sh`).
