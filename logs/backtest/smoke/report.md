# Backtest report — smoke

- pairs: 3 | tfs: ['1h'] | range: 2026-05-01 → now | gate: v1
- total emissions: 336
- caveat: confidence-gate path uses indicator-only confidence (news/research not backtestable); NWE/trend paths exact.

## Per-group metrics (tf | regime | trigger)
| group | n | tp | sl | t/o | TB precision | 95% CI | expectancy R | fixed hit |
|---|---|---|---|---|---|---|---|---|
| 1h|all|nwe_direct | 336 | 62 | 125 | 149 | 33.2% | [26.8%, 40.2%] | 0.080 | 42.6% |

## Gate funnel
| outcome | bars |
|---|---|
| suppressed:no_trigger | 2694 |
| nwe_direct | 336 |
