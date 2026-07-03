# Backtest report — gate-v2

- pairs: 12 | tfs: ['1h', '4h', '1d'] | range: 2024-07-01 → now | gate: v1
- total emissions: 30969
- caveat: confidence-gate path uses indicator-only confidence (news/research not backtestable); NWE/trend paths exact.

## Per-group metrics (tf | regime | trigger)
| group | n | tp | sl | t/o | TB precision | 95% CI | expectancy R | fixed hit |
|---|---|---|---|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 190 | 10 | 12 | 167 | 45.5% | [26.9%, 65.3%] | 0.088 | 16.9% |
| 1d|ranging|conf_over_80 | 486 | 20 | 68 | 397 | 22.7% | [15.2%, 32.5%] | -0.041 | 13.2% |
| 1d|trend_down|conf_over_80 | 718 | 44 | 71 | 603 | 38.3% | [29.9%, 47.4%] | 0.056 | 25.5% |
| 1d|trend_up|conf_over_80 | 170 | 11 | 33 | 125 | 25.0% | [14.6%, 39.4%] | -0.178 | 11.8% |
| 1h|mixed|nwe_direct | 1488 | 241 | 587 | 660 | 29.1% | [26.1%, 32.3%] | -0.034 | 41.0% |
| 1h|ranging|nwe_direct | 760 | 116 | 271 | 373 | 30.0% | [25.6%, 34.7%] | 0.017 | 45.0% |
| 4h|mixed|conf_over_80 | 3402 | 507 | 798 | 2095 | 38.9% | [36.2%, 41.5%] | 0.020 | 24.7% |
| 4h|mixed|nwe_direct | 274 | 31 | 91 | 152 | 25.4% | [18.5%, 33.8%] | -0.001 | 35.8% |
| 4h|ranging|conf_over_80 | 11207 | 1416 | 2707 | 7078 | 34.3% | [32.9%, 35.8%] | 0.007 | 25.3% |
| 4h|ranging|nwe_direct | 192 | 12 | 81 | 99 | 12.9% | [7.5%, 21.2%] | -0.243 | 25.5% |
| 4h|trend_down|conf_over_80 | 6718 | 1008 | 1386 | 4324 | 42.1% | [40.1%, 44.1%] | 0.034 | 28.4% |
| 4h|trend_up|conf_over_80 | 5364 | 874 | 1456 | 3025 | 37.5% | [35.6%, 39.5%] | 0.023 | 28.7% |

## Gate funnel
| outcome | bars |
|---|---|
| suppressed:no_trigger | 222021 |
| conf_over_80 | 29695 |
| nwe_direct | 2714 |

## vs baseline
| group | ΔTB precision | Δexpectancy | Δn | sig@95% |
|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 0.0% | -0.003 | 1 | no |
| 1d|ranging|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|trend_down|conf_over_80 | 8.4% | -0.008 | -734 | no |
| 1d|trend_up|conf_over_80 | -0.3% | -0.088 | -305 | no |
| 1h|mixed|nwe_direct | 0.0% | 0.000 | 0 | no |
| 1h|ranging|nwe_direct | 0.0% | 0.000 | 0 | no |
| 4h|mixed|conf_over_80 | 0.0% | -0.000 | 2 | no |
| 4h|mixed|nwe_direct | 0.0% | 0.000 | 0 | no |
| 4h|ranging|conf_over_80 | 0.0% | -0.000 | 3 | no |
| 4h|ranging|nwe_direct | 0.0% | 0.000 | 0 | no |
| 4h|trend_down|conf_over_80 | 7.4% | 0.019 | -8226 | yes |
| 4h|trend_up|conf_over_80 | 4.8% | 0.026 | -5213 | yes |
