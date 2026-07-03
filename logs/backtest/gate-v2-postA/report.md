# Backtest report — gate-v2-postA

- pairs: 12 | tfs: ['1h', '4h', '1d'] | range: 2024-07-01 → now | gate: v2
- total emissions: 28565
- caveat: confidence-gate path uses indicator-only confidence (news/research not backtestable); NWE/trend paths exact.

## Per-group metrics (tf | regime | trigger)
| group | n | tp | sl | t/o | TB precision | 95% CI | expectancy R | fixed hit |
|---|---|---|---|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 190 | 10 | 12 | 167 | 45.5% | [26.9%, 65.3%] | 0.088 | 16.9% |
| 1d|ranging|conf_over_80 | 486 | 20 | 68 | 397 | 22.7% | [15.2%, 32.5%] | -0.041 | 13.2% |
| 1d|trend_down|conf_over_80 | 508 | 28 | 40 | 440 | 41.2% | [30.3%, 53.0%] | 0.057 | 24.6% |
| 1d|trend_down|trend_continuation | 151 | 15 | 29 | 107 | 34.1% | [21.9%, 48.9%] | 0.083 | 32.5% |
| 1d|trend_up|conf_over_80 | 89 | 5 | 18 | 66 | 21.7% | [9.7%, 41.9%] | -0.259 | 7.9% |
| 1d|trend_up|trend_continuation | 83 | 6 | 20 | 56 | 23.1% | [11.0%, 42.1%] | -0.147 | 8.5% |
| 1h|mixed|nwe_mixed | 587 | 97 | 243 | 247 | 28.5% | [24.0%, 33.5%] | -0.042 | 39.9% |
| 1h|ranging|nwe_ranging | 594 | 106 | 215 | 273 | 33.0% | [28.1%, 38.3%] | 0.047 | 46.0% |
| 4h|mixed|conf_over_80 | 3545 | 533 | 834 | 2177 | 39.0% | [36.4%, 41.6%] | 0.021 | 25.1% |
| 4h|ranging|conf_over_80 | 11295 | 1438 | 2732 | 7119 | 34.5% | [33.1%, 35.9%] | 0.008 | 25.4% |
| 4h|trend_down|conf_over_80 | 4343 | 589 | 803 | 2951 | 42.3% | [39.7%, 44.9%] | 0.037 | 28.4% |
| 4h|trend_down|trend_continuation | 1707 | 337 | 467 | 903 | 41.9% | [38.6%, 45.4%] | 0.017 | 28.6% |
| 4h|trend_up|conf_over_80 | 3198 | 477 | 807 | 1906 | 37.1% | [34.5%, 39.8%] | 0.029 | 28.7% |
| 4h|trend_up|trend_continuation | 1789 | 364 | 552 | 870 | 39.7% | [36.6%, 42.9%] | 0.036 | 30.2% |

## Gate funnel
| outcome | bars |
|---|---|
| suppressed:no_trigger | 221394 |
| conf_over_80 | 24735 |
| trend_continuation | 3730 |
| suppressed:low_volume | 1932 |
| suppressed:no_brain_agreement | 673 |
| nwe_ranging | 594 |
| nwe_mixed | 587 |
| suppressed:counter_trend_conf | 487 |
| suppressed:nwe_higher_tf_disabled | 242 |
| suppressed:counter_trend_no_flip | 108 |
| suppressed:reversal_disabled | 40 |

## vs baseline
| group | ΔTB precision | Δexpectancy | Δn | sig@95% |
|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|ranging|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|trend_down|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 1d|trend_up|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|trend_up|trend_continuation | 0.0% | 0.000 | 0 | no |
| 1h|mixed|nwe_mixed | 0.0% | -0.000 | 0 | no |
| 1h|ranging|nwe_ranging | 0.0% | 0.000 | 0 | no |
| 4h|mixed|conf_over_80 | -0.1% | 0.000 | 4 | no |
| 4h|ranging|conf_over_80 | 0.1% | 0.001 | 72 | no |
| 4h|trend_down|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 4h|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 4h|trend_up|conf_over_80 | 0.0% | 0.001 | 7 | no |
| 4h|trend_up|trend_continuation | 0.0% | 0.000 | 3 | no |
