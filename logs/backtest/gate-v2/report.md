# Backtest report — gate-v2

- pairs: 12 | tfs: ['1h', '4h', '1d'] | range: 2024-07-01 → now | gate: v2
- total emissions: 28775
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
| 1d|trend_up|trend_reversal | 1 | 0 | 0 | 1 | — | — | 0.300 | 100.0% |
| 1h|mixed|nwe_mixed | 587 | 97 | 243 | 247 | 28.5% | [24.0%, 33.5%] | -0.042 | 39.9% |
| 1h|ranging|nwe_ranging | 594 | 106 | 215 | 273 | 33.0% | [28.1%, 38.3%] | 0.047 | 46.0% |
| 4h|mixed|conf_over_80 | 3541 | 533 | 832 | 2174 | 39.0% | [36.5%, 41.7%] | 0.021 | 25.1% |
| 4h|mixed|nwe_mixed | 111 | 14 | 37 | 60 | 27.5% | [17.1%, 40.9%] | 0.031 | 43.2% |
| 4h|ranging|conf_over_80 | 11223 | 1418 | 2711 | 7088 | 34.3% | [32.9%, 35.8%] | 0.007 | 25.3% |
| 4h|ranging|nwe_ranging | 145 | 10 | 70 | 65 | 12.5% | [6.9%, 21.5%] | -0.286 | 24.8% |
| 4h|trend_down|conf_over_80 | 4343 | 589 | 803 | 2951 | 42.3% | [39.7%, 44.9%] | 0.037 | 28.4% |
| 4h|trend_down|trend_continuation | 1707 | 337 | 467 | 903 | 41.9% | [38.6%, 45.4%] | 0.017 | 28.6% |
| 4h|trend_down|trend_reversal | 26 | 0 | 5 | 21 | 0.0% | [0.0%, 43.4%] | -0.214 | 15.4% |
| 4h|trend_up|conf_over_80 | 3191 | 477 | 807 | 1899 | 37.1% | [34.5%, 39.8%] | 0.028 | 28.7% |
| 4h|trend_up|trend_continuation | 1786 | 364 | 552 | 869 | 39.7% | [36.6%, 42.9%] | 0.035 | 30.1% |
| 4h|trend_up|trend_reversal | 13 | 0 | 1 | 12 | 0.0% | [0.0%, 79.3%] | 0.115 | 46.2% |

## Gate funnel
| outcome | bars |
|---|---|
| suppressed:no_trigger | 221336 |
| conf_over_80 | 24652 |
| trend_continuation | 3727 |
| suppressed:low_volume | 1983 |
| nwe_ranging | 739 |
| nwe_mixed | 698 |
| suppressed:no_brain_agreement | 678 |
| suppressed:counter_trend_conf | 487 |
| suppressed:counter_trend_no_flip | 108 |
| trend_reversal | 40 |

## vs baseline
| group | ΔTB precision | Δexpectancy | Δn | sig@95% |
|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 0.0% | -0.003 | 1 | no |
| 1d|ranging|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|trend_down|conf_over_80 | 11.3% | -0.006 | -944 | no |
| 1d|trend_up|conf_over_80 | -3.5% | -0.170 | -386 | no |
| 4h|mixed|conf_over_80 | 0.2% | 0.002 | 141 | no |
| 4h|ranging|conf_over_80 | -0.0% | 0.000 | 19 | no |
| 4h|trend_down|conf_over_80 | 7.6% | 0.021 | -10601 | yes |
| 4h|trend_up|conf_over_80 | 4.4% | 0.031 | -7386 | yes |
