# Backtest report — v34-mfi

- pairs: 12 | tfs: ['1h', '4h', '1d'] | range: 2024-07-01 → 2026-07-04 | gate: v2
- total emissions: 28366
- caveat: confidence-gate path uses indicator-only confidence (news/research not backtestable); NWE/trend paths exact.

## Per-group metrics (tf | regime | trigger)
| group | n | tp | sl | t/o | TB precision | 95% CI | expectancy R | fixed hit |
|---|---|---|---|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 190 | 10 | 13 | 166 | 43.5% | [25.6%, 63.2%] | 0.081 | 16.9% |
| 1d|ranging|conf_over_80 | 487 | 20 | 68 | 398 | 22.7% | [15.2%, 32.5%] | -0.036 | 13.6% |
| 1d|trend_down|conf_over_80 | 508 | 28 | 40 | 440 | 41.2% | [30.3%, 53.0%] | 0.057 | 24.6% |
| 1d|trend_down|trend_continuation | 151 | 15 | 29 | 107 | 34.1% | [21.9%, 48.9%] | 0.083 | 32.5% |
| 1d|trend_up|conf_over_80 | 89 | 5 | 18 | 66 | 21.7% | [9.7%, 41.9%] | -0.259 | 7.9% |
| 1d|trend_up|trend_continuation | 83 | 6 | 20 | 57 | 23.1% | [11.0%, 42.1%] | -0.146 | 8.4% |
| 1h|mixed|nwe_mixed | 374 | 71 | 166 | 137 | 30.0% | [24.5%, 36.1%] | -0.032 | 42.0% |
| 1h|ranging|nwe_ranging | 513 | 98 | 192 | 223 | 33.8% | [28.6%, 39.4%] | 0.046 | 46.2% |
| 4h|mixed|conf_over_80 | 3597 | 541 | 852 | 2201 | 38.8% | [36.3%, 41.4%] | 0.019 | 25.1% |
| 4h|ranging|conf_over_80 | 11323 | 1447 | 2738 | 7136 | 34.6% | [33.1%, 36.0%] | 0.009 | 25.5% |
| 4h|trend_down|conf_over_80 | 4339 | 588 | 801 | 2950 | 42.3% | [39.8%, 44.9%] | 0.037 | 28.4% |
| 4h|trend_down|trend_continuation | 1707 | 337 | 467 | 903 | 41.9% | [38.6%, 45.4%] | 0.017 | 28.6% |
| 4h|trend_up|conf_over_80 | 3212 | 481 | 807 | 1917 | 37.3% | [34.7%, 40.0%] | 0.033 | 29.0% |
| 4h|trend_up|trend_continuation | 1793 | 365 | 552 | 872 | 39.8% | [36.7%, 43.0%] | 0.037 | 30.2% |

## Gate funnel
| outcome | bars |
|---|---|
| suppressed:no_trigger | 222188 |
| conf_over_80 | 24824 |
| trend_continuation | 3734 |
| suppressed:low_volume | 1773 |
| suppressed:no_brain_agreement | 514 |
| nwe_ranging | 513 |
| suppressed:counter_trend_conf | 481 |
| nwe_mixed | 374 |
| suppressed:nwe_higher_tf_disabled | 163 |
| suppressed:counter_trend_no_flip | 108 |
| suppressed:reversal_disabled | 40 |

## vs baseline
| group | ΔTB precision | Δexpectancy | Δn | sig@95% |
|---|---|---|---|---|
| 1d|mixed|conf_over_80 | -2.0% | -0.005 | -1 | no |
| 1d|ranging|conf_over_80 | 0.0% | 0.004 | 0 | no |
| 1d|trend_down|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 1d|trend_up|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|trend_up|trend_continuation | 0.0% | 0.000 | 0 | no |
| 1h|mixed|nwe_mixed | 0.3% | 0.004 | 1 | no |
| 1h|ranging|nwe_ranging | 0.0% | 0.000 | 0 | no |
| 4h|mixed|conf_over_80 | -0.0% | -0.001 | -1 | no |
| 4h|ranging|conf_over_80 | 0.0% | -0.000 | 4 | no |
| 4h|trend_down|conf_over_80 | 0.0% | 0.000 | -4 | no |
| 4h|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 4h|trend_up|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 4h|trend_up|trend_continuation | 0.0% | 0.000 | 0 | no |
