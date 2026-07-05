# Backtest report — v34-fib

- pairs: 12 | tfs: ['1h', '4h', '1d'] | range: 2024-07-01 → 2026-07-04 | gate: v2
- total emissions: 28118
- caveat: confidence-gate path uses indicator-only confidence (news/research not backtestable); NWE/trend paths exact.

## Per-group metrics (tf | regime | trigger)
| group | n | tp | sl | t/o | TB precision | 95% CI | expectancy R | fixed hit |
|---|---|---|---|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 190 | 10 | 12 | 167 | 45.5% | [26.9%, 65.3%] | 0.086 | 16.9% |
| 1d|ranging|conf_over_80 | 478 | 20 | 68 | 389 | 22.7% | [15.2%, 32.5%] | -0.038 | 13.6% |
| 1d|trend_down|conf_over_80 | 508 | 28 | 40 | 440 | 41.2% | [30.3%, 53.0%] | 0.057 | 24.6% |
| 1d|trend_down|trend_continuation | 151 | 15 | 29 | 107 | 34.1% | [21.9%, 48.9%] | 0.083 | 32.5% |
| 1d|trend_up|conf_over_80 | 87 | 4 | 18 | 65 | 18.2% | [7.3%, 38.5%] | -0.281 | 6.9% |
| 1d|trend_up|trend_continuation | 83 | 6 | 20 | 57 | 23.1% | [11.0%, 42.1%] | -0.146 | 8.4% |
| 1h|mixed|nwe_mixed | 373 | 70 | 166 | 137 | 29.7% | [24.2%, 35.8%] | -0.037 | 42.1% |
| 1h|ranging|nwe_ranging | 513 | 98 | 192 | 223 | 33.8% | [28.6%, 39.4%] | 0.046 | 46.2% |
| 4h|mixed|conf_over_80 | 3596 | 539 | 851 | 2203 | 38.8% | [36.2%, 41.4%] | 0.018 | 25.0% |
| 4h|ranging|conf_over_80 | 11096 | 1419 | 2676 | 6999 | 34.7% | [33.2%, 36.1%] | 0.010 | 25.5% |
| 4h|trend_down|conf_over_80 | 4338 | 588 | 803 | 2947 | 42.3% | [39.7%, 44.9%] | 0.036 | 28.4% |
| 4h|trend_down|trend_continuation | 1707 | 337 | 467 | 903 | 41.9% | [38.6%, 45.4%] | 0.017 | 28.6% |
| 4h|trend_up|conf_over_80 | 3205 | 480 | 804 | 1914 | 37.4% | [34.8%, 40.1%] | 0.033 | 28.9% |
| 4h|trend_up|trend_continuation | 1793 | 365 | 552 | 872 | 39.8% | [36.7%, 43.0%] | 0.037 | 30.2% |

## Gate funnel
| outcome | bars |
|---|---|
| suppressed:no_trigger | 222042 |
| conf_over_80 | 24804 |
| trend_continuation | 3734 |
| suppressed:low_volume | 1773 |
| suppressed:counter_trend_conf | 647 |
| suppressed:no_brain_agreement | 515 |
| nwe_ranging | 513 |
| nwe_mixed | 373 |
| suppressed:nwe_higher_tf_disabled | 163 |
| suppressed:counter_trend_no_flip | 108 |
| suppressed:reversal_disabled | 40 |

## vs baseline
| group | ΔTB precision | Δexpectancy | Δn | sig@95% |
|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 0.0% | -0.000 | -1 | no |
| 1d|ranging|conf_over_80 | 0.0% | 0.003 | -9 | no |
| 1d|trend_down|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 1d|trend_up|conf_over_80 | -3.6% | -0.022 | -2 | no |
| 1d|trend_up|trend_continuation | 0.0% | 0.000 | 0 | no |
| 1h|mixed|nwe_mixed | 0.0% | 0.000 | 0 | no |
| 1h|ranging|nwe_ranging | 0.0% | 0.000 | 0 | no |
| 4h|mixed|conf_over_80 | -0.1% | -0.001 | -2 | no |
| 4h|ranging|conf_over_80 | 0.1% | 0.001 | -223 | no |
| 4h|trend_down|conf_over_80 | -0.0% | -0.000 | -5 | no |
| 4h|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 4h|trend_up|conf_over_80 | 0.0% | 0.001 | -7 | no |
| 4h|trend_up|trend_continuation | 0.0% | 0.000 | 0 | no |
