# Backtest report — v34-cci

- pairs: 12 | tfs: ['1h', '4h', '1d'] | range: 2024-07-01 → 2026-07-04 | gate: v2
- total emissions: 28540
- caveat: confidence-gate path uses indicator-only confidence (news/research not backtestable); NWE/trend paths exact.

## Per-group metrics (tf | regime | trigger)
| group | n | tp | sl | t/o | TB precision | 95% CI | expectancy R | fixed hit |
|---|---|---|---|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 196 | 10 | 14 | 171 | 41.7% | [24.5%, 61.2%] | 0.076 | 16.9% |
| 1d|ranging|conf_over_80 | 491 | 18 | 72 | 400 | 20.0% | [13.0%, 29.4%] | -0.054 | 12.9% |
| 1d|trend_down|conf_over_80 | 501 | 28 | 37 | 436 | 43.1% | [31.8%, 55.2%] | 0.064 | 25.0% |
| 1d|trend_down|trend_continuation | 151 | 15 | 29 | 107 | 34.1% | [21.9%, 48.9%] | 0.083 | 32.5% |
| 1d|trend_up|conf_over_80 | 86 | 4 | 18 | 64 | 18.2% | [7.3%, 38.5%] | -0.278 | 7.0% |
| 1d|trend_up|trend_continuation | 83 | 6 | 20 | 57 | 23.1% | [11.0%, 42.1%] | -0.146 | 8.4% |
| 1h|mixed|nwe_mixed | 383 | 74 | 169 | 140 | 30.5% | [25.0%, 36.5%] | -0.026 | 42.0% |
| 1h|ranging|nwe_ranging | 513 | 98 | 192 | 223 | 33.8% | [28.6%, 39.4%] | 0.046 | 46.2% |
| 4h|mixed|conf_over_80 | 3636 | 545 | 853 | 2235 | 39.0% | [36.5%, 41.6%] | 0.022 | 25.1% |
| 4h|ranging|conf_over_80 | 11488 | 1440 | 2788 | 7258 | 34.1% | [32.6%, 35.5%] | 0.006 | 25.4% |
| 4h|trend_down|conf_over_80 | 4317 | 584 | 799 | 2934 | 42.2% | [39.6%, 44.8%] | 0.036 | 28.4% |
| 4h|trend_down|trend_continuation | 1707 | 337 | 467 | 903 | 41.9% | [38.6%, 45.4%] | 0.017 | 28.6% |
| 4h|trend_up|conf_over_80 | 3195 | 477 | 800 | 1911 | 37.4% | [34.7%, 40.0%] | 0.033 | 29.0% |
| 4h|trend_up|trend_continuation | 1793 | 365 | 552 | 872 | 39.8% | [36.7%, 43.0%] | 0.037 | 30.2% |

## Gate funnel
| outcome | bars |
|---|---|
| suppressed:no_trigger | 222370 |
| conf_over_80 | 24733 |
| trend_continuation | 3734 |
| suppressed:low_volume | 1773 |
| nwe_ranging | 513 |
| suppressed:no_brain_agreement | 505 |
| suppressed:counter_trend_conf | 398 |
| nwe_mixed | 383 |
| suppressed:nwe_higher_tf_disabled | 155 |
| suppressed:counter_trend_no_flip | 108 |
| suppressed:reversal_disabled | 40 |

## vs baseline
| group | ΔTB precision | Δexpectancy | Δn | sig@95% |
|---|---|---|---|---|
| 1d|mixed|conf_over_80 | -3.8% | -0.010 | 5 | no |
| 1d|ranging|conf_over_80 | -2.7% | -0.014 | 4 | no |
| 1d|trend_down|conf_over_80 | 1.9% | 0.006 | -7 | no |
| 1d|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 1d|trend_up|conf_over_80 | -3.6% | -0.019 | -3 | no |
| 1d|trend_up|trend_continuation | 0.0% | 0.000 | 0 | no |
| 1h|mixed|nwe_mixed | 0.8% | 0.011 | 10 | no |
| 1h|ranging|nwe_ranging | 0.0% | 0.000 | 0 | no |
| 4h|mixed|conf_over_80 | 0.1% | 0.002 | 38 | no |
| 4h|ranging|conf_over_80 | -0.5% | -0.002 | 169 | no |
| 4h|trend_down|conf_over_80 | -0.1% | -0.000 | -26 | no |
| 4h|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 4h|trend_up|conf_over_80 | 0.0% | 0.000 | -17 | no |
| 4h|trend_up|trend_continuation | 0.0% | 0.000 | 0 | no |
