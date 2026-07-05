# Backtest report — v34-ichimoku

- pairs: 12 | tfs: ['1h', '4h', '1d'] | range: 2024-07-01 → 2026-07-04 | gate: v2
- total emissions: 28551
- caveat: confidence-gate path uses indicator-only confidence (news/research not backtestable); NWE/trend paths exact.

## Per-group metrics (tf | regime | trigger)
| group | n | tp | sl | t/o | TB precision | 95% CI | expectancy R | fixed hit |
|---|---|---|---|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 195 | 10 | 12 | 172 | 45.5% | [26.9%, 65.3%] | 0.095 | 17.5% |
| 1d|ranging|conf_over_80 | 493 | 17 | 69 | 406 | 19.8% | [12.7%, 29.4%] | -0.049 | 13.0% |
| 1d|trend_down|conf_over_80 | 508 | 28 | 40 | 440 | 41.2% | [30.3%, 53.0%] | 0.057 | 24.6% |
| 1d|trend_down|trend_continuation | 151 | 15 | 29 | 107 | 34.1% | [21.9%, 48.9%] | 0.083 | 32.5% |
| 1d|trend_up|conf_over_80 | 89 | 5 | 18 | 66 | 21.7% | [9.7%, 41.9%] | -0.259 | 7.9% |
| 1d|trend_up|trend_continuation | 83 | 6 | 20 | 57 | 23.1% | [11.0%, 42.1%] | -0.146 | 8.4% |
| 1h|mixed|nwe_mixed | 358 | 68 | 159 | 131 | 30.0% | [24.4%, 36.2%] | -0.030 | 42.5% |
| 1h|ranging|nwe_ranging | 513 | 98 | 192 | 223 | 33.8% | [28.6%, 39.4%] | 0.046 | 46.2% |
| 4h|mixed|conf_over_80 | 3664 | 549 | 862 | 2250 | 38.9% | [36.4%, 41.5%] | 0.021 | 25.0% |
| 4h|ranging|conf_over_80 | 11424 | 1456 | 2775 | 7191 | 34.4% | [33.0%, 35.9%] | 0.009 | 25.4% |
| 4h|trend_down|conf_over_80 | 4346 | 592 | 802 | 2952 | 42.5% | [39.9%, 45.1%] | 0.038 | 28.5% |
| 4h|trend_down|trend_continuation | 1707 | 337 | 467 | 903 | 41.9% | [38.6%, 45.4%] | 0.017 | 28.6% |
| 4h|trend_up|conf_over_80 | 3227 | 482 | 808 | 1930 | 37.4% | [34.8%, 40.0%] | 0.032 | 28.9% |
| 4h|trend_up|trend_continuation | 1793 | 365 | 552 | 872 | 39.8% | [36.7%, 43.0%] | 0.037 | 30.2% |

## Gate funnel
| outcome | bars |
|---|---|
| suppressed:no_trigger | 222032 |
| conf_over_80 | 24993 |
| trend_continuation | 3734 |
| suppressed:low_volume | 1773 |
| suppressed:no_brain_agreement | 530 |
| nwe_ranging | 513 |
| suppressed:counter_trend_conf | 464 |
| nwe_mixed | 358 |
| suppressed:nwe_higher_tf_disabled | 167 |
| suppressed:counter_trend_no_flip | 108 |
| suppressed:reversal_disabled | 40 |

## vs baseline
| group | ΔTB precision | Δexpectancy | Δn | sig@95% |
|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 0.0% | 0.009 | 4 | no |
| 1d|ranging|conf_over_80 | -3.0% | -0.009 | 6 | no |
| 1d|trend_down|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 1d|trend_up|conf_over_80 | 0.0% | 0.000 | 0 | no |
| 1d|trend_up|trend_continuation | 0.0% | 0.000 | 0 | no |
| 1h|mixed|nwe_mixed | 0.3% | 0.006 | -15 | no |
| 1h|ranging|nwe_ranging | 0.0% | -0.000 | 0 | no |
| 4h|mixed|conf_over_80 | 0.0% | 0.002 | 66 | no |
| 4h|ranging|conf_over_80 | -0.1% | 0.000 | 105 | no |
| 4h|trend_down|conf_over_80 | 0.2% | 0.001 | 3 | no |
| 4h|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 4h|trend_up|conf_over_80 | 0.0% | -0.000 | 15 | no |
| 4h|trend_up|trend_continuation | 0.0% | 0.000 | 0 | no |
