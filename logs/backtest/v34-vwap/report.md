# Backtest report — v34-vwap

- pairs: 12 | tfs: ['1h', '4h', '1d'] | range: 2024-07-01 → 2026-07-04 | gate: v2
- total emissions: 27954
- caveat: confidence-gate path uses indicator-only confidence (news/research not backtestable); NWE/trend paths exact.

## Per-group metrics (tf | regime | trigger)
| group | n | tp | sl | t/o | TB precision | 95% CI | expectancy R | fixed hit |
|---|---|---|---|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 191 | 10 | 12 | 168 | 45.5% | [26.9%, 65.3%] | 0.086 | 16.8% |
| 1d|ranging|conf_over_80 | 487 | 20 | 68 | 398 | 22.7% | [15.2%, 32.5%] | -0.040 | 13.2% |
| 1d|trend_down|conf_over_80 | 508 | 28 | 40 | 440 | 41.2% | [30.3%, 53.0%] | 0.057 | 24.6% |
| 1d|trend_down|trend_continuation | 151 | 15 | 29 | 107 | 34.1% | [21.9%, 48.9%] | 0.083 | 32.5% |
| 1d|trend_up|conf_over_80 | 89 | 5 | 18 | 66 | 21.7% | [9.7%, 41.9%] | -0.259 | 7.9% |
| 1d|trend_up|trend_continuation | 83 | 6 | 20 | 57 | 23.1% | [11.0%, 42.1%] | -0.146 | 8.4% |
| 1h|mixed|nwe_mixed | 340 | 66 | 151 | 123 | 30.4% | [24.7%, 36.8%] | -0.019 | 43.5% |
| 1h|ranging|nwe_ranging | 513 | 98 | 192 | 223 | 33.8% | [28.6%, 39.4%] | 0.046 | 46.2% |
| 4h|mixed|conf_over_80 | 3616 | 548 | 844 | 2221 | 39.4% | [36.8%, 42.0%] | 0.025 | 25.2% |
| 4h|ranging|conf_over_80 | 10882 | 1401 | 2547 | 6932 | 35.5% | [34.0%, 37.0%] | 0.015 | 25.4% |
| 4h|trend_down|conf_over_80 | 4367 | 593 | 804 | 2970 | 42.4% | [39.9%, 45.1%] | 0.037 | 28.3% |
| 4h|trend_down|trend_continuation | 1707 | 337 | 467 | 903 | 41.9% | [38.6%, 45.4%] | 0.017 | 28.6% |
| 4h|trend_up|conf_over_80 | 3227 | 483 | 808 | 1929 | 37.4% | [34.8%, 40.1%] | 0.033 | 28.9% |
| 4h|trend_up|trend_continuation | 1793 | 365 | 552 | 872 | 39.8% | [36.7%, 43.0%] | 0.037 | 30.2% |

## Gate funnel
| outcome | bars |
|---|---|
| suppressed:no_trigger | 220872 |
| conf_over_80 | 25085 |
| trend_continuation | 3734 |
| suppressed:low_volume | 1773 |
| suppressed:counter_trend_conf | 1530 |
| suppressed:no_brain_agreement | 548 |
| nwe_ranging | 513 |
| nwe_mixed | 340 |
| suppressed:nwe_higher_tf_disabled | 169 |
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
| 1h|mixed|nwe_mixed | 0.8% | 0.017 | -33 | no |
| 1h|ranging|nwe_ranging | 0.0% | 0.000 | 0 | no |
| 4h|mixed|conf_over_80 | 0.5% | 0.006 | 18 | no |
| 4h|ranging|conf_over_80 | 0.9% | 0.007 | -437 | no |
| 4h|trend_down|conf_over_80 | 0.1% | 0.000 | 24 | no |
| 4h|trend_down|trend_continuation | 0.0% | 0.000 | 0 | no |
| 4h|trend_up|conf_over_80 | 0.1% | 0.001 | 15 | no |
| 4h|trend_up|trend_continuation | 0.0% | 0.000 | 0 | no |
