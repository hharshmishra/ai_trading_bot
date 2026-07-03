# Backtest report — baseline

- pairs: 12 | tfs: ['1h', '4h', '1d'] | range: 2024-07-01 → now | gate: v1
- total emissions: 65430
- caveat: confidence-gate path uses indicator-only confidence (news/research not backtestable); NWE/trend paths exact.

## Per-group metrics (tf | regime | trigger)
| group | n | tp | sl | t/o | TB precision | 95% CI | expectancy R | fixed hit |
|---|---|---|---|---|---|---|---|---|
| 1d|mixed|conf_over_80 | 189 | 10 | 12 | 166 | 45.5% | [26.9%, 65.3%] | 0.090 | 17.0% |
| 1d|ranging|conf_over_80 | 486 | 20 | 68 | 397 | 22.7% | [15.2%, 32.5%] | -0.041 | 13.2% |
| 1d|trend_down|conf_over_80 | 1452 | 57 | 134 | 1253 | 29.8% | [23.8%, 36.7%] | 0.063 | 24.9% |
| 1d|trend_down|nwe_direct | 53 | 3 | 16 | 34 | 15.8% | [5.5%, 37.6%] | -0.147 | 17.0% |
| 1d|trend_up|conf_over_80 | 475 | 22 | 65 | 386 | 25.3% | [17.3%, 35.3%] | -0.090 | 15.4% |
| 1d|trend_up|nwe_direct | 12 | 3 | 4 | 5 | 42.9% | [15.8%, 75.0%] | 0.141 | 50.0% |
| 1h|mixed|nwe_direct | 1488 | 241 | 587 | 660 | 29.1% | [26.1%, 32.3%] | -0.034 | 41.0% |
| 1h|ranging|nwe_direct | 760 | 116 | 271 | 373 | 30.0% | [25.6%, 34.7%] | 0.017 | 45.0% |
| 1h|trend_down|nwe_direct | 8207 | 1065 | 3049 | 4093 | 25.9% | [24.6%, 27.2%] | -0.033 | 43.0% |
| 1h|trend_up|nwe_direct | 8145 | 1315 | 2903 | 3927 | 31.2% | [29.8%, 32.6%] | -0.001 | 40.3% |
| 4h|mixed|conf_over_80 | 3400 | 507 | 798 | 2093 | 38.9% | [36.2%, 41.5%] | 0.020 | 24.7% |
| 4h|mixed|nwe_direct | 274 | 31 | 91 | 152 | 25.4% | [18.5%, 33.8%] | -0.001 | 35.8% |
| 4h|ranging|conf_over_80 | 11204 | 1416 | 2707 | 7075 | 34.3% | [32.9%, 35.8%] | 0.007 | 25.3% |
| 4h|ranging|nwe_direct | 192 | 12 | 81 | 99 | 12.9% | [7.5%, 21.2%] | -0.243 | 25.5% |
| 4h|trend_down|conf_over_80 | 14944 | 1649 | 3104 | 10191 | 34.7% | [33.4%, 36.1%] | 0.015 | 27.3% |
| 4h|trend_down|nwe_direct | 1630 | 147 | 480 | 1003 | 23.4% | [20.3%, 26.9%] | 0.014 | 42.1% |
| 4h|trend_up|conf_over_80 | 10577 | 1265 | 2601 | 6703 | 32.7% | [31.3%, 34.2%] | -0.003 | 27.4% |
| 4h|trend_up|nwe_direct | 1942 | 243 | 689 | 1006 | 26.1% | [23.4%, 29.0%] | -0.047 | 36.8% |

## Gate funnel
| outcome | bars |
|---|---|
| suppressed:no_trigger | 187369 |
| conf_over_80 | 44245 |
| nwe_direct | 22703 |
