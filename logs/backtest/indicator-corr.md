# Indicator redundancy & standalone win rates — 4h, 4 pairs

## Spearman correlation (|rho| > 0.75 = redundant pair)

| | nwe_state | supertrend_dir | macd_sign | rsi_zone | bb_pos | ma_ribbon | chandelier | alpha_trend | rsi_div | obv_div |
|---|---|---|---|---|---|---|---|---|---|---|
| nwe_state | 1.0 | -0.14 | -0.15 | 0.28 | 0.23 | -0.17 | -0.04 | -0.2 | 0.0 | 0.0 |
| supertrend_dir | -0.14 | 1.0 | 0.15 | -0.34 | -0.28 | 0.65 | 0.04 | 0.51 | 0.0 | -0.02 |
| macd_sign | -0.15 | 0.15 | 1.0 | -0.3 | -0.34 | 0.47 | 0.14 | 0.47 | 0.01 | 0.0 |
| rsi_zone | 0.28 | -0.34 | -0.3 | 1.0 | 0.44 | -0.37 | -0.01 | -0.48 | -0.0 | 0.0 |
| bb_pos | 0.23 | -0.28 | -0.34 | 0.44 | 1.0 | -0.39 | -0.22 | -0.43 | -0.0 | 0.01 |
| ma_ribbon | -0.17 | 0.65 | 0.47 | -0.37 | -0.39 | 1.0 | 0.14 | 0.6 | 0.0 | -0.02 |
| chandelier | -0.04 | 0.04 | 0.14 | -0.01 | -0.22 | 0.14 | 1.0 | 0.05 | 0.0 | 0.0 |
| alpha_trend | -0.2 | 0.51 | 0.47 | -0.48 | -0.43 | 0.6 | 0.05 | 1.0 | 0.01 | -0.01 |
| rsi_div | 0.0 | 0.0 | 0.01 | -0.0 | -0.0 | 0.0 | 0.0 | 0.01 | 1.0 | 0.24 |
| obv_div | 0.0 | -0.02 | 0.0 | 0.0 | 0.01 | -0.02 | 0.0 | -0.01 | 0.24 | 1.0 |

**Redundant pairs:** none at |rho|>0.75

## Standalone TB win rates (decided signals only; production barriers)

| indicator | n_decided (sum) | mean win rate |
|---|---|---|
| alpha_trend | 751 | 0.401 |
| bb_pos | 1097 | 0.258 |
| chandelier | 177 | 0.405 |
| ma_ribbon | 626 | 0.381 |
| macd_sign | 590 | 0.393 |
| nwe_state | 330 | 0.749 |
| obv_div | 5 | 0.167 |
| rsi_div | 2 | 0.5 |
| rsi_zone | 856 | 0.243 |
| supertrend_dir | 603 | 0.395 |

_Report-only (D3): breakeven at tp1.5/sl1.0 is 40% win rate. Nothing in production reads this file._
