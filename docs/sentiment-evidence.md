# Sentiment-feature evidence (offline IC study)

_Generated 2026-07-05 06:53 UTC · Spearman IC, * = p<0.05. This measures FEATURES vs forward returns; the voter itself is judged live in shadow (harness replays indicator+gate only — stated per project honesty rule)._

## Daily market features vs BTC forward returns (2y)

| feature | IC vs fwd 1d | IC vs fwd 7d |
|---|---|---|
| fng_level | -0.001 (p=0.985, n=729) | +0.021 (p=0.566, n=723) |
| fng_roc_7d | -0.035 (p=0.344, n=722) | +0.003 (p=0.938, n=716) |
| fng_extreme | -0.016 (p=0.676, n=729) | +0.020 (p=0.596, n=723) |
| fee_pressure_z | -0.005 (p=0.893, n=699) | +0.030 (p=0.430, n=693) |
| tx_momentum | -0.052 (p=0.169, n=715) | -0.026 (p=0.489, n=709) |
| onchain_divergence | -0.015 (p=0.686, n=717) | -0.060 (p=0.108, n=711) |

### Contrarian table — forward BTC return after extreme days

| state | days | fwd 7d mean | fwd 7d >0 | fwd 30d mean |
|---|---|---|---|---|
| extreme fear (F&G<20) | 109 | +0.38% | 53% | +4.22% |
| neutral (40–60) | 160 | -0.18% | 51% | -1.02% |
| extreme greed (F&G>80) | 20 | +1.06% | 55% | +2.51% |

## Taker buy-ratio z (live feature math) vs forward returns

| tf | pooled IC (fwd k bars) | Q5−Q1 spread | n |
|---|---|---|---|
| 1h (k=3) | -0.016 (p=0.000, n=215592) * | -0.044% | 215592 |
| 4h (k=2) | +0.006 (p=0.146, n=59604) | +0.033% | 59604 |
| 1d (k=1) | -0.010 (p=0.096, n=29860) | +0.086% | 29860 |

_Reading: |IC| 0.02–0.05 is normal for a single crypto feature; the bandit weighs and signs features from graded outcomes — this table only checks none of them is pure noise before go-live._

