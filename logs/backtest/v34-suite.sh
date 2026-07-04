#!/bin/bash
# v3.4 evidence suite: gate-v2 baseline + one run per candidate vote.
# Fixed --end pins the bar set so every A/B compares identical history.
set -e
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
# numpy's small-window ops thread-thrash (456% cpu for 0.7x the speed of one
# capped core) — cap BLAS to 1 thread per worker and scale workers instead
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PAIRS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,NEARUSDT,ARBUSDT,GALAUSDT
COMMON="--pairs $PAIRS --tfs 1h,4h,1d --start 2024-07-01 --end 2026-07-04 --gate v2 --window 500 --workers 8"

echo "=== v34-baseline $(date) ==="
./venv/bin/python scripts/run_backtest.py $COMMON --label v34-baseline

for k in rsi30 mfi cci vwap fib ichimoku; do
  echo "=== v34-$k $(date) ==="
  T2_EXTRA_VOTES=$k ./venv/bin/python scripts/run_backtest.py $COMMON \
    --label "v34-$k" --baseline logs/backtest/v34-baseline/report.json
done
echo "=== suite done $(date) ==="
