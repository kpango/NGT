#!/usr/bin/env bash
# bench_dataset.sh <dataset> <metric>
# Build QBG + QG (if missing), wait for the NGTAQ fp16 index, then bench all three
# single-thread (BLAS pinned) and emit a compare_grid table. Idempotent: skips
# builds whose index dir already exists.
set -u
ds="$1"; m="$2"
DATA=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks
IDX=/home/kpango/go/src/github.com/kpango/NGT/data/indices
WTB=/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/ngtaq-beat-qbg/build_ngtaq
LOGS=/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/ngtaq-beat-qbg/benchmarks/results/grid
export LD_LIBRARY_PATH=$WTB/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # search is SIMD; >1 pollutes 1-thread timing + risks OpenBLAS alloc fail
cd "$WTB/tests/ngtaq"
hdf5="$DATA/$ds.hdf5"
ge=0.20; [ "$m" = angular ] && ge=0.50    # angular needs a wider enqueue gate
echo "[$ds] === start $(date) ==="

# 1) QBG (refined-capable: GenuineDataType=Float stored by default)
if [ ! -d "$IDX/qbg_$ds" ]; then
  ./qbg_build "$hdf5" "$IDX/qbg_$ds" "$m" 4 16 > "$LOGS/build_qbg_$ds.log" 2>&1
fi
echo "[$ds] QBG ready $(date)"

# 2) QG (produces _anng/_qg4/_qsg1/_qsg2)
if [ ! -d "${IDX}/qg_${ds}_qg4" ]; then
  ./qg_build "$hdf5" "$IDX/qg_$ds" "$m" 100 16 > "$LOGS/build_qg_$ds.log" 2>&1
fi
echo "[$ds] QG ready $(date)"

# 3) wait for the NGTAQ fp16 index (may still be building in another job)
while [ ! -s "$IDX/ngtaq_$ds/aqindex" ]; do sleep 30; done
sleep 5
echo "[$ds] NGTAQ index ready $(date)"

# 4) benches (single-thread, BLAS=1)
NLOG="$LOGS/ngtaq_${ds}_t1.log"; : > "$NLOG"
for gt in 0.02 0.05 0.10 0.15 0.20 0.30 0.40 0.55 0.70 0.90; do
  ./ann_bench "$IDX/ngtaq_$ds" "$hdf5" 10 "$ge" "$gt" 3 1 0 2>/dev/null >> "$NLOG"
done
./ann_bench "$IDX/ngtaq_$ds" "$hdf5" 10 "$ge" 0.70 10 1 20 2>/dev/null >> "$NLOG"
./ann_bench "$IDX/ngtaq_$ds" "$hdf5" 10 "$ge" 0.90 20 1 40 2>/dev/null >> "$NLOG"
echo "[$ds] NGTAQ benched ($(grep -c agg_QPS "$NLOG") pts) $(date)"

./qbg_bench "$IDX/qbg_$ds" "$hdf5" 10 1 "$m" 2>/dev/null > "$LOGS/qbg_${ds}_t1.log"
echo "[$ds] QBG benched ($(grep -c agg_QPS "$LOGS/qbg_${ds}_t1.log") pts) $(date)"

QLOG="$LOGS/qg_${ds}_t1.log"; : > "$QLOG"
for v in qg4 qsg2 qsg1; do
  ./qg_bench "${IDX}/qg_${ds}_$v" "$hdf5" 10 1 "$m" 2>/dev/null >> "$QLOG"
done
echo "[$ds] QG benched ($(grep -c agg_QPS "$QLOG") pts) $(date)"

# 5) grid
python3 "$LOGS/compare_grid.py" "$NLOG" "$LOGS/qbg_${ds}_t1.log" "$QLOG" "$ds (1T)" | tee "$LOGS/grid_${ds}.txt"
echo "[$ds] === DONE $(date) ==="
