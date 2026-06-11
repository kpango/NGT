#!/usr/bin/env bash
# bench_dataset_big.sh <dataset> <metric>
# Like bench_dataset.sh but BUILDS run with thread parallelism (graph build needs OMP);
# only the BENCH phase pins threads to 1 (correct single-thread QPS + avoids OpenBLAS
# buffer explosion). For large datasets (GloVe, GIST) where OMP=1 builds are too slow.
set -u
ds="$1"; m="$2"
DATA=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks
IDX=/home/kpango/go/src/github.com/kpango/NGT/data/indices
WTB=/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/ngtaq-beat-qbg/build_ngtaq
LOGS=/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/ngtaq-beat-qbg/benchmarks/results/grid
export LD_LIBRARY_PATH=$WTB/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
cd "$WTB/tests/ngtaq"
hdf5="$DATA/$ds.hdf5"
ge=0.20; [ "$m" = angular ] && ge=0.50
echo "[$ds] === start $(date) ==="

# Builds: moderate parallelism (graph OMP + bounded BLAS to avoid 128-thread buffer blowup)
if [ ! -d "$IDX/qbg_$ds" ]; then
  OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=8 \
    ./qbg_build "$hdf5" "$IDX/qbg_$ds" "$m" 4 32 > "$LOGS/build_qbg_$ds.log" 2>&1
fi
echo "[$ds] QBG ready $(date)"
if [ ! -d "${IDX}/qg_${ds}_qg4" ]; then
  OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=8 \
    ./qg_build "$hdf5" "$IDX/qg_$ds" "$m" 100 32 > "$LOGS/build_qg_$ds.log" 2>&1
fi
echo "[$ds] QG ready $(date)"

while [ ! -s "$IDX/ngtaq_$ds/aqindex" ]; do sleep 30; done
sleep 5
echo "[$ds] NGTAQ index ready $(date)"

# Benches: single-thread, BLAS pinned to 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
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
python3 "$LOGS/compare_grid.py" "$NLOG" "$LOGS/qbg_${ds}_t1.log" "$QLOG" "$ds (1T)" | tee "$LOGS/grid_${ds}.txt"
echo "[$ds] === DONE $(date) ==="
