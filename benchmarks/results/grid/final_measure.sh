#!/usr/bin/env bash
# final_measure.sh — definitive after-all-optimizations grid (dec7f18 binary, build_ngtaq2).
# NGTAQ swept with the CORRECT knobs: L2 -> visit-cap (max_visits) as the low-recall lever;
# angular -> seeds_per_cluster (arg 11) as the lever. Compared to QBG (refined) + QG (qg4+qsg2).
# ann_bench args: <idx> <hdf5> k gamma_enq gamma_term rerank_factor n_threads n_probe [max_visits] [seeds_per_cluster]
set -u
DATA=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks
IDX=/home/kpango/go/src/github.com/kpango/NGT/data/indices
WTB=/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/ngtaq-beat-qbg/build_ngtaq2
LOGS=/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/ngtaq-beat-qbg/benchmarks/results/grid
export LD_LIBRARY_PATH=$WTB/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$WTB" && cmake --build . --target qbg_bench qg_bench -j8 >/dev/null 2>&1
cd "$WTB/tests/ngtaq"

l2_ngtaq () { # $1=ds  -> visit-cap frontier
  local ds="$1" hdf5="$DATA/$1.hdf5" idx="$IDX/ngtaq_$1" log="$LOGS/ngtaq_final_${1}_t1.log"; : > "$log"
  for gt in 0.10 0.40 0.90; do
    for mv in 15 30 60 150 400 1000 0; do
      ./ann_bench "$idx" "$hdf5" 10 0.20 "$gt" 3 1 0 "$mv" 2>/dev/null >> "$log"
    done
  done
  echo "[$ds] NGTAQ-final $(grep -c agg_QPS "$log") pts"
}
ang_ngtaq () { # $1=ds  -> seeds_per_cluster frontier
  local ds="$1" hdf5="$DATA/$1.hdf5" idx="$IDX/ngtaq_$1" log="$LOGS/ngtaq_final_${1}_t1.log"; : > "$log"
  for gt in 0.40 0.90; do
    for sc in 16 32 64 128; do
      for mv in 200 600 0; do
        ./ann_bench "$idx" "$hdf5" 10 0.50 "$gt" 3 1 20 "$mv" "$sc" 2>/dev/null >> "$log"
      done
    done
  done
  echo "[$ds] NGTAQ-final $(grep -c agg_QPS "$log") pts"
}
comp () { # $1=ds $2=metric
  local ds="$1" m="$2" hdf5="$DATA/$1.hdf5"
  local qbgidx="$IDX/qbg_$1"; [ -d "${qbgidx}_fixed" ] && qbgidx="${qbgidx}_fixed"
  ./qbg_bench "$qbgidx" "$hdf5" 10 1 "$m" 2>/dev/null > "$LOGS/qbg_final_${1}_t1.log"
  local qlog="$LOGS/qg_final_${1}_t1.log"; : > "$qlog"
  for v in qg4 qsg2; do ./qg_bench "${IDX}/qg_${1}_$v" "$hdf5" 10 1 "$m" 2>/dev/null >> "$qlog"; done
  python3 "$LOGS/compare_grid.py" "$LOGS/ngtaq_final_${1}_t1.log" "$LOGS/qbg_final_${1}_t1.log" "$qlog" "$ds FINAL (1T)" | tee "$LOGS/grid_final_${1}.txt"
}

echo "=== final_measure start $(date) ==="
l2_ngtaq sift-128-euclidean;  comp sift-128-euclidean l2
ang_ngtaq glove-100-angular;  comp glove-100-angular angular
l2_ngtaq gist-960-euclidean;  comp gist-960-euclidean l2
echo "=== final_measure DONE $(date) ==="
