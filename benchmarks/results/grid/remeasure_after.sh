#!/usr/bin/env bash
# remeasure_after.sh — clean (uncontended) re-measure of all 5 datasets after the
# searchV2 visit-cap/bucketing/bounded-rerank optimization. ArcFlare swept over
# gamma_term AND max_visits (the new knob); QBG (refined) and QG (qg4+qsg2) re-benched
# clean for a fair iso-recall comparison. Single-thread, BLAS pinned. Sequential (no
# concurrent benches) so QPS is uncontended.
set -u
DATA=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks
IDX=/home/kpango/go/src/github.com/kpango/NGT/data/indices
WTB=/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/arcflare-beat-qbg/build_arcflare
LOGS=/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/arcflare-beat-qbg/benchmarks/results/grid
export LD_LIBRARY_PATH=$WTB/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$WTB/tests/arcflare"

# ann_bench args: <idx> <hdf5> k gamma_enq gamma_term rerank_factor n_threads n_probe [max_visits]
ng_sweep () { # $1=ds $2=ge $3=np   -> writes arcflare_after_<ds>_t1.log
  local ds="$1" ge="$2" np="$3" hdf5="$DATA/$1.hdf5" idx="$IDX/arcflare_$1"
  local log="$LOGS/arcflare_after_${ds}_t1.log"; : > "$log"
  # (A) recall-ceiling curve: gamma sweep, unlimited visits
  for gt in 0.02 0.05 0.10 0.20 0.40 0.70 0.90; do
    ./ann_bench "$idx" "$hdf5" 10 "$ge" "$gt" 3 1 "$np" 0 2>/dev/null >> "$log"
  done
  # (B) visit-cap frontier at high gamma (fills faster high-recall points)
  for mv in 4000 3000 2000 1500 1000 700 400 200 100; do
    ./ann_bench "$idx" "$hdf5" 10 "$ge" 0.90 3 1 "$np" "$mv" 2>/dev/null >> "$log"
  done
  echo "[$ds] ArcFlare after: $(grep -c agg_QPS "$log") pts"
}

# datasets: ds:metric:ge:np  (angular ge=0.50 np=20; L2 ge=0.20 np=0)
for spec in \
  sift-128-euclidean:l2:0.20:0 \
  fashion-mnist-784-euclidean:l2:0.20:0 \
  gist-960-euclidean:l2:0.20:0 \
  nytimes-256-angular:angular:0.50:20 \
  glove-100-angular:angular:0.50:20 ; do
  ds="${spec%%:*}"; r="${spec#*:}"; m="${r%%:*}"; r="${r#*:}"; ge="${r%%:*}"; np="${r##*:}"
  echo "=== $ds ($m) $(date) ==="
  ng_sweep "$ds" "$ge" "$np"
  # QBG (refined) — use _fixed index for angular (dimension bug fix), canonical for L2
  qbgidx="$IDX/qbg_$ds"; [ -d "${qbgidx}_fixed" ] && qbgidx="${qbgidx}_fixed"
  ./qbg_bench "$qbgidx" "$DATA/$ds.hdf5" 10 1 "$m" 2>/dev/null > "$LOGS/qbg_after_${ds}_t1.log"
  echo "[$ds] QBG after: $(grep -c agg_QPS "$LOGS/qbg_after_${ds}_t1.log") pts"
  # QG: qg4 + qsg2 (skip impractically-slow qsg1)
  qlog="$LOGS/qg_after_${ds}_t1.log"; : > "$qlog"
  for v in qg4 qsg2; do ./qg_bench "${IDX}/qg_${ds}_$v" "$DATA/$ds.hdf5" 10 1 "$m" 2>/dev/null >> "$qlog"; done
  echo "[$ds] QG after: $(grep -c agg_QPS "$qlog") pts"
  python3 "$LOGS/compare_grid.py" "$LOGS/arcflare_after_${ds}_t1.log" "$LOGS/qbg_after_${ds}_t1.log" "$qlog" "$ds AFTER (1T)" | tee "$LOGS/grid_after_${ds}.txt"
done
echo "=== remeasure_after DONE $(date) ==="
