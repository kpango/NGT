#!/usr/bin/env bash
# ci_bench.sh — fair single-thread recall→QPS sweep, ArcFlare vs QG, for one dataset.
#
# ArcFlare: ann_bench swept over AQ_EF (the recall-QPS knob — see ArcFlareIndex.cpp:1692;
#        AQ_EF overrides the max_visits-derived frontier). Single-thread, BLAS pinned.
# QG:    qg_bench on the qsg2 index, which internally sweeps epsilon × result_expansion
#        (the fair-comparison fix, commit 5dff002 / 5dff... — result_expansion>1 reranks
#        against full-precision fp32, which is what lets QG reach high recall). We narrow
#        both grids via QG_EPS / QG_RE to keep CI time bounded while spanning the range.
#
# Output:
#   $LOGS/arcflare_<ds>_t1.log   raw ann_bench blocks
#   $LOGS/qg_<ds>_t1.log      raw qg_bench blocks
#   $LOGS/grid_<ds>.txt       compare_grid.py per-recall-band ArcFlare/QG ratio table
#   appends a markdown section to $GITHUB_STEP_SUMMARY (if set)
#
# Usage: ci_bench.sh <build_dir> <data_dir> <idx_dir> <logs_dir> <dataset> <metric>
set -euo pipefail

BUILD_DIR="${1:?build_dir}"
DATA_DIR="${2:?data_dir}"
IDX_DIR="${3:?idx_dir}"
LOGS="${4:?logs_dir}"
DS="${5:?dataset}"
METRIC="${6:?metric}"

HDF5="$DATA_DIR/$DS.hdf5"
BIN="$BUILD_DIR/tests/arcflare"
LIBDIR="$BUILD_DIR/lib/NGT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMP="$HERE/../../benchmarks/results/grid/compare_grid.py"
[ -f "$CMP" ] || CMP="$BUILD_DIR/../benchmarks/results/grid/compare_grid.py"

export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}"
# Single-thread, BLAS pinned: search is SIMD; >1 BLAS thread pollutes 1-thread timing.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

mkdir -p "$LOGS"
arcflare_idx="$IDX_DIR/arcflare_$DS"
qg_idx="$IDX_DIR/qg_${DS}_qsg2"
[ -s "$arcflare_idx/aqindex" ] || { echo "ERROR: ArcFlare index missing: $arcflare_idx" >&2; exit 1; }
[ -d "$qg_idx" ]            || { echo "ERROR: QG qsg2 index missing: $qg_idx"   >&2; exit 1; }

# metric-specific enqueue gate + per-cluster probe (mirrors the local grid scripts)
GE=0.20; NP=0
if [ "$METRIC" = angular ] || [ "$METRIC" = cosine ]; then GE=0.50; NP=20; fi

# AQ_EF frontier: spans low (ef=15) -> high (ef=2000) recall. Override via CI_AQ_EF.
AQ_EF_GRID="${CI_AQ_EF:-15 20 30 50 80 120 200 320 560 1000 2000}"
# QG grids (narrowed from the 13x5 default to keep CI bounded). Override via CI_QG_*.
QG_EPS_GRID="${CI_QG_EPS:-0.01,0.05,0.1,0.2,0.3,0.5,0.8,1.0}"
QG_RE_GRID="${CI_QG_RE:-1.0,1.5,2.0,3.0,5.0,8.0,12.0,18.0}"

echo "::group::[${DS}] ArcFlare AQ_EF sweep (single-thread)"
NLOG="$LOGS/arcflare_${DS}_t1.log"; : > "$NLOG"
# ann_bench <idx> <hdf5> k gamma_enq gamma_term rerank_factor n_threads n_probe [max_visits]
# gamma_term=0.90 (recall ceiling) + AQ_EF as the swept frontier knob; max_visits=0 (AQ_EF wins).
for ef in $AQ_EF_GRID; do
  AQ_EF="$ef" "$BIN/ann_bench" "$arcflare_idx" "$HDF5" 10 "$GE" 0.90 3 1 "$NP" 0 2>/dev/null >> "$NLOG" || true
done
echo "[${DS}] ArcFlare: $(grep -c agg_QPS "$NLOG") points"
echo "::endgroup::"

echo "::group::[${DS}] QG qsg2 epsilon x result_expansion sweep (single-thread)"
QLOG="$LOGS/qg_${DS}_t1.log"; : > "$QLOG"
# qg_bench <qg_idx> <hdf5> k threads metric ; sweeps QG_EPS x QG_RE internally.
QG_EPS="$QG_EPS_GRID" QG_RE="$QG_RE_GRID" \
  "$BIN/qg_bench" "$qg_idx" "$HDF5" 10 1 "$METRIC" 2>/dev/null >> "$QLOG" || true
echo "[${DS}] QG: $(grep -c agg_QPS "$QLOG") points"
echo "::endgroup::"

# Per-band ratio table. QBG arg = a path that does not exist -> compare_grid prints
# N/A for QBG and computes the ratio against QG alone (ArcFlare/QG).
GRID="$LOGS/grid_${DS}.txt"
python3 "$CMP" "$NLOG" "$LOGS/__no_qbg__.log" "$QLOG" "$DS ArcFlare-vs-QG (1T, CI)" | tee "$GRID"

# ---- markdown summary ----
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### ${DS} (${METRIC}) — ArcFlare vs QG (single-thread)"
    echo ""
    echo "ArcFlare points: $(grep -c agg_QPS "$NLOG")  ·  QG points: $(grep -c agg_QPS "$QLOG")"
    echo ""
    echo '```'
    cat "$GRID"
    echo '```'
    echo ""
  } >> "$GITHUB_STEP_SUMMARY"
fi
