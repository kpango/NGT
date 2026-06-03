#!/usr/bin/env bash
# ci_build_indexes.sh — build the NGTAQ and QG indexes for one ANN-Benchmarks dataset.
#
# NGTAQ: M=128 (max_edges=128) + ONNG graph reconstruction (AQ_ONNG=1) + GORDER
#        cache-locality reorder (the consolidated build-time default; no env needed).
# QG:    ANNG base + qsg2 quantization (D_sub=2) — the strongest fair NGTQG baseline
#        with full-precision result_expansion refinement enabled in qg_bench.
#
# Usage: ci_build_indexes.sh <build_dir> <data_dir> <idx_dir> <dataset> <metric>
#   <build_dir>  CMake build tree (contains tests/ngtaq/<tools> and lib/NGT/libngt.so)
#   <data_dir>   dir holding <dataset>.hdf5
#   <idx_dir>    output dir for indexes (ngtaq_<ds>, qg_<ds>_*)
#   <dataset>    e.g. sift-128-euclidean   (no .hdf5 suffix)
#   <metric>     l2 | angular
#
# Idempotent: skips a build whose output index dir already exists (so a restored
# cache short-circuits the expensive graph construction).
set -euo pipefail

BUILD_DIR="${1:?build_dir}"
DATA_DIR="${2:?data_dir}"
IDX_DIR="${3:?idx_dir}"
DS="${4:?dataset}"
METRIC="${5:?metric}"

HDF5="$DATA_DIR/$DS.hdf5"
BIN="$BUILD_DIR/tests/ngtaq"
LIBDIR="$BUILD_DIR/lib/NGT"

export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}"
# Index build uses internal threads (createIndex(8) / ONNG); let it use all cores.
export OMP_NUM_THREADS="${BUILD_THREADS:-$(nproc)}"
unset OPENBLAS_NUM_THREADS || true

mkdir -p "$IDX_DIR"
[ -f "$HDF5" ] || { echo "ERROR: dataset not found: $HDF5" >&2; exit 1; }

ngtaq_idx="$IDX_DIR/ngtaq_$DS"
qg_base="$IDX_DIR/qg_$DS"

echo "::group::[${DS}] NGTAQ index (M=128, ONNG, GORDER) build"
if [ -s "$ngtaq_idx/aqindex" ]; then
  echo "[skip] NGTAQ index already present: $ngtaq_idx"
else
  t0=$(date +%s)
  # build_ngtaqv2_ann <hdf5> <out_dir> <metric> <k_clusters=0> <max_edges> <alpha> <edge_size>
  # max_edges=128 = "M=128". ONNG reconstruction is env-gated (QG-style o=32/i=64).
  AQ_ONNG=1 AQ_ONNG_O="${AQ_ONNG_O:-32}" AQ_ONNG_I="${AQ_ONNG_I:-64}" \
    "$BIN/build_ngtaqv2_ann" "$HDF5" "$ngtaq_idx" "$METRIC" 0 128 1.2 10
  echo "[ok] NGTAQ index built in $(( $(date +%s) - t0 ))s -> $ngtaq_idx"
fi
echo "::endgroup::"

echo "::group::[${DS}] QG index (ANNG + qsg2) build"
if [ -d "${qg_base}_qsg2" ]; then
  echo "[skip] QG qsg2 index already present: ${qg_base}_qsg2"
else
  t0=$(date +%s)
  # qg_build <hdf5> <out_base> <metric> <edge_size=100> <threads>
  # Produces <base>_anng, _qg4, _qsg1, _qsg2. We bench _qsg2 (best fair baseline).
  "$BIN/qg_build" "$HDF5" "$qg_base" "$METRIC" 100 "${BUILD_THREADS:-$(nproc)}"
  echo "[ok] QG index built in $(( $(date +%s) - t0 ))s -> ${qg_base}_qsg2"
fi
echo "::endgroup::"

echo "[${DS}] index build complete:"
du -sh "$ngtaq_idx" "${qg_base}_qsg2" 2>/dev/null || true
