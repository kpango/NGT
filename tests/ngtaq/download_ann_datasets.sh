#!/usr/bin/env bash
# tests/ngtaq/download_ann_datasets.sh
# Download ANN-Benchmarks HDF5 files from ann-benchmarks.com
set -euo pipefail

BASE_DIR="${1:-/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks}"
mkdir -p "$BASE_DIR"

BASE_URL="http://ann-benchmarks.com"
FILES=(
    "sift-128-euclidean.hdf5"
    "glove-100-angular.hdf5"
    "nytimes-256-angular.hdf5"
    "gist-960-euclidean.hdf5"
    "fashion-mnist-784-euclidean.hdf5"
)

for file in "${FILES[@]}"; do
    dst="$BASE_DIR/$file"
    if [ -f "$dst" ]; then
        size=$(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst")
        if [ "$size" -gt 1000000 ]; then
            echo "[SKIP] $file already present (${size} bytes)"
            continue
        fi
    fi
    echo "[DOWNLOAD] $file ..."
    wget -q --show-progress -O "$dst" "$BASE_URL/$file" || \
        curl -# -L -o "$dst" "$BASE_URL/$file"
    echo "[OK] $file"
done
echo "All datasets ready in $BASE_DIR"
