// tests/ngtaq/qbg_build.cpp
// Build QBG (Quantized Blob Graph) index from HDF5 train vectors.
// 3-phase build: HierarchicalKmeans → Optimizer → QBG blob graph.
//
// Usage: qbg_build <hdf5_path> <out_dir> [metric=l2|angular] [D_sub=4] [threads=8]
//
// Produces:
//   <out_dir>   — fully built QBG index (searchable after phase 3)
//
// Notes:
//   - D_sub is the PQ sub-vector dimension (numOfSubvectors = D / D_sub).
//   - Angular metric: L2-normalize vectors before insertion.
//   - If <out_dir>/grp (or global/grp) already exists, build is skipped.
#include "bench_common.hpp"
#include "NGT/NGTQ/QuantizedBlobGraph.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static double elapsed_s(double t0) { return (bc_now_us() - t0) / 1e6; }

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <hdf5_path> <out_dir> [metric=l2|angular] [D_sub=4] [threads=8]\n"
            "  D_sub: PQ sub-vector dimension (numOfSubvectors = D / D_sub)\n",
            argv[0]);
        return 1;
    }
    const char* hdf5_path  = argv[1];
    const char* out_dir    = argv[2];
    const char* metric_str = (argc > 3) ? argv[3] : "l2";
    int D_sub              = (argc > 4) ? std::stoi(argv[4]) : 4;
    int n_threads          = (argc > 5) ? std::stoi(argv[5]) : 8;
    const bool is_angular  = (strcmp(metric_str, "angular") == 0 ||
                               strcmp(metric_str, "cosine")  == 0);

    // QBG graph existence check — built index has grp file at top level or in global/
    const bool prebuilt = fs::exists(std::string(out_dir) + "/grp") ||
                          fs::exists(std::string(out_dir) + "/global/grp");
    if (prebuilt) {
        fprintf(stderr, "[Skip] QBG index already built: %s\n", out_dir);
        return 0;
    }

    // ── Step 1: Read train vectors ─────────────────────────────────────────
    fprintf(stderr, "[Load] HDF5 train from: %s\n", hdf5_path);
    H5FloatDataset train = h5_read_float(hdf5_path, "train");
    const int N = train.n_rows, D = train.n_cols;
    fprintf(stderr, "  N=%d  D=%d  D_sub=%d  metric=%s  threads=%d\n",
            N, D, D_sub, metric_str, n_threads);

    if (D % D_sub != 0) {
        fprintf(stderr, "[Error] D=%d is not divisible by D_sub=%d\n", D, D_sub);
        return 1;
    }

    // ── Step 2: Normalize if angular ──────────────────────────────────────
    if (is_angular) {
        for (int i = 0; i < N; ++i)
            l2_normalize(train.data.data() + (size_t)i * D, D);
        fprintf(stderr, "[Prep] L2-normalized %d vectors\n", N);
    }

    // ── Step 3: Create QBG index skeleton ────────────────────────────────
    QBG::BuildParameters bp;
    bp.creation.genuineDimension   = static_cast<size_t>(D);
    // QBG's global codebook ObjectSpace pads the working dimension up to a
    // multiple of 16 (getPaddedDimension()), but the genuine object list stores
    // vectors at genuineDimension. If genuineDimension is not already 16-aligned
    // (e.g. GloVe D=100 -> padded 112), Phase 3 residual generation compares the
    // unpadded object (100) against the padded codebook dim (112) and throws
    // "The dimensionalities are inconsitent. 100:112". The official `qbg create`
    // CLI avoids this by setting the working `dimension` to the 16-aligned value
    // while keeping genuineDimension at the true data dim (QbgCli.cpp:158-162).
    // 16-aligned datasets (SIFT 128, NYTimes 256, GIST 960, FashionMNIST 784)
    // worked only by coincidence; replicate the CLI's padding here so non-aligned
    // dims (GloVe 100) build correctly too.
    bp.creation.dimension          = ((static_cast<size_t>(D) + 15) / 16) * 16;
    bp.creation.dimensionOfSubvector = static_cast<size_t>(D_sub);
    bp.creation.distanceType       = NGTQ::DistanceType::DistanceTypeL2;
    bp.creation.dataType           = NGTQ::DataTypeFloat;
    bp.creation.threadSize         = static_cast<size_t>(n_threads);

    fprintf(stderr, "[QBG] Creating index skeleton at: %s\n", out_dir);
    try {
        QBG::Index::create(out_dir, bp);
    } catch (const std::exception& e) {
        fprintf(stderr, "[Error] QBG::Index::create failed: %s\n", e.what());
        return 1;
    }

    // ── Step 4: Append vectors ────────────────────────────────────────────
    fprintf(stderr, "[QBG] Appending %d vectors...\n", N);
    {
        const double t0 = bc_now_us();
        QBG::Index qbg(out_dir, /*prebuilt=*/false);
        for (int i = 0; i < N; ++i) {
            std::vector<float> vec(train.data.data() + (size_t)i * D,
                                   train.data.data() + (size_t)(i+1) * D);
            qbg.append(vec);
            if ((i+1) % 100000 == 0 || i == N-1)
                fprintf(stderr, "  appended %d/%d (%.1fs)\n", i+1, N, elapsed_s(t0));
        }
        fprintf(stderr, "[QBG] Saving...\n");
        qbg.save();
        fprintf(stderr, "[QBG] Saved (%.1fs)\n", elapsed_s(t0));
    }

    // ── Step 5: Phase 1 — Hierarchical k-means clustering ─────────────────
    fprintf(stderr, "[QBG] Phase 1/3: HierarchicalKmeans clustering...\n");
    {
        const double t0 = bc_now_us();
        try {
            QBG::HierarchicalKmeans hkm(bp);
            hkm.clustering(out_dir);
        } catch (const std::exception& e) {
            fprintf(stderr, "[Error] Phase 1 (clustering) failed: %s\n", e.what());
            return 1;
        }
        fprintf(stderr, "[QBG] Phase 1 done (%.1fs)\n", elapsed_s(t0));
    }

    // ── Step 6: Phase 2 — Optimizer (PQ codebook rotation) ────────────────
    fprintf(stderr, "[QBG] Phase 2/3: Optimizer (PQ codebook + rotation)...\n");
    {
        const double t0 = bc_now_us();
        try {
            QBG::Optimizer opt(bp);
            opt.optimize(out_dir, /*threads=*/0);  // 0 = all available
        } catch (const std::exception& e) {
            fprintf(stderr, "[Error] Phase 2 (optimizer) failed: %s\n", e.what());
            return 1;
        }
        fprintf(stderr, "[QBG] Phase 2 done (%.1fs)\n", elapsed_s(t0));
    }

    // ── Step 7: Phase 3 — Build NGTQ + QBG blob graph ─────────────────────
    fprintf(stderr, "[QBG] Phase 3/3: Building blob graph...\n");
    {
        const double t0 = bc_now_us();
        try {
            QBG::Index::build(out_dir, /*verbose=*/true);
        } catch (const std::exception& e) {
            fprintf(stderr, "[Error] Phase 3 (build) failed: %s\n", e.what());
            return 1;
        }
        fprintf(stderr, "[QBG] Phase 3 done (%.1fs)\n", elapsed_s(t0));
    }

    fprintf(stderr, "[All done] QBG index built at: %s\n", out_dir);
    return 0;
}
