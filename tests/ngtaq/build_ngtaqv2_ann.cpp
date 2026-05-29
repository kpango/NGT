// tests/ngtaq/build_ngtaqv2_ann.cpp
// Build NGTAQv2 index from an ANN-Benchmarks HDF5 file.
//
// Usage: build_ngtaqv2_ann <hdf5_path> <out_dir> [metric=l2|angular|cosine]
//                          [k_clusters=0] [max_edges=64] [alpha=1.2] [edge_size=10]
//
// Workflow:
//   1. Read "train" dataset from HDF5 (float32, shape [N, D])
//   2. D_eff = pad_dim_for_v2(D)  — next power-of-2 >= 64, divisible by 64
//   3. If angular/cosine: L2-normalize each vector
//   4. Zero-pad D -> D_eff
//   5. Build NGT ANNG index from padded vectors (tmpdir = out_dir + "_ngt_tmp")
//   6. Call fromNGTv2() to build AQv2 index, save to out_dir
//   7. Remove NGT tmpdir
#include "NGT/NGTAQ/AQIndex.h"
#include "NGT/NGTAQ/DimUtils.h"
#include "NGT/Index.h"
#include "hdf5_io.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

static double elapsed_s(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <hdf5_path> <out_dir> [metric=l2|angular|cosine]"
            " [k_clusters=0] [max_edges=64] [alpha=1.2] [edge_size=10]\n",
            argv[0]);
        return 1;
    }
    const char* hdf5_path  = argv[1];
    const char* out_dir    = argv[2];
    std::string metric_str = (argc > 3) ? argv[3] : "l2";
    int   k_clusters       = (argc > 4) ? std::stoi(argv[4]) : 0;
    int   max_edges        = (argc > 5) ? std::stoi(argv[5]) : 64;
    float alpha            = (argc > 6) ? std::stof(argv[6]) : 1.2f;
    int   edge_size        = (argc > 7) ? std::stoi(argv[7]) : 10;

    const bool is_angular  = (metric_str == "angular" || metric_str == "cosine");
    const auto t0 = std::chrono::steady_clock::now();

    // 1. Read train vectors
    fprintf(stderr, "[Load] HDF5 train from: %s\n", hdf5_path);
    H5FloatDataset train_ds;
    try { train_ds = h5_read_float(hdf5_path, "train"); }
    catch (const std::exception& e) { fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
    const int N     = train_ds.n_rows;
    const int D     = train_ds.n_cols;
    const int D_eff = NGT::NGTAQ::pad_dim_for_v2(D);
    fprintf(stderr, "  N=%d  D=%d  D_eff=%d  metric=%s  (%.1fs)\n",
        N, D, D_eff, metric_str.c_str(), elapsed_s(t0));

    // 2. Pad + optionally normalize
    std::vector<float> padded((size_t)N * D_eff, 0.f);
    for (int i = 0; i < N; ++i) {
        const float* src = train_ds.data.data() + (size_t)i * D;
        float* dst = padded.data() + (size_t)i * D_eff;
        std::copy(src, src + D, dst);
        // [D, D_eff) stays 0 from initialization
        if (is_angular) {
            float norm2 = 0.f;
            for (int d = 0; d < D; ++d) norm2 += dst[d] * dst[d];
            if (norm2 > 1e-12f) {
                float inv = 1.f / std::sqrt(norm2);
                for (int d = 0; d < D_eff; ++d) dst[d] *= inv;
            }
        }
    }
    fprintf(stderr, "[Prep] %d vectors padded%s (%.1fs)\n",
        N, is_angular ? " + normalized" : "", elapsed_s(t0));

    // 3. Build NGT ANNG index
    const std::string ngt_tmp = std::string(out_dir) + "_ngt_tmp";
    std::filesystem::remove_all(ngt_tmp);

    NGT::Property ngt_prop;
    ngt_prop.dimension           = D_eff;
    ngt_prop.objectType          = NGT::ObjectSpace::ObjectType::Float;
    ngt_prop.distanceType        = NGT::ObjectSpace::DistanceType::DistanceTypeL2;
    ngt_prop.edgeSizeForCreation = edge_size;

    fprintf(stderr, "[NGT] Creating ANNG index (D=%d, edge_size=%d) ...\n", D_eff, edge_size);
    try {
        NGT::Index::create(ngt_tmp, ngt_prop);
        NGT::Index ngt(ngt_tmp);
        const int batch = 50000;
        for (int i = 0; i < N; i += batch) {
            const int end = std::min(i + batch, N);
            for (int j = i; j < end; ++j) {
                std::vector<float> v(padded.data() + (size_t)j * D_eff,
                                     padded.data() + (size_t)j * D_eff + D_eff);
                ngt.append(v);
            }
            ngt.createIndex(/*threads=*/8);
            fprintf(stderr, "  indexed %d/%d (%.1fs)\n", end, N, elapsed_s(t0));
        }
        ngt.save();
    } catch (const std::exception& e) {
        fprintf(stderr, "NGT error: %s\n", e.what()); return 1;
    }
    fprintf(stderr, "[NGT] ANNG done (%.1fs)\n", elapsed_s(t0));

    // 4. Build AQv2 index
    NGTAQ::NGTAQIndex::Property aq_prop;
    aq_prop.dimension  = D_eff;
    aq_prop.k_clusters = k_clusters;
    aq_prop.max_edges  = max_edges;
    aq_prop.alpha      = alpha;
    aq_prop.metric     = is_angular
        ? NGT::ObjectSpace::DistanceTypeCosine
        : NGT::ObjectSpace::DistanceTypeL2;

    fprintf(stderr, "[NGTAQv2] Building (k_clusters=%d, max_edges=%d, alpha=%.2f)...\n",
        k_clusters, max_edges, alpha);
    NGTAQ::NGTAQIndex idx = NGTAQ::NGTAQIndex::fromNGTv2(ngt_tmp, aq_prop);
    fprintf(stderr, "[NGTAQv2] Build done (%.1fs)\n", elapsed_s(t0));

    // 5. Save
    std::filesystem::create_directories(out_dir);
    idx.save(std::string(out_dir) + "/aqindex");
    idx.saveV2(out_dir);
    fprintf(stderr, "[Done] Saved to: %s (%.1fs total)\n", out_dir, elapsed_s(t0));

    // 6. Cleanup
    std::filesystem::remove_all(ngt_tmp);
    return 0;
}
