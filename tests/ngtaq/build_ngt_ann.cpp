// tests/ngtaq/build_ngt_ann.cpp
// Build ONLY the NGT ANNG from an ANN-Benchmarks HDF5 file.
// Used to build a denser NGT source for rebuild_graph without re-training AQv2.
//
// Usage: build_ngt_ann <hdf5_path> <out_ngt_dir> [metric=l2|angular] [edge_size=100]
#include "NGT/Index.h"
#include "hdf5_io.h"
#include "NGT/NGTAQ/DimUtils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

static double elapsed_s(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <hdf5_path> <out_ngt_dir> [metric=l2|angular] [edge_size=100]\n",
            argv[0]);
        return 1;
    }
    const char* hdf5_path  = argv[1];
    const char* out_dir    = argv[2];
    std::string metric_str = (argc > 3) ? argv[3] : "angular";
    int         edge_size  = (argc > 4) ? std::stoi(argv[4]) : 100;

    const bool is_angular = (metric_str == "angular" || metric_str == "cosine");
    const auto t0 = std::chrono::steady_clock::now();

    fprintf(stderr, "[Load] HDF5 train from: %s\n", hdf5_path);
    H5FloatDataset train_ds = h5_read_float(hdf5_path, "train");
    const int N    = train_ds.n_rows;
    const int D    = train_ds.n_cols;
    const int D_eff = NGT::NGTAQ::pad_dim_for_v2(D);
    fprintf(stderr, "  N=%d  D=%d  D_eff=%d  metric=%s  edge_size=%d  (%.1fs)\n",
        N, D, D_eff, metric_str.c_str(), edge_size, elapsed_s(t0));

    // Pad + normalize
    std::vector<float> padded((size_t)N * D_eff, 0.f);
    int n_holes = 0;
    for (int i = 0; i < N; ++i) {
        const float* src = train_ds.data.data() + (size_t)i * D;
        float* dst = padded.data() + (size_t)i * D_eff;
        std::copy(src, src + D, dst);
        if (is_angular) {
            float norm2 = 0.f;
            for (int d = 0; d < D; ++d) norm2 += dst[d] * dst[d];
            if (norm2 > 1e-12f) {
                float inv = 1.f / std::sqrt(norm2);
                for (int d = 0; d < D_eff; ++d) dst[d] *= inv;
            } else {
                ++n_holes;
            }
        }
    }
    fprintf(stderr, "[Prep] %d vectors padded%s (%d holes) (%.1fs)\n",
        N, is_angular ? " + normalized" : "", n_holes, elapsed_s(t0));

    // Build NGT ANNG
    std::filesystem::remove_all(out_dir);

    NGT::Property ngt_prop;
    ngt_prop.dimension           = D_eff;
    ngt_prop.objectType          = NGT::ObjectSpace::ObjectType::Float;
    ngt_prop.distanceType        = NGT::ObjectSpace::DistanceType::DistanceTypeL2;
    ngt_prop.edgeSizeForCreation = edge_size;

    fprintf(stderr, "[NGT] Creating ANNG (D=%d, edge_size=%d) ...\n", D_eff, edge_size);
    try {
        NGT::Index::create(out_dir, ngt_prop);
        NGT::Index ngt(out_dir);
        const int batch = 50000;
        const int threads = 16;
        for (int i = 0; i < N; i += batch) {
            const int end = std::min(i + batch, N);
            for (int j = i; j < end; ++j) {
                std::vector<float> v(padded.data() + (size_t)j * D_eff,
                                     padded.data() + (size_t)j * D_eff + D_eff);
                ngt.append(v);
            }
            ngt.createIndex(threads);
            fprintf(stderr, "  indexed %d/%d (%.1fs)\n", end, N, elapsed_s(t0));
        }
        ngt.save();
        fprintf(stderr, "[NGT] ANNG done (%.1fs)\n", elapsed_s(t0));
    } catch (const std::exception& e) {
        fprintf(stderr, "NGT error: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "[Done] NGT ANNG saved to: %s\n", out_dir);
    return 0;
}
