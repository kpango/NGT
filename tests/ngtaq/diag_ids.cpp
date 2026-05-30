// tests/ngtaq/diag_ids.cpp
// Diagnostic: verify raw_flat_ ID mapping and check what searchV2 returns for query 0
// Usage: diag_ids <aq_index_dir> <hdf5_path>
#include "NGT/NGTAQ/AQIndex.h"
#include "hdf5_io.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <aq_index_dir> <hdf5_path>\n", argv[0]);
        return 1;
    }
    const std::string idx_dir  = argv[1];
    const std::string hdf5_path = argv[2];

    fprintf(stderr, "Loading index...\n");
    NGTAQ::NGTAQIndex idx = NGTAQ::NGTAQIndex::load(idx_dir + "/aqindex");
    idx.loadV2(idx_dir);
    const int D = idx.dim();
    const int D_eff = idx.dEff();
    const size_t N = idx.size();
    fprintf(stderr, "N=%zu D=%d D_eff=%d\n", N, D, D_eff);

    fprintf(stderr, "Loading HDF5...\n");
    H5FloatDataset test_ds = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt_ds   = h5_read_int(hdf5_path, "neighbors");

    // Query 0
    std::vector<float> q(D_eff, 0.f);
    const float* src = test_ds.data.data();  // query 0 raw vector
    int D_raw = test_ds.n_cols;
    std::copy(src, src + std::min(D_raw, D_eff), q.begin());

    // Print query vector stats
    float q_norm = 0.f;
    for (float x : q) q_norm += x * x;
    fprintf(stderr, "Query 0 norm = %.4f (D_raw=%d)\n", std::sqrt(q_norm), D_raw);

    // Ground truth IDs for query 0 (first 10)
    printf("=== Ground Truth (query 0) ===\n");
    std::vector<int> gt_ids(10);
    for (int i = 0; i < 10; ++i) {
        gt_ids[i] = gt_ds.data[i];
        printf("  GT[%d] = ID %d\n", i, gt_ids[i]);
    }

    const float* raw = idx.rawFlat();

    // Check what raw_flat_ has for the true NN (GT[0])
    printf("\n=== raw_flat_ lookup for GT[0] = ID %d ===\n", gt_ids[0]);
    {
        const float* vec = raw + static_cast<size_t>(gt_ids[0]) * D_eff;
        float dot = 0.f, vn = 0.f;
        for (int d = 0; d < D_eff; ++d) {
            dot += q[d] * vec[d];
            vn += vec[d] * vec[d];
        }
        printf("  vec_norm = %.4f, dot_with_q = %.4f\n", std::sqrt(vn), dot);
    }

    // Run searchV2 for query 0
    printf("\n=== searchV2 results (query 0, k=10, rf=10) ===\n");
    auto results = idx.searchV2(q, 10, 0.50f, 1.0f, 10);
    for (int i = 0; i < (int)results.size(); ++i) {
        int id = (int)results[i].id;
        bool in_gt = false;
        for (int g : gt_ids) if (g == id) { in_gt = true; break; }
        printf("  result[%d] = ID %d  dist=%.4f  %s\n",
               i, id, results[i].distance, in_gt ? "✓GT" : "");
    }

    printf("\n=== Checking GT IDs in raw_flat_ ===\n");
    for (int i = 0; i < 10; ++i) {
        const float* vec = raw + static_cast<size_t>(gt_ids[i]) * D_eff;
        float dot = 0.f, vn = 0.f;
        float qn = 0.f;
        for (int d = 0; d < D_eff; ++d) {
            dot += q[d] * vec[d];
            vn += vec[d] * vec[d];
            qn += q[d] * q[d];
        }
        float cos = dot / (std::sqrt(qn) * std::sqrt(vn) + 1e-10f);
        printf("  GT[%d] ID=%d: vec_norm=%.4f  cos_sim=%.4f\n",
               i, gt_ids[i], std::sqrt(vn), cos);
    }

    return 0;
}
