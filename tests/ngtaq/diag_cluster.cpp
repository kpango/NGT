// tests/ngtaq/diag_cluster.cpp
// Diagnostic: Check cluster membership of GT NNs vs query cluster
// Usage: diag_cluster <aq_index_dir> <hdf5_path>
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
    const std::string idx_dir   = argv[1];
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

    // Trigger cluster table initialization by running a dummy search
    std::vector<float> q(D_eff, 0.f);
    const float* src = test_ds.data.data();
    int D_raw = test_ds.n_cols;
    std::copy(src, src + std::min(D_raw, D_eff), q.begin());
    // normalize query
    float qnorm = 0.f;
    for (float x : q) qnorm += x*x;
    if (qnorm > 1e-12f) { float inv = 1.f/std::sqrt(qnorm); for (float& x : q) x *= inv; }

    idx.searchV2(q, 1, 0.1f, 0.1f, 1);  // warm up cluster tables

    const float* raw = idx.rawFlat();
    if (!raw) { fprintf(stderr, "rawFlat() is null!\n"); return 1; }

    // Get cluster assignments via record lookup
    // Use the AQv2 cluster assignment stored in the graph records
    printf("=== Cluster membership for Q0 neighbors ===\n");
    printf("GT ID        cos_sim  cluster_id\n");

    for (int i = 0; i < 20; ++i) {
        int gt_id = gt_ds.data[i];
        if (gt_id < 0 || (size_t)gt_id >= N) {
            printf("  GT[%d]: invalid ID %d\n", i, gt_id);
            continue;
        }
        uint32_t cid = idx.nodeCluster(static_cast<uint32_t>(gt_id));
        const float* vec = raw + static_cast<size_t>(gt_id) * D_eff;
        float dot = 0.f;
        for (int d = 0; d < D_eff; ++d) dot += q[d] * vec[d];
        printf("  GT[%2d]: ID=%6d  cos_sim=%.4f  cluster=%u\n", i, gt_id, dot, cid);
    }

    // Get query cluster
    uint32_t q_cid = idx.queryCluster(q);
    printf("\nQuery cluster: %u\n", q_cid);

    // Get cluster sizes
    auto cluster_sizes = idx.clusterSizes();
    if (!cluster_sizes.empty()) {
        int q_size = (q_cid < cluster_sizes.size()) ? cluster_sizes[q_cid] : -1;
        printf("Query cluster size: %d\n", q_size);
        int total = 0; int max_sz = 0; int min_sz = INT_MAX;
        for (int s : cluster_sizes) { total += s; max_sz = std::max(max_sz, s); min_sz = std::min(min_sz, s); }
        printf("Total indexed: %d, cluster min/avg/max = %d/%d/%d\n",
               total, min_sz, total/(int)cluster_sizes.size(), max_sz);
    }

    return 0;
}
