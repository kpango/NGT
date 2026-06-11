// Skip-rerank top-k ROUTING recall of the GPQ4 batch quantizer.
//
// For each query: build the GPQ4 per-query LUT, score EVERY indexed node with the
// batch route distance (q_ns + recon_norm^2 - 2<q,x_pq>), take the top-k by that
// distance ALONE (no graph traversal, no exact rerank), and measure overlap with the
// HDF5 ground-truth top-k. This isolates codebook fidelity from graph navigation:
// it is the ceiling the batch router's ordering can achieve. M=16 caps ~12%; finer M
// (set at build via AQ_GPQ4_DSUB) should push it toward QG's ~80%.
//
// Usage: diag_route_recall <index_dir> <hdf5> [k=10] [n_queries=200]
#include "NGT/ArcFlare/ArcFlareIndex.h"
#include "NGT/ArcFlare/GlobalPQ4.h"
#include "hdf5_io.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unordered_set>
#include <vector>

using namespace NGT::ArcFlare;

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <index_dir> <hdf5> [k] [nq]\n", argv[0]); return 1; }
    const std::string dir = argv[1];
    const char* h5 = argv[2];
    const int k  = (argc > 3) ? std::atoi(argv[3]) : 10;
    const int NQ = (argc > 4) ? std::atoi(argv[4]) : 200;

    ::ArcFlare::ArcFlareIndex idx = ::ArcFlare::ArcFlareIndex::load(dir + "/aqindex");
    idx.loadV2(dir);
    const int M = idx.gpq4MPQ();
    fprintf(stderr, "hasGPQ4=%d dEff=%d gpq4MPQ=%d gpq4DSub=%d\n",
            (int)idx.hasGPQ4(), idx.dEff(), M, idx.gpq4DSub());

    // Flat per-node GPQ4 codes + recon norms (gpq4MPQ()-wide).
    std::ifstream f(dir + "/v2_gpq4_codes.bin", std::ios::binary);
    if (!f) { fprintf(stderr, "missing v2_gpq4_codes.bin\n"); return 1; }
    uint64_t n = 0; f.read((char*)&n, 8);
    std::vector<uint8_t> codes((size_t)n * M);
    std::vector<float>   norms(n);
    f.read((char*)codes.data(), codes.size());
    f.read((char*)norms.data(), norms.size() * 4);

    H5FloatDataset test = h5_read_float(h5, "test");
    H5IntDataset   gt   = h5_read_int(h5, "neighbors");
    const int D    = test.n_cols;
    const int nq   = std::min(NQ, test.n_rows);
    const int gtcw = gt.n_cols;

    GlobalPQ4LUT lut;
    std::vector<float> ipt((size_t)M * GPQ4_K);
    std::vector<std::pair<float, uint32_t>> scored;
    scored.reserve(n);

    double recall_sum = 0.0;
    ArcFlare::SearchContext ctx;
    for (int q = 0; q < nq; ++q) {
        std::vector<float> query(test.data.data() + (size_t)q * D,
                                 test.data.data() + (size_t)q * D + D);
        float q_ns = idx.buildGlobalLUT16(query, lut, ctx, ipt.data());

        // Exhaustive route-distance scoring over all nodes.
        scored.clear();
        for (uint64_t id = 0; id < n; ++id) {
            const uint8_t* c = codes.data() + (size_t)id * M;
            float ip = 0.f;
            for (int s = 0; s < M; ++s) ip += ipt[(size_t)s * GPQ4_K + c[s]];
            scored.push_back({q_ns + norms[id] - 2.0f * ip, (uint32_t)id});
        }
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });

        std::unordered_set<uint32_t> truth;
        for (int j = 0; j < k; ++j) truth.insert((uint32_t)gt.data[(size_t)q * gtcw + j]);
        int hit = 0;
        for (int j = 0; j < k; ++j) if (truth.count(scored[j].second)) ++hit;
        recall_sum += (double)hit / k;
    }
    printf("route_recall@%d = %.4f  (M=%d D_sub=%d, nq=%d, N=%llu)\n",
           k, recall_sum / nq, M, idx.gpq4DSub(), nq, (unsigned long long)n);
    return 0;
}
