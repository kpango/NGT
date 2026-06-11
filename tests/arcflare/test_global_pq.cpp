// tests/arcflare/test_global_pq.cpp
// Stage A correctness test: is the GLOBAL PQ tier an accurate enough ROUTING signal?
//
// Usage: test_global_pq <aq_index_dir> <hdf5_path> [n_query_sample=200] [k_truth=10]
//
// For a sample of test queries it ranks EVERY index node by globalPQDist() (one LUT
// per query, scoring any node — the Stage A property), then measures how highly the
// dataset's exact k-NN (HDF5 "neighbors") rank under that global-PQ ordering.
//
// Go/no-go signal for Stages B/C: do the exact top-k fall within the global-PQ
// top-N for small N? If yes, batch routing on global PQ can preserve recall (the
// existing fp16 exact rerank recovers final accuracy). Exit 0 always (diagnostic).
#include "NGT/ArcFlare/ArcFlareIndex.h"
#include "hdf5_io.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <aq_index_dir> <hdf5_path> [n_query_sample=200] [k_truth=10]\n",
            argv[0]);
        return 2;
    }
    const std::string idx_dir   = argv[1];
    const std::string hdf5_path = argv[2];
    const int n_q_sample = (argc > 3) ? std::stoi(argv[3]) : 200;
    const int k_truth    = (argc > 4) ? std::stoi(argv[4]) : 10;

    // ---- Load index ----
    fprintf(stderr, "[Load] AQv2 from: %s\n", idx_dir.c_str());
    ArcFlare::ArcFlareIndex idx = ArcFlare::ArcFlareIndex::load(idx_dir + "/aqindex");
    idx.loadV2(idx_dir);
    if (!idx.hasGlobalPQ()) {
        fprintf(stderr, "FAIL: index has no global PQ tier (rebuild with Stage A).\n");
        return 1;
    }
    const int    M_PQ = idx.mPQ();
    const size_t N    = idx.size();
    fprintf(stderr, "[Load] N=%zu  M_PQ=%d  D_eff=%d  angular=%d\n",
            N, M_PQ, idx.dEff(), (int)idx.isAngular());

    // ---- Load queries + ground-truth neighbors ----
    H5FloatDataset test_ds = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt_ds   = h5_read_int(hdf5_path, "neighbors");
    const int nq   = std::min(test_ds.n_rows, gt_ds.n_rows);
    const int Dq   = test_ds.n_cols;
    const int gt_k = gt_ds.n_cols;
    const int K    = std::min(k_truth, gt_k);
    const int sample = std::min(n_q_sample, nq);
    fprintf(stderr, "[Load] nq=%d Dq=%d gt_k=%d -> sampling %d queries, k_truth=%d\n",
            nq, Dq, gt_k, sample, K);

    // N thresholds at which we measure "exact top-K within global-PQ top-N".
    const std::vector<int> Ns = {10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
    std::vector<double> recall_at(Ns.size(), 0.0);  // mean over sampled queries
    // Rank stats of the GT top-1 and the worst (Kth) GT neighbor under global-PQ order.
    std::vector<long> rank_top1;       // rank of GT[0]
    std::vector<long> rank_worst_topK; // max rank among GT[0..K-1]
    rank_top1.reserve(sample);
    rank_worst_topK.reserve(sample);

    std::vector<float> lut((size_t)M_PQ * 256);
    std::vector<float> dist(N);
    std::vector<uint32_t> order(N);

    ArcFlare::SearchContext ctx;
    for (int s = 0; s < sample; ++s) {
        // Spread the sample across the query set (stride) rather than first-`sample`.
        const int qi = (long long)s * nq / sample;
        std::vector<float> q(test_ds.row(qi), test_ds.row(qi) + Dq);

        const float q_norm_sq = idx.buildGlobalLUT(q, lut.data(), ctx);
        for (size_t n = 0; n < N; ++n)
            dist[n] = idx.globalPQDist((uint32_t)n, lut.data(), q_norm_sq);

        // Argsort node ids by ascending global-PQ distance.
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) { return dist[a] < dist[b]; });

        // rank[node] = position in the global-PQ ordering (0 = closest).
        // Build a reverse lookup only for the GT ids we care about by scanning order
        // until we've located all K (cheaper than a full N-sized rank vector).
        // Collect the GT ids for this query.
        std::vector<int> gt_ids(K);
        for (int j = 0; j < K; ++j)
            gt_ids[j] = (int)gt_ds.data[(size_t)qi * gt_k + j];

        // Locate each GT id's rank. Single pass over `order`, early-exit once all found.
        std::vector<long> ranks(K, -1);
        int found_cnt = 0;
        for (size_t pos = 0; pos < N && found_cnt < K; ++pos) {
            uint32_t id = order[pos];
            for (int j = 0; j < K; ++j) {
                if (ranks[j] < 0 && gt_ids[j] == (int)id) {
                    ranks[j] = (long)pos; ++found_cnt; break;
                }
            }
        }
        // Any GT id not present in the index (e.g., a hole) → treat rank as N (worst).
        long worst = 0; long r0 = ranks[0] < 0 ? (long)N : ranks[0];
        for (int j = 0; j < K; ++j) {
            long r = (ranks[j] < 0) ? (long)N : ranks[j];
            worst = std::max(worst, r);
        }
        rank_top1.push_back(r0);
        rank_worst_topK.push_back(worst);

        // recall@N: fraction of GT top-K whose rank < N.
        for (size_t ni = 0; ni < Ns.size(); ++ni) {
            const long thr = Ns[ni];
            int hit = 0;
            for (int j = 0; j < K; ++j) {
                long r = (ranks[j] < 0) ? (long)N : ranks[j];
                if (r < thr) ++hit;
            }
            recall_at[ni] += (double)hit / K;
        }

        if ((s + 1) % 50 == 0)
            fprintf(stderr, "  ...%d/%d queries\n", s + 1, sample);
    }

    for (auto& r : recall_at) r /= sample;

    auto pct = [](std::vector<long>& v, double p) -> long {
        if (v.empty()) return 0;
        std::vector<long> t = v;
        std::sort(t.begin(), t.end());
        size_t i = (size_t)(p * (t.size() - 1));
        return t[i];
    };

    printf("\n==== Stage A GLOBAL-PQ routing quality (N=%zu, %d queries, k_truth=%d) ====\n",
           N, sample, K);
    printf("exact top-%d recall within global-PQ top-N:\n", K);
    for (size_t ni = 0; ni < Ns.size(); ++ni)
        printf("  top-%-5d : %.4f\n", Ns[ni], recall_at[ni]);

    printf("\nrank of GT[0] (closest exact NN) under global-PQ order:\n");
    printf("  p50=%ld  p90=%ld  p95=%ld  p99=%ld  max=%ld\n",
           pct(rank_top1,0.50), pct(rank_top1,0.90), pct(rank_top1,0.95),
           pct(rank_top1,0.99), pct(rank_top1,1.0));
    printf("worst rank among exact top-%d under global-PQ order:\n", K);
    printf("  p50=%ld  p90=%ld  p95=%ld  p99=%ld  max=%ld\n",
           pct(rank_worst_topK,0.50), pct(rank_worst_topK,0.90), pct(rank_worst_topK,0.95),
           pct(rank_worst_topK,0.99), pct(rank_worst_topK,1.0));

    // Heuristic verdict for the report (the spec's go/no-go).
    // "usable routing signal" ~ exact top-K mostly land within global-PQ top-~200.
    size_t i200 = 0; for (; i200 < Ns.size(); ++i200) if (Ns[i200] == 200) break;
    double r200 = (i200 < Ns.size()) ? recall_at[i200] : 0.0;
    printf("\nVERDICT: exact top-%d within global-PQ top-200 = %.4f -> %s\n",
           K, r200, (r200 >= 0.90 ? "USABLE routing signal (proceed to Stage B/C)"
                                  : "WEAK — reconsider global-tier before B/C"));
    return 0;
}
