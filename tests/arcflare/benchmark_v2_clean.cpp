// tests/arcflare/benchmark_v2_clean.cpp
// ArcFlare clean benchmark: pre-built cache + real SIFT-1M test queries + exact GT
// No index build overhead → no thermal throttling from build.
//
// Usage:
//   ./benchmark_v2_clean <v2_cache_dir> <query.fvecs> <gt.ivecs> [k=10] [warmup=200]
#include "NGT/ArcFlare/ArcFlareIndex.h"
#include "fvecs_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>

static double computeRecall(const std::vector<uint32_t>& result_ids,
                            const std::vector<int32_t>&  gt, int k) {
    const int gt_k = std::min((int)gt.size(), k);
    if (gt_k == 0) return 0.0;
    std::unordered_set<int32_t> gt_set(gt.begin(), gt.begin() + gt_k);
    int hits = 0;
    const int res_k = std::min((int)result_ids.size(), k);
    for (int j = 0; j < res_k; ++j)
        if (gt_set.count((int32_t)result_ids[j])) ++hits;
    return (double)hits / gt_k;
}

static double pct(std::vector<double>& sv, double p) {
    if (sv.empty()) return 0.0;
    std::sort(sv.begin(), sv.end());
    size_t idx = (size_t)(p / 100.0 * sv.size());
    if (idx >= sv.size()) idx = sv.size() - 1;
    return sv[idx];
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <v2_cache_dir> <query.fvecs> <gt.ivecs> [k=10] [warmup=200]\n";
        return 1;
    }
    const std::string cache_dir  = argv[1];
    const std::string query_path = argv[2];
    const std::string gt_path    = argv[3];
    const int k       = argc > 4 ? std::stoi(argv[4]) : 10;
    const int n_warm  = argc > 5 ? std::stoi(argv[5]) : 200;

    // ---- Load pre-built ArcFlare index ----
    std::cout << "[Load] ArcFlare from: " << cache_dir << "\n"; std::cout.flush();
    ArcFlare::ArcFlareIndex idx = [&]() {
        auto i = ArcFlare::ArcFlareIndex::load(cache_dir + "/aqindex");
        i.loadV2(cache_dir);
        return i;
    }();
    std::cout << "  N=" << idx.size() << "  D=" << idx.dim() << "\n\n"; std::cout.flush();

    // ---- Load queries ----
    std::cout << "[Load] queries: " << query_path << "\n"; std::cout.flush();
    auto queries_2d = ArcFlare::loadFvecs(query_path);
    const int nq = (int)queries_2d.size();
    std::cout << "  " << nq << " queries\n"; std::cout.flush();

    // ---- Load ground truth ----
    std::cout << "[Load] ground truth: " << gt_path << "\n"; std::cout.flush();
    auto gt_2d = ArcFlare::loadIvecs(gt_path);
    std::cout << "  " << gt_2d.size() << " GT entries\n\n"; std::cout.flush();

    // ---- Warmup ----
    std::cout << "[Warmup] " << n_warm << " queries at gamma=0.20\n"; std::cout.flush();
    for (int i = 0; i < n_warm; ++i)
        idx.searchV2(queries_2d[i % nq], k, 0.2f, 0.20f);
    std::cout << "  Done\n\n"; std::cout.flush();

    // ---- Gamma sweep ----
    static const float GAMMAS[] = {0.10f, 0.12f, 0.14f, 0.16f, 0.18f,
                                    0.20f, 0.25f, 0.30f, 0.40f, 0.50f};
    constexpr int NG = (int)(sizeof(GAMMAS)/sizeof(GAMMAS[0]));

    std::cout << "=== ArcFlare gamma sweep (k=" << k << ", nq=" << nq << ") ===\n";
    std::cout << std::left
              << std::setw(12) << "gamma"
              << std::setw(14) << "recall@" + std::to_string(k)
              << std::setw(10) << "QPS"
              << std::setw(12) << "P50(us)"
              << std::setw(12) << "P99(us)"
              << "\n";
    std::cout.flush();

    // Run heavier gammas first to warm caches, then measure all from heavy→light
    struct Res { float gamma, recall, qps, p50, p99; };
    std::vector<Res> results(NG);

    for (int gi = NG - 1; gi >= 0; --gi) {
        float gamma = GAMMAS[gi];
        std::vector<double> lats(nq);
        double total_recall = 0.0;

        for (int i = 0; i < nq; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            auto res = idx.searchV2(queries_2d[i], k, 0.2f, gamma);
            const auto t1 = std::chrono::high_resolution_clock::now();
            lats[i] = std::chrono::duration<double,std::micro>(t1-t0).count();

            std::vector<uint32_t> ids;
            ids.reserve(res.size());
            for (auto& r : res) ids.push_back(r.id);
            if (i < (int)gt_2d.size())
                total_recall += computeRecall(ids, gt_2d[i], k);
        }

        double total_us = std::accumulate(lats.begin(), lats.end(), 0.0);
        double recall   = total_recall / nq;
        double qps      = nq / (total_us / 1e6);
        double p50_val  = pct(lats, 50.0);
        double p99_val  = pct(lats, 99.0);
        results[gi] = {gamma, (float)recall, (float)qps, (float)p50_val, (float)p99_val};
    }

    for (int gi = 0; gi < NG; ++gi) {
        const auto& r = results[gi];
        std::cout << "  gamma=" << std::fixed << std::setprecision(3) << r.gamma
                  << "  recall=" << std::setprecision(4) << r.recall
                  << "  QPS=" << std::setprecision(0) << r.qps
                  << "  P50=" << std::setprecision(1) << r.p50 << "us"
                  << "  P99=" << std::setprecision(1) << r.p99 << "us"
                  << (r.recall >= 0.80f ? "  *** >=0.80 ***" : "")
                  << "\n";
        std::cout.flush();
    }

    return 0;
}
