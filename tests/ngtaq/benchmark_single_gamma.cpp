// benchmark_single_gamma.cpp
// Focused single-gamma benchmark to avoid thermal noise from sweep ordering.
// Usage: ./bench_sg <cache_dir> <query.fvecs> <gt.ivecs> <gamma_term> [k=10] [nq=10000] [warmup=500] [gamma_enq=gamma_term]
#include "NGT/NGTAQ/AQIndex.h"
#include "fvecs_io.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>

static double computeRecall(const std::vector<uint32_t>& ids,
                             const std::vector<int32_t>&  gt, int k) {
    const int gk = std::min((int)gt.size(), k);
    if (!gk) return 0.0;
    std::unordered_set<int32_t> s(gt.begin(), gt.begin() + gk);
    int hits = 0;
    for (int j = 0; j < std::min((int)ids.size(), k); ++j)
        if (s.count((int32_t)ids[j])) ++hits;
    return (double)hits / gk;
}

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t)(p / 100.0 * v.size());
    if (i >= v.size()) i = v.size() - 1;
    return v[i];
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <cache_dir> <query.fvecs> <gt.ivecs> <gamma_term>"
                  << " [k=10] [nq=10000] [warmup=500] [gamma_enq=gamma_term]\n";
        return 1;
    }
    const std::string cache_dir  = argv[1];
    const std::string query_path = argv[2];
    const std::string gt_path    = argv[3];
    const float gamma_term = std::stof(argv[4]);
    const int k            = argc > 5 ? std::stoi(argv[5]) : 10;
    const int nq_limit     = argc > 6 ? std::stoi(argv[6]) : 10000;
    const int n_warm       = argc > 7 ? std::stoi(argv[7]) : 500;
    const float gamma_enq      = argc > 8 ? std::stof(argv[8]) : gamma_term;
    const int   n_cluster_seeds = argc > 9 ? std::stoi(argv[9]) : 32;

    NGTAQ::NGTAQIndex idx = [&]() {
        auto i = NGTAQ::NGTAQIndex::load(cache_dir + "/aqindex");
        i.loadV2(cache_dir);
        return i;
    }();
    std::cerr << "  N=" << idx.size() << "  D=" << idx.dim() << "\n";

    // Per-query tuning injected via the thread-safe SearchParameters/SearchContext API.
    NGTAQ::SearchParameters params;
    params.k               = k;
    params.gamma_enq       = gamma_enq;
    params.gamma_term      = gamma_term;
    params.n_cluster_seeds = n_cluster_seeds;
    NGTAQ::SearchContext ctx;

    auto queries_2d = NGTAQ::loadFvecs(query_path);
    auto gt_2d      = NGTAQ::loadIvecs(gt_path);
    const int nq = std::min((int)queries_2d.size(), nq_limit);

    // Warmup
    for (int i = 0; i < n_warm; ++i)
        idx.searchV2(queries_2d[i % nq], params, ctx);
    std::cerr << "Warmup done\n";

    // Measure
    std::vector<double> lats(nq);
    double total_recall = 0.0;
    for (int i = 0; i < nq; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = idx.searchV2(queries_2d[i], params, ctx);
        auto t1 = std::chrono::high_resolution_clock::now();
        lats[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::vector<uint32_t> ids;
        for (auto& r : res) ids.push_back(r.id);
        if (i < (int)gt_2d.size()) total_recall += computeRecall(ids, gt_2d[i], k);
    }

    double total_us = std::accumulate(lats.begin(), lats.end(), 0.0);
    double recall   = total_recall / nq;
    double qps      = nq / (total_us / 1e6);
    double p50      = pct(lats, 50.0);
    double p99      = pct(lats, 99.0);

    std::cout << std::fixed
              << "gamma_term=" << std::setprecision(3) << gamma_term
              << "  gamma_enq=" << std::setprecision(3) << gamma_enq
              << "  recall=" << std::setprecision(4) << recall
              << "  QPS=" << std::setprecision(0) << qps
              << "  P50=" << std::setprecision(1) << p50 << "us"
              << "  P99=" << std::setprecision(1) << p99 << "us"
              << (recall >= 0.90 ? "  *** >=0.90 ***" : recall >= 0.80 ? "  *** >=0.80 ***" : "")
              << "\n";
    return 0;
}
