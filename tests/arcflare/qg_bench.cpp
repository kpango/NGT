// tests/arcflare/qg_bench.cpp
// ANN-Benchmarks benchmark for NGTQG (QG / QSG) sweeping BOTH epsilon (quantized graph
// exploration) AND result_expansion (full-precision exact-L2 refinement) — the two params
// the official ann-benchmarks NGT-qg module sweeps. result_expansion>1.0 reranks
// result_expansion*k candidates against the FULL fp32 base vectors (QuantizedGraph.h:385-407),
// which is what lets QG reach high recall (~0.99). The pure-epsilon sweep (re=1.0) is
// quantized-only and caps at ~0.85-0.93. The fair Pareto frontier = best QPS at each recall
// across the epsilon × result_expansion grid.
// Usage: qg_bench <qg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]
#include "bench_common.hpp"
#include "NGT/NGTQ/QuantizedGraph.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <qg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]\n",
            argv[0]);
        return 1;
    }
    const char* idx_dir    = argv[1];
    const char* hdf5_path  = argv[2];
    int   k                = (argc > 3) ? std::stoi(argv[3]) : 10;
    int   n_threads        = (argc > 4) ? std::stoi(argv[4]) : 4;
    const char* metric_str = (argc > 5) ? argv[5] : "l2";
    const bool is_angular  = (strcmp(metric_str, "angular") == 0 ||
                               strcmp(metric_str, "cosine")  == 0);

    fprintf(stderr, "[Load] NGTQG from: %s\n", idx_dir);
    NGTQG::Index index(idx_dir);

    fprintf(stderr, "[Load] HDF5 from: %s\n", hdf5_path);
    H5FloatDataset test_ds = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt_ds   = h5_read_int(hdf5_path, "neighbors");
    const int nq = test_ds.n_rows, D = test_ds.n_cols;
    fprintf(stderr, "  nq=%d  D=%d  k=%d  threads=%d  metric=%s\n",
            nq, D, k, n_threads, metric_str);

    // Prepare queries (normalize if angular)
    std::vector<std::vector<float>> queries(nq, std::vector<float>(D));
    for (int qi = 0; qi < nq; ++qi) {
        const float* src = test_ds.data.data() + (size_t)qi * D;
        std::copy(src, src + D, queries[qi].begin());
        if (is_angular)
            l2_normalize(queries[qi].data(), D);
    }

    // result_expansion grid (1.0 == off == quantized-only, the prior behavior).
    // QG_RE env (comma-separated) overrides for targeted high-recall runs.
    std::vector<double> re_grid = { 1.0, 2.0, 3.0, 5.0, 10.0 };
    if (const char* e = std::getenv("QG_RE")) {
        re_grid.clear(); std::string s(e); size_t p = 0;
        while (p < s.size()) { size_t c = s.find(',', p); double v = std::stod(s.substr(p, c==std::string::npos?std::string::npos:c-p));
            re_grid.push_back(v); if (c==std::string::npos) break; p = c+1; }
    }
    // QG_EPS env (comma-separated) overrides the epsilon grid for targeted runs.
    std::vector<double> eps_grid(EPSILON_GRID, EPSILON_GRID + EPSILON_GRID_SIZE);
    if (const char* e = std::getenv("QG_EPS")) {
        eps_grid.clear(); std::string s(e); size_t p = 0;
        while (p < s.size()) { size_t c = s.find(',', p); double v = std::stod(s.substr(p, c==std::string::npos?std::string::npos:c-p));
            eps_grid.push_back(v); if (c==std::string::npos) break; p = c+1; }
    }

    // Warmup with middle epsilon + a refining result_expansion
    const int WARMUP = std::min(200, nq);
    for (int wi = 0; wi < WARMUP; ++wi) {
        NGTQG::SearchQuery sq(queries[wi % nq]);
        NGT::ObjectDistances objs;
        sq.setResults(&objs); sq.setSize(k); sq.setEpsilon(0.1f); sq.setResultExpansion(3.0f);
        index.search(sq);
    }

    // Sweep epsilon × result_expansion (the official ann-benchmarks NGT-qg grid).
    for (size_t gi = 0; gi < eps_grid.size(); ++gi) {
        const float eps = (float)eps_grid[gi];
      for (size_t ri = 0; ri < re_grid.size(); ++ri) {
        const float re = (float)re_grid[ri];

        std::vector<std::vector<int>> results(nq);
        std::vector<double> latencies(nq, 0.0);
        std::atomic<int> next_qi{0};
        std::atomic<uint64_t> sum_visit{0}, sum_distcomp{0};  // QG node-touch diagnostics
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        const double t_start = bc_now_us();
        for (int t = 0; t < n_threads; ++t) {
            threads.emplace_back([&, eps, re]() {
                for (;;) {
                    int qi = next_qi.fetch_add(1, std::memory_order_relaxed);
                    if (qi >= nq) break;
                    NGTQG::SearchQuery sq(queries[qi]);
                    NGT::ObjectDistances objs;
                    sq.setResults(&objs); sq.setSize(k); sq.setEpsilon(eps);
                    sq.setResultExpansion(re);
                    const double t0 = bc_now_us();
                    index.search(sq);
                    const double t1 = bc_now_us();
                    latencies[qi] = t1 - t0;
                    sum_visit.fetch_add((uint64_t)sq.visitCount, std::memory_order_relaxed);
                    sum_distcomp.fetch_add((uint64_t)sq.distanceComputationCount, std::memory_order_relaxed);
                    results[qi].resize(objs.size());
                    for (int i = 0; i < (int)objs.size(); ++i)
                        results[qi][i] = (int)objs[i].id - 1; // NGTQG IDs are 1-based
                }
            });
        }
        for (auto& th : threads) th.join();
        const double elapsed_s = (bc_now_us() - t_start) / 1e6;

        const double qps    = nq / elapsed_s;
        const double recall = compute_recall_k(gt_ds, results, k);
        std::vector<double> lats = latencies;
        std::sort(lats.begin(), lats.end());
        const double p50 = lats[(size_t)(nq * 0.50)];
        const double p99 = lats[(size_t)(nq * 0.99)];

        printf("=== NGTQG Benchmark Result ===\n");
        printf("Index   : %s\n", idx_dir);
        printf("Dataset : %s  nq=%d  D=%d\n", hdf5_path, nq, D);
        printf("Config  : k=%d  epsilon=%.3f  result_expansion=%.1f  threads=%d\n", k, eps, re, n_threads);
        printf("recall@%d  = %.4f\n", k, recall);
        printf("agg_QPS   = %.0f\n", qps);
        printf("visit/q   = %.1f\n", (double)sum_visit.load() / nq);     // QG nodes visited
        printf("distcomp/q= %.1f\n", (double)sum_distcomp.load() / nq);  // QG distance computations
        printf("P50(us)   = %.1f\n", p50);
        printf("P99(us)   = %.1f\n\n", p99);
        fflush(stdout);
      }
    }
    return 0;
}
