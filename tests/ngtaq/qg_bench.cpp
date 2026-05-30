// tests/ngtaq/qg_bench.cpp
// ANN-Benchmarks benchmark for NGTQG (QG / QSG) with epsilon sweep.
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

    // Warmup with middle epsilon
    const int WARMUP = std::min(200, nq);
    for (int wi = 0; wi < WARMUP; ++wi) {
        NGTQG::SearchQuery sq(queries[wi % nq]);
        NGT::ObjectDistances objs;
        sq.setResults(&objs); sq.setSize(k); sq.setEpsilon(0.1f);
        index.search(sq);
    }

    // Sweep
    for (int gi = 0; gi < EPSILON_GRID_SIZE; ++gi) {
        const float eps = (float)EPSILON_GRID[gi];

        std::vector<std::vector<int>> results(nq);
        std::vector<double> latencies(nq, 0.0);
        std::atomic<int> next_qi{0};
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        const double t_start = bc_now_us();
        for (int t = 0; t < n_threads; ++t) {
            threads.emplace_back([&, eps]() {
                for (;;) {
                    int qi = next_qi.fetch_add(1, std::memory_order_relaxed);
                    if (qi >= nq) break;
                    NGTQG::SearchQuery sq(queries[qi]);
                    NGT::ObjectDistances objs;
                    sq.setResults(&objs); sq.setSize(k); sq.setEpsilon(eps);
                    const double t0 = bc_now_us();
                    index.search(sq);
                    const double t1 = bc_now_us();
                    latencies[qi] = t1 - t0;
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
        printf("Config  : k=%d  epsilon=%.3f  threads=%d\n", k, eps, n_threads);
        printf("recall@%d  = %.4f\n", k, recall);
        printf("agg_QPS   = %.0f\n", qps);
        printf("P50(us)   = %.1f\n", p50);
        printf("P99(us)   = %.1f\n\n", p99);
        fflush(stdout);
    }
    return 0;
}
