// tests/ngtaq/qg_bench_re.cpp
// ANN-Benchmarks benchmark for NGTQG sweeping result_expansion (the real
// recall lever for quantized graphs), at a small fixed epsilon. Single-thread.
// Usage: qg_bench_re <qg_idx_dir> <hdf5_path> [k=10] [threads=1] [metric=l2|angular] [epsilon=0.02]
#include "bench_common.hpp"
#include "NGT/NGTQ/QuantizedGraph.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static const double RE_GRID[] = {1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0};
static const int RE_GRID_SIZE = 9;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <qg_idx_dir> <hdf5_path> [k=10] [threads=1] [metric=l2|angular] [epsilon=0.02]\n",
            argv[0]);
        return 1;
    }
    const char* idx_dir    = argv[1];
    const char* hdf5_path  = argv[2];
    int   k                = (argc > 3) ? std::stoi(argv[3]) : 10;
    int   n_threads        = (argc > 4) ? std::stoi(argv[4]) : 1;
    const char* metric_str = (argc > 5) ? argv[5] : "l2";
    const float epsilon    = (argc > 6) ? std::stof(argv[6]) : 0.02f;
    const bool is_angular  = (strcmp(metric_str, "angular") == 0 ||
                               strcmp(metric_str, "cosine")  == 0);

    fprintf(stderr, "[Load] NGTQG from: %s\n", idx_dir);
    NGTQG::Index index(idx_dir);

    fprintf(stderr, "[Load] HDF5 from: %s\n", hdf5_path);
    H5FloatDataset test_ds = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt_ds   = h5_read_int(hdf5_path, "neighbors");
    const int nq = test_ds.n_rows, D = test_ds.n_cols;
    fprintf(stderr, "  nq=%d  D=%d  k=%d  threads=%d  metric=%s  epsilon=%.3f\n",
            nq, D, k, n_threads, metric_str, epsilon);

    std::vector<std::vector<float>> queries(nq, std::vector<float>(D));
    for (int qi = 0; qi < nq; ++qi) {
        const float* src = test_ds.data.data() + (size_t)qi * D;
        std::copy(src, src + D, queries[qi].begin());
        if (is_angular)
            l2_normalize(queries[qi].data(), D);
    }

    const int WARMUP = std::min(200, nq);
    for (int wi = 0; wi < WARMUP; ++wi) {
        NGTQG::SearchQuery sq(queries[wi % nq]);
        NGT::ObjectDistances objs;
        sq.setResults(&objs); sq.setSize(k); sq.setEpsilon(epsilon);
        sq.setResultExpansion(3.0f);
        index.search(sq);
    }

    for (int gi = 0; gi < RE_GRID_SIZE; ++gi) {
        const float re = (float)RE_GRID[gi];

        std::vector<std::vector<int>> results(nq);
        std::vector<double> latencies(nq, 0.0);
        std::atomic<int> next_qi{0};
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        const double t_start = bc_now_us();
        for (int t = 0; t < n_threads; ++t) {
            threads.emplace_back([&, re]() {
                for (;;) {
                    int qi = next_qi.fetch_add(1, std::memory_order_relaxed);
                    if (qi >= nq) break;
                    NGTQG::SearchQuery sq(queries[qi]);
                    NGT::ObjectDistances objs;
                    sq.setResults(&objs); sq.setSize(k); sq.setEpsilon(epsilon);
                    sq.setResultExpansion(re);
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
        printf("Config  : k=%d  epsilon=%.3f  result_expansion=%.1f  threads=%d\n",
               k, epsilon, re, n_threads);
        printf("recall@%d  = %.4f\n", k, recall);
        printf("agg_QPS   = %.0f\n", qps);
        printf("P50(us)   = %.1f\n", p50);
        printf("P99(us)   = %.1f\n\n", p99);
        fflush(stdout);
    }
    return 0;
}
