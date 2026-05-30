// tests/ngtaq/qbg_bench.cpp
// ANN-Benchmarks benchmark for QBG (Quantized Blob Graph) with numOfProbes sweep.
//
// Usage: qbg_bench <qbg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]
//
// Sweeps over a fixed grid of numOfProbes values (analogous to the epsilon grid for QG).
// For each probe count, reports recall@k, aggregate QPS, P50 latency, P99 latency.
#include "bench_common.hpp"
#include "NGT/NGTQ/QuantizedBlobGraph.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Fixed numOfProbes sweep grid — covers full recall-QPS curve.
static const int PROBE_GRID[] = {
    1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 150, 290
};
static const int PROBE_GRID_SIZE = 13;

// graphExplorationSize: scales with probes to avoid being the bottleneck.
static int ges_for(int probes) {
    if (probes <=  5) return  10;
    if (probes <= 10) return  20;
    if (probes <= 30) return  50;
    if (probes <= 80) return 100;
    return 200;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <qbg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]\n",
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

    fprintf(stderr, "[Load] QBG from: %s\n", idx_dir);
    QBG::Index index(idx_dir, /*prebuilt=*/true);
    fprintf(stderr, "[Load] QBG ready.\n");

    fprintf(stderr, "[Load] HDF5 from: %s\n", hdf5_path);
    H5FloatDataset test_ds = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt_ds   = h5_read_int(hdf5_path, "neighbors");
    const int nq = test_ds.n_rows, D = test_ds.n_cols;
    fprintf(stderr, "  nq=%d  D=%d  k=%d  threads=%d  metric=%s\n",
            nq, D, k, n_threads, metric_str);

    // Prepare query vectors (normalize if angular)
    std::vector<std::vector<float>> queries(nq, std::vector<float>(D));
    for (int qi = 0; qi < nq; ++qi) {
        const float* src = test_ds.data.data() + (size_t)qi * D;
        std::copy(src, src + D, queries[qi].begin());
        if (is_angular)
            l2_normalize(queries[qi].data(), D);
    }

    // Warmup: middle probe count
    {
        const int W = std::min(200, nq);
        for (int wi = 0; wi < W; ++wi) {
            std::vector<float> qv = queries[wi % nq];
            QBG::SearchContainer sc;
            sc.setObjectVector(qv);
            sc.setSize(k);
            sc.setEpsilon(0.1f);
            sc.setBlobEpsilon(0.0f);
            sc.setNumOfProbes(20);
            sc.setGraphExplorationSize(50);
            NGT::ObjectDistances res;
            sc.setResults(&res);
            try { index.searchInTwoSteps(sc); } catch (const std::exception& e) {
                fprintf(stderr, "[Warn] warmup search: %s\n", e.what());
            }
        }
    }

    // Sweep over numOfProbes
    for (int gi = 0; gi < PROBE_GRID_SIZE; ++gi) {
        const int probes = PROBE_GRID[gi];
        const int ges    = ges_for(probes);

        std::vector<std::vector<int>> results(nq);
        std::vector<double> latencies(nq, 0.0);
        std::atomic<int> next_qi{0};
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        const double t_start = bc_now_us();
        for (int t = 0; t < n_threads; ++t) {
            threads.emplace_back([&, probes, ges]() {
                for (;;) {
                    int qi = next_qi.fetch_add(1, std::memory_order_relaxed);
                    if (qi >= nq) break;

                    std::vector<float> qv = queries[qi];  // mutable copy required
                    QBG::SearchContainer sc;
                    sc.setObjectVector(qv);
                    sc.setSize(k);
                    sc.setEpsilon(0.1f);
                    sc.setBlobEpsilon(0.0f);
                    sc.setNumOfProbes(static_cast<size_t>(probes));
                    sc.setGraphExplorationSize(static_cast<size_t>(ges));
                    NGT::ObjectDistances objs;
                    sc.setResults(&objs);

                    const double t0 = bc_now_us();
                    try { index.searchInTwoSteps(sc); } catch (const std::exception& e) {
                        fprintf(stderr, "[Error] query %d: %s\n", qi, e.what());
                    }
                    const double t1 = bc_now_us();
                    latencies[qi] = t1 - t0;

                    results[qi].resize(objs.size());
                    for (int i = 0; i < (int)objs.size(); ++i)
                        results[qi][i] = (int)objs[i].id - 1;  // QBG IDs are 1-based
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

        printf("=== QBG Benchmark Result ===\n");
        printf("Index   : %s\n", idx_dir);
        printf("Dataset : %s  nq=%d  D=%d\n", hdf5_path, nq, D);
        printf("Config  : k=%d  numOfProbes=%d  graphExplorationSize=%d  threads=%d\n",
               k, probes, ges, n_threads);
        printf("recall@%d  = %.4f\n", k, recall);
        printf("agg_QPS   = %.0f\n", qps);
        printf("P50(us)   = %.1f\n", p50);
        printf("P99(us)   = %.1f\n\n", p99);
        fflush(stdout);
    }
    return 0;
}
