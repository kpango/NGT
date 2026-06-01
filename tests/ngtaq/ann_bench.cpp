// tests/ngtaq/ann_bench.cpp
// Unified ANN-Benchmarks benchmark for NGTAQv2.
// Usage: ann_bench <aq_index_dir> <hdf5_path> [k=10] [gamma_enq=0.20] [gamma_term=0.40]
//                  [rerank_factor=3] [n_threads=4] [n_probe=0] [max_visits=0]
//
// Queries are zero-padded to D_eff (same as training).
// Angular/Cosine: queries are L2-normalized before search (handled inside searchV2).
// Outputs: recall@k, agg_QPS (N_queries / elapsed_s), P50(us), P99(us).
// Exit code: 0 if recall >= 0.99, else 1.
#include "NGT/NGTAQ/AQIndex.h"
#include "hdf5_io.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(
        steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <aq_index_dir> <hdf5_path> [k=10] [gamma_enq=0.20]"
            " [gamma_term=0.40] [rerank_factor=3] [n_threads=4] [n_probe=0]"
            " [max_visits=0] [seeds_per_cluster=-1]\n"
            "  seeds_per_cluster: angular per-cluster seed cap. -1=use index default,"
            " 0=unbounded (legacy full scan), >0=cap.\n",
            argv[0]);
        return 1;
    }
    const char* idx_dir     = argv[1];
    const char* hdf5_path   = argv[2];
    int   k             = (argc > 3) ? std::stoi(argv[3]) : 10;
    float gamma_enq     = (argc > 4) ? std::stof(argv[4]) : 0.20f;
    float gamma_term    = (argc > 5) ? std::stof(argv[5]) : 0.40f;
    int   rerank_factor = (argc > 6) ? std::stoi(argv[6]) : 3;
    int   n_threads     = (argc > 7) ? std::stoi(argv[7]) : 4;
    int   n_probe       = (argc > 8) ? std::stoi(argv[8]) : 0;
    int   max_visits    = (argc > 9) ? std::stoi(argv[9]) : 0;
    int   seeds_per_cluster = (argc > 10) ? std::stoi(argv[10]) : -1;

    // Load AQv2 index
    fprintf(stderr, "[Load] AQv2 from: %s\n", idx_dir);
    NGTAQ::NGTAQIndex idx = NGTAQ::NGTAQIndex::load(std::string(idx_dir) + "/aqindex");
    idx.loadV2(idx_dir);
    if (n_probe > 0) idx.setNProbe(n_probe);
    if (seeds_per_cluster >= 0) idx.setSeedsPerCluster(seeds_per_cluster);
    const int D_eff = idx.dEff();

    // Load test queries + ground truth
    fprintf(stderr, "[Load] HDF5 from: %s\n", hdf5_path);
    H5FloatDataset test_ds = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt_ds   = h5_read_int(hdf5_path, "neighbors");
    const int nq = test_ds.n_rows;
    const int D  = test_ds.n_cols;
    fprintf(stderr, "  nq=%d  D=%d  D_eff=%d  k=%d  rerank_factor=%d  threads=%d\n",
        nq, D, D_eff, k, rerank_factor, n_threads);

    // Prepare padded queries (zero-pad D → D_eff)
    std::vector<std::vector<float>> queries(nq, std::vector<float>(D_eff, 0.f));
    for (int qi = 0; qi < nq; ++qi) {
        const float* src = test_ds.data.data() + (size_t)qi * D;
        std::copy(src, src + D, queries[qi].begin());
        // Angular normalization is handled inside searchV2 if is_angular_=true
    }

    // Warmup
    const int WARMUP = std::min(200, nq);
    for (int qi = 0; qi < WARMUP; ++qi)
        idx.searchV2(queries[qi % nq], k, gamma_enq, gamma_term, rerank_factor, max_visits);

    // Parallel benchmark
    std::vector<std::vector<int>>  results(nq);
    std::vector<double>            latencies(nq, 0.0);
    std::atomic<int> next_qi{0};
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    const double t_start = now_us();
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&]() {
            for (;;) {
                int qi = next_qi.fetch_add(1, std::memory_order_relaxed);
                if (qi >= nq) break;
                const double t0 = now_us();
                auto sr = idx.searchV2(queries[qi], k, gamma_enq, gamma_term, rerank_factor, max_visits);
                const double t1 = now_us();
                latencies[qi] = t1 - t0;
                results[qi].resize(sr.size());
                for (int i = 0; i < (int)sr.size(); ++i)
                    results[qi][i] = (int)sr[i].id;
            }
        });
    }
    for (auto& th : threads) th.join();
    const double elapsed_s = (now_us() - t_start) / 1e6;

    // Metrics
    const double qps    = nq / elapsed_s;
    const double recall = compute_recall_k(gt_ds, results, k);
    std::vector<double> lats = latencies;
    std::sort(lats.begin(), lats.end());
    const double p50 = lats[(size_t)(nq * 0.50)];
    const double p99 = lats[(size_t)(nq * 0.99)];

    printf("=== ANN-Benchmarks Result ===\n");
    printf("Index   : %s\n", idx_dir);
    printf("Dataset : %s  nq=%d  D=%d  D_eff=%d\n", hdf5_path, nq, D, D_eff);
    printf("Config  : k=%d  gamma_enq=%.3f  gamma_term=%.3f  rerank_factor=%d  threads=%d  n_probe=%d  max_visits=%d  seeds_per_cluster=%d\n",
        k, gamma_enq, gamma_term, rerank_factor, n_threads, n_probe, max_visits, seeds_per_cluster);
    printf("recall@%d  = %.4f%s\n", k, recall, recall >= 0.99 ? "  *** >=0.99 ***" : "");
    printf("agg_QPS   = %.0f\n", qps);
    printf("P50(us)   = %.1f\n", p50);
    printf("P99(us)   = %.1f\n", p99);

    return (recall >= 0.99) ? 0 : 1;
}
