// tests/ngtaq/qbg_bench.cpp
// ANN-Benchmarks benchmark for QBG (Quantized Blob Graph).
//
// Usage: qbg_bench <qbg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]
//
// Sweeps a 2-D grid of (numOfProbes, resultExpansion). resultExpansion enables
// QBG's refinement reranking: the two-step search collects size*resultExpansion
// quantized candidates, then re-ranks them with EXACT distances against the
// genuine (float) vectors stored in the index object list, keeping the top-k.
// This is the same mechanism the `qbg search -p <result_expansion>` CLI uses and
// is what lifts SIFT recall@10 from the PQ-only ceiling (~0.89) to ~0.99.
//
// Refinement requires the index to expose a refinement object space. QBG always
// builds `refinementObjectSpaceForObjectList` from the genuine float object list
// (GenuineDataType=Float), so no rebuild / dedicated refinement data is needed;
// resultExpansion>=1.0 turns it on. resultExpansion=0 reproduces the PQ-only path.
//
// For each operating point, reports recall@k, aggregate QPS, P50/P99 latency.
//
// NOTE on threading: QBG's refinement path indexes per-thread object-list streams
// via omp_get_thread_num(); outside an OpenMP region that returns 0 for every
// std::thread, so refinement reranking is only safe single-threaded. Run with
// threads=1 (and OPENBLAS_NUM_THREADS=1/OMP_NUM_THREADS=1) for the canonical
// recall->QPS curve. threads>1 is left intact for the PQ-only (resultExpansion=0)
// rows but should not be trusted for refined rows.
#include "bench_common.hpp"
#include "NGT/NGTQ/QuantizedBlobGraph.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// One operating point on the recall->QPS curve.
struct OpPoint {
    int   probes;            // numOfProbes (blobs explored)
    int   ges;               // graphExplorationSize for the global blob graph
    float result_expansion;  // refinementExpansion; <1.0 => PQ-only (no rerank)
};

// graphExplorationSize: scales with probes so the blob graph search isn't the
// bottleneck for the requested number of probes.
static int ges_for(int probes) {
    if (probes <=  5) return  10;
    if (probes <= 10) return  20;
    if (probes <= 30) return  50;
    if (probes <= 80) return 100;
    return 200;
}

// Build the sweep. Two regimes, concatenated so recall ascends roughly monotonically:
//   1) PQ-only baseline (result_expansion = 0): a probe sweep that shows the
//      quantization ceiling (~0.89 on SIFT) and the high-QPS end of the curve.
//   2) Refined: a probe ladder at a fixed result_expansion. With refinement the
//      candidate-pool size (numOfProbes) is the dominant recall lever -- once
//      result_expansion is large enough to rerank the whole pool with exact
//      distances, recall is set by how many candidates were collected. We fix
//      result_expansion=2.0 (the QPS-optimal value that still fully reranks the
//      top-k pool on SIFT) and sweep probes to trace a clean recall->QPS curve up
//      to ~0.999. A couple of larger result_expansion points at the high end show
//      the (small) extra recall available from a deeper rerank pool.
static std::vector<OpPoint> build_sweep() {
    std::vector<OpPoint> pts;

    // 1) PQ-only ceiling (no refinement) -- the high-QPS / low-recall regime.
    static const int PQ_PROBES[] = {1, 2, 3, 5, 10, 20, 50, 100};
    for (int p : PQ_PROBES)
        pts.push_back({p, ges_for(p), 0.0f});

    // 2) Refinement-reranked curve at a fixed result_expansion: clean monotone
    //    recall up to ~0.999.
    static const int RE_PROBES[] = {2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256};
    for (int p : RE_PROBES)
        pts.push_back({p, ges_for(p), 2.0f});

    // 3) Deeper-rerank points at the high-recall end (diminishing returns).
    pts.push_back({128, ges_for(128), 5.0f});
    pts.push_back({256, ges_for(256), 5.0f});

    return pts;
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
    // prebuilt=true; refinementDataType defaults to DataTypeAny, which resolves to
    // the index's stored RefinementDataType (None here). Refinement still works via
    // refinementObjectSpaceForObjectList (genuine float object list) -- no extra RAM.
    QBG::Index index(idx_dir, /*prebuilt=*/true);
    fprintf(stderr, "[Load] QBG ready.\n");

    fprintf(stderr, "[Load] HDF5 from: %s\n", hdf5_path);
    H5FloatDataset test_ds = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt_ds   = h5_read_int(hdf5_path, "neighbors");
    const int nq = test_ds.n_rows, D = test_ds.n_cols;
    fprintf(stderr, "  nq=%d  D=%d  k=%d  threads=%d  metric=%s\n",
            nq, D, k, n_threads, metric_str);
    if (n_threads > 1)
        fprintf(stderr, "[Warn] threads>1: refinement (resultExpansion>0) rows are "
                        "only reliable single-threaded; use threads=1.\n");

    // Prepare query vectors (normalize if angular)
    std::vector<std::vector<float>> queries(nq, std::vector<float>(D));
    for (int qi = 0; qi < nq; ++qi) {
        const float* src = test_ds.data.data() + (size_t)qi * D;
        std::copy(src, src + D, queries[qi].begin());
        if (is_angular)
            l2_normalize(queries[qi].data(), D);
    }

    // Warmup: a mid-curve refined point to fault in the object list / streams.
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
            sc.setRefinementExpansion(3.0f);
            NGT::ObjectDistances res;
            sc.setResults(&res);
            try { index.searchInTwoSteps(sc); } catch (const std::exception& e) {
                fprintf(stderr, "[Warn] warmup search: %s\n", e.what());
            }
        }
    }

    const std::vector<OpPoint> sweep = build_sweep();

    for (const OpPoint& op : sweep) {
        const int   probes = op.probes;
        const int   ges    = op.ges;
        const float re     = op.result_expansion;

        std::vector<std::vector<int>> results(nq);
        std::vector<double> latencies(nq, 0.0);
        std::atomic<int> next_qi{0};
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        const double t_start = bc_now_us();
        for (int t = 0; t < n_threads; ++t) {
            threads.emplace_back([&, probes, ges, re]() {
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
                    sc.setRefinementExpansion(re);  // >=1.0 => exact-distance rerank
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
        printf("Config  : k=%d  numOfProbes=%d  graphExplorationSize=%d  resultExpansion=%.1f  threads=%d\n",
               k, probes, ges, re, n_threads);
        printf("recall@%d  = %.4f\n", k, recall);
        printf("agg_QPS   = %.0f\n", qps);
        printf("P50(us)   = %.1f\n", p50);
        printf("P99(us)   = %.1f\n\n", p99);
        fflush(stdout);
    }
    return 0;
}
