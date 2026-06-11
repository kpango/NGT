// tests/arcflare/benchmark_sift1m.cpp
// SIFT-1M benchmark: measures QPS, P50/P99 latency, and recall@k for ArcFlareIndex.
//
// Usage:
//   ./benchmark_sift1m <ngt_path> <query.fvecs> <groundtruth.ivecs> [k=10] [gamma_term=0.35]
//
// Ground truth IDs in SIFT-1M ivecs are 0-indexed and match ArcFlare node IDs directly.
#include "NGT/ArcFlare/ArcFlareIndex.h"
#include "fvecs_io.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <ngt_path> <query.fvecs> <groundtruth.ivecs>"
                  << " [k=10] [gamma_term=0.35] [k_prime_factor=2.0] [aq_cache_path]\n"
                  << "  aq_cache_path: if given, load ArcFlare index from cache (or build+save on first run)\n";
        return 1;
    }

    const std::string ngt_path  = argv[1];
    const std::string query_path = argv[2];
    const std::string gt_path    = argv[3];

    int k = 10;
    if (argc > 4) {
        try { k = std::stoi(argv[4]); }
        catch (const std::exception& e) {
            std::cerr << "Invalid k='" << argv[4] << "': " << e.what() << "\n";
            return 1;
        }
    }

    float gamma_term = 0.35f;
    if (argc > 5) {
        try { gamma_term = std::stof(argv[5]); }
        catch (const std::exception& e) {
            std::cerr << "Invalid gamma_term='" << argv[5] << "': " << e.what() << "\n";
            return 1;
        }
    }

    float k_prime_factor = 2.0f;
    if (argc > 6) {
        try { k_prime_factor = std::stof(argv[6]); }
        catch (const std::exception& e) {
            std::cerr << "Invalid k_prime_factor='" << argv[6] << "': " << e.what() << "\n";
            return 1;
        }
    }

    // Optional ArcFlare index cache: load if exists, else build + save
    std::string aq_cache_path;
    if (argc > 7) aq_cache_path = argv[7];

    // --- Load data ---
    std::cout << "Loading queries from: " << query_path << "\n";
    std::vector<std::vector<float>> queries;
    try { queries = ArcFlare::loadFvecs(query_path); }
    catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 1; }

    std::cout << "Loading ground truth from: " << gt_path << "\n";
    std::vector<std::vector<int32_t>> ground_truth;
    try { ground_truth = ArcFlare::loadIvecs(gt_path); }
    catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 1; }

    if (queries.empty()) { std::cerr << "No queries loaded.\n"; return 1; }
    if (ground_truth.empty()) { std::cerr << "No ground truth loaded.\n"; return 1; }
    if (queries.size() != ground_truth.size()) {
        std::cerr << "Mismatch: " << queries.size() << " queries vs "
                  << ground_truth.size() << " GT entries\n";
        return 1;
    }

    const int D = static_cast<int>(queries[0].size());
    std::cout << "Loaded " << queries.size() << " queries, D=" << D
              << ", k=" << k << ", gamma_term=" << gamma_term
              << ", k_prime_factor=" << k_prime_factor << "\n";

    // Flush so progress is visible even when stdout is file-buffered
    std::cout.flush();

    // --- Build or load ArcFlareIndex ---
    ArcFlare::ArcFlareIndex::Property prop;
    prop.dimension      = D;
    prop.n_tau_samples  = 10000;
    prop.metric         = NGT::ObjectSpace::DistanceTypeL2;
    prop.gamma_term     = gamma_term;
    prop.gamma_enq      = gamma_term * 0.43f;
    prop.k_prime_factor = k_prime_factor;

    ArcFlare::ArcFlareIndex aq = [&]() -> ArcFlare::ArcFlareIndex {
        // Use cached index if available (avoids ~30-min fromNGT rebuild)
        if (!aq_cache_path.empty()) {
            std::ifstream probe(aq_cache_path);
            if (probe.good()) {
                probe.close();
                std::cout << "Loading cached ArcFlareIndex from: " << aq_cache_path << " ...\n";
                std::cout.flush();
                try {
                    auto idx = ArcFlare::ArcFlareIndex::load(aq_cache_path);
                    std::cout << "Loaded. Size=" << idx.size() << "\n";
                    std::cout.flush();
                    return idx;
                } catch (const std::exception& e) {
                    std::cerr << "Cache load failed (" << e.what() << "), rebuilding...\n";
                }
            }
        }
        std::cout << "Building ArcFlareIndex from: " << ngt_path << " (this may take ~30 min for 1M vecs) ...\n";
        std::cout.flush();
        auto t0 = std::chrono::steady_clock::now();
        auto idx = ArcFlare::ArcFlareIndex::fromNGT(ngt_path, prop);
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "Built in " << elapsed << "s. Size=" << idx.size() << "\n";
        std::cout.flush();
        if (!aq_cache_path.empty()) {
            std::cout << "Saving to cache: " << aq_cache_path << " ...\n";
            std::cout.flush();
            try { idx.save(aq_cache_path); std::cout << "Saved.\n"; std::cout.flush(); }
            catch (const std::exception& e) { std::cerr << "Save failed: " << e.what() << "\n"; }
        }
        return idx;
    }();
    std::cout << "Index built. Size=" << aq.size() << "\n";

    // --- Warmup: 100 queries (untimed) ---
    const size_t warmup_n = std::min(queries.size(), size_t(100));
    std::cout << "Warming up with " << warmup_n << " queries...\n";
    for (size_t i = 0; i < warmup_n; ++i) {
        aq.search(queries[i], k);
    }

    // --- Timed run: all queries ---
    const size_t nq = queries.size();
    std::vector<double> latencies_us(nq);
    std::vector<std::vector<ArcFlare::SearchResult>> all_results(nq);

    std::cout << "Running timed benchmark over " << nq << " queries...\n";
    for (size_t i = 0; i < nq; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        all_results[i] = aq.search(queries[i], k);
        const auto t1 = std::chrono::high_resolution_clock::now();
        latencies_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // --- Compute QPS ---
    double total_us = 0.0;
    for (double l : latencies_us) total_us += l;
    const double total_s = total_us / 1e6;
    const double qps = static_cast<double>(nq) / total_s;

    // --- Compute P50, P99 ---
    std::vector<double> sorted_lat = latencies_us;
    std::sort(sorted_lat.begin(), sorted_lat.end());

    // Index for Pnn: ceil(n/100 * nq) - 1, clamped
    auto percentile_idx = [&](double p) -> size_t {
        size_t idx = static_cast<size_t>(p / 100.0 * static_cast<double>(nq));
        if (idx >= nq) idx = nq - 1;
        return idx;
    };
    const double p50 = sorted_lat[percentile_idx(50.0)];
    const double p99 = sorted_lat[percentile_idx(99.0)];

    // --- Compute recall@k ---
    // GT IDs are 0-indexed; ArcFlareIndex also returns 0-based IDs (fromNGT converts 1-based NGT IDs).
    double total_recall = 0.0;
    for (size_t i = 0; i < nq; ++i) {
        const auto& gt   = ground_truth[i];
        const auto& res  = all_results[i];

        // Build set of GT top-k IDs (up to k entries)
        const int gt_k = std::min(static_cast<int>(gt.size()), k);
        std::unordered_set<int32_t> gt_set(gt.begin(), gt.begin() + gt_k);

        // Count how many result IDs appear in GT set
        int hits = 0;
        const int res_k = std::min(static_cast<int>(res.size()), k);
        for (int j = 0; j < res_k; ++j) {
            if (gt_set.count(static_cast<int32_t>(res[j].id))) ++hits;
        }

        total_recall += static_cast<double>(hits) / static_cast<double>(gt_k);
    }
    const double recall = total_recall / static_cast<double>(nq);

    // --- Output ---
    std::cout << "\n=== Results ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  QPS       = " << qps << "\n";
    std::cout << "  P50       = " << p50 << " µs\n";
    std::cout << "  P99       = " << p99 << " µs\n";
    std::cout << std::setprecision(4);
    std::cout << "  recall@" << k << "  = " << recall << "\n";

    // --- Batch search benchmark ---
    const int batch_threads = omp_get_max_threads();
    std::cout << "\n=== Batch Search (n_threads=" << batch_threads << ") ===\n";
    std::cout.flush();

    // Warm up batch search
    {
        ArcFlare::ArcFlareIndex::Property bp = prop;
        bp.n_search_threads = batch_threads;
        // Just measure against current aq object (n_search_threads=0 → max threads)
    }

    const auto batch_t0 = std::chrono::high_resolution_clock::now();
    auto batch_results = aq.searchBatch(queries, k);
    const auto batch_t1 = std::chrono::high_resolution_clock::now();

    const double batch_total_us =
        std::chrono::duration<double, std::micro>(batch_t1 - batch_t0).count();
    const double batch_qps = static_cast<double>(nq) / (batch_total_us / 1e6);

    // Sanity-check recall for batch results
    double batch_recall = 0.0;
    for (size_t i = 0; i < nq; ++i) {
        const auto& gt  = ground_truth[i];
        const auto& res = batch_results[i];
        const int gt_k  = std::min(static_cast<int>(gt.size()), k);
        std::unordered_set<int32_t> gt_set(gt.begin(), gt.begin() + gt_k);
        int hits = 0;
        const int res_k = std::min(static_cast<int>(res.size()), k);
        for (int j = 0; j < res_k; ++j) {
            if (gt_set.count(static_cast<int32_t>(res[j].id))) ++hits;
        }
        batch_recall += static_cast<double>(hits) / static_cast<double>(gt_k);
    }
    batch_recall /= static_cast<double>(nq);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Batch QPS   = " << batch_qps << "\n";
    std::cout << "  Total time  = " << batch_total_us / 1e6 << " s\n";
    std::cout << std::setprecision(4);
    std::cout << "  recall@" << k << "    = " << batch_recall << "\n";
    std::cout << "  Speedup vs single-thread = "
              << std::fixed << std::setprecision(2)
              << (batch_qps / qps) << "x\n";

    return 0;
}
