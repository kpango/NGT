// tests/arcflare/benchmark_aqindex.cpp
// Benchmark harness: measures QPS at 3 gamma_term values on a given NGT index.
//
// Usage: ./benchmark_aqindex <ngt_index_path> <query_tsv> [k=10]
//
// Example (SIFT-5k):
//   ./benchmark_aqindex /tmp/arcflare_test_ngt data/sift-query-3.tsv 10
//
// Output per gamma_term: QPS and average result count (proxy for recall depth).
#include "NGT/ArcFlare/ArcFlareIndex.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<std::vector<float>> loadTSV(const std::string& p) {
    std::ifstream f(p);
    if (!f) { std::cerr << "Cannot open: " << p << "\n"; return {}; }
    std::vector<std::vector<float>> vecs;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        float v;
        std::vector<float> row;
        while (ss >> v) row.push_back(v);
        if (!row.empty()) vecs.push_back(std::move(row));
    }
    return vecs;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <ngt_index> <query_tsv> [k=10]\n";
        return 1;
    }
    const std::string ngt_path  = argv[1];
    const std::string query_path = argv[2];
    int k = 10;
    if (argc > 3) {
        try { k = std::stoi(argv[3]); }
        catch (const std::exception& e) {
            std::cerr << "Invalid k argument '" << argv[3] << "': " << e.what() << "\n";
            return 1;
        }
    }

    auto queries = loadTSV(query_path);
    if (queries.empty()) { std::cerr << "No queries loaded from: " << query_path << "\n"; return 1; }
    const int D = static_cast<int>(queries[0].size());

    std::cout << "Loaded " << queries.size() << " queries, D=" << D << ", k=" << k << "\n";

    // Benchmark at 3 gamma_term values (controls search aggressiveness).
    // gamma_enq ≈ 0.43 * gamma_term is the recommended ratio.
    const float gamma_values[] = {0.20f, 0.35f, 0.50f};

    for (float gamma_term : gamma_values) {
        ArcFlare::ArcFlareIndex::Property prop;
        prop.dimension      = D;
        prop.n_tau_samples  = 10000;
        prop.metric         = NGT::ObjectSpace::DistanceTypeL2;
        prop.gamma_term     = gamma_term;
        prop.gamma_enq      = gamma_term * 0.43f;

        std::cout << "\nBuilding index with gamma_term=" << gamma_term
                  << " gamma_enq=" << prop.gamma_enq << "...\n";

        auto aq = ArcFlare::ArcFlareIndex::fromNGT(ngt_path, prop);

        // Warmup: first 3 queries, not timed (warms instruction cache + BQ encoder)
        const size_t warmup_n = std::min(queries.size(), size_t(3));
        std::cout << "  Warming up with " << warmup_n << " queries (untimed)...\n";
        for (size_t i = 0; i < warmup_n; ++i) {
            aq.search(queries[i], k);
        }

        // Timed runs
        const auto t0 = std::chrono::high_resolution_clock::now();
        size_t total_results = 0;
        for (const auto& q : queries) {
            auto results = aq.search(q, k);
            total_results += results.size();
        }
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double qps = (static_cast<double>(queries.size()) / elapsed_ms) * 1000.0;

        std::cout << std::fixed << std::setprecision(1)
                  << "  gamma_term=" << gamma_term
                  << "  QPS=" << qps
                  << "  avg_results=" << (static_cast<double>(total_results) / queries.size())
                  << "  elapsed_ms=" << elapsed_ms
                  << "\n";
    }

    std::cout << "\nBenchmark complete.\n";
    return 0;
}
