// tests/ngtaq/test_dabs.cpp
#include "NGT/NGTAQ/DABSSearcher.h"
#include "NGT/NGTAQ/SoAGraph.h"
#include "NGT/NGTAQ/BinaryQuantizer.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <algorithm>

static int failures = 0;
#define EXPECT_TRUE(c) do { if(!(c)){std::cerr<<__FILE__<<":"<<__LINE__<<" FAIL "#c"\n";++failures;}}while(0)
#define EXPECT_EQ(a,b) do { if((a)!=(b)){std::cerr<<__FILE__<<":"<<__LINE__<<" FAIL "#a"=="#b" got "<<(a)<<" vs "<<(b)<<"\n";++failures;}}while(0)

// Build a trivial N-node graph where all nodes are mutually connected.
// Returns via unique_ptr because SoAGraph holds a shared_mutex (non-copyable).
std::unique_ptr<NGTAQ::SoAGraph> buildFullyConnectedBQGraph(int N, int D) {
    int words = D / 64;
    auto gp = std::make_unique<NGTAQ::SoAGraph>(words);
    NGTAQ::SoAGraph& g = *gp;
    std::mt19937 rng(99);
    std::uniform_int_distribution<uint64_t> ud;

    std::vector<uint64_t> s(words), m(words);
    for (int i = 0; i < N; i++) {
        for (auto& x : s) x = ud(rng);
        for (auto& x : m) x = ud(rng);
        g.addNode(s.data(), m.data());
    }
    g.finalizeCSR();
    // Connect each node to all others
    for (int i = 0; i < N; i++) {
        std::vector<uint32_t> nbrs;
        for (int j = 0; j < N; j++) if (j != i) nbrs.push_back(j);
        g.setNeighbors(i, nbrs);
    }
    return gp;
}

void testRouteReturnsKResults() {
    const int D = 64, N = 10, k = 3;
    auto gp = buildFullyConnectedBQGraph(N, D);

    NGTAQ::DABSSearcher searcher;
    searcher.gamma_enq  = 0.15f;
    searcher.gamma_term = 0.35f;

    std::vector<uint64_t> query_sign(1, 0xAAAAAAAAAAAAAAAAULL);
    std::vector<uint64_t> query_mag(1, 0xFFFFFFFFFFFFFFFFULL);

    auto candidates = searcher.route(
        query_sign.data(), query_mag.data(), k, *gp, {0} /* entry points */);

    // Should return at least k candidates for refinement (k' = 2k)
    EXPECT_TRUE(candidates.size() >= static_cast<size_t>(k));
    EXPECT_TRUE(candidates.size() <= static_cast<size_t>(std::min(k * 2, N)));

    // All returned IDs must be valid
    for (uint32_t id : candidates) {
        EXPECT_TRUE(id < static_cast<uint32_t>(N));
    }
}

void testRouteNoDuplicates() {
    const int D = 64, N = 8, k = 4;
    auto gp = buildFullyConnectedBQGraph(N, D);

    NGTAQ::DABSSearcher searcher;
    searcher.gamma_enq  = 0.1f;
    searcher.gamma_term = 0.3f;

    std::vector<uint64_t> qs(1, 0), qm(1, 0xFFFFFFFFFFFFFFFFULL);
    auto cands = searcher.route(qs.data(), qm.data(), k, *gp, {0});

    std::sort(cands.begin(), cands.end());
    auto uniq_end = std::unique(cands.begin(), cands.end());
    EXPECT_TRUE(uniq_end == cands.end()); // no duplicates
}

void testColdStartExploresAll() {
    // Sub-test A: N <= k_prime → all N nodes should appear in results.
    // With gamma gates very large, no early termination → result_q fills to min(k_prime, N)=N.
    {
        const int D = 64, N = 5, k = 3;
        // N=5, k_prime=6 → min(6,5)=5=N, so all N nodes fit in result_q.
        static_assert(N <= k * 2, "sub-test A requires N <= k_prime");
        auto gp = buildFullyConnectedBQGraph(N, D);

        NGTAQ::DABSSearcher searcher;
        searcher.gamma_enq  = 1000.0f;
        searcher.gamma_term = 1000.0f;

        std::vector<uint64_t> qs(1, 0), qm(1, 0xFFFFFFFFFFFFFFFFULL);
        auto cands = searcher.route(qs.data(), qm.data(), k, *gp, {0});

        // All N=5 nodes fit in k_prime=6 slots; verify count and membership.
        EXPECT_EQ(cands.size(), static_cast<size_t>(N));
        std::vector<uint32_t> sorted_cands = cands;
        std::sort(sorted_cands.begin(), sorted_cands.end());
        for (int i = 0; i < N; ++i) {
            EXPECT_TRUE(std::binary_search(sorted_cands.begin(), sorted_cands.end(),
                                           static_cast<uint32_t>(i)));
        }
    }

    // Sub-test B: N > k_prime → result_q is capped at k_prime, not all N nodes returned.
    // Verify exactly k_prime distinct IDs are returned and no duplicates.
    {
        const int D = 64, N = 20, k = 4;  // k_prime = 8 < 20
        auto gp = buildFullyConnectedBQGraph(N, D);

        NGTAQ::DABSSearcher searcher;
        searcher.gamma_enq  = 1000.0f;
        searcher.gamma_term = 1000.0f;

        std::vector<uint64_t> qs(1, 0), qm(1, 0xFFFFFFFFFFFFFFFFULL);
        auto cands = searcher.route(qs.data(), qm.data(), k, *gp, {0});

        // k_prime = 8, N = 20 → exactly k_prime results (cap applies)
        const int k_prime = k * 2;
        EXPECT_EQ(cands.size(), static_cast<size_t>(std::min(k_prime, N)));

        // No duplicates
        std::vector<uint32_t> sorted_cands = cands;
        std::sort(sorted_cands.begin(), sorted_cands.end());
        auto uniq_end = std::unique(sorted_cands.begin(), sorted_cands.end());
        EXPECT_TRUE(uniq_end == sorted_cands.end());

        // All IDs valid
        for (uint32_t id : cands) EXPECT_TRUE(id < static_cast<uint32_t>(N));
    }
}

int main() {
    testRouteReturnsKResults();
    testRouteNoDuplicates();
    testColdStartExploresAll();
    if (failures > 0) { std::cerr << failures << " test(s) FAILED\n"; return 1; }
    std::cout << "All DABSSearcher tests PASSED\n";
    return 0;
}
