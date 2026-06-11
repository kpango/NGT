// tests/arcflare/test_alpha_cg.cpp
#include "NGT/ArcFlare/AlphaCGPruner.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static int failures = 0;
#define EXPECT_EQ(a,b) do { if((a)!=(b)){std::cerr<<__FILE__<<":"<<__LINE__<<" FAIL "#a"=="#b"\n";++failures;}}while(0)
#define EXPECT_TRUE(c) do { if(!(c)){std::cerr<<__FILE__<<":"<<__LINE__<<" FAIL "#c"\n";++failures;}}while(0)

void testNoPruneWhenNoCandidates() {
    ArcFlare::AlphaCGPruner pruner(1.2f, 1.0f);
    // Empty candidate list → empty result
    std::vector<std::pair<uint32_t, float>> candidates; // {id, bq_dist}
    auto result = pruner.prune(candidates, 0.05f);
    EXPECT_EQ(result.size(), 0u);
}

void testKeepAllWhenFarApart() {
    // If no candidate is closer to another than the α-CG threshold, keep all.
    ArcFlare::AlphaCGPruner pruner(1.2f, 0.0f /* kappa=0: no tau adjustment */);
    // Two candidates at distance 0.3 from query, and 0.9 from each other → no pruning
    std::vector<std::pair<uint32_t, float>> candidates = {{1, 0.3f}, {2, 0.3f}};
    std::vector<float> inter_dists = {0.9f}; // dist(1,2) = 0.9

    // pruneWithDistances: inter_dists[i] = dist(accepted[i], u_current)
    auto result = pruner.pruneWithDistances(candidates, inter_dists, 0.0f);
    EXPECT_EQ(result.size(), 2u);
}

void testPruneRedundantCandidate() {
    // Candidate 2 is "behind" candidate 1 (v=1 covers u=2 for p=query).
    // dist(p, u=2) = 0.5, dist(p, v=1) = 0.3, dist(v=1, u=2) = 0.1
    // α_eff = 1.2, tau = 0
    // threshold = (1/1.2) * (0.5 - (1.2+1)*0) = 0.417
    // dist(v=1, u=2) = 0.1 < 0.417 → prune u=2
    ArcFlare::AlphaCGPruner pruner(1.2f, 0.0f);
    std::vector<std::pair<uint32_t, float>> candidates = {{1, 0.3f}, {2, 0.5f}};
    // inter_dists[0] = dist(accepted[0]=1, u=2) = 0.1
    std::vector<float> inter_dists = {0.1f};
    auto result = pruner.pruneWithDistances(candidates, inter_dists, 0.0f);
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 1u); // only v=1 kept
}

void testTauMakesThresholdMoreConservative() {
    // With tau > 0, α_eff increases → threshold (1/α_eff)*(d_pu - (α_eff+1)*tau) decreases
    // → harder to prune (more conservative)
    ArcFlare::AlphaCGPruner pruner(1.2f, 1.0f); // kappa=1
    float tau = 0.05f;
    float alpha_eff = 1.2f + 1.0f * tau; // = 1.25
    float d_pu = 0.5f;
    // threshold = (1/1.25)*(0.5 - (1.25+1)*0.05) = 0.8*(0.5 - 0.1125) = 0.8*0.3875 = 0.31
    float threshold = pruner.pruningThreshold(d_pu, tau);
    EXPECT_TRUE(std::abs(threshold - 0.31f) < 0.01f);
}

int main() {
    testNoPruneWhenNoCandidates();
    testKeepAllWhenFarApart();
    testPruneRedundantCandidate();
    testTauMakesThresholdMoreConservative();
    if (failures > 0) { std::cerr << failures << " test(s) FAILED\n"; return 1; }
    std::cout << "All AlphaCGPruner tests PASSED\n";
    return 0;
}
