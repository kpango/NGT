// lib/NGT/NGTAQ/DABSSearcher.h
//
// DABS (Dual-Adaptive Beam Search) searcher for AQ-DABS.
//
// Algorithm:
//   1. Encode query to BQ (caller's responsibility).
//   2. Initialize candidate min-heap with entry points.
//   3. Pop closest unvisited x:
//      - if k results found AND δ_BQ(q,x) > (1+γ_term)*d_k → TERMINATE
//   4. For each neighbor u of x:
//      - if k results found AND δ_BQ(q,u) > (1+γ_enq)*d_k → skip
//      - if not visited: push to candidate queue + update result heap
//   5. Return top-k' = 2k candidate IDs sorted by BQ distance ascending.
//
// Cold start: γ gates don't fire until k results are accumulated.
// d_k = k-th best BQ distance (max-heap top when result_q has exactly k entries).
//
#pragma once

#include "NGT/NGTAQ/BQDistance.h"
#include "NGT/NGTAQ/SoAGraph.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

namespace NGTAQ {

struct SearchResult {
    uint32_t id;
    float    distance;     // exact distance (after refinement)
    float    bq_distance;  // BQ routing distance
};

class DABSSearcher {
public:
    float gamma_enq      = 0.15f;  // enqueue gate
    float gamma_term     = 0.35f;  // termination gate
    float k_prime_factor = 2.0f;   // refinement candidates = k * k_prime_factor

    // BQ-only routing. Returns up to k' = k*k_prime_factor candidate IDs sorted
    // by BQ distance ascending.
    std::vector<uint32_t> route(
        const uint64_t* query_sign,
        const uint64_t* query_mag,
        int k,
        const SoAGraph& graph,
        const std::vector<uint32_t>& entry_points) const
    {
        const int k_prime = static_cast<int>(k * k_prime_factor);
        const int words   = graph.words();
        const int D       = words * 64;

        using Entry = std::pair<float, uint32_t>;
        // Min-heap: closest candidate first
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> cand_q;
        // Max-heap of size k_prime: farthest on top, used for result collection
        std::priority_queue<Entry> result_q;
        std::unordered_set<uint32_t> visited;

        // d_k: k-th best BQ distance seen so far; gates don't fire until k results found
        float d_k = std::numeric_limits<float>::infinity();

        // Seed from entry points
        for (uint32_t ep : entry_points) {
            if (ep >= static_cast<uint32_t>(graph.size()) || graph.isTombstone(ep))
                continue;
            if (visited.count(ep)) continue;
            float d = bqDistance(
                query_sign, query_mag,
                graph.getSignPlane(ep), graph.getMagPlane(ep),
                words, D);
            cand_q.push({d, ep});
            visited.insert(ep);
        }

        while (!cand_q.empty()) {
            auto [dist_qx, x] = cand_q.top();
            cand_q.pop();

            // Termination gate: only after k results accumulated
            if (static_cast<int>(result_q.size()) >= k &&
                dist_qx > (1.0f + gamma_term) * d_k) {
                break;
            }

            // Update result heap
            if (static_cast<int>(result_q.size()) < k_prime) {
                result_q.push({dist_qx, x});
                // When we first accumulate k results, pin d_k to the k-th best
                // (max-heap top = k-th worst = k-th best when size == k)
                if (static_cast<int>(result_q.size()) == k) {
                    d_k = result_q.top().first;
                }
            } else if (dist_qx < result_q.top().first) {
                result_q.pop();
                result_q.push({dist_qx, x});
                // After k_prime results, top = k_prime-th worst >= k-th worst.
                // Conservative: gates may fire slightly later, but correctness maintained.
            }

            // Explore neighbors
            auto neighbors = graph.getNeighbors(x);
            for (uint32_t u : neighbors) {
                if (graph.isTombstone(u)) continue;
                if (visited.count(u)) continue;
                visited.insert(u);

                float d_qu = bqDistance(
                    query_sign, query_mag,
                    graph.getSignPlane(u), graph.getMagPlane(u),
                    words, D);

                // Enqueue gate: skip if clearly too far (only after k results)
                if (static_cast<int>(result_q.size()) >= k &&
                    d_qu > (1.0f + gamma_enq) * d_k) {
                    continue;
                }

                cand_q.push({d_qu, u});
            }
        }

        // Drain result_q into sorted vector (ascending by BQ distance)
        std::vector<Entry> rv;
        rv.reserve(result_q.size());
        while (!result_q.empty()) {
            rv.push_back(result_q.top());
            result_q.pop();
        }
        std::sort(rv.begin(), rv.end());  // ascending by distance

        std::vector<uint32_t> ids;
        ids.reserve(rv.size());
        for (auto& [d, id] : rv) ids.push_back(id);
        return ids;
    }
};

} // namespace NGTAQ
