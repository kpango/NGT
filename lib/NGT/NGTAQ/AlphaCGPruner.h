#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace NGTAQ {

// Noise-robust α-Convergent Graph pruner.
// Operates in BQ normalized distance space [0, 1].
// α_eff = alpha_base + kappa * tau
// Prune u if ∃ accepted neighbor v: δ_BQ(v,u) < (1/α_eff) * (δ_BQ(p,u) - (α_eff+1)*τ)
class AlphaCGPruner {
public:
    float alpha_base;  // default 1.2
    float kappa;       // default 1.0

    AlphaCGPruner(float alpha = 1.2f, float kap = 1.0f)
        : alpha_base(alpha), kappa(kap) {}

    float effectiveAlpha(float tau) const {
        return alpha_base + kappa * tau;
    }

    // Pruning threshold: if δ_BQ(v, u) < threshold, v "covers" u and u is pruned.
    // Returns a negative value when τ is large enough that the rule never fires.
    float pruningThreshold(float dist_pu, float tau) const {
        float ae = effectiveAlpha(tau);
        return (1.0f / ae) * (dist_pu - (ae + 1.0f) * tau);
    }

    // Prune a sorted candidate list using the α-CG rule.
    // candidates: sorted ascending by BQ distance from p, each element is {node_id, bq_dist_from_p}
    // tau: current BQ quantization error bound
    // dist_fn: function(uint32_t id_v, uint32_t id_u) → float bq_dist(v, u)
    //          If nullptr, no pruning is applied (all candidates are kept).
    // Returns: vector of kept node IDs (in ascending-distance order).
    std::vector<uint32_t> prune(
        const std::vector<std::pair<uint32_t, float>>& candidates,
        float tau,
        std::function<float(uint32_t, uint32_t)> dist_fn = nullptr) const
    {
        std::vector<uint32_t> accepted;
        accepted.reserve(candidates.size());
        for (auto& [uid, d_pu] : candidates) {
            bool dominated = false;
            if (dist_fn) {
                float threshold = pruningThreshold(d_pu, tau);
                if (threshold > 0.0f) {
                    for (uint32_t v : accepted) {
                        if (dist_fn(v, uid) < threshold) {
                            dominated = true;
                            break;
                        }
                    }
                }
            }
            if (!dominated) accepted.push_back(uid);
        }
        return accepted;
    }

    // Test-friendly variant: takes explicit pairwise distances instead of a distance function.
    // inter_dists layout: for each candidate ci (in order), provides dist(accepted[ai], cand[ci])
    // for ai = 0..accepted.size()-1 (at time ci is processed, i.e., BEFORE potentially adding ci).
    // Total entries needed: sum over all candidates of accepted.size() at time of processing.
    std::vector<uint32_t> pruneWithDistances(
        const std::vector<std::pair<uint32_t, float>>& candidates,
        const std::vector<float>& inter_dists,
        float tau) const
    {
        std::vector<uint32_t> accepted;
        accepted.reserve(candidates.size());
        size_t inter_idx = 0;

        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            auto [uid, d_pu] = candidates[ci];
            float threshold = pruningThreshold(d_pu, tau);
            bool dominated = false;
            size_t n_accepted = accepted.size();  // count BEFORE this candidate

            if (threshold > 0.0f) {
                for (size_t ai = 0; ai < n_accepted; ++ai) {
                    if (inter_dists[inter_idx + ai] < threshold) {
                        dominated = true;
                        break;
                    }
                }
            }
            if (!dominated) accepted.push_back(uid);
            inter_idx += n_accepted;  // advance by distances consumed (before-count)
        }
        return accepted;
    }
};

} // namespace NGTAQ
