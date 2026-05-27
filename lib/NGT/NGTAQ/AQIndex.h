// lib/NGT/NGTAQ/AQIndex.h
#pragma once

#include "NGT/Index.h"
#include "NGT/NGTAQ/AlphaCGPruner.h"
#include "NGT/NGTAQ/BinaryQuantizer.h"
#include "NGT/NGTAQ/DABSSearcher.h"
#include "NGT/NGTAQ/SoAGraph.h"

#include <fstream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

namespace NGTAQ {

class NGTAQIndex {
public:
    struct Property {
        int    dimension      = 128;
        float  alpha          = 1.2f;
        float  kappa          = 1.0f;
        float  gamma_enq      = 0.15f;
        float  gamma_term     = 0.35f;
        float  k_prime_factor = 2.0f;
        int    n_tau_samples  = 10000;
        int    n_entry_points = 8;
        int    max_edges      = 64;
        NGT::ObjectSpace::DistanceType metric =
            NGT::ObjectSpace::DistanceTypeL2;
    };

    // Build from an existing NGT float32 index.
    static NGTAQIndex fromNGT(const std::string& ngt_path, const Property& prop);

    // Returns top-k results sorted by exact distance.
    std::vector<SearchResult> search(const std::vector<float>& query, int k) const;

    // Encode and insert vec into the AQ graph. Returns 0-based node ID.
    uint32_t insert(const std::vector<float>& vec);

    // Mark node as tombstone (removed on next rebuild()).
    void remove(uint32_t id);

    // Compact tombstones and re-select entry points.
    void rebuild();

    size_t size() const;

    // Thread safety: must be called without concurrent insert()/remove()/rebuild().
    void save(const std::string& path) const;
    static NGTAQIndex load(const std::string& path);

private:
    // Concurrency model:
    //   graph_->mutex() (std::shared_mutex) protects: graph_, raw_vecs_, entry_points_
    //   Shared lock: search(), size()
    //   Unique lock: insert(), remove(), rebuild()
    //   No lock: save() — caller must ensure no concurrent mutations
    //
    // prop_, bq_, pruner_, searcher_ are effectively immutable after construction.
    Property                        prop_;
    BinaryQuantizer                 bq_;
    std::unique_ptr<SoAGraph>       graph_;       // unique_ptr because SoAGraph is non-movable
    AlphaCGPruner                   pruner_;
    DABSSearcher                    searcher_;
    std::vector<uint32_t>           entry_points_;
    std::vector<std::vector<float>> raw_vecs_;    // exact float vectors for refinement

    // Private constructor used by fromNGT() and load().
    NGTAQIndex(Property prop, BinaryQuantizer bq,
               std::unique_ptr<SoAGraph> graph,
               std::vector<uint32_t> eps,
               std::vector<std::vector<float>> raw_vecs);

    static std::vector<uint32_t> selectEntryPoints(
        const SoAGraph& graph, int n, uint32_t seed = 42);
};

} // namespace NGTAQ
