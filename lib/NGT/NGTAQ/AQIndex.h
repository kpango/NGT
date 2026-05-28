// lib/NGT/NGTAQ/AQIndex.h
#pragma once

#include "NGT/Index.h"
#include "NGT/NGTAQ/AlphaCGPruner.h"
#include "NGT/NGTAQ/BinaryQuantizer.h"
#include "NGT/NGTAQ/DABSSearcher.h"
#include "NGT/NGTAQ/SoAGraph.h"
#include "NGT/NGTAQ/SRHT.h"
#include "NGT/NGTAQ/KMeansCentering.h"
#include "NGT/NGTAQ/PCAProjector.h"
#include "NGT/NGTAQ/ADCTable.h"
#include "NGT/NGTAQ/ADCDistance.h"

#include <fstream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

namespace NGTAQ {

class NGTAQIndex {
public:
    struct Property {
        int    dimension        = 128;
        float  alpha            = 1.2f;
        float  kappa            = 1.0f;
        float  gamma_enq        = 0.15f;
        float  gamma_term       = 0.35f;
        float  k_prime_factor   = 2.0f;
        int    n_tau_samples    = 10000;
        int    n_entry_points   = 8;
        int    max_edges        = 64;
        int    n_search_threads = 0;   // 0 = omp_get_max_threads() at call time
        NGT::ObjectSpace::DistanceType metric =
            NGT::ObjectSpace::DistanceTypeL2;
    };

    // Build from an existing NGT float32 index.
    static NGTAQIndex fromNGT(const std::string& ngt_path, const Property& prop);

    // Returns top-k results sorted by exact distance.
    std::vector<SearchResult> search(const std::vector<float>& query, int k) const;

    // Override search-time gamma gates and k_prime_factor (not thread-safe while search() is running).
    // Used for recall-QPS sweep benchmarks.
    void setSearchGammas(float gamma_enq, float gamma_term, float k_prime_factor = -1.0f) {
        searcher_.gamma_enq      = gamma_enq;
        searcher_.gamma_term     = gamma_term;
        if (k_prime_factor > 0.0f) searcher_.k_prime_factor = k_prime_factor;
    }

    // Batch search over multiple queries. Returns one result-vector per query.
    // Uses OpenMP with prop_.n_search_threads (0 = all available threads).
    std::vector<std::vector<SearchResult>> searchBatch(
        const std::vector<std::vector<float>>& queries, int k) const;

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

    // ---- v2 ADC path ----

    // Build v2 index from existing NGT float32 index.
    // SRHT rotation + K-means centering + PCA top-32 + VectorRecord encoding.
    static NGTAQIndex fromNGTv2(const std::string& ngt_path, const Property& prop);

    // ADC search using routeV2(). Requires is_v2_ == true.
    std::vector<SearchResult> searchV2(
        const std::vector<float>& query, int k,
        float gamma_enq = 0.2f, float gamma_term = 0.4f) const;

    // Save/load v2 state to directory (separate from v1 state).
    void saveV2(const std::string& dir) const;
    void loadV2(const std::string& dir);

    bool isV2() const { return is_v2_; }

private:
    // Concurrency model:
    //   graph_->mutex() (std::shared_mutex) protects: graph_, raw_flat_, entry_points_
    //   Shared lock: search(), searchBatch(), size()
    //   Unique lock: insert(), remove(), rebuild()
    //   No lock: save() — caller must ensure no concurrent mutations
    //
    // prop_, bq_, pruner_, searcher_ are effectively immutable after construction.
    Property                        prop_;
    BinaryQuantizer                 bq_;
    std::unique_ptr<SoAGraph>       graph_;
    AlphaCGPruner                   pruner_;
    DABSSearcher                    searcher_;
    std::vector<uint32_t>           entry_points_;
    std::vector<float>              raw_flat_;     // flat [N*D] exact float vectors

    // v2 ADC state (null/empty if not built via fromNGTv2)
    bool                                      is_v2_ = false;
    std::unique_ptr<NGT::NGTAQ::SRHT>         srht_v2_;
    std::unique_ptr<NGT::NGTAQ::KMeansCentering> kmeans_v2_;
    std::unique_ptr<NGT::NGTAQ::PCAProjector> pca_v2_;
    std::vector<float>                        tier2_codebook_; // [16][32] = 512 floats
    std::vector<uint32_t>                     v2_entry_points_;

    // Private constructor used by fromNGT() and load().
    NGTAQIndex(Property prop, BinaryQuantizer bq,
               std::unique_ptr<SoAGraph> graph,
               std::vector<uint32_t> eps,
               std::vector<float> raw_flat);

    static std::vector<uint32_t> selectEntryPoints(
        const SoAGraph& graph, int n, uint32_t seed = 42);
};

} // namespace NGTAQ
