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
#include <mutex>
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
        int    k_clusters       = 0;   // 0 = use select_k(N); >0 overrides select_k
        int    n_cluster_seeds  = 32;  // seeds per cluster for DABS warm-start (larger → tighter d_k init)
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

    // Override cluster seed count at search time (not thread-safe while searchV2 is running).
    void setNClusterSeeds(int n) { prop_.n_cluster_seeds = n; }

    // Override n_probe (number of clusters to probe) at search time.
    // 0 = use default (is_angular ? 20 : 3). Higher values improve recall at cost of QPS.
    void setNProbe(int n) { n_probe_override_ = n; }

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
    // rerank_factor=0 or 1: standard beam width k (existing behavior).
    // rerank_factor>1: internally searches k*rerank_factor candidates, returns top k.
    std::vector<SearchResult> searchV2(
        const std::vector<float>& query, int k,
        float gamma_enq = 0.2f, float gamma_term = 0.4f,
        int rerank_factor = 0) const;

    // Save/load v2 state to directory (separate from v1 state).
    void saveV2(const std::string& dir) const;
    void loadV2(const std::string& dir);

    bool isV2()      const { return is_v2_; }
    bool isAngular() const { return is_angular_; }
    int  dEff()      const { return d_eff_ > 0 ? d_eff_ : prop_.dimension; }
    int  mPQ()       const { return m_pq_; }

    // Accessors for raw float vectors and dimension (used by standalone benchmarks)
    const float* rawFlat() const { return raw_flat_.empty() ? nullptr : raw_flat_.data(); }
    int dim() const { return prop_.dimension; }

    // Rebuild v2 graph edges from a (denser) NGT source index without re-training
    // SRHT/K-means/PCA/PQ (those are reused from the existing index).
    // Only the graph construction + entry point selection are re-run.
    // ~50s vs ~400s for a full fromNGTv2 rebuild.
    // new_alpha=-1 keeps existing prop_.alpha; new_max_edges=-1 keeps existing prop_.max_edges.
    void rebuildGraphFromNGT(const std::string& ngt_path,
                              float new_alpha    = -1.0f,
                              int   new_max_edges = -1);

    // Rebuild v2 graph edges using the index's own searchV2 to find high-quality
    // candidate neighbors for each node. Avoids needing a denser NGT source.
    // Runs k_search searchV2 calls per node (parallelized with n_threads).
    // gamma controls the search-time recall-QPS tradeoff for candidate generation.
    // ~30s with n_threads=16 at gamma=0.30 on N=1M SIFT-1M.
    void rebuildGraphSelf(int   k_search   = 40,
                          float gamma      = 0.30f,
                          int   n_threads  = 16,
                          float new_alpha  = -1.0f,
                          int   new_max_edges = -1);

    // Post-hoc tombstone fix: scan raw_flat_ for zero-norm vectors and tombstone them.
    // Use to repair indices built with code that did not properly tombstone hole nodes.
    // Returns the count of newly tombstoned nodes.
    int fixHoleTombstones();

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

    int                                       n_probe_override_ = 0;  // 0 = default

    // During rebuildGraphSelf, limit total seeds per searchV2 call to avoid queue
    // flooding on angular data (large clusters). 0 = no limit (full cluster scan).
    // Set before parallel loop, reset after. mutable so searchV2 (const) can read it.
    mutable int                               rebuild_max_seeds_ = 0;

    // v2 ADC state (null/empty if not built via fromNGTv2)
    bool                                      is_v2_      = false;
    bool                                      is_angular_ = false;  // true when metric is Angle or Cosine
    int                                       d_eff_      = 0;      // effective dimension (0 = prop_.dimension)
    int                                       m_pq_       = 16;     // PQ sub-codebook count
    std::unique_ptr<NGT::NGTAQ::SRHT>         srht_v2_;
    std::unique_ptr<NGT::NGTAQ::KMeansCentering> kmeans_v2_;
    std::unique_ptr<NGT::NGTAQ::PCAProjector> pca_v2_;
    std::vector<float>                        tier2_codebook_;   // [M][K][D_sub] = row-major (original layout)
    std::vector<float>                        tier2_codebook_T_; // [M][D_sub][K] = transposed (for fast AVX2 LUT build)
    std::vector<uint32_t>                     v2_entry_points_;

    // Lazy-built inverted list + cluster neighbor table for cluster-aware seeding.
    // Built once on first searchV2 call; unique_ptr keeps NGTAQIndex movable (once_flag is non-movable).
    mutable std::unique_ptr<std::once_flag>    cluster_members_once_{std::make_unique<std::once_flag>()};
    mutable std::vector<std::vector<uint32_t>> cluster_members_v2_;  // cluster_id → [node_ids]
    mutable std::vector<std::vector<uint32_t>> cluster_neighbors_v2_; // cluster_id → [nearest cluster_ids]

    // Private constructor used by fromNGT() and load().
    NGTAQIndex(Property prop, BinaryQuantizer bq,
               std::unique_ptr<SoAGraph> graph,
               std::vector<uint32_t> eps,
               std::vector<float> raw_flat);

    static std::vector<uint32_t> selectEntryPoints(
        const SoAGraph& graph, int n, uint32_t seed = 42);
};

} // namespace NGTAQ
