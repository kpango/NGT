// lib/NGT/ArcFlare/ArcFlareIndex.h
#pragma once

#include "NGT/Index.h"
#include "NGT/ArcFlare/AlphaCGPruner.h"
#include "NGT/ArcFlare/BinaryQuantizer.h"
#include "NGT/ArcFlare/DABSSearcher.h"
#include "NGT/ArcFlare/SoAGraph.h"
#include "NGT/ArcFlare/SRHT.h"
#include "NGT/ArcFlare/KMeansCentering.h"
#include "NGT/ArcFlare/PCAProjector.h"
#include "NGT/ArcFlare/ADCTable.h"
#include "NGT/ArcFlare/ADCDistance.h"
#include "NGT/ArcFlare/GlobalPQ4.h"
#include "NGT/ArcFlare/SearchContext.h"

#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace ArcFlare {

// Immutable per-index resolution of the structural AQ_* env flags. These six knobs
// describe the index's search topology (routing path, LUT form, SQ8 tier, graph entry,
// visited-set form) and NEVER vary per query — unlike the per-query search budget knobs
// in AQConfig (aq_cfg()). Populated ONCE at construction from aq_cfg() so searchV2 reads
// them from a member instead of the process-global singleton. Defaults mirror AQConfig.
struct IndexRuntimeConfig {
    bool use_global_routing = false; // AQ_USE_GLOBAL_ROUTING
    bool batch_routing      = true;  // AQ_BATCH_ROUTING
    bool dist_lut           = true;  // AQ_DIST_LUT
    bool use_sq8            = false; // AQ_SQ8
    bool graph_entry        = false; // AQ_GRAPH_ENTRY
    bool versioned_vis      = false; // AQ_VERSIONED_VIS
};

class ArcFlareIndex {
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
        // Angular only: cap on SEEDS scored/enqueued per probed cluster. Angular seeding
        // otherwise scans ALL members of each of n_probe clusters (~20k seeds/query for
        // GloVe), which floods the DABS beam and pins QPS. Bounding seeds per cluster
        // (while still probing n_probe clusters for COVERAGE) is the primary angular
        // recall-QPS knob. 0 = unbounded (legacy: scan full cluster). L2 is unaffected
        // (it already caps at n_cluster_seeds). Placed LAST so the serialized Property
        // layout stays positionally compatible with pre-cap index files (load() defaults
        // this to 64 when the trailing field is absent).
        int    seeds_per_cluster = 64;
    };

    // Build from an existing NGT float32 index.
    static ArcFlareIndex fromNGT(const std::string& ngt_path, const Property& prop);

    // Returns top-k results sorted by exact distance.
    std::vector<SearchResult> search(const std::vector<float>& query, int k) const;

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
    static ArcFlareIndex load(const std::string& path);

    // ---- v2 ADC path ----

    // Build v2 index from existing NGT float32 index.
    // SRHT rotation + K-means centering + PCA top-32 + VectorRecord encoding.
    static ArcFlareIndex fromNGTv2(const std::string& ngt_path, const Property& prop);

    // ADC search using routeV2(). Requires is_v2_ == true.
    // rerank_factor=0 or 1: standard beam width k (existing behavior).
    // rerank_factor>1: internally searches k*rerank_factor candidates, returns top k.
    // rerank_factor<0: skip exact rerank entirely; return top-k by approx ADC distance.
    // max_visits: HNSW ef-style cap on the number of DABS nodes popped/processed.
    //   0 = unlimited (baseline behavior); >0 bounds the beam loop (primary recall-QPS knob).
    std::vector<SearchResult> searchV2(
        const std::vector<float>& query, int k,
        float gamma_enq = 0.2f, float gamma_term = 0.4f,
        int rerank_factor = 0, int max_visits = 0) const;

    // Injected form: pure function of (params, ctx). The legacy overload above
    // populates a SearchParameters from aq_cfg()/members and delegates here.
    std::vector<SearchResult> searchV2(const std::vector<float>& query,
                                       const SearchParameters& params,
                                       SearchContext& ctx) const;

    // Save/load v2 state to directory (separate from v1 state).
    void saveV2(const std::string& dir) const;
    void loadV2(const std::string& dir);

    bool isV2()      const { return is_v2_; }
    bool isAngular() const { return is_angular_; }
    int  dEff()      const { return d_eff_ > 0 ? d_eff_ : prop_.dimension; }
    int  mPQ()       const { return m_pq_; }
    // GPQ4 (batch vpshufb routing) subspace count — DECOUPLED from the legacy
    // tier-2/global-PQ m_pq_ (which stays D/8). Finer M (smaller D_sub) → more
    // accurate routing, matching QG's D_sub=1 → M=D default. K stays 16 (4-bit).
    int  gpq4MPQ()   const { return gpq4_m_pq_ > 0 ? gpq4_m_pq_ : m_pq_; }
    int  gpq4DSub()  const { int m = gpq4MPQ(); return m > 0 ? dEff() / m : 8; }

    // ---- Stage A: GLOBAL PQ routing tier ----
    // One PQ codebook trained over ALL SRHT-rotated vectors (no per-cluster residual),
    // so a single per-query LUT scores ANY node. Foundation for batch DABS routing.
    bool hasGlobalPQ() const { return has_global_pq_; }

    // Build one global LUT from a raw (unrotated, unnormalized) query.
    // Rotates the query with the index SRHT internally, then fills lut (M_PQ*256 floats).
    // Returns ||q_rot||^2 (== ||q||^2, SRHT is orthogonal) for use by globalPQDist().
    // Caller must pre-size lut to mPQ()*256. Requires hasGlobalPQ().
    float buildGlobalLUT(const std::vector<float>& query, float* lut,
                         SearchContext& ctx) const;

    // Approximate squared-L2 distance from the query (whose LUT + q_norm_sq came from
    // buildGlobalLUT) to node_id, via its global PQ code. One LUT scores any node —
    // no per-cluster LUT rebuild. Requires hasGlobalPQ().
    float globalPQDist(uint32_t node_id, const float* lut, float q_norm_sq) const;

    // ---- Stage B/C: 16-centroid (4-bit) GLOBAL PQ for batch vpshufb routing ----
    // hasGPQ4(): a 16-centroid global PQ codebook + per-node contiguous neighbor-code
    // store are present (built by fromNGTv2, persisted by saveV2/loadV2).
    bool hasGPQ4() const { return has_gpq4_ && graph_ && graph_->hasGPQ4(); }

    // Build the per-query uint8 batch LUT from a raw (unrotated) query. Rotates with the
    // index SRHT, fills `lut` (interleaved planes), and returns ||q_rot||^2. Requires
    // a 16-centroid codebook (has_gpq4_). If `ip_out` != nullptr, also writes the
    // dequantized float table [M_PQ*16] (for single-node scoring).
    // dist_lut=false (default): the table/LUT holds <q_sub,centroid> (IP form; caller
    //   assembles L2 = ||q||^2 + ||x||^2 - 2*IP per neighbor).
    // dist_lut=true: the table/LUT holds ||q_sub - centroid||^2 (QG form); the kernel
    //   accumulates L2 DIRECTLY — no per-neighbor norm read, no IP->L2 assembly.
    float buildGlobalLUT16(const std::vector<float>& query,
                           NGT::ArcFlare::GlobalPQ4LUT& lut,
                           SearchContext& ctx,
                           float* ip_out = nullptr,
                           bool dist_lut = false) const;

    // Accessors for raw fp16 vectors and dimension (used by standalone benchmarks).
    // Elements are fp16-packed; decode with NGT::ArcFlare::fp16_to_float.
    const uint16_t* rawFlat() const { return raw_flat_.empty() ? nullptr : raw_flat_.data(); }
    int dim() const { return prop_.dimension; }

    // Diagnostic-only accessors (validate the centroid-assignment accelerator offline).
    const NGT::ArcFlare::KMeansCentering* kmeansForDiag() const { return kmeans_v2_.get(); }
    // Rotate a raw (unpadded) query through the index SRHT into out[dEff()], matching
    // the searchV2 setup (zero-pad to D, then SRHT apply). out must be sized dEff().
    void rotateForDiag(const float* q, int q_dim, float* out) const;
    // Diagnostic-only: read-only view of the loaded graph for the standalone rabitq_bench
    // (P1 measure-first microbench). Does NOT touch searchV2; mirrors the diag-accessor pattern.
    const SoAGraph* graphForDiag() const { return graph_.get(); }

    // Diagnostic: mean |neighbor_id - node_id| over the CSR (graph ID-locality proxy).
    // Large gap => walked nodes' gpq4 blocks are scattered (reordering may help); small
    // gap => already cache-local (reordering won't help). Returns {mean_gap, mean_degree}.
    std::pair<double,double> graphLocalityStats() const;

    // Reorder node IDs for walk cache-locality. Pure permutation — remaps CSR, gpq4 store/
    // codes/norms, sq8, raw_flat_, global codes/norms, cluster members, entry points
    // consistently → recall byte-identical, only memory layout changes. Call before saveV2.
    //   BFS    : breadth-first from entry points (the shipped baseline, commit 2766713).
    //   RCM    : reverse Cuthill-McKee (degree-ascending BFS, reversed) — bandwidth-minimizing.
    //   GORDER : window-based neighbor co-occurrence greedy order (Coleman NeurIPS'22) —
    //            maximizes shared-neighbor locality over a sliding window.
    enum class ReorderMode { BFS, RCM, GORDER };
    void reorderForLocality(ReorderMode mode = ReorderMode::BFS);

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
    // Build the SearchParameters that reproduces the legacy searchV2(query, k, ...)
    // behavior (aq_cfg() + loaded searcher_/prop_ defaults; n_probe=0 => worker derives
    // the default). Shared by the legacy delegator and rebuildGraphSelf's internal
    // searches so the two stay in lockstep.
    SearchParameters buildLegacyParams(int k, float gamma_enq, float gamma_term,
                                       int rerank_factor, int max_visits) const;

    // Concurrency model:
    //   graph_->mutex() (std::shared_mutex) protects: graph_, raw_flat_, entry_points_
    //   Shared lock: search(), searchBatch(), size()
    //   Unique lock: insert(), remove(), rebuild()
    //   No lock: save() — caller must ensure no concurrent mutations
    //
    // prop_, bq_, pruner_, searcher_ are effectively immutable after construction.
    Property                        prop_;
    // Six structural AQ_* flags resolved once from aq_cfg() at construction (set in the
    // private ctor body, the single path for both fromNGT() and load()). Immutable after.
    IndexRuntimeConfig              rt_cfg_;
    BinaryQuantizer                 bq_;
    std::unique_ptr<SoAGraph>       graph_;
    AlphaCGPruner                   pruner_;
    DABSSearcher                    searcher_;
    std::vector<uint32_t>           entry_points_;
    std::vector<uint16_t>           raw_flat_;     // flat [N*D] exact vectors, fp16-packed (rerank via l2_sq_f32_fp16)
    // Task 2 (BFS reorder): when node IDs are permuted for cache locality, results carry
    // INTERNAL ids; this maps internal -> original (insertion-order) id so returned ids
    // still match the caller's ground truth. Empty == identity (no reorder). Serialized.
    std::vector<uint32_t>           id_to_external_;

    // Pool of reusable per-thread search scratch (SearchContext). Held by pointer so
    // ArcFlareIndex stays movable (SearchContextPool deletes copy/move; the factories
    // fromNGT()/fromNGTv2()/load() return by value). mutable because searchV2 is const;
    // the pool is internally mutex-guarded.
    mutable std::unique_ptr<SearchContextPool> ctx_pool_ =
        std::make_unique<SearchContextPool>();

    // During rebuildGraphSelf, limit total seeds per searchV2 call to avoid queue
    // flooding on angular data (large clusters). 0 = no limit (full cluster scan).
    // Set before parallel loop, reset after. mutable so searchV2 (const) can read it.
    mutable int                               rebuild_max_seeds_ = 0;

    // v2 ADC state (null/empty if not built via fromNGTv2)
    bool                                      is_v2_      = false;
    bool                                      is_angular_ = false;  // true when metric is Angle or Cosine
    int                                       d_eff_      = 0;      // effective dimension (0 = prop_.dimension)
    int                                       m_pq_       = 16;     // PQ sub-codebook count
    std::unique_ptr<NGT::ArcFlare::SRHT>         srht_v2_;
    std::unique_ptr<NGT::ArcFlare::KMeansCentering> kmeans_v2_;
    std::unique_ptr<NGT::ArcFlare::PCAProjector> pca_v2_;
    std::vector<float>                        tier2_codebook_;   // [M][K][D_sub] = row-major (original layout)
    std::vector<float>                        tier2_codebook_T_; // [M][D_sub][K] = transposed (for fast AVX2 LUT build)
    std::vector<uint32_t>                     v2_entry_points_;

    // ---- Stage A: GLOBAL PQ tier (one codebook over all rotated vectors) ----
    bool                                      has_global_pq_ = false;
    std::vector<float>   global_pq_codebook_;    // [M_PQ][256][D_sub] row-major (trained on rotated vectors)
    std::vector<float>   global_pq_codebook_T_;  // [M_PQ][D_sub][256] transposed (fast LUT build)
    std::vector<uint8_t> global_codes_;          // [N*M_PQ] per-vector global PQ codes
    std::vector<float>   global_pq_norm_sq_;     // [N] ||reconstructed rotated PQ vector||^2 per node

    // ---- Stage B/C: 16-centroid (4-bit) GLOBAL PQ for batch vpshufb routing ----
    // The K=16 codebook is REQUIRED for the single-shuffle 16-entry lookup (the K=256
    // global PQ above is incompatible with vpshufb). Per-node 4-bit codes + recon-norms
    // live in the SoAGraph contiguous neighbor-code store (graph_->buildGPQ4).
    bool                 has_gpq4_ = false;
    // GPQ4 uses its OWN subspace count (gpq4_m_pq_), independent of the legacy m_pq_.
    // Finer = smaller D_sub = larger M = more vpshufb subspace iterations (QG-style).
    // gpq4_d_sub_ = dEff() / gpq4_m_pq_. 0 → fall back to m_pq_ (old indices).
    int                  gpq4_m_pq_ = 0;    // #subspaces for the K=16 batch PQ (0 = use m_pq_)
    std::vector<float>   gpq4_codebook_;    // [gpq4_m_pq_][16][D_sub] row-major (rotated vectors)
    std::vector<float>   gpq4_codebook_T_;  // [gpq4_m_pq_][D_sub][16] transposed (fast LUT build)
    std::vector<uint8_t> gpq4_codes_;       // [N*gpq4_m_pq_] per-node 4-bit codes (single-node scoring)
    std::vector<float>   gpq4_norm_sq_;     // [N] per-node reconstructed-norm^2

    // ---- Tech 1: symmetric SQ8 in-loop routing distance (pyglass SQ8P scheme) ----
    // Per-node int8 code of the SRHT-rotated vector (D bytes, padded to mult of 64 for the
    // VNNI kernel) + per-node max scale + per-node ||x||^2. The batch loop routes with a
    // single signed-int8 dot (one vpdpbusd pass) instead of the gpq4 IP + recon-norm gather.
    // Built in fromNGTv2 from the rotated vectors; persisted to v2_sq8.bin (loaded at loadV2).
    bool                 has_sq8_ = false;
    int                  sq8_dim_align_ = 0;  // D padded up to a multiple of 64 (code stride)
    std::vector<int8_t>  sq8_codes_;          // [N * sq8_dim_align_] symmetric int8 codes
    std::vector<float>   sq8_max_;            // [N] per-node max|x| scale
    std::vector<float>   sq8_norm_;           // [N] per-node ||x||^2 (rotated) for exact L2
    bool hasSQ8() const { return has_sq8_ && !sq8_codes_.empty(); }

    // Symmetric SQ8 routing distance from an int8-encoded query (q_i8 with scale max_q and
    // ||q||^2 q_nsq) to node_id. Returns approximate squared-L2:
    //   ||q||^2 + ||x||^2 - 2 * <q_i8,x_i8> * max_q * max_x / 127^2.
    float sq8Dist(uint32_t node_id, const int8_t* q_i8, float max_q, float q_nsq) const {
        const int8_t* xc = sq8_codes_.data() + (size_t)node_id * sq8_dim_align_;
        int32_t dot = NGT::ArcFlare::dot_s8_s8(q_i8, xc, sq8_dim_align_);
        float scale = max_q * sq8_max_[node_id] * (1.0f / (127.0f * 127.0f));
        return q_nsq + sq8_norm_[node_id] - 2.0f * (float)dot * scale;
    }

    // Single-node batch-PQ distance (seeds / expansion): ||q_rot - x_pq16||^2 via the
    // node's own 4-bit code, scored against the per-query LUT's dequantized float table.
    // `ip_table` is the M*16 float IP table from gpq4_ip_table (NOT the uint8 LUT).
    float gpq4Dist(uint32_t node_id, const float* ip_table, float q_norm_sq) const {
        const int M_PQ = gpq4MPQ();
        const uint8_t* codes = gpq4_codes_.data() + (size_t)node_id * M_PQ;
        float ip = 0.f;
        for (int s = 0; s < M_PQ; ++s) ip += ip_table[(size_t)s * NGT::ArcFlare::GPQ4_K + codes[s]];
        return q_norm_sq + gpq4_norm_sq_[node_id] - 2.0f * ip;
    }

    // Distance-LUT single-node variant (seeds/expansion): the table is squared-distance
    // (gpq4_dist_table), so the per-subspace contributions sum to L2 directly — no norm
    // read, no assembly. Matches the distance-LUT batch kernel's metric.
    float gpq4DistL2(uint32_t node_id, const float* dist_table) const {
        const int M_PQ = gpq4MPQ();
        const uint8_t* codes = gpq4_codes_.data() + (size_t)node_id * M_PQ;
        float d = 0.f;
        for (int s = 0; s < M_PQ; ++s) d += dist_table[(size_t)s * NGT::ArcFlare::GPQ4_K + codes[s]];
        return d;
    }

    // Lazy-built inverted list + cluster neighbor table for cluster-aware seeding.
    // Built once on first searchV2 call; unique_ptr keeps ArcFlareIndex movable (once_flag is non-movable).
    mutable std::unique_ptr<std::once_flag>    cluster_members_once_{std::make_unique<std::once_flag>()};
    mutable std::vector<std::vector<uint32_t>> cluster_members_v2_;  // cluster_id → [node_ids]
    mutable std::vector<std::vector<uint32_t>> cluster_neighbors_v2_; // cluster_id → [nearest cluster_ids]

    // Private constructor used by fromNGT() and load().
    // raw_flat is fp16-packed [N*D] (see raw_flat_).
    ArcFlareIndex(Property prop, BinaryQuantizer bq,
               std::unique_ptr<SoAGraph> graph,
               std::vector<uint32_t> eps,
               std::vector<uint16_t> raw_flat);

    static std::vector<uint32_t> selectEntryPoints(
        const SoAGraph& graph, int n, uint32_t seed = 42);
};

} // namespace ArcFlare
