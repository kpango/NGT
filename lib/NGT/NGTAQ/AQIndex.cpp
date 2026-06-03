// lib/NGT/NGTAQ/AQIndex.cpp
#include "NGT/NGTAQ/AQIndex.h"
#include "NGT/NGTAQ/DimUtils.h"

#include "NGT/Graph.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <omp.h>
#include <queue>
#include <random>
#include <shared_mutex>
#include <stdexcept>
// unordered_set removed: visited tracking uses flat bitvector (see searchV2)

// ---------------------------------------------------------------------------
// AQ_PROFILE: compile-time-gated per-region timers for searchV2 (diagnostic).
// Zero effect on normal builds (macros expand to nothing when AQ_PROFILE undefined).
// Accumulates per-region microseconds into thread_local counters, printed once at
// thread/process exit to stderr.
// ---------------------------------------------------------------------------
#ifdef AQ_PROFILE
#include <chrono>
namespace { struct AqProf { double seed=0,dabs=0,refine=0,expand=0,rerank=0,setup=0;
  double srht=0,lut=0; double hops=0,npops=0; long n=0;
  // Task 1: result-set stabilization-hop = max pop-index among the final top-10 winners.
  double stab_sum=0; long stab_n=0; std::vector<int> stab_samples;
  ~AqProf(){ if(n) { fprintf(stderr,
    "[AQ_PROFILE] n=%ld setup=%.1f (srht=%.1f lut=%.1f) seed=%.1f dabs=%.1f refine=%.1f expand=%.1f rerank=%.1f us/query | hops=%.1f npops=%.1f ns/hop=%.1f\n",
    n, setup/n, srht/n, lut/n, seed/n, dabs/n, refine/n, expand/n, rerank/n,
    hops/n, npops/n, hops>0?dabs*1000.0/hops:0.0);
    if (stab_n) { auto s = stab_samples; std::sort(s.begin(), s.end());
      auto pc=[&](double p){ return s.empty()?0:s[(size_t)(p*(s.size()-1))]; };
      fprintf(stderr, "[AQ_STAB] stabilization-hop mean=%.1f p50=%d p90=%d p99=%d  (vs hops=%.1f) "
        "=> wasted-frac mean=%.2f\n", stab_sum/stab_n, pc(0.50), pc(0.90), pc(0.99),
        hops/n, 1.0 - (stab_sum/stab_n)/(hops/n>0?hops/n:1)); } } } };
  thread_local AqProf g_aqprof; }
#define AQ_T0() auto _t=std::chrono::steady_clock::now(); auto _ts=_t
#define AQ_ADD(f) do{auto _e=std::chrono::steady_clock::now(); \
  g_aqprof.f+=std::chrono::duration<double,std::micro>(_e-_t).count(); _t=_e;}while(0)
// Sub-timers use an INDEPENDENT clock (_ts) so they don't disturb the main region (_t)
// accumulation — they measure a slice within a region without double-counting.
#define AQ_MARK() do{ _ts=std::chrono::steady_clock::now(); }while(0)
#define AQ_SUB(f) do{auto _e=std::chrono::steady_clock::now(); \
  g_aqprof.f+=std::chrono::duration<double,std::micro>(_e-_ts).count(); _ts=_e;}while(0)
#define AQ_CNT(f,v) do{ g_aqprof.f += (double)(v); }while(0)
#else
#define AQ_T0()
#define AQ_ADD(f)
#define AQ_MARK()
#define AQ_SUB(f)
#define AQ_CNT(f,v)
#endif

namespace NGTAQ {

// ---------------------------------------------------------------------------
// LinearPool: pyglass-style ef-bounded sorted-array candidate frontier.
// (port of glass/neighbor.hpp LinearPool, glass/searcher/graph_searcher.hpp loop)
//
// Replaces the DABS std::priority_queue + dk_tracker heap + ef_gate max-heap + the two
// gamma-termination gates with a single flat, distance-sorted {id,dist} array of
// capacity `ef`. The cursor `cur_` marks the exploration frontier; the candidate-order
// "checked" bit is packed into id bit31. insert() = binary-search + memmove (and rewinds
// cur_ when inserting before it, so a newly-found-closer node is auto re-explored).
// has_next() == (cur_ < size_ && cur_ < ef_) is the ONLY termination test: no gamma
// gates, no per-pop tier-2 gather. Visited tracking stays external (the existing t_vis
// bitvector) so we don't double-allocate.
struct AQLinearPool {
    struct Node { uint32_t id; float dist; };  // id bit31 = "checked" (popped) flag
    std::vector<Node> data_;                    // sorted ascending by dist, size capacity_+1
    int size_ = 0, cur_ = 0, ef_ = 0, capacity_ = 0;

    static constexpr uint32_t kMask = 0x7fffffffu;
    static inline uint32_t rawid(uint32_t id)   { return id & kMask; }
    static inline bool      checked(uint32_t id){ return (id >> 31) & 1u; }

    void reset(int ef) {
        ef_ = ef;
        capacity_ = ef;            // pyglass uses cap=max(k,ef); ef caps the live frontier
        size_ = cur_ = 0;
        if ((int)data_.size() < capacity_ + 1) data_.resize(capacity_ + 1);
    }

    // Binary search for the insertion slot (first index with data_[i].dist > dist).
    inline int find_bsearch(float dist) const {
        int lo = 0, hi = size_;
        while (lo < hi) { int mid = (lo + hi) >> 1;
            if (data_[mid].dist > dist) hi = mid; else lo = mid + 1; }
        return lo;
    }

    // Returns true if the candidate entered the pool (caller may then prefetch it).
    inline bool insert(uint32_t u, float dist) {
        if (size_ == capacity_ && dist >= data_[size_ - 1].dist) return false;
        int lo = find_bsearch(dist);
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Node));
        data_[lo] = {u, dist};
        if (size_ < capacity_) ++size_;
        if (lo < cur_) cur_ = lo;   // inserted before frontier → rewind for re-exploration
        return true;
    }

    // Pop the current frontier node (marks it checked), advance cursor past checked nodes.
    inline uint32_t pop() {
        data_[cur_].id |= (1u << 31);
        int pre = cur_;
        while (cur_ < size_ && checked(data_[cur_].id)) ++cur_;
        return rawid(data_[pre].id);
    }
    inline bool has_next() const { return cur_ < size_ && cur_ < ef_; }
};

// ---------------------------------------------------------------------------
// Private constructor
// ---------------------------------------------------------------------------
NGTAQIndex::NGTAQIndex(Property prop, BinaryQuantizer bq,
                       std::unique_ptr<SoAGraph> graph,
                       std::vector<uint32_t> eps,
                       std::vector<uint16_t> raw_flat)
    : prop_(prop)
    , bq_(std::move(bq))
    , graph_(std::move(graph))
    , pruner_(prop.alpha, prop.kappa)
    , entry_points_(std::move(eps))
    , raw_flat_(std::move(raw_flat))
{
    searcher_.gamma_enq      = prop_.gamma_enq;
    searcher_.gamma_term     = prop_.gamma_term;
    searcher_.k_prime_factor = prop_.k_prime_factor;
}

// ---------------------------------------------------------------------------
// fromNGT
// ---------------------------------------------------------------------------
NGTAQIndex NGTAQIndex::fromNGT(const std::string& ngt_path, const Property& prop) {
    if (prop.dimension % 64 != 0)
        throw std::invalid_argument("NGTAQIndex: dimension must be a multiple of 64");

    NGT::Index ngt(ngt_path);
    NGT::ObjectSpace& objspace = ngt.getObjectSpace();
    const size_t repo_size = objspace.getRepository().size();
    const size_t N = repo_size - 1;
    const int D = prop.dimension;
    const int words = D / 64;

    // Load all float vectors into flat array: raw_flat[i*D .. i*D+D-1] = vec i
    std::vector<float> raw_flat(N * static_cast<size_t>(D), 0.0f);
    std::vector<bool> is_hole(N, false);
    std::vector<float> tmp(D);
    for (size_t i = 1; i <= N; i++) {
        try {
            objspace.getObject(static_cast<NGT::ObjectID>(i), tmp);
            std::copy(tmp.begin(), tmp.end(), raw_flat.begin() + static_cast<ptrdiff_t>((i - 1) * D));
        } catch (...) {
            is_hole[i - 1] = true;
        }
    }

    // Pre-normalize for cosine metric
    if (prop.metric == NGT::ObjectSpace::DistanceTypeAngle ||
        prop.metric == NGT::ObjectSpace::DistanceTypeCosine) {
        for (size_t i = 0; i < N; ++i) {
            float* v = raw_flat.data() + i * D;
            float norm_sq = 0.0f;
            for (int j = 0; j < D; ++j) norm_sq += v[j] * v[j];
            if (norm_sq > 0.0f) {
                float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (int j = 0; j < D; ++j) v[j] *= inv_norm;
            }
        }
    }

    BinaryQuantizer bq;
    bq.init(D);
    bq.setRandomRotation();

    std::vector<const float*> ptrs(N);
    for (size_t i = 0; i < N; i++) ptrs[i] = raw_flat.data() + i * D;
    bq.calibrateTau(ptrs, prop.n_tau_samples, prop.metric);

    // Encode all vectors and build SoAGraph
    auto graph = std::make_unique<SoAGraph>(words);
    std::vector<uint64_t> bq_buf(static_cast<size_t>(words) * 2);
    for (size_t i = 0; i < N; i++) {
        bq.encode(raw_flat.data() + i * D, bq_buf.data());
        graph->addNode(bq_buf.data());
    }
    graph->finalizeCSR();

    // Tombstone ghost nodes
    for (size_t i = 0; i < N; ++i) {
        if (is_hole[i]) graph->removeNode(static_cast<uint32_t>(i));
    }

    // Build alpha-CG graph from NGT edges (O(N·k) via resetEdges)
    AlphaCGPruner pruner(prop.alpha, prop.kappa);
    const float tau = bq.tau();
    NGT::GraphIndex& gi = static_cast<NGT::GraphIndex&>(ngt.getIndex());

    std::vector<std::vector<uint32_t>> adj(N);
    for (size_t i = 1; i <= N; i++) {
        uint32_t aq_id = static_cast<uint32_t>(i - 1);
        NGT::GraphNode* node = nullptr;
        try {
            node = gi.getNode(static_cast<NGT::ObjectID>(i));
        } catch (...) {
            continue;
        }
        if (!node || node->empty()) continue;

        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(node->size());
        for (auto& edge : *node) {
            if (edge.id == 0 || edge.id > static_cast<unsigned int>(N)) continue;
            uint32_t nbr = static_cast<uint32_t>(edge.id - 1);
            float d = bqDistance(graph->getNodeBQ(aq_id), graph->getNodeBQ(nbr), words, D);
            candidates.push_back({nbr, d});
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (static_cast<int>(candidates.size()) > prop.max_edges)
            candidates.resize(static_cast<size_t>(prop.max_edges));

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(graph->getNodeBQ(v), graph->getNodeBQ(u), words, D);
        };
        adj[aq_id] = pruner.prune(candidates, tau, dist_fn);
    }
    graph->resetEdges(adj);

    int n_ep = std::min(prop.n_entry_points, static_cast<int>(N));
    auto entry_points = selectEntryPoints(*graph, n_ep);

    // Pack the fp32 working buffer into fp16 for the exact-rerank store (Task 0.2).
    std::vector<uint16_t> raw_flat_h(raw_flat.size());
    for (size_t i = 0; i < raw_flat.size(); ++i)
        raw_flat_h[i] = NGT::NGTAQ::float_to_fp16(raw_flat[i]);

    return NGTAQIndex(prop, std::move(bq), std::move(graph),
                      std::move(entry_points), std::move(raw_flat_h));
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------
std::vector<SearchResult> NGTAQIndex::search(
    const std::vector<float>& query, int k) const
{
    if (static_cast<int>(query.size()) < prop_.dimension)
        throw std::invalid_argument("NGTAQIndex::search: query dimension mismatch");
    const int D = prop_.dimension;
    const int words = D / 64;

    // Encode query to interleaved BQ
    std::vector<uint64_t> q_bq(static_cast<size_t>(words) * 2);
    bq_.encode(query.data(), q_bq.data());

    std::shared_lock<std::shared_mutex> lock(graph_->mutex());

    auto cand_ids = searcher_.route(q_bq.data(), k, *graph_, entry_points_);

    // Prefetch raw float vectors for refinement (PREFETCH_AHEAD=8 candidates)
    constexpr int PREFETCH_AHEAD = 8;
    const int n_cands = static_cast<int>(cand_ids.size());
    for (int ci = 0; ci < n_cands; ++ci) {
        if (ci + PREFETCH_AHEAD < n_cands) {
            uint32_t nxt = cand_ids[static_cast<size_t>(ci + PREFETCH_AHEAD)];
            if (nxt * static_cast<size_t>(D) < raw_flat_.size()) {
                __builtin_prefetch(raw_flat_.data() + nxt * static_cast<size_t>(D), 0, 1);
            }
        }
    }

    // Cosine: raw_flat_ stores pre-normalized vectors; normalize the query once
    // (identical for all candidates) and reuse inside the refinement loop.
    const bool is_cosine = (prop_.metric != NGT::ObjectSpace::DistanceTypeL2);
    std::vector<float> qn;
    if (is_cosine) {
        qn.assign(query.data(), query.data() + D);
        float norm_sq = 0.0f;
        for (float x : qn) norm_sq += x * x;
        if (norm_sq > 0.0f) {
            float inv_norm = 1.0f / std::sqrt(norm_sq);
            for (float& x : qn) x *= inv_norm;
        }
    }

    // Exact-distance refinement
    std::vector<SearchResult> results;
    results.reserve(cand_ids.size());
    for (uint32_t id : cand_ids) {
        if (static_cast<size_t>(id) * D + D > raw_flat_.size()) continue;
        // raw_flat_ is fp16-packed; decode lazily via the F16C helpers.
        const uint16_t* vec = raw_flat_.data() + static_cast<size_t>(id) * D;
        float exact_dist = 0.0f;
        if (!is_cosine) {
            exact_dist = std::sqrt(NGT::NGTAQ::l2_sq_f32_fp16(query.data(), vec, D));
        } else {
            float dot = 0.0f;
            for (int j = 0; j < D; ++j) dot += qn[j] * NGT::NGTAQ::fp16_to_float(vec[j]);
            exact_dist = 1.0f - dot;
        }
        float bq_dist = bqDistance(q_bq.data(), graph_->getNodeBQ(id), words, D);
        results.push_back({id, exact_dist, bq_dist});
    }

    std::sort(results.begin(), results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        });
    if (static_cast<int>(results.size()) > k)
        results.resize(static_cast<size_t>(k));
    return results;
}

// ---------------------------------------------------------------------------
// searchBatch
// ---------------------------------------------------------------------------
std::vector<std::vector<SearchResult>> NGTAQIndex::searchBatch(
    const std::vector<std::vector<float>>& queries, int k) const
{
    const int nq = static_cast<int>(queries.size());
    std::vector<std::vector<SearchResult>> out(static_cast<size_t>(nq));

    const int nt = (prop_.n_search_threads <= 0)
                   ? omp_get_max_threads()
                   : prop_.n_search_threads;

    // Note: each search() acquires a shared_lock; multiple shared locks coexist safely.
    // If the caller is already in an OMP parallel region, set n_search_threads=1
    // (or omp_set_nested(false)) to avoid nested parallelism overhead.
#pragma omp parallel for schedule(dynamic, 8) num_threads(nt)
    for (int qi = 0; qi < nq; ++qi) {
        out[static_cast<size_t>(qi)] = search(queries[static_cast<size_t>(qi)], k);
    }
    return out;
}

// ---------------------------------------------------------------------------
// insert
// ---------------------------------------------------------------------------
uint32_t NGTAQIndex::insert(const std::vector<float>& vec) {
    if (static_cast<int>(vec.size()) < prop_.dimension)
        throw std::invalid_argument("NGTAQIndex::insert: vector dimension mismatch");
    const int D = prop_.dimension;
    const int words = D / 64;

    std::vector<uint64_t> bq_buf(static_cast<size_t>(words) * 2);
    bq_.encode(vec.data(), bq_buf.data());

    std::unique_lock<std::shared_mutex> lock(graph_->mutex());
    uint32_t new_id = graph_->addNode(bq_buf.data());

    // Append raw vector to flat array, fp16-packed.
    raw_flat_.reserve(raw_flat_.size() + static_cast<size_t>(D));
    for (int j = 0; j < D; ++j)
        raw_flat_.push_back(NGT::NGTAQ::float_to_fp16(vec[j]));
    // Normalize in-place for cosine metric (fp16 round-trip).
    if (prop_.metric == NGT::ObjectSpace::DistanceTypeAngle ||
        prop_.metric == NGT::ObjectSpace::DistanceTypeCosine) {
        uint16_t* h = raw_flat_.data() + static_cast<size_t>(new_id) * D;
        // Norm is computed over the fp16-rounded values (not the original fp32) so the
        // stored vector is normalized consistently with how search-time decode reads it.
        float norm_sq = 0.0f;
        for (int j = 0; j < D; ++j) {
            float f = NGT::NGTAQ::fp16_to_float(h[j]);
            norm_sq += f * f;
        }
        if (norm_sq > 0.0f) {
            float inv_norm = 1.0f / std::sqrt(norm_sq);
            for (int j = 0; j < D; ++j)
                h[j] = NGT::NGTAQ::float_to_fp16(NGT::NGTAQ::fp16_to_float(h[j]) * inv_norm);
        }
    }

    if (graph_->size() > 1) {
        graph_->finalizeCSR();

        auto cand_ids = searcher_.route(bq_buf.data(),
            std::min(prop_.max_edges, static_cast<int>(graph_->size()) - 1),
            *graph_, entry_points_);

        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(cand_ids.size());
        for (uint32_t cid : cand_ids) {
            if (cid == new_id) continue;
            float d = bqDistance(graph_->getNodeBQ(new_id), graph_->getNodeBQ(cid), words, D);
            candidates.push_back({cid, d});
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(graph_->getNodeBQ(v), graph_->getNodeBQ(u), words, D);
        };
        auto pruned = pruner_.prune(candidates, bq_.tau(), dist_fn);
        graph_->setNeighbors(new_id, pruned);
    } else {
        graph_->finalizeCSR();
    }
    return new_id;
}

// ---------------------------------------------------------------------------
// remove
// ---------------------------------------------------------------------------
void NGTAQIndex::remove(uint32_t id) {
    std::unique_lock<std::shared_mutex> lock(graph_->mutex());
    graph_->removeNode(id);
}

// ---------------------------------------------------------------------------
// rebuild
// ---------------------------------------------------------------------------
void NGTAQIndex::rebuild() {
    std::unique_lock<std::shared_mutex> lock(graph_->mutex());

    const size_t N = graph_->size();
    const int D = prop_.dimension;
    std::vector<uint32_t> old_to_new(N, static_cast<uint32_t>(-1));
    uint32_t next_id = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
        if (!graph_->isTombstone(i)) old_to_new[i] = next_id++;
    }

    // Reorder raw_flat_ to match post-rebuild node ordering (fp16 elements)
    std::vector<uint16_t> new_flat(static_cast<size_t>(next_id) * D);
    for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
        if (old_to_new[i] == static_cast<uint32_t>(-1)) continue;
        size_t src_off = static_cast<size_t>(i) * D;
        size_t dst_off = static_cast<size_t>(old_to_new[i]) * D;
        if (src_off + D <= raw_flat_.size()) {
            std::copy(raw_flat_.begin() + static_cast<ptrdiff_t>(src_off),
                      raw_flat_.begin() + static_cast<ptrdiff_t>(src_off + D),
                      new_flat.begin() + static_cast<ptrdiff_t>(dst_off));
        }
    }
    raw_flat_ = std::move(new_flat);

    graph_->rebuild();

    int n_ep = std::min(prop_.n_entry_points, static_cast<int>(graph_->size()));
    entry_points_ = selectEntryPoints(*graph_, n_ep);
}

// ---------------------------------------------------------------------------
// size
// ---------------------------------------------------------------------------
size_t NGTAQIndex::size() const {
    return graph_->activeCount();
}

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------
// Magic number identifying the versioned binary format.
// Oldest format:  first 4 bytes = prop_.dimension (small positive int, typically 128); fp32 raw_flat_.
// fp32 versioned:  first 4 bytes = kPropMagicFp32, then [prop_size][prop_bytes]; fp32 raw_flat_.
// fp16 versioned:  first 4 bytes = kPropMagicFp16 (Task 0.2), same prop header but raw_flat_ is uint16_t.
// The fp16 magic distinguishes the new layout; older indices are rejected in load()
// (no fp32→fp16 conversion — indices are rebuilt; see Task 0.3).
static constexpr uint32_t kPropMagicFp32 = 0xAE17AE17u;
static constexpr uint32_t kPropMagicFp16 = 0xAE17AE18u;

// raw_flat_ I/O writes n_elems * sizeof(uint16_t) bytes via a std::streamsize cast;
// on a 64-bit streamsize this never overflows for any feasible index size.
static_assert(sizeof(std::streamsize) >= 8, "NGTAQ index I/O assumes 64-bit streamsize");

void NGTAQIndex::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("NGTAQIndex::save: cannot open " + path);

    // 1. Property — versioned format: [magic][prop_size][prop_bytes]
    //    kPropMagicFp16 marks the fp16 raw_flat_ layout (Task 0.2).
    const uint32_t prop_size = sizeof(prop_);
    os.write(reinterpret_cast<const char*>(&kPropMagicFp16), 4);
    os.write(reinterpret_cast<const char*>(&prop_size),  4);
    os.write(reinterpret_cast<const char*>(&prop_), prop_size);

    // 2. BinaryQuantizer
    bq_.serialize(os);

    // 3. SoAGraph
    graph_->serialize(os);

    // 4. Entry points
    uint32_t n_ep = static_cast<uint32_t>(entry_points_.size());
    os.write(reinterpret_cast<const char*>(&n_ep), sizeof(n_ep));
    if (n_ep > 0)
        os.write(reinterpret_cast<const char*>(entry_points_.data()),
                 n_ep * sizeof(uint32_t));

    // 5. Raw flat vectors: uint64_t n_elems, then fp16 (uint16_t) array
    uint64_t n_elems = static_cast<uint64_t>(raw_flat_.size());
    os.write(reinterpret_cast<const char*>(&n_elems), sizeof(n_elems));
    if (n_elems > 0)
        os.write(reinterpret_cast<const char*>(raw_flat_.data()),
                 static_cast<std::streamsize>(n_elems * sizeof(uint16_t)));

    os.flush();
    if (!os) throw std::runtime_error("NGTAQIndex::save: write error on " + path);
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
NGTAQIndex NGTAQIndex::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("NGTAQIndex::load: cannot open " + path);

    // 1. Property — detect format by magic number.
    //    fp16 versioned (current): first 4 bytes = kPropMagicFp16, then [prop_size][prop_bytes].
    //    fp32 versioned (legacy):  first 4 bytes = kPropMagicFp32  → rejected (rebuild required).
    //    oldest (pre-magic):       first 4 bytes = dimension (small int) → rejected (rebuild required).
    Property prop;
    memset(&prop, 0, sizeof(prop));
    prop.k_clusters       = 0;   // default: use select_k(N)
    prop.n_cluster_seeds  = 32;  // default
    prop.seeds_per_cluster = 64; // default for pre-cap indices (trailing field absent in old files)

    uint32_t maybe_magic;
    is.read(reinterpret_cast<char*>(&maybe_magic), 4);
    if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read header from " + path);

    if (maybe_magic != kPropMagicFp16) {
        // Either the legacy fp32 versioned magic, or the oldest dimension-prefixed
        // format — both store raw_flat_ as fp32. We do not convert (YAGNI; indices are
        // rebuilt in Task 0.3). Reject with a clear, actionable error.
        throw std::runtime_error(
            "AQ index uses fp32 raw_flat_; rebuild required for fp16 (Task 0.2)");
    }

    // fp16 versioned format: [magic(4)][prop_size(4)][prop_bytes(prop_size)]
    {
        uint32_t stored_size;
        is.read(reinterpret_cast<char*>(&stored_size), 4);
        if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read prop_size from " + path);
        const uint32_t read_size = std::min(stored_size, static_cast<uint32_t>(sizeof(prop)));
        is.read(reinterpret_cast<char*>(&prop), read_size);
        if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read property from " + path);
        // Skip unknown future fields (forward compatibility)
        if (stored_size > static_cast<uint32_t>(sizeof(prop)))
            is.seekg(static_cast<std::streamoff>(stored_size - sizeof(prop)), std::ios::cur);
    }

    if (prop.dimension <= 0 || prop.dimension > 65536)
        throw std::runtime_error("NGTAQIndex::load: invalid dimension in file");

    // 2. BinaryQuantizer
    BinaryQuantizer bq;
    bq.deserialize(is);

    // 3. SoAGraph
    auto graph = std::make_unique<SoAGraph>(prop.dimension / 64);
    graph->deserialize(is);

    // 4. Entry points
    uint32_t n_ep = 0;
    is.read(reinterpret_cast<char*>(&n_ep), sizeof(n_ep));
    if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read entry point count");
    // Upper bound: graph node count (already loaded above). Corrupt files may set n_ep
    // to an absurdly large value — cap it at graph size as a sanity check.
    if (n_ep > static_cast<uint64_t>(graph->size()))
        throw std::runtime_error("NGTAQIndex::load: n_ep exceeds graph size (file corrupt?)");
    std::vector<uint32_t> entry_points(n_ep);
    if (n_ep > 0)
        is.read(reinterpret_cast<char*>(entry_points.data()), n_ep * sizeof(uint32_t));

    // 5. Raw flat vectors (fp16 / uint16_t elements)
    uint64_t n_elems = 0;
    is.read(reinterpret_cast<char*>(&n_elems), sizeof(n_elems));
    if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read vec count");
    // Upper bound = dimension × maximum reasonable vector count (20M vectors).
    // Full consistency check (n_elems == prop_.dimension * graph->size()) is done
    // implicitly: the read below will fail or produce wrong results if corrupt.
    const uint64_t max_reasonable = static_cast<uint64_t>(prop.dimension) * 20000000ULL;
    if (n_elems > max_reasonable)
        throw std::runtime_error("NGTAQIndex::load: n_elems exceeds limit (file corrupt?)");
    std::vector<uint16_t> raw_flat(n_elems);
    if (n_elems > 0)
        is.read(reinterpret_cast<char*>(raw_flat.data()),
                static_cast<std::streamsize>(n_elems * sizeof(uint16_t)));

    if (!is)
        throw std::runtime_error("NGTAQIndex::load: stream error reading " + path);

    return NGTAQIndex(prop, std::move(bq), std::move(graph),
                      std::move(entry_points), std::move(raw_flat));
}

// ---------------------------------------------------------------------------
// selectEntryPoints
// ---------------------------------------------------------------------------
std::vector<uint32_t> NGTAQIndex::selectEntryPoints(
    const SoAGraph& graph, int n, uint32_t seed)
{
    if (n <= 0 || graph.size() == 0) return {};

    const int words = graph.words();
    const int D = words * 64;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> pick(
        0, static_cast<uint32_t>(graph.size()) - 1);

    std::vector<uint32_t> selected;
    selected.reserve(static_cast<size_t>(n));

    for (int attempt = 0; attempt < 1000 && selected.empty(); ++attempt) {
        uint32_t c = pick(rng);
        if (!graph.isTombstone(c)) selected.push_back(c);
    }
    if (selected.empty()) return {};

    while (static_cast<int>(selected.size()) < n) {
        const int cand_size = std::min(200, static_cast<int>(graph.activeCount()));
        if (cand_size == 0) break;
        float best_min_dist = -1.0f;
        uint32_t best_id = selected[0];

        for (int t = 0; t < cand_size; ++t) {
            uint32_t c = pick(rng);
            if (graph.isTombstone(c)) continue;

            bool already = false;
            for (uint32_t s : selected) {
                if (s == c) { already = true; break; }
            }
            if (already) continue;

            float min_d = std::numeric_limits<float>::infinity();
            for (uint32_t s : selected) {
                float d = bqDistance(graph.getNodeBQ(s), graph.getNodeBQ(c), words, D);
                if (d < min_d) min_d = d;
            }
            if (min_d > best_min_dist) {
                best_min_dist = min_d;
                best_id = c;
            }
        }
        if (best_min_dist < 0.0f) break;
        selected.push_back(best_id);
    }
    return selected;
}

// ---------------------------------------------------------------------------
// fromNGTv2: SRHT + K-means + PCA + VectorRecord + cluster-aware graph
// ---------------------------------------------------------------------------
NGTAQIndex NGTAQIndex::fromNGTv2(const std::string& ngt_path, const Property& prop) {
    const int D_orig = prop.dimension;
    const int D = NGT::NGTAQ::pad_dim_for_v2(D_orig);  // pad to next power-of-2 divisible by 64
    // D_orig may differ from D (e.g., D_orig=100 → D=128, D_orig=960 → D=1024);
    // raw vectors are zero-padded when loaded.

    // ---- 1. Load all float vectors from NGT ----
    NGT::Index ngt(ngt_path);
    NGT::ObjectSpace& objspace = ngt.getObjectSpace();
    const size_t repo_size = objspace.getRepository().size();
    const size_t N = repo_size - 1;
    const int words = D / 64;

    std::vector<float> raw_flat(N * static_cast<size_t>(D), 0.f);  // zero-padded to D
    std::vector<bool> is_hole(N, false);
    std::vector<float> tmp(D_orig);  // NGT stores D_orig dims; extras stay zero
    for (size_t i = 1; i <= N; i++) {
        try {
            objspace.getObject(static_cast<NGT::ObjectID>(i), tmp);
            // Copy D_orig dims into the D-padded slot; padding dims remain 0
            std::copy(tmp.begin(), tmp.end(),
                      raw_flat.begin() + static_cast<ptrdiff_t>((i - 1) * D));
        } catch (...) {
            is_hole[i - 1] = true;
        }
    }
    fprintf(stderr, "[NGTAQv2] Loaded %zu vectors D_orig=%d D_eff=%d\n", N, D_orig, D);

    // ---- 1b. Angular/Cosine: L2-normalize all raw vectors ----
    // Zero/degenerate vectors (norm <= 1e-6) are marked as holes: they cannot be
    // meaningfully normalized and would pollute cluster centroids and graph edges,
    // causing probe queries to become trapped in zero-vector clusters with distance=q_norm_sq.
    int n_zero = 0;
    if (prop.metric == NGT::ObjectSpace::DistanceTypeAngle ||
        prop.metric == NGT::ObjectSpace::DistanceTypeCosine) {
        for (size_t i = 0; i < N; ++i) {
            float* v = raw_flat.data() + static_cast<ptrdiff_t>(i * static_cast<size_t>(D));
            float norm2 = 0.f;
            for (int d = 0; d < D; ++d) norm2 += v[d] * v[d];
            if (norm2 > 1e-12f) {
                float inv = 1.f / std::sqrt(norm2);
                for (int d = 0; d < D; ++d) v[d] *= inv;
            } else {
                // Degenerate vector: mark as hole so it is excluded from the graph
                is_hole[i] = true;
                ++n_zero;
            }
        }
        fprintf(stderr, "[NGTAQv2] Angular: L2-normalized %zu vectors (%d degenerate holes)\n",
                N, n_zero);
    }

    // ---- 2. SRHT: rotate all vectors ----
    const uint64_t seed = 0xCAFEBABE12345678ULL;
    auto srht = std::make_unique<NGT::NGTAQ::SRHT>(D, seed);
    std::vector<float> rotated(N * static_cast<size_t>(D));
    for (size_t i = 0; i < N; ++i)
        srht->apply(raw_flat.data() + i*D, rotated.data() + i*D);

    // ---- 3. K-means on rotated vectors ----
    uint32_t K = (prop.k_clusters > 0)
                 ? static_cast<uint32_t>(prop.k_clusters)
                 : NGT::NGTAQ::select_k(N);
    auto kmeans = std::make_unique<NGT::NGTAQ::KMeansCentering>(K, D, seed ^ 0xFFFF);
    fprintf(stderr, "[NGTAQv2] K-means K=%u...\n", K);
    kmeans->train(rotated.data(), N);

    // ---- 4. Assign and compute residuals ----
    std::vector<uint32_t> centroid_ids(N);
    kmeans->assign(rotated.data(), N, centroid_ids.data());
    std::vector<float> residuals(N * static_cast<size_t>(D));
    for (size_t i = 0; i < N; ++i)
        kmeans->get_residual(rotated.data() + i*D, centroid_ids[i], residuals.data() + i*D);

    // ---- 5. PCA top-32 on residuals ----
    auto pca = std::make_unique<NGT::NGTAQ::PCAProjector>(D, 32, seed ^ 0x1234);
    size_t fit_n = std::min(N, (size_t)262144);
    fprintf(stderr, "[NGTAQv2] PCA fit on %zu residuals...\n", fit_n);
    pca->fit(residuals.data(), fit_n);

    // ---- 6. PCA-project all residuals ----
    std::vector<float> pca_residuals(N * 32);
    for (size_t i = 0; i < N; ++i)
        pca->project(residuals.data() + i*D, pca_residuals.data() + i*32);

    // ---- 7. Tier-2 PQ: M_PQ sub-codebooks on SRHT residuals (all D dims) ----
    // M_PQ = D/8 sub-spaces × D_sub=8 dims each. K=256 centroids (8 bit/sub).
    // Layout: tier2_cb[(sub*256 + code)*D_sub + dim]
    // M_PQ sub-spaces × 8 bits → uses all M_PQ bytes of tier2 storage.
    // SRHT isotropizes data → equal per-sub-space variance → balanced PQ.
    // D=128: M_PQ=16, D=256: M_PQ=32, D=1024: M_PQ=128.
    const int M_PQ  = D / 8;   // D_sub = 8 fixed; M_PQ scales with D
    const int K_PQ  = 256;
    const int D_sub = 8;       // always 8 dims per sub-space
    std::vector<float> tier2_cb((size_t)M_PQ * K_PQ * D_sub, 0.f);
    fprintf(stderr, "[NGTAQv2] Training %d PQ sub-codebooks (K=%d, D_sub=%d) on SRHT residuals...\n",
            M_PQ, K_PQ, D_sub);
    for (int sub = 0; sub < M_PQ; ++sub) {
        std::vector<float> sub_data(N * (size_t)D_sub);
        for (size_t i = 0; i < N; ++i)
            memcpy(sub_data.data() + i*D_sub,
                   residuals.data() + i*D + sub*D_sub,
                   (size_t)D_sub * sizeof(float));
        NGT::NGTAQ::KMeansCentering sub_km(K_PQ, D_sub, seed ^ (0xABCD1234ULL + (uint64_t)sub));
        sub_km.train(sub_data.data(), N, 262144, 50);
        for (int code = 0; code < K_PQ; ++code)
            memcpy(tier2_cb.data() + (sub*K_PQ + code)*D_sub,
                   sub_km.centroid(code),
                   (size_t)D_sub * sizeof(float));
    }
    fprintf(stderr, "[NGTAQv2] PQ sub-codebooks done.\n");

    // ---- 7b. GLOBAL PQ tier: one codebook over the ROTATED vectors directly ----
    // (Stage A) Unlike tier-2 (per-cluster residuals), this trains M_PQ sub-codebooks
    // on the SRHT-rotated vectors WITHOUT centroid subtraction. A single per-query LUT
    // can then score ANY node via its global code — no per-cluster LUT rebuild.
    // Layout matches tier-2: global_cb[(sub*256 + code)*D_sub + dim].
    std::vector<float> global_cb((size_t)M_PQ * K_PQ * D_sub, 0.f);
    std::vector<uint8_t> global_codes(N * (size_t)M_PQ, 0);
    std::vector<float>   global_norm_sq(N, 0.f);
    {
        fprintf(stderr, "[NGTAQv2] Training %d GLOBAL PQ sub-codebooks on rotated vectors...\n", M_PQ);
        for (int sub = 0; sub < M_PQ; ++sub) {
            std::vector<float> sub_data(N * (size_t)D_sub);
            for (size_t i = 0; i < N; ++i)
                memcpy(sub_data.data() + i*D_sub,
                       rotated.data() + i*D + sub*D_sub,
                       (size_t)D_sub * sizeof(float));
            NGT::NGTAQ::KMeansCentering sub_km(K_PQ, D_sub, seed ^ (0x5EED9001ULL + (uint64_t)sub));
            sub_km.train(sub_data.data(), N, 262144, 50);
            for (int code = 0; code < K_PQ; ++code)
                memcpy(global_cb.data() + (sub*K_PQ + code)*D_sub,
                       sub_km.centroid(code),
                       (size_t)D_sub * sizeof(float));
        }
        // Encode each rotated vector's global PQ code + reconstructed squared norm.
        // Independent per vector → parallelize (mirrors the build's other O(N) passes).
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            if (is_hole[i]) continue;  // holes keep code 0 / norm 0
            const float* xr = rotated.data() + i*D;
            uint8_t* codes = global_codes.data() + i*(size_t)M_PQ;
            float recon_norm_sq = 0.f;
            for (int sub = 0; sub < M_PQ; ++sub) {
                const float* sv = xr + sub*D_sub;
                float best_d = std::numeric_limits<float>::max();
                uint8_t best_code = 0;
                for (int code = 0; code < K_PQ; ++code) {
                    const float* c = global_cb.data() + (sub*K_PQ + code) * D_sub;
                    float dist = 0.f;
                    for (int dd = 0; dd < D_sub; ++dd) { float df = sv[dd]-c[dd]; dist += df*df; }
                    if (dist < best_d) { best_d = dist; best_code = (uint8_t)code; }
                }
                codes[sub] = best_code;
                // accumulate ||reconstructed sub-vector||^2 (centroid the code points to)
                const float* bc = global_cb.data() + (sub*K_PQ + best_code) * D_sub;
                for (int dd = 0; dd < D_sub; ++dd) recon_norm_sq += bc[dd]*bc[dd];
            }
            global_norm_sq[i] = recon_norm_sq;
        }
        fprintf(stderr, "[NGTAQv2] GLOBAL PQ done.\n");
    }

    // ---- 7c. GLOBAL PQ-16 tier (Stage B/C): K=16 (4-bit) codebook over rotated vectors ----
    // REQUIRED for the QG-style vpshufb batch kernel (single _mm256_shuffle_epi8 = 16-entry
    // lookup). K=16; per-node 4-bit codes + reconstructed-norm^2 feed the SoAGraph
    // contiguous neighbor-code store (built below).
    //
    // GPQ4 uses its OWN subspace dimension, DECOUPLED from the legacy tier-2 D_sub=8.
    // QG (D<=400) defaults to D_sub=1 → M=D subspaces (fine routing); our coarse D_sub=8
    // → M=D/8 caps skip-rerank routing recall at ~12%. AQ_GPQ4_DSUB selects D_sub
    // (must divide D). Default 1 (M=D, finest) — the consolidated campaign default: best
    // walk-routing ordering at high recall; M=64 (dsub=2) was an iso-recall wash (coarser
    // quant needs proportionally more hops) but a smaller index. AQ_GPQ4_DSUB=2 for the
    // small-memory tradeoff. K stays 16.
    int gpq4_dsub = 1;
    {
        const char* e = std::getenv("AQ_GPQ4_DSUB");
        if (e) { int v = std::atoi(e); if (v > 0) gpq4_dsub = v; }
        // Snap to a divisor of D so M*D_sub == D exactly (kernel/store assume this).
        while (gpq4_dsub > 1 && (D % gpq4_dsub) != 0) --gpq4_dsub;
        if (gpq4_dsub < 1) gpq4_dsub = 1;
    }
    const int GM   = D / gpq4_dsub;   // GPQ4 subspace count (e.g. D=128: dsub=2→M=64, dsub=1→M=128)
    const int GDsub = gpq4_dsub;
    std::vector<float> gpq4_cb((size_t)GM * NGT::NGTAQ::GPQ4_K * GDsub, 0.f);
    std::vector<uint8_t> gpq4_codes(N * (size_t)GM, 0);
    std::vector<float>   gpq4_norm_sq(N, 0.f);
    {
        fprintf(stderr, "[NGTAQv2] Training %d GLOBAL PQ-16 sub-codebooks (K=16, D_sub=%d) on rotated vectors...\n",
                GM, GDsub);
        for (int sub = 0; sub < GM; ++sub) {
            std::vector<float> sub_data(N * (size_t)GDsub);
            for (size_t i = 0; i < N; ++i)
                memcpy(sub_data.data() + i*GDsub,
                       rotated.data() + i*D + sub*GDsub,
                       (size_t)GDsub * sizeof(float));
            NGT::NGTAQ::KMeansCentering sub_km(NGT::NGTAQ::GPQ4_K, GDsub,
                                               seed ^ (0x4B16C0DEULL + (uint64_t)sub));
            sub_km.train(sub_data.data(), N, 262144, 50);
            for (int code = 0; code < NGT::NGTAQ::GPQ4_K; ++code)
                memcpy(gpq4_cb.data() + (sub*NGT::NGTAQ::GPQ4_K + code)*GDsub,
                       sub_km.centroid(code),
                       (size_t)GDsub * sizeof(float));
        }
        // ScaNN anisotropic encoding knob (AQ_GPQ4_ETA, default 1.0). eta=1 => plain
        // per-subspace L2-argmin (bit-identical to gpq4_encode_sub). eta>1 => noise-shaped
        // coordinate descent that sharpens RANKING at the same 4-bit budget (codebook,
        // LUT, kernel, storage all unchanged — only code selection differs).
        const float gpq4_eta = []{ const char* e = std::getenv("AQ_GPQ4_ETA");
            float v = e ? (float)std::atof(e) : 1.0f; return v < 1.0f ? 1.0f : v; }();
        fprintf(stderr, "[NGTAQv2] GPQ4 encoding with eta=%.2f (%s)\n", gpq4_eta,
                gpq4_eta > 1.0f ? "ScaNN anisotropic" : "L2-argmin");
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            if (is_hole[i]) continue;
            const float* xr = rotated.data() + i*D;
            uint8_t* codes = gpq4_codes.data() + i*(size_t)GM;
            float rns = 0.f;
            if (gpq4_eta > 1.0f) {
                NGT::NGTAQ::gpq4_encode_anisotropic(xr, gpq4_cb.data(), GM, GDsub, gpq4_eta, codes, rns);
            } else {
                for (int sub = 0; sub < GM; ++sub)
                    codes[sub] = NGT::NGTAQ::gpq4_encode_sub(
                        xr + sub*GDsub, gpq4_cb.data() + (size_t)sub*NGT::NGTAQ::GPQ4_K*GDsub, GDsub, rns);
            }
            gpq4_norm_sq[i] = rns;
        }
        fprintf(stderr, "[NGTAQv2] GLOBAL PQ-16 done.\n");
    }

    // ---- 7d. Tech 1: symmetric SQ8 codes over the SRHT-rotated vectors ----
    // Per-node int8 (max-scaled) code + max + ||x||^2, for the cheap in-loop routing dot.
    const int sq8_align = ((D + 63) / 64) * 64;   // pad to multiple of 64 for the VNNI dot
    std::vector<int8_t> sq8_codes((size_t)N * sq8_align, 0);
    std::vector<float>  sq8_max(N, 0.f);
    std::vector<float>  sq8_norm(N, 0.f);
    {
        fprintf(stderr, "[NGTAQv2] Encoding symmetric SQ8 codes (dim_align=%d)...\n", sq8_align);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            if (is_hole[i]) continue;
            const float* xr = rotated.data() + i * D;
            int8_t* code = sq8_codes.data() + i * (size_t)sq8_align;
            sq8_max[i] = NGT::NGTAQ::sq8_encode_sym(xr, code, D);  // [D, sq8_align) stays 0
            float ns = 0.f;
            for (int d = 0; d < D; ++d) ns += xr[d] * xr[d];
            sq8_norm[i] = ns;
        }
        fprintf(stderr, "[NGTAQv2] SQ8 codes done.\n");
    }

    // ---- 8. Encode all vectors into VectorRecord ----
    // Build the BQ-compatible SoAGraph (needed for existing graph infra + v1 compat)
    BinaryQuantizer bq;
    bq.init(D);
    bq.setRandomRotation();
    {
        std::vector<const float*> ptrs(N);
        for (size_t i = 0; i < N; i++) ptrs[i] = raw_flat.data() + i * D;
        bq.calibrateTau(ptrs, prop.n_tau_samples, prop.metric);
    }
    auto graph = std::make_unique<SoAGraph>(words);
    {
        std::vector<uint64_t> bq_buf(static_cast<size_t>(words) * 2);
        for (size_t i = 0; i < N; i++) {
            bq.encode(raw_flat.data() + i * D, bq_buf.data());
            graph->addNode(bq_buf.data());
        }
    }
    graph->finalizeCSR();
    for (size_t i = 0; i < N; ++i)
        if (is_hole[i]) graph->removeNode(static_cast<uint32_t>(i));

    // Fill v2 VectorRecords using VectorRecordView (variable-D safe)
    graph->reserveV2(N, D/8, D/8);  // tier1_n = D/8, tier2_n = D/8
    for (size_t i = 0; i < N; ++i) {
        if (is_hole[i]) continue;
        auto view = graph->getRecordView(static_cast<uint32_t>(i));
        view.set_centroid_id(centroid_ids[i]);

        // tier-1: sign bits of SRHT residual (D bits → D/8 bytes)
        const float* res = residuals.data() + i*D;
        for (int b = 0; b < D; ++b)
            view.set_tier1_bit(b, res[b] >= 0.f);

        // norm_fp16: L2 norm of residual
        float norm2 = 0.f;
        for (int d = 0; d < D; ++d) norm2 += res[d] * res[d];
        view.set_norm_fp16(NGT::NGTAQ::float_to_fp16(std::sqrt(norm2)));

        // tier-2: M_PQ independent PQ codes (M_PQ bytes, 8-bit each)
        // Each byte encodes nearest centroid (0-255) for D_sub-dim sub-vector of SRHT residual
        const float* sv_base = residuals.data() + i * D;
        for (int sub = 0; sub < M_PQ; ++sub) {
            const float* sv = sv_base + sub * D_sub;
            float best_d = std::numeric_limits<float>::max();
            uint8_t best_code = 0;
            for (int code = 0; code < K_PQ; ++code) {
                const float* c = tier2_cb.data() + (sub*K_PQ + code) * D_sub;
                float dist = 0.f;
                for (int dd = 0; dd < D_sub; ++dd) {
                    float diff = sv[dd] - c[dd]; dist += diff * diff;
                }
                if (dist < best_d) { best_d = dist; best_code = (uint8_t)code; }
            }
            view.set_tier2_byte(sub, best_code);
        }
    }

    // ---- 9. Build cluster-aware graph from NGT edges ----
    NGT::GraphIndex& gi = static_cast<NGT::GraphIndex&>(ngt.getIndex());
    AlphaCGPruner pruner(prop.alpha, prop.kappa);
    const float tau = bq.tau();

    std::vector<std::vector<uint32_t>> adj(N);
    for (size_t i = 1; i <= N; i++) {
        uint32_t aq_id = static_cast<uint32_t>(i - 1);
        if (is_hole[aq_id]) continue;
        NGT::GraphNode* node = nullptr;
        try { node = gi.getNode(static_cast<NGT::ObjectID>(i)); }
        catch (...) { continue; }
        if (!node || node->empty()) continue;

        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(node->size());
        for (auto& edge : *node) {
            if (edge.id == 0 || edge.id > static_cast<unsigned int>(N)) continue;
            uint32_t nbr = static_cast<uint32_t>(edge.id - 1);
            float d_bq = bqDistance(graph->getNodeBQ(aq_id), graph->getNodeBQ(nbr), words, D);
            candidates.push_back({nbr, d_bq});
        }
        // Pure BQ distance sort (no cluster priority — hurts navigability).
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (static_cast<int>(candidates.size()) > prop.max_edges)
            candidates.resize(static_cast<size_t>(prop.max_edges));

        // Tech 3-FULL: when the source NGT index is already an ONNG (reconstruct-graph
        // produced a navigability-optimized graph), the BQ-space AlphaCG occlusion prune
        // can UNDO that navigability. AQ_KEEP_ONNG=1 keeps the ONNG edges verbatim
        // (BQ-sorted + truncated to max_edges) — no re-prune.
        static const bool keep_onng = [] {
            const char* e = std::getenv("AQ_KEEP_ONNG");
            return e && std::atoi(e) != 0;
        }();
        // Tech 3-FULL (route 2): SSG-style ANGLE occlusion (pyglass NSG sync_prune).
        // Prune u if an accepted neighbor v subtends an angle ∠(v,p,u) below `angle` degrees,
        // i.e. cos∠ > cos(angle). cos∠ = (d_pu + d_pv − d_vu)/(2·sqrt(d_pu·d_pv)), all in BQ
        // distance space. A wider min-angle → more spread-out (navigable) edges. AQ_SSG_ANGLE
        // sets the degrees (default 60, pyglass NSG default). AQ_SSG=1 enables.
        static const bool use_ssg = [] {
            const char* e = std::getenv("AQ_SSG");
            return e && std::atoi(e) != 0;
        }();
        static const float ssg_cos = [] {
            const char* e = std::getenv("AQ_SSG_ANGLE");
            float deg = e ? std::atof(e) : 60.0f;
            return std::cos(deg * 3.14159265358979f / 180.0f);
        }();
        if (keep_onng) {
            adj[aq_id].reserve(candidates.size());
            for (auto& c : candidates) adj[aq_id].push_back(c.first);
        } else if (use_ssg) {
            std::vector<std::pair<uint32_t, float>> acc;  // (id, d_pu) of accepted
            acc.reserve(candidates.size());
            for (auto& [uid, d_pu] : candidates) {
                bool occ = false;
                for (auto& [vid, d_pv] : acc) {
                    float d_vu = bqDistance(graph->getNodeBQ(vid), graph->getNodeBQ(uid), words, D);
                    float denom = 2.0f * std::sqrt(std::max(d_pu * d_pv, 1e-12f));
                    float cos_vu = (d_pu + d_pv - d_vu) / denom;
                    if (cos_vu > ssg_cos) { occ = true; break; }
                }
                if (!occ) {
                    acc.push_back({uid, d_pu});
                    if ((int)acc.size() >= prop.max_edges) break;
                }
            }
            adj[aq_id].reserve(acc.size());
            for (auto& a : acc) adj[aq_id].push_back(a.first);
        } else {
            auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
                return bqDistance(graph->getNodeBQ(v), graph->getNodeBQ(u), words, D);
            };
            adj[aq_id] = pruner.prune(candidates, tau, dist_fn);
        }
    }
    graph->resetEdges(adj);

    int n_ep = std::min(prop.n_entry_points, static_cast<int>(N));
    auto entry_points = selectEntryPoints(*graph, n_ep);

    // Pack the fp32 working buffer into fp16 for the exact-rerank store (Task 0.2).
    // raw_flat (fp32) was consumed by SRHT/BQ/normalization above; only fp16 is kept.
    std::vector<uint16_t> raw_flat_h(raw_flat.size());
    for (size_t i = 0; i < raw_flat.size(); ++i)
        raw_flat_h[i] = NGT::NGTAQ::float_to_fp16(raw_flat[i]);

    // Construct index
    NGTAQIndex idx(prop, std::move(bq), std::move(graph),
                   std::move(entry_points), std::move(raw_flat_h));
    idx.is_v2_ = true;
    idx.srht_v2_        = std::move(srht);
    idx.kmeans_v2_      = std::move(kmeans);
    idx.pca_v2_         = std::move(pca);
    idx.tier2_codebook_ = std::move(tier2_cb);
    // Build transposed codebook [M][D_sub][K] for AVX2 FMA LUT build
    {
        const int Dv2 = D;
        const int Mv2 = M_PQ, Kv2 = K_PQ, DSv2 = D_sub;
        idx.tier2_codebook_T_.resize((size_t)Mv2 * DSv2 * Kv2);
        NGT::NGTAQ::build_tier2_codebook_T(
            idx.tier2_codebook_.data(), Mv2, Kv2, DSv2,
            idx.tier2_codebook_T_.data());
        (void)Dv2;
    }
    idx.v2_entry_points_ = idx.entry_points_;  // reuse existing entry points for v2

    // GLOBAL PQ tier (Stage A): codebook + transposed copy + per-vector codes/norms.
    idx.global_pq_codebook_ = std::move(global_cb);
    idx.global_codes_       = std::move(global_codes);
    idx.global_pq_norm_sq_  = std::move(global_norm_sq);
    {
        idx.global_pq_codebook_T_.resize((size_t)M_PQ * D_sub * K_PQ);
        NGT::NGTAQ::build_tier2_codebook_T(
            idx.global_pq_codebook_.data(), M_PQ, K_PQ, D_sub,
            idx.global_pq_codebook_T_.data());
    }
    idx.has_global_pq_ = true;

    // GLOBAL PQ-16 tier (Stage B/C): codebook (+ transposed) + contiguous neighbor store.
    // buildGPQ4 reads the FINAL CSR edges (resetEdges already ran above), so the per-node
    // neighbor blocks reflect the pruned graph used at search time.
    idx.gpq4_codebook_ = std::move(gpq4_cb);
    idx.gpq4_m_pq_ = GM;    // GPQ4's OWN subspace count (decoupled from legacy m_pq_)
    {
        idx.gpq4_codebook_T_.resize((size_t)GM * GDsub * NGT::NGTAQ::GPQ4_K);
        NGT::NGTAQ::build_tier2_codebook_T(
            idx.gpq4_codebook_.data(), GM, NGT::NGTAQ::GPQ4_K, GDsub,
            idx.gpq4_codebook_T_.data());
    }
    idx.graph_->buildGPQ4(GM, gpq4_codes.data(), gpq4_norm_sq.data());
    idx.gpq4_codes_   = std::move(gpq4_codes);    // flat per-node codes (seeds/expansion)
    idx.gpq4_norm_sq_ = std::move(gpq4_norm_sq);
    idx.has_gpq4_ = true;
    fprintf(stderr, "[NGTAQv2] GPQ4 neighbor-code store built (M=%d, D_sub=%d).\n", GM, GDsub);

    // Tech 1: symmetric SQ8 codes (per-node int8 + max + ||x||^2) for the in-loop dot.
    idx.sq8_dim_align_ = sq8_align;
    idx.sq8_codes_ = std::move(sq8_codes);
    idx.sq8_max_   = std::move(sq8_max);
    idx.sq8_norm_  = std::move(sq8_norm);
    idx.has_sq8_   = true;

    idx.is_angular_ = (prop.metric == NGT::ObjectSpace::DistanceTypeAngle ||
                       prop.metric == NGT::ObjectSpace::DistanceTypeCosine);
    idx.d_eff_ = D;
    idx.m_pq_  = M_PQ;

    // Consolidated default: GORDER cache-locality node-ID reorder (commit 8e6d4d2). Pure
    // permutation, recall byte-stable, dynamic-safe (re-run after bulk updates). Folds the
    // former post-build reorder_index step into the default build. AQ_NO_REORDER=1 skips it;
    // AQ_REORDER=bfs|rcm selects an alternate mode. Requires a tombstone-free fresh build (true).
    {
        const char* skip = std::getenv("AQ_NO_REORDER");
        if (!(skip && std::atoi(skip) != 0)) {
            ReorderMode rm = ReorderMode::GORDER;
            if (const char* m = std::getenv("AQ_REORDER")) {
                std::string ms(m);
                if (ms == "bfs") rm = ReorderMode::BFS;
                else if (ms == "rcm") rm = ReorderMode::RCM;
            }
            fprintf(stderr, "[NGTAQv2] Reordering for cache locality (mode=%s)...\n",
                    rm == ReorderMode::GORDER ? "gorder" : rm == ReorderMode::RCM ? "rcm" : "bfs");
            idx.reorderForLocality(rm);
        }
    }

    fprintf(stderr, "[NGTAQv2] Build complete. N=%zu K=%u\n", N, K);
    return idx;
}

// ---------------------------------------------------------------------------
// Stage A: GLOBAL PQ routing tier — single per-query LUT scores any node.
// ---------------------------------------------------------------------------
float NGTAQIndex::buildGlobalLUT(const std::vector<float>& query, float* lut) const {
    if (!has_global_pq_)
        throw std::runtime_error("buildGlobalLUT: index has no global PQ tier");
    const int D      = (d_eff_ > 0) ? d_eff_ : prop_.dimension;
    const int M_PQ   = (m_pq_ > 0) ? m_pq_ : 16;
    const int D_orig = (int)query.size();

    // Angular/Cosine: L2-normalize the query exactly as searchV2 does, so the rotated
    // query lives in the same space as the (normalized→rotated) stored vectors.
    static thread_local std::vector<float> q_norm_tl;
    const float* q_src = query.data();
    if (is_angular_) {
        q_norm_tl.assign(query.begin(), query.begin() + std::min(D_orig, D));
        float n2 = 0.f;
        for (float x : q_norm_tl) n2 += x * x;
        if (n2 > 1e-12f) { float inv = 1.f/std::sqrt(n2); for (float& x : q_norm_tl) x *= inv; }
        q_src = q_norm_tl.data();
    }

    // Zero-pad to D, then SRHT-rotate (codebook was trained on rotated vectors directly).
    static thread_local std::vector<float> q_padded_tl, q_rot_tl;
    q_padded_tl.assign(static_cast<size_t>(D), 0.f);
    std::copy(q_src, q_src + std::min(D_orig, D), q_padded_tl.begin());
    q_rot_tl.resize(static_cast<size_t>(D));
    srht_v2_->apply(q_padded_tl.data(), q_rot_tl.data());

    // LUT[sub][code] = <q_rot_sub, global_centroid_sub>. Reuse the tier-2 fast builder.
    NGT::NGTAQ::build_tier2_lut_fast_m(q_rot_tl.data(), M_PQ,
                                        global_pq_codebook_T_.data(), lut);

    float q_norm_sq = 0.f;
    for (int d = 0; d < D; ++d) q_norm_sq += q_rot_tl[d] * q_rot_tl[d];
    return q_norm_sq;
}

float NGTAQIndex::globalPQDist(uint32_t node_id, const float* lut, float q_norm_sq) const {
    const int M_PQ = (m_pq_ > 0) ? m_pq_ : 16;
    const uint8_t* codes = global_codes_.data() + (size_t)node_id * M_PQ;
    // <q_rot, x_rot_pq> via the global LUT (one LUT scores any node).
    float ip = NGT::NGTAQ::tier2_adc_pq_m(lut, codes, M_PQ);
    // ||q_rot - x_rot_pq||^2 = ||q_rot||^2 + ||x_rot_pq||^2 - 2<q_rot, x_rot_pq>.
    return q_norm_sq + global_pq_norm_sq_[node_id] - 2.0f * ip;
}

// ---------------------------------------------------------------------------
// Stage B/C: build the per-query uint8 batch LUT (K=16) and return ||q_rot||^2.
// ---------------------------------------------------------------------------
float NGTAQIndex::buildGlobalLUT16(const std::vector<float>& query,
                                   NGT::NGTAQ::GlobalPQ4LUT& lut,
                                   float* ip_out, bool dist_lut) const {
    const int D      = (d_eff_ > 0) ? d_eff_ : prop_.dimension;
    const int M_PQ   = gpq4MPQ();          // GPQ4's own (fine) subspace count
    const int D_sub  = (M_PQ > 0) ? D / M_PQ : 8;
    const int D_orig = (int)query.size();

    static thread_local std::vector<float> q_norm_tl;
    const float* q_src = query.data();
    if (is_angular_) {
        q_norm_tl.assign(query.begin(), query.begin() + std::min(D_orig, D));
        float n2 = 0.f;
        for (float x : q_norm_tl) n2 += x * x;
        if (n2 > 1e-12f) { float inv = 1.f/std::sqrt(n2); for (float& x : q_norm_tl) x *= inv; }
        q_src = q_norm_tl.data();
    }
    static thread_local std::vector<float> q_padded_tl, q_rot_tl, ip_tl;
    q_padded_tl.assign(static_cast<size_t>(D), 0.f);
    std::copy(q_src, q_src + std::min(D_orig, D), q_padded_tl.begin());
    q_rot_tl.resize(static_cast<size_t>(D));
    srht_v2_->apply(q_padded_tl.data(), q_rot_tl.data());

    ip_tl.resize((size_t)M_PQ * NGT::NGTAQ::GPQ4_K);
    if (dist_lut) {
        // QG-style squared-distance table: kernel accumulates L2 directly.
        NGT::NGTAQ::gpq4_dist_table(q_rot_tl.data(), M_PQ, gpq4_codebook_T_.data(), D_sub, ip_tl.data());
    } else {
        NGT::NGTAQ::gpq4_ip_table(q_rot_tl.data(), M_PQ, gpq4_codebook_T_.data(), D_sub, ip_tl.data());
    }
    NGT::NGTAQ::gpq4_build_lut(ip_tl.data(), M_PQ, lut);
    if (ip_out) std::copy(ip_tl.begin(), ip_tl.end(), ip_out);

    float q_norm_sq = 0.f;
    for (int d = 0; d < D; ++d) q_norm_sq += q_rot_tl[d] * q_rot_tl[d];
    return q_norm_sq;
}

// Diagnostic: mean |neighbor_id - node_id| over the CSR (graph ID-locality proxy).
std::pair<double,double> NGTAQIndex::graphLocalityStats() const {
    const size_t N = graph_->size();
    double sum_gap = 0.0, sum_deg = 0.0; size_t cnt = 0;
    for (uint32_t i = 0; i < (uint32_t)N; ++i) {
        if (graph_->isTombstone(i)) continue;
        auto nbrs = graph_->getNeighbors(i);
        sum_deg += (double)nbrs.size();
        for (size_t k = 0; k < nbrs.size(); ++k) {
            uint32_t u = nbrs[k];
            sum_gap += (u > i) ? (double)(u - i) : (double)(i - u);
            ++cnt;
        }
    }
    return { cnt ? sum_gap / cnt : 0.0, N ? sum_deg / N : 0.0 };
}

// Node-ID reorder for walk cache-locality (Task 2). Pure permutation; recall-invariant.
void NGTAQIndex::reorderForLocality(ReorderMode mode) {
    const size_t N = graph_->size();
    if (N == 0) return;
    if (graph_->hasTombstones())
        throw std::runtime_error("reorderForLocality: rebuild() to compact tombstones first");

    // 1. Compute old_to_new (o2n): old node i's data lands at new id o2n[i].
    std::vector<uint32_t> o2n(N, UINT32_MAX);
    uint32_t next_id = 0;
    auto place = [&](uint32_t old_id) {
        if (o2n[old_id] == UINT32_MAX) { o2n[old_id] = next_id++; return true; }
        return false;
    };

    if (mode == ReorderMode::GORDER) {
        // GORDER (Coleman et al., NeurIPS'22): greedy window-based co-occurrence ordering.
        // Maintain a sliding window of the last W placed nodes; the next node to place is the
        // unplaced candidate with the most edges to the window (shared-neighbor locality), so
        // nodes co-visited during a walk get nearby IDs. Candidates = neighbors of windowed
        // nodes (so the frontier stays bounded). Score = #edges into the window.
        const int W = 5;  // sliding window size (Coleman reports 5 near-optimal)
        std::vector<uint32_t> order; order.reserve(N);
        // seed with the highest-degree node (most central) for a strong locality core.
        uint32_t seed = 0, best_deg = 0;
        for (uint32_t i = 0; i < (uint32_t)N; ++i) {
            uint32_t d = graph_->neighborCount(i);
            if (d > best_deg) { best_deg = d; seed = i; }
        }
        std::vector<uint8_t> in_window(N, 0);
        std::vector<int>     gain(N, 0);            // #edges from candidate to current window
        std::vector<uint32_t> cand;                 // candidates with gain>0 (deduped via gain>0)
        auto add_to_window = [&](uint32_t x) {
            in_window[x] = 1;
            for (uint32_t u : graph_->getNeighbors(x)) {
                if (u >= N || o2n[u] != UINT32_MAX) continue;
                if (gain[u] == 0) cand.push_back(u);
                ++gain[u];
            }
        };
        auto remove_from_window = [&](uint32_t x) {
            in_window[x] = 0;
            for (uint32_t u : graph_->getNeighbors(x)) {
                if (u >= N || o2n[u] != UINT32_MAX) continue;
                if (gain[u] > 0) --gain[u];
            }
        };
        std::vector<uint32_t> window;  // ring of last W placed
        place(seed); order.push_back(seed); add_to_window(seed); window.push_back(seed);
        for (size_t step = 1; step < N; ++step) {
            // pick the unplaced candidate with max gain (ties: smallest id for determinism).
            uint32_t best = UINT32_MAX; int bg = -1;
            for (uint32_t c : cand) {
                if (o2n[c] != UINT32_MAX) continue;
                if (gain[c] > bg) { bg = gain[c]; best = c; }
            }
            if (best == UINT32_MAX) {  // window frontier exhausted → next unplaced by id
                for (uint32_t i = 0; i < (uint32_t)N; ++i) if (o2n[i] == UINT32_MAX) { best = i; break; }
            }
            place(best); order.push_back(best);
            // slide window: evict oldest if full, then add the new node.
            if ((int)window.size() == W) { remove_from_window(window.front());
                                           window.erase(window.begin()); }
            add_to_window(best); window.push_back(best);
            // compact cand occasionally (drop placed) to bound the scan.
            if ((step & 1023) == 0) {
                cand.erase(std::remove_if(cand.begin(), cand.end(),
                    [&](uint32_t c){ return o2n[c] != UINT32_MAX || gain[c] == 0; }), cand.end());
            }
        }
    } else {
        // BFS / RCM: queue-based. RCM visits neighbors in ASCENDING degree (bandwidth-min);
        // BFS visits in CSR order. RCM additionally reverses the final order.
        std::vector<uint32_t> queue; queue.reserve(N);
        auto push = [&](uint32_t old_id){ if (place(old_id)) queue.push_back(old_id); };
        if (mode == ReorderMode::RCM) {
            // RCM seeds from the minimum-degree node; expands neighbors degree-ascending.
            uint32_t seed = 0, mind = UINT32_MAX;
            for (uint32_t i = 0; i < (uint32_t)N; ++i) {
                uint32_t d = graph_->neighborCount(i);
                if (d < mind) { mind = d; seed = i; }
            }
            push(seed);
            for (size_t head = 0; head < queue.size(); ++head) {
                uint32_t x = queue[head];
                std::vector<uint32_t> nb;
                for (uint32_t u : graph_->getNeighbors(x)) if (u < N && o2n[u] == UINT32_MAX) nb.push_back(u);
                std::sort(nb.begin(), nb.end(), [&](uint32_t a, uint32_t b){
                    return graph_->neighborCount(a) < graph_->neighborCount(b); });
                for (uint32_t u : nb) push(u);
            }
        } else {  // BFS (default, shipped baseline)
            for (uint32_t ep : entry_points_) if (ep < N) push(ep);
            if (next_id == 0) push(0);  // fallback seed
            for (size_t head = 0; head < queue.size(); ++head) {
                uint32_t x = queue[head];
                for (uint32_t u : graph_->getNeighbors(x)) if (u < N) push(u);
            }
        }
        // Disconnected nodes (not reached) keep relative order at the tail.
        for (uint32_t i = 0; i < (uint32_t)N; ++i) if (o2n[i] == UINT32_MAX) o2n[i] = next_id++;
        if (mode == ReorderMode::RCM) {
            // Reverse: new id k -> (N-1-k). Cuthill-McKee reversed = RCM.
            for (uint32_t i = 0; i < (uint32_t)N; ++i) o2n[i] = (uint32_t)(N - 1) - o2n[i];
        }
    }

    // 2. Permute the graph (CSR + bq + v2 records) and invalidate gpq4/packed stores.
    graph_->applyPermutation(o2n);

    // 3. Permute NGTAQIndex node-keyed arrays. Helper for a row-major [N*stride] vector.
    auto permute_rows = [&](auto& vec, size_t stride) {
        if (vec.empty()) return;
        using T = typename std::decay_t<decltype(vec)>::value_type;
        std::vector<T> nv(vec.size());
        for (uint32_t i = 0; i < (uint32_t)N; ++i)
            std::memcpy(&nv[(size_t)o2n[i] * stride], &vec[(size_t)i * stride], stride * sizeof(T));
        vec.swap(nv);
    };
    permute_rows(raw_flat_,          (size_t)dEff());
    permute_rows(gpq4_codes_,        (size_t)gpq4MPQ());
    permute_rows(gpq4_norm_sq_,      1);
    if (has_global_pq_) { permute_rows(global_codes_, (size_t)m_pq_); permute_rows(global_pq_norm_sq_, 1); }
    if (has_sq8_) { permute_rows(sq8_codes_, (size_t)sq8_dim_align_); permute_rows(sq8_max_, 1); permute_rows(sq8_norm_, 1); }

    // 4. Remap entry points; clear lazy cluster tables (rebuilt on next search with new IDs).
    for (auto& ep : entry_points_) if (ep < N) ep = o2n[ep];
    v2_entry_points_ = entry_points_;
    cluster_members_v2_.clear();
    cluster_neighbors_v2_.clear();
    cluster_members_once_ = std::make_unique<std::once_flag>();

    // 5. Rebuild the gpq4 neighbor-code store from the permuted CSR + permuted per-node codes.
    if (has_gpq4_)
        graph_->buildGPQ4(gpq4MPQ(), gpq4_codes_.data(), gpq4_norm_sq_.data());

    // 6. Build internal->external id map (compose with any prior map). new id o2n[old]
    //    holds old node's data → its external id is the OLD node's external id.
    std::vector<uint32_t> prev_ext = id_to_external_;  // empty == identity
    id_to_external_.assign(N, 0);
    for (uint32_t old_i = 0; old_i < (uint32_t)N; ++old_i) {
        uint32_t ext = prev_ext.empty() ? old_i : prev_ext[old_i];
        id_to_external_[o2n[old_i]] = ext;
    }
}

// Diagnostic: rotate a raw query through SRHT exactly as searchV2's setup does.
void NGTAQIndex::rotateForDiag(const float* q, int q_dim, float* out) const {
    const int D = (d_eff_ > 0) ? d_eff_ : prop_.dimension;
    std::vector<float> padded(static_cast<size_t>(D), 0.f);
    std::copy(q, q + std::min(q_dim, D), padded.begin());
    srht_v2_->apply(padded.data(), out);
}

// ---------------------------------------------------------------------------
// searchV2: ADC search with lazy centroid switch
// ---------------------------------------------------------------------------
std::vector<SearchResult> NGTAQIndex::searchV2(
    const std::vector<float>& query, int k,
    float gamma_enq, float gamma_term,
    int rerank_factor, int max_visits) const
{
    if (!is_v2_)
        throw std::runtime_error("searchV2: call fromNGTv2() first");
    if (static_cast<int>(query.size()) < prop_.dimension)
        throw std::invalid_argument("searchV2: query dimension mismatch");

    AQ_T0();  // [AQ_PROFILE] start timing the searchV2 body (after early-return guards)

    // rerank_factor: widen beam by searching for k_beam candidates, return top k_out.
    // rerank_factor <= 1: standard behavior (k_beam == k).
    // rerank_factor >  1: search k*rerank_factor internally, trim to k at output.
    const int k_beam = (rerank_factor > 1) ? k * rerank_factor : k;
    const int k_out  = k;

    const int D     = (d_eff_ > 0) ? d_eff_ : prop_.dimension;
    const int M_PQ  = (m_pq_ > 0) ? m_pq_ : 16;
    const int D_orig = (int)query.size();

    // Coarse-routing detection, hoisted ABOVE the per-query setup so we can skip the
    // tier-1 ADC encode + tier-2 LUT build that the batch/global-PQ path never uses
    // (all per-cluster ADC scoring is gated behind !use_coarse). Definitions of the
    // env statics below reuse these. AQ_PROFILE showed setup (~20us) becomes the
    // dominant fixed cost once dabs is lean, and tier-1/tier-2 encode is its bulk.
    static const bool use_global_env_e = [] {
        const char* e = std::getenv("AQ_USE_GLOBAL_ROUTING");
        return e && std::atoi(e) != 0;
    }();
    // Default ON: QG-style contiguous-neighbor 16-wide vpshufb batch path. A popped node's
    // whole neighbor block is scored in one SIMD sweep over the co-located GPQ4 code store
    // (dist-LUT form: no per-neighbor norm read, NO gather). Measured ns/hop 1533->995 (-35%)
    // and ~1.4-1.8x iso-recall vs the legacy per-neighbor cand_q/per-cluster-ADC path. The
    // recall knob for this path is AQ_EF (or max_visits); set AQ_BATCH_ROUTING=0 to revert to
    // the legacy path (gamma_term knob). (Mirrors the search-body selector below; must agree.)
    static const bool use_batch_env_e = [] {
        const char* e = std::getenv("AQ_BATCH_ROUTING");
        return e ? (std::atoi(e) != 0) : true;  // default ON (gather-free contiguous sweep)
    }();
    const bool use_batch_e  = hasGPQ4() && use_batch_env_e;
    const bool use_global_e = (has_global_pq_ && use_global_env_e) && !use_batch_e;
    const bool use_coarse_e = use_batch_e || use_global_e;

    // 0. Angular/Cosine: L2-normalize query (over D_orig dims)
    static thread_local std::vector<float> q_normalized_tl;
    const float* q_src = query.data();
    if (is_angular_) {
        q_normalized_tl.assign(query.begin(), query.begin() + std::min(D_orig, D));
        float norm2 = 0.f;
        for (float x : q_normalized_tl) norm2 += x * x;
        if (norm2 > 1e-12f) {
            float inv = 1.f / std::sqrt(norm2);
            for (float& x : q_normalized_tl) x *= inv;
        }
        q_src = q_normalized_tl.data();
    }

    // Query padding: zero-pad to D (for D_orig < D, e.g., GloVe-100 → D=128)
    // Thread-local scratch buffers: reused across queries on same thread to avoid
    // per-query heap allocation overhead (~50KB for D=1024, significant for QPS).
    static thread_local std::vector<float> q_padded_tl;
    static thread_local std::vector<float> q_rot_tl;
    static thread_local std::vector<float> q_res_tl;
    static thread_local std::vector<float> q_res_init_tl;
    // ADC state: q_int8 buffer lives in a thread_local struct (one alloc per thread).
    // Using a reference alias so all existing `adc.*` accesses compile unchanged.
    static thread_local NGT::NGTAQ::ADCQueryState adc_tl(0);

    q_padded_tl.assign(static_cast<size_t>(D), 0.f);
    std::copy(q_src, q_src + std::min(D_orig, D), q_padded_tl.begin());
    const float* q_ptr = q_padded_tl.data();

    // 1. Rotate query
    q_rot_tl.resize(static_cast<size_t>(D));
    AQ_MARK();  // [AQ_PROFILE] start SRHT sub-timer (within setup)
    srht_v2_->apply(q_ptr, q_rot_tl.data());
    AQ_SUB(srht);  // [AQ_PROFILE] SRHT rotation cost

    // 2. Find query's nearest centroid. On the coarse (batch/global-PQ) path the centroid
    // only selects seed clusters — the graph walk corrects near-ties — so use the 2-level
    // coarse quantizer (scan ~sqrt(K) super-centroids + the members of the top-`probe`,
    // ~3.5x fewer distance evals than the brute K-way scan). probe=4 hits 96.6% exact
    // agreement at ~3.6us vs ~9.8us fp16 / ~12.8us fp32 — the brute scan was the dominant
    // setup cost. The exact (non-coarse) path keeps the full-precision fp32 scan.
    // AQ_CQ_PROBE tunes the super-centroid probe count (accuracy/speed knob).
    static const int cq_probe = [] {
        const char* e = std::getenv("AQ_CQ_PROBE");
        int v = e ? std::atoi(e) : 4;
        return v > 0 ? v : 4;
    }();
    uint32_t active_cid = use_coarse_e ? kmeans_v2_->nearest_2level(q_rot_tl.data(), cq_probe)
                                       : kmeans_v2_->nearest_public(q_rot_tl.data());
    q_res_tl.resize(static_cast<size_t>(D));

    // 3. Build initial ADC state (tier-1 + tier-2 PQ on SRHT residuals)
    // Grow q_int8 once on first use (or if D changed); no allocation on steady-state.
    if (static_cast<int>(adc_tl.q_int8.size()) < D)
        adc_tl.q_int8.assign(static_cast<size_t>(D), int8_t(0));
    NGT::NGTAQ::ADCQueryState& adc = adc_tl;
    adc.q_norm_sq = 0.f; adc.q_norm = 0.f; adc.q_sum = 0;
    float q_norm_sq = 0.f;
    // Coarse routing (batch/global-PQ) scores every node through the single per-query
    // PQ LUT (gpq4 vpshufb / global LUT); it never touches the per-cluster tier-1 ADC
    // (adc_dist/maybe_rebuild_adc) nor the tier-2 LUT. So the tier-1 residual encode
    // here — residual subtract + int8 quantize + norm over D dims — is dead work on the
    // batch path. Skip it; the centroid was still needed for cluster-seed selection.
    if (!use_coarse_e) {
        NGT::NGTAQ::compute_residual_and_tier1(
            q_rot_tl.data(), kmeans_v2_->centroid(active_cid), D,
            q_res_tl.data(), adc.q_norm_sq, adc.q_int8.data(), adc.q_sum);
        adc.q_norm = std::sqrt(adc.q_norm_sq);
        q_norm_sq = adc.q_norm_sq;
    }
    const float inv_sqrt_D = 1.f / std::sqrt((float)D);

    // Save initial cluster residual for tier-2 LUT build post-routing.
    // maybe_rebuild_adc overwrites q_res/q_norm_sq on cluster transitions.
    // Coarse path never builds the tier-2 LUT, so skip the residual snapshot copy too.
    const uint32_t initial_cid = active_cid;
    if (!use_coarse_e)
        q_res_init_tl.assign(q_res_tl.begin(), q_res_tl.end());
    const float q_norm_sq_initial = q_norm_sq;

    std::shared_lock<std::shared_mutex> lock(graph_->mutex());
    const size_t N = graph_->size();

    // Lazy-build cluster inverted list + precomputed cluster neighbors (once per index lifetime).
    // Done under shared_lock (N stable). call_once provides thread-safe one-shot semantics.
    std::call_once(*cluster_members_once_, [this, N]() {
        const uint32_t K = kmeans_v2_->num_clusters();
        const int Dim    = prop_.dimension;

        // 1. Build inverted list: cluster_id → [node_ids]
        cluster_members_v2_.resize(K);
        for (size_t i = 0; i < N; ++i) {
            if (graph_->isTombstone(static_cast<uint32_t>(i))) continue;
            uint32_t cid = graph_->getRecordConstView(static_cast<uint32_t>(i)).centroid_id();
            if (cid < K)
                cluster_members_v2_[cid].push_back(static_cast<uint32_t>(i));
        }

        // 2. Precompute top-N_EXTRA_CLUSTERS nearest clusters for each cluster.
        // Cost: K² × D scalar ops (K=2000, D=256 → 1B ops, ~50ms) — one-time amortized.
        // Angular data: use 20 neighbor clusters for accurate multi-cluster seeding.
        // DABS graph traversal bridges inter-cluster gaps via ANNG edges.
        // Precompute enough cluster neighbors to support any n_probe value.
        // n_probe_override_ may be set before the first search triggers this call_once.
        const int default_nbrs = is_angular_ ? 20 : 4;
        const int CLUSTER_NBRS = std::max(default_nbrs, n_probe_override_);
        cluster_neighbors_v2_.resize(K);
        using CD = std::pair<float, uint32_t>;
        std::vector<CD> dists;
        dists.reserve(K);
        for (uint32_t c = 0; c < K; ++c) {
            const float* cc = kmeans_v2_->centroid(c);
            dists.clear();
            for (uint32_t c2 = 0; c2 < K; ++c2) {
                if (c2 == c) continue;
                const float* cc2 = kmeans_v2_->centroid(c2);
                float d2 = NGT::NGTAQ::KMeansCentering::l2sq(cc, cc2, Dim);
                dists.push_back({d2, c2});
            }
            int take = std::min((int)dists.size(), CLUSTER_NBRS);
            std::partial_sort(dists.begin(), dists.begin() + take, dists.end());
            cluster_neighbors_v2_[c].resize(static_cast<size_t>(take));
            for (int i = 0; i < take; ++i)
                cluster_neighbors_v2_[c][i] = dists[i].second;
        }

        // 3. Build the 2-level coarse-quantizer accelerator over the centroids here (under
        // call_once) so the lazy in-method build never races across concurrent query
        // threads. One sqrt(K)-way k-means on K centroids (~ms, one-time amortized).
        kmeans_v2_->buildCoarseQuantizer();
    });

    // 4. DABS search with asymmetric PQ ADC + lazy centroid rebuild
    using Entry = std::pair<float, uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> cand_q;
    std::priority_queue<float> dk_tracker;
    // Tech 2: pyglass LinearPool replaces cand_q/dk_tracker/ef_gate + gamma gates on the
    // BATCH path. thread_local so its backing array persists across queries (no malloc).
    static thread_local AQLinearPool lp;
    // Thread-local results buffer: capacity persists across queries (no malloc on steady-state).
    static thread_local std::vector<std::pair<float, uint32_t>> results_tl;
    results_tl.clear();
    if (results_tl.capacity() < static_cast<size_t>(k_beam * 30))
        results_tl.reserve(static_cast<size_t>(k_beam * 30));
    auto& results = results_tl;
    // Visited tracking. Two backends (Lever 1):
    //  - versioned (default): a uint16 version stamp per node + a per-thread cur_version.
    //    visited(u) == (vis[u]==cur), mark(u) == (vis[u]=cur). Per query just ++cur — NO
    //    125KB memset (only a rare full clear on the 65536-query wrap). hnswlib/pyglass
    //    VisitedListPool pattern. Costs 2MB (uint16) vs 125KB bitvector — measured net win
    //    since the memset dominated and the stamp array is touched only at visited nodes.
    //  - bitvector (default): the 125KB flat bitvector with a per-query memset.
    // MEASURED: versioned is a WASH-to-slight-loss on SIFT-1M — the 125KB bitvector stays
    // L2-resident and its memset is a cheap streaming write, while the 2MB stamp array's
    // random in-walk visited check misses cache more (profile: dabs 36->44us). So the
    // memset was NOT the setup bottleneck (that's the 2-level centroid scan). Default OFF;
    // AQ_VERSIONED_VIS=1 enables it (may help on larger N where the memset grows).
    static const bool use_versioned_vis = [] {
        const char* e = std::getenv("AQ_VERSIONED_VIS");
        return e && std::atoi(e) != 0;  // default false (bitvector)
    }();
    static thread_local std::vector<uint64_t> t_vis;        // legacy bitvector backend
    static thread_local std::vector<uint16_t> t_vis_ver;    // versioned backend (stamps)
    static thread_local uint16_t t_vis_cur = 0;
    if (use_versioned_vis) {
        if (t_vis_ver.size() < N) t_vis_ver.assign(N, 0);
        if (++t_vis_cur == 0) {                              // wrap: clear stamps, restart at 1
            std::fill(t_vis_ver.begin(), t_vis_ver.end(), uint16_t(0));
            t_vis_cur = 1;
        }
    } else {
        t_vis.assign((N + 63) / 64, 0ULL);
    }
    const uint16_t vis_cur = t_vis_cur;
    uint16_t* __restrict vis_ver = t_vis_ver.data();
    auto is_visited = [&](uint32_t id) -> bool {
        return use_versioned_vis ? (vis_ver[id] == vis_cur)
                                 : ((t_vis[id >> 6] >> (id & 63)) & 1ULL);
    };
    auto mark_visited = [&](uint32_t id) {
        if (use_versioned_vis) vis_ver[id] = vis_cur;
        else                   t_vis[id >> 6] |= 1ULL << (id & 63);
    };
    float d_k = std::numeric_limits<float>::infinity();

    // Visit-budget cap (HNSW ef-style): bound the number of DABS nodes we pop and
    // process. Profiling showed the beam loop over-visits nodes (78-94% of query
    // time), especially for angular data where the loose fast-path gate lets the
    // queue drain far past the useful frontier. The cap is the primary recall-QPS
    // knob: smaller → fewer visits → higher QPS at some recall cost.
    //
    // max_visits == 0 ⇒ unlimited (baseline behavior preserved exactly: angular
    // high-recall configs genuinely pop >>6k nodes, so any fixed "generous" default
    // like 200*k_beam silently clips recall). The knob is therefore strictly opt-in:
    // pass max_visits > 0 to trade recall for QPS along the curve.
    const size_t visit_budget = (max_visits > 0)
        ? static_cast<size_t>(max_visits)
        : std::numeric_limits<size_t>::max();
    size_t n_visits = 0;

    // Bounded-ef candidate frontier (coarse/batch path only). The unbounded min-heap
    // pushed ~10x more candidates than it ever popped within visit_budget (measured ~556
    // pushes per 30 visits): every wasted push pays an O(log n) heapify on a heap that
    // grows to ~500. We cap the LIVE frontier to `ef` entries by tracking the ef-th best
    // candidate distance in a small max-heap (ef_gate); once the frontier is full, a
    // candidate worse than the current ef-th best is dropped before it ever enters the
    // exploration heap. This shrinks the heap (cheaper push/pop) and skips the heapify of
    // hopeless candidates. ef defaults to a multiple of the visit budget (enough slack
    // that recall is preserved) and is overridable via AQ_EF for the recall-QPS sweep.
    static const int ef_env = [] {
        const char* e = std::getenv("AQ_EF");
        return e ? std::atoi(e) : 0;
    }();
    // ef ≈ the visit budget is the sweet spot: the frontier never usefully holds more
    // live candidates than nodes we will pop, so capping at ~visit_budget cuts the bulk of
    // the wasted heapify (~556 pushes -> ~mv) while preserving recall (verified iso-recall
    // across the curve). Floor at 2*k_beam for tiny budgets. AQ_EF overrides for sweeps.
    const size_t ef_cap =
        (ef_env > 0) ? static_cast<size_t>(ef_env)
        : (max_visits > 0 ? std::max<size_t>(static_cast<size_t>(k_beam) * 2,
                                             static_cast<size_t>(max_visits))
                          : std::numeric_limits<size_t>::max());
    std::priority_queue<float> ef_gate;  // max-heap of the ef best candidate distances
    // Tech 2: the LinearPool capacity == ef. ef collapses gamma_enq/gamma_term/max_visits
    // into one knob: explore until the frontier cursor reaches ef (cur_ >= ef). When the
    // caller passes no budget (max_visits==0, AQ_EF unset) fall back to a generous cap so
    // recall isn't silently clipped. Floored at k_out so we can always return k results.
    int lp_ef = (ef_env > 0) ? ef_env
              : (max_visits > 0 ? std::max(k_beam * 2, max_visits)
                                : 4096);
    if (lp_ef < k_out) lp_ef = k_out;
    if (use_batch_e) lp.reset(lp_ef);

    // ADC state cache: skip get_residual + q_norm_sq loop + build_tier1_query on
    // repeated visits to the same cluster.
    // Thread_local flat int8 buffer (ADC_SLOTS × D) + scalar metadata avoids
    // per-query heap allocation (~8KB for D=1024).
    constexpr int ADC_SLOTS = 8;
    struct ADCSlotMeta {
        uint32_t cid       = UINT32_MAX;
        float    q_norm_sq = 0.f;
        float    q_norm    = 0.f;
        int32_t  q_sum     = 0;
    };
    static thread_local std::vector<int8_t> adc_int8_tl;  // ADC_SLOTS * D int8 entries
    static thread_local std::array<ADCSlotMeta, ADC_SLOTS> adc_meta_tl;
    int adc_cache_hand = 1;
    // The 8-slot per-cluster ADC cache is only consulted by maybe_rebuild_adc/adc_dist,
    // both gated behind !use_coarse. Skip its per-query reset (resize + meta clear +
    // memcpy of the D-byte tier-1 vector) on the batch/global-PQ path.
    if (!use_coarse_e) {
        adc_int8_tl.resize(static_cast<size_t>(ADC_SLOTS) * static_cast<size_t>(D));
        for (auto& m : adc_meta_tl) m.cid = UINT32_MAX;
        // Slot 0 = initial cluster (ADC state already computed above)
        adc_meta_tl[0] = {active_cid, adc.q_norm_sq, adc.q_norm, adc.q_sum};
        std::memcpy(adc_int8_tl.data(), adc.q_int8.data(), static_cast<size_t>(D));
    }

    // Tier-1 routing: fast RaBitQ ADC for DABS beam search.
    // Tier-2 contributes via seeding (pre-sorts cluster candidates) AND d_k tracking:
    // using the fixed initial-cluster LUT for dk_tracker gives ~10% noise vs tier-1's
    // ~44%, allowing smaller gamma_term at same recall → higher QPS. No per-cluster
    // LUT rebuild needed — costs only ~5μs (1000 tier2_adc_pq calls × ~5ns) per query.
    // Note: full tier-2 heap routing was tested and regressed QPS 8x — accurate distances
    // delay DABS stopping criterion, causing many more node visits than tier-1 allows.
    auto adc_dist = [&](const NGT::NGTAQ::VectorRecordConstView& rec) -> float {
        float norm_x = NGT::NGTAQ::fp16_to_float(rec.norm_fp16());
        float t1 = NGT::NGTAQ::tier1_adc_fast_d(adc.q_int8.data(), rec.tier1(), adc.q_sum, D);
        float t1_ip = adc.q_norm * norm_x * NGT::NGTAQ::RABITQ_SCALE * t1 * inv_sqrt_D / 127.f;
        return adc.q_norm_sq + norm_x * norm_x - 2.0f * t1_ip;
    };

    // Rebuild tier-1 ADC tables on cluster boundary crossing, with cache lookup.
    // Cache hit (~10ns): restore from thread_local L1-resident flat buffer, skip recompute.
    // Cache miss: full rebuild (get_residual + q_norm_sq + sqrt + build_tier1_query ~150ns).
    auto maybe_rebuild_adc = [&](uint32_t cid) {
        if (cid == active_cid) return;
        for (int s = 0; s < ADC_SLOTS; ++s) {
            if (adc_meta_tl[s].cid == cid) {
                active_cid    = cid;
                q_norm_sq     = adc_meta_tl[s].q_norm_sq;
                adc.q_norm_sq = adc_meta_tl[s].q_norm_sq;
                adc.q_norm    = adc_meta_tl[s].q_norm;
                adc.q_sum     = adc_meta_tl[s].q_sum;
                std::memcpy(adc.q_int8.data(),
                            adc_int8_tl.data() + static_cast<size_t>(s) * static_cast<size_t>(D),
                            static_cast<size_t>(D));
                return;
            }
        }
        // Cache miss: full rebuild
        active_cid = cid;
        NGT::NGTAQ::compute_residual_and_tier1(
            q_rot_tl.data(), kmeans_v2_->centroid(active_cid), D,
            q_res_tl.data(), adc.q_norm_sq, adc.q_int8.data(), adc.q_sum);
        adc.q_norm = std::sqrt(adc.q_norm_sq);
        q_norm_sq = adc.q_norm_sq;
        // Store in cache (round-robin eviction)
        const int slot = adc_cache_hand;
        adc_cache_hand = (adc_cache_hand + 1) % ADC_SLOTS;
        adc_meta_tl[slot] = {cid, adc.q_norm_sq, adc.q_norm, adc.q_sum};
        std::memcpy(adc_int8_tl.data() + static_cast<size_t>(slot) * static_cast<size_t>(D),
                    adc.q_int8.data(), static_cast<size_t>(D));
    };

    // Tier-2 seed scoring: build LUT from initial cluster, pre-sort all cluster seeds
    // by tier-2 distance. Better seeds → routing converges faster → higher recall at
    // same gamma_term. Cost: 1 LUT build (~0.5μs AVX2) + N_seeds × 10ns ADC.
    // LUT built from q_res_initial (saved before routing modifies q_res).
    // Coarse routing (batch/global-PQ) never reads t2_lut_tl: seeds are scored via
    // coarse_dist, and the DABS termination gate uses the single coarse metric (break
    // on use_coarse). Skip the 256*M_PQ LUT build entirely on the batch path.
    static thread_local std::vector<float> t2_lut_tl;
    if (!use_coarse_e) {
        t2_lut_tl.resize(static_cast<size_t>(M_PQ) * 256);
        NGT::NGTAQ::build_tier2_lut_fast_m(q_res_init_tl.data(), M_PQ,
                                            tier2_codebook_T_.data(),
                                            t2_lut_tl.data());
    }

    // Stage C-lite: GLOBAL-PQ routing. Build ONE LUT per query (good for ANY node)
    // so the DABS loop never rebuilds a per-cluster ADC table — the per-neighbor
    // maybe_rebuild_adc that dominated the loop (78-94% of query time) is eliminated.
    // route_dist(id) returns an approx sq-L2 via the node's global PQ code; the d_k
    // tracker, enqueue gate (gamma_enq) and termination gate (gamma_term) all use this
    // single consistent metric. fp16 exact rerank below still recovers true k-NN.
    //
    // DEFAULT = OFF (legacy per-cluster routing). Empirically the global-PQ-routed DABS
    // is 1.4-1.9x SLOWER iso-recall than the legacy per-cluster path: the coarse global
    // codebook routes worse per visit, and the legacy 8-slot ADC cache + neighbor-cluster
    // bucketing already made maybe_rebuild_adc cheap. The global-PQ path is kept reachable
    // behind an opt-in flag (documented dead-end; may revisit for a batch ADC kernel).
    // AQ_USE_GLOBAL_ROUTING=1 opts INTO global-PQ routing on a Stage-A index; unset (or 0)
    // uses the LEGACY per-cluster ADC path — byte-identical in behavior to commit 99af263.
    static const bool use_global_env = [] {
        const char* e = std::getenv("AQ_USE_GLOBAL_ROUTING");
        return e && std::atoi(e) != 0;
    }();
    // Stage C-full: QG-style 16-wide vpshufb BATCH routing. AQ_BATCH_ROUTING=1 opts in
    // (requires a meta_version>=3 index with the K=16 GPQ4 store). When on, the popped
    // node's WHOLE neighbor block is scored in one _mm256_shuffle_epi8 pass over a single
    // shared per-query LUT — no per-neighbor maybe_rebuild_adc, no gather. The fp16 exact
    // rerank below recovers true k-NN. Implies global routing for seeds/popped/expansion
    // (single-node gpq4Dist with the same codebook), so d_k uses one consistent metric.
    static const bool use_batch_env = [] {
        const char* e = std::getenv("AQ_BATCH_ROUTING");
        return e ? (std::atoi(e) != 0) : true;  // default ON (see use_batch_env_e above)
    }();
    const bool use_batch = hasGPQ4() && use_batch_env;
    // Batch implies global-PQ semantics (no per-cluster ADC); legacy global PQ (K=256)
    // is the fallback opt-in. They are mutually exclusive; batch wins if both set.
    const bool use_global_pq = (has_global_pq_ && use_global_env) && !use_batch;

    static thread_local std::vector<float> global_lut_tl;
    float q_ns_global = 0.f;
    if (use_global_pq) {
        global_lut_tl.resize(static_cast<size_t>(M_PQ) * 256);
        q_ns_global = buildGlobalLUT(query, global_lut_tl.data());
    }
    // Distance-LUT gpq4 (matches QG's createFloatL2DistanceLookup): build a per-subspace
    // squared-DISTANCE LUT so the vpshufb kernel accumulates L2 directly — dropping the
    // per-neighbor fused-norm read and the ||q||^2+||x||^2-2*IP assembly from the hot loop.
    // Default ON for the batch path; AQ_DIST_LUT=0 reverts to the IP-LUT form for A/B.
    static const bool use_dist_lut_env = [] {
        const char* e = std::getenv("AQ_DIST_LUT");
        return !(e && std::atoi(e) == 0);  // default true
    }();
    // Batch (K=16) LUT + dequantized float table (the latter for single-node seeds).
    static thread_local NGT::NGTAQ::GlobalPQ4LUT batch_lut_tl;
    static thread_local std::vector<float> batch_ip_tl;
    float q_ns_batch = 0.f;
    const bool use_dist_lut = use_batch && use_dist_lut_env;
    if (use_batch) {
        // GPQ4 has its OWN (finer) subspace count — size by gpq4MPQ(), NOT the legacy
        // M_PQ, or buildGlobalLUT16 overflows this buffer when gpq4MPQ() > m_pq_.
        batch_ip_tl.resize((size_t)gpq4MPQ() * NGT::NGTAQ::GPQ4_K);
        AQ_MARK();  // [AQ_PROFILE] start dist-LUT build sub-timer (within setup)
        q_ns_batch = buildGlobalLUT16(query, batch_lut_tl, batch_ip_tl.data(), use_dist_lut);
        AQ_SUB(lut);  // [AQ_PROFILE] global dist-LUT build cost (incl. its internal SRHT)
    }
    // Tech 1: symmetric SQ8 in-loop routing distance. Encode the rotated query to int8 once
    // (per-vector symmetric max scale), then route every neighbor with a single signed-int8
    // dot (one vpdpbusd pass over D bytes) — no LUT, no gpq4_norm_sq_ gather. Opt-in
    // (AQ_SQ8=1) so we can A/B it against the gpq4 vpshufb routing.
    static const bool use_sq8_env = [] {
        const char* e = std::getenv("AQ_SQ8");
        return e && std::atoi(e) != 0;
    }();
    const bool use_sq8 = use_batch && use_sq8_env && hasSQ8();
    static thread_local std::vector<int8_t> q_sq8_tl;
    float q_max_sq8 = 0.f, q_nsq_sq8 = 0.f;
    if (use_sq8) {
        q_sq8_tl.assign((size_t)sq8_dim_align_, 0);
        q_max_sq8 = NGT::NGTAQ::sq8_encode_sym(q_rot_tl.data(), q_sq8_tl.data(), D);
        for (int d = 0; d < D; ++d) q_nsq_sq8 += q_rot_tl[d] * q_rot_tl[d];
    }
    auto sq8_dist1 = [&](uint32_t id) -> float {
        return sq8Dist(id, q_sq8_tl.data(), q_max_sq8, q_nsq_sq8);
    };
    auto route_dist = [&](uint32_t id) -> float {
        return globalPQDist(id, global_lut_tl.data(), q_ns_global);
    };
    // Single-node batch-PQ distance (seeds / popped-fallback / expansion).
    auto batch_dist1 = [&](uint32_t id) -> float {
        if (use_sq8)      return sq8_dist1(id);
        if (use_dist_lut) return gpq4DistL2(id, batch_ip_tl.data());  // distance-LUT: L2 directly
        return gpq4Dist(id, batch_ip_tl.data(), q_ns_batch);
    };
    // Unified single-node "coarse route" used wherever route_dist was used: batch wins.
    auto coarse_dist = [&](uint32_t id) -> float {
        return use_batch ? batch_dist1(id) : route_dist(id);
    };
    const bool use_coarse = use_batch || use_global_pq;  // any single-LUT coarse routing

    AQ_ADD(setup);  // [AQ_PROFILE] query-encode + ADC-init + initial tier-2 LUT (incl. one-time call_once)

    // Cluster-aware seeding: probe top-n_probe nearest clusters with PER-CLUSTER ADC.
    //
    // Root cause of angular data failure (NYTimes-256, GloVe-100):
    //   The original code scored ALL seeds with the initial cluster's LUT. Cross-cluster
    //   seeds are computed with the wrong residual basis → inaccurate ADC estimates →
    //   d_k initialized too small from primary-cluster-only hits → gamma gate prunes
    //   all cross-cluster graph edges → recall collapses to ~cluster_size/N * k.
    //
    // Fix: per-cluster probing. For each probed cluster c_i:
    //   1. rebuild ADC state to centroid(c_i) → correct q_res
    //   2. build tier-2 LUT from q_res(c_i)   → accurate per-cluster ADC
    //   3. score ALL cluster members with accurate LUT
    // This mirrors IVF nprobe: angular data requires more probing because unit vectors
    // spread true neighbors across many clusters (no magnitude diversity).
    //
    // After seeding, restore t2_lut_tl to initial cluster (used by DABS termination gate).
    // Coarse-path seeding: the batch graph walk now routes accurately (QG-class), so it
    // needs only a small warm-start. The legacy 32-seeds/cluster scan is pure overhead —
    // measured strictly worse than 4-8 seeds across the ENTIRE recall curve (r=0.50:
    // +15% QPS, r=0.80: +21%, r=0.92: +14%, recall within noise). Default the coarse
    // per-cluster seed count to 8 (n_probe=3 → ~24 total, vs ~96). AQ_SEEDS overrides
    // for sweeps. Legacy/global-PQ uses prop_.n_cluster_seeds unchanged.
    static const int coarse_seeds_env = [] {
        const char* e = std::getenv("AQ_SEEDS");
        return e ? std::atoi(e) : 0;
    }();
    const int N_CLUSTER_SEEDS =
        use_coarse ? (coarse_seeds_env > 0 ? coarse_seeds_env : 8)
                   : prop_.n_cluster_seeds;
    // n_probe: number of IVF clusters probed for seeding.
    // L2: 3 neighbor clusters (magnitude diversity keeps true NNs local).
    // Angular (BATCH path): MEASURED — the legacy n_probe=20 cost ~225us/query of seeding
    // (>50% of the GloVe query) for NO recall benefit over n_probe=1: a single probed
    // cluster's seeds warm-start the now-accurate gpq4 graph walk, which recovers the rest.
    // n_probe=1 is ~3x faster at iso-recall (GloVe r=0.66: 4737->14219 QPS) and flips angular
    // low-recall from a loss to a WIN vs fair QG-qsg2 (r<=0.70: 1.08-1.16x). So default the
    // BATCH angular path to n_probe=1. Legacy (non-batch) angular keeps 20 (its per-cluster
    // ADC seeding genuinely needs the cross-cluster coverage). AQ_NPROBE / setNProbe override.
    const bool batch_angular = is_angular_ && use_coarse;
    const int angular_default = batch_angular ? 1 : 20;
    const int n_probe = (n_probe_override_ > 0) ? n_probe_override_
                                                 : (is_angular_ ? angular_default : 3);
    struct SeedScore { float score; uint32_t id; };
    // Phase 3: graph-only entry (QG-style). When AQ_GRAPH_ENTRY=1 (coarse/batch path only),
    // skip the IVF cluster-probe seeding entirely — no 2-level centroid scan, no per-cluster
    // member scan — and seed the LinearPool from a small fixed set of entry points scored by
    // the coarse (gpq4 dist-LUT) metric. Saves the ~6us setup+seed at the cost of warm-start
    // quality (the now-accurate graph walk recovers it on L2; risky for angular / very low ef).
    static const bool graph_entry = [] {
        const char* e = std::getenv("AQ_GRAPH_ENTRY");
        return e && std::atoi(e) != 0;
    }();
    if (graph_entry && use_batch) {
        static thread_local std::vector<SeedScore> scored;
        scored.clear();
        for (uint32_t ep : entry_points_) {
            if (ep >= N || graph_->isTombstone(ep)) continue;
            scored.push_back({coarse_dist(ep), ep});
        }
        if (scored.empty() && N > 0) scored.push_back({coarse_dist(0u), 0u});
        std::sort(scored.begin(), scored.end(),
                  [](const SeedScore& a, const SeedScore& b){ return a.score < b.score; });
        for (const auto& s : scored) {
            if (is_visited(s.id)) continue;
            mark_visited(s.id);
            lp.insert(s.id, s.score);
        }
        goto seeds_done;  // skip the IVF seeding block below
    }
    {
        // Build list of clusters to probe: primary cluster + top-(n_probe-1) neighbors.
        // cluster_neighbors_v2_[active_cid] is sorted by cluster-centroid L2 distance.
        // thread_local buffers: the seed phase ran ~12-19us/query and these two vectors
        // were heap-allocated+freed EVERY query (probe_clusters ~n_probe, scored reserve
        // n_probe*200). Reuse persistent thread_local storage instead.
        static thread_local std::vector<uint32_t> probe_clusters;
        probe_clusters.clear();
        probe_clusters.push_back(initial_cid);  // primary cluster first
        if (initial_cid < cluster_neighbors_v2_.size()) {
            for (uint32_t c2 : cluster_neighbors_v2_[initial_cid]) {
                if (static_cast<int>(probe_clusters.size()) >= n_probe) break;
                probe_clusters.push_back(c2);
            }
        }

        static thread_local std::vector<SeedScore> scored;
        scored.clear();
        if (scored.capacity() < static_cast<size_t>(n_probe) * 200)
            scored.reserve(static_cast<size_t>(n_probe) * 200);

        // Score seeds from each probed cluster with the CORRECT centroid's LUT.
        // For each cluster c_i: rebuild ADC → build LUT(c_i) → score all members.
        // The tier-2 probe LUT is only used by the angular per-cluster path
        // (is_angular_ && !use_coarse); skip the M_PQ*256 alloc for the batch path.
        static thread_local std::vector<float> t2_lut_probe;
        if (is_angular_ && !use_coarse)
            t2_lut_probe.resize(static_cast<size_t>(M_PQ) * 256);
        for (uint32_t cid_p : probe_clusters) {
            if (cid_p >= cluster_members_v2_.size()) continue;
            const auto& members = cluster_members_v2_[cid_p];
            if (members.empty()) continue;

            // Angular: rebuild ADC + tier-2 LUT per cluster for accurate cross-cluster scoring.
            // L2: initial-cluster LUT is accurate (magnitude diversity makes cross-cluster
            //     residual error negligible for seeding); skip expensive per-cluster rebuild.
            // Global-PQ: skip the per-cluster rebuild entirely — score with the single
            //     global LUT (route_dist) so seeds are ranked by the SAME metric the DABS
            //     loop uses, giving a consistent d_k initialization.
            if (is_angular_ && !use_coarse) {
                maybe_rebuild_adc(cid_p);
                NGT::NGTAQ::build_tier2_lut_fast_m(q_res_tl.data(), M_PQ,
                                                    tier2_codebook_T_.data(),
                                                    t2_lut_probe.data());
            }
            const float q_ns  = is_angular_ ? adc.q_norm_sq : q_norm_sq_initial;
            const float* lut_p = is_angular_ ? t2_lut_probe.data() : t2_lut_tl.data();
            // Per-cluster seed cap.
            //   L2:      already bounded at N_CLUSTER_SEEDS (legacy behavior, unchanged).
            //   Angular: legacy scanned ALL members of every probed cluster (~20k seeds/query
            //            for GloVe). The full-cluster tier-2 SCAN — not just the resulting beam
            //            flood — is the QPS floor (A/B: at np=20 the scan pins QPS at ~600
            //            regardless of max_visits). We bound the seeds CONTRIBUTED per cluster
            //            to `seed_cap` while still probing all n_probe clusters (coverage
            //            preserved). Members are not pre-sorted, so two strategies exist:
            //              (default) scan only the FIRST `seed_cap` members — skips the
            //                  full-cluster scan entirely. ~5x QPS at near-iso-recall (GloVe
            //                  r≈0.77: 528→2238 QPS) since the scan no longer dominates.
            //              (AQ_SEED_CAP_TOPK=1) SCORE all members, keep the top-`seed_cap` by
            //                  tier-2 LUT estimate. Slightly higher recall per visit but pays
            //                  the full scan, so it stays at the ~600 QPS floor — opt-in only.
            //   seeds_per_cluster == 0 ⇒ unbounded (legacy full-cluster scan).
            static const bool seed_cap_topk = [] {
                const char* e = std::getenv("AQ_SEED_CAP_TOPK");
                return e && std::atoi(e) != 0;
            }();
            const int seed_cap = prop_.seeds_per_cluster;
            const bool angular_capped =
                is_angular_ && seed_cap > 0 &&
                members.size() > static_cast<size_t>(seed_cap);
            // For L2: scan ≤ N_CLUSTER_SEEDS. For angular: default scans only the first
            // `seed_cap` members; AQ_SEED_CAP_TOPK scans the full cluster (to keep top-k).
            const size_t take = is_angular_
                ? ((angular_capped && !seed_cap_topk)
                       ? static_cast<size_t>(seed_cap)
                       : members.size())
                : std::min(members.size(), static_cast<size_t>(N_CLUSTER_SEEDS));
            // Prefetch only what we'll score: avoids cache pollution for L2 (full-cluster
            // prefetch evicts DABS hot data from L1/L2 when only 32/N_CLUSTER_SEEDS are used).
            // Coarse (batch/global-PQ) seed scoring reads the node's per-node PQ code +
            // recon-norm (gpq4_codes_/gpq4_norm_sq_ or global codes), NOT its V2 record.
            // Prefetching the V2 record there is pure pollution — it evicts the DABS hot
            // data and brings in bytes the seed scan never touches. Skip it for coarse.
            if (!use_coarse) {
                for (size_t mi = 0; mi < take; ++mi)
                    if (members[mi] < N) graph_->prefetchRecord(members[mi]);
            } else if (use_batch) {
                // Batch seed scoring (gpq4Dist) gathers gpq4_norm_sq_[member] — a random
                // 4MB-array access that misses cache on every seed. Prefetch them up front
                // so the per-seed code-lookup loop overlaps the DRAM latency.
                for (size_t mi = 0; mi < take; ++mi)
                    if (members[mi] < N) __builtin_prefetch(&gpq4_norm_sq_[members[mi]], 0, 1);
            }
            // The score-all-keep-topK path needs a per-cluster staging buffer. The default
            // scan-first path (and L2) push directly into `scored` (no extra trim pass).
            const bool stage_topk = angular_capped && seed_cap_topk;
            static thread_local std::vector<SeedScore> clus_buf;
            if (stage_topk) { clus_buf.clear(); clus_buf.reserve(take); }
            for (size_t mi = 0; mi < take; ++mi) {
                uint32_t ep = members[mi];
                if (ep >= N || graph_->isTombstone(ep)) continue;
                float d_approx;
                if (use_coarse) {
                    d_approx = coarse_dist(ep);
                } else {
                    auto rec = graph_->getRecordConstView(ep);
                    float norm_x = NGT::NGTAQ::fp16_to_float(rec.norm_fp16());
                    float t2_ip  = NGT::NGTAQ::tier2_adc_pq_m(lut_p, rec.tier2(), M_PQ);
                    d_approx = q_ns + norm_x * norm_x - 2.0f * t2_ip;
                }
                (stage_topk ? clus_buf : scored).push_back({d_approx, ep});
            }
            // Strategy (a): keep this cluster's best `seed_cap` seeds (smallest d_approx).
            if (stage_topk) {
                if (clus_buf.size() > static_cast<size_t>(seed_cap)) {
                    std::nth_element(clus_buf.begin(),
                                     clus_buf.begin() + seed_cap,
                                     clus_buf.end(),
                                     [](const SeedScore& a, const SeedScore& b){
                                         return a.score < b.score; });
                    clus_buf.resize(static_cast<size_t>(seed_cap));
                }
                scored.insert(scored.end(), clus_buf.begin(), clus_buf.end());
            }
        }

        // Fall back to static entry points if cluster membership is empty
        if (scored.empty()) {
            for (uint32_t ep : entry_points_) {
                if (ep < N && !graph_->isTombstone(ep)) {
                    auto rec = graph_->getRecordConstView(ep);
                    scored.push_back({std::numeric_limits<float>::infinity(), ep});
                }
            }
        }

        // Sort seeds by accurate per-cluster ADC estimate; best-first ensures d_k
        // is initialized from the true near-neighbors rather than random seeds.
        std::sort(scored.begin(), scored.end(),
                  [](const SeedScore& a, const SeedScore& b){ return a.score < b.score; });

        // During rebuildGraphSelf on angular data, limit total seeds to avoid queue
        // flooding (n_probe=20 × avg_cluster_members can be 20k+ seeds per query).
        // rebuild_max_seeds_ = 0 (default) means no limit (normal search path).
        if (rebuild_max_seeds_ > 0 &&
            static_cast<int>(scored.size()) > rebuild_max_seeds_)
            scored.resize(static_cast<size_t>(rebuild_max_seeds_));

        {
            // ── DABS path for all metrics ─────────────────────────────────────────
            // Per-cluster tier-2 seeding (above) gives accurate cross-cluster d_k
            // initialization. DABS graph traversal then bridges inter-cluster gaps
            // via ANNG edges (built on true angular proximity for normalized vectors).
            // gamma_enq ≥ 0.50 (caller should use this for angular) ensures tier-1
            // noise (~44% max deflation) doesn't prune true near-neighbors.
            for (const auto& s : scored) {
                if (is_visited(s.id)) continue;
                mark_visited(s.id);
                float d;
                if (use_coarse) {
                    // Coarse routing scored every seed above (s.score == coarse_dist(s.id),
                    // a deterministic PQ-code lookup). Reuse it instead of recomputing —
                    // the seed scan paid for it once already.
                    d = s.score;
                } else {
                    auto rec = graph_->getRecordConstView(s.id);
                    maybe_rebuild_adc(rec.centroid_id());
                    d = adc_dist(rec);
                }
                if (use_batch) lp.insert(s.id, d);     // Tech 2: seed the LinearPool
                else           cand_q.push({d, s.id});
                graph_->prefetchOffset(s.id);
            }

            // Angular only: restore t2_lut_tl and ADC to initial cluster.
            // The DABS termination gate (two-gate tier-2 check) uses t2_lut_tl with the
            // initial-cluster residual. L2 never rebuilt LUT/ADC during seeding → no restore.
            // Coarse routing (global-PQ / batch) never touched per-cluster ADC/LUT → no restore.
            if (is_angular_ && !use_coarse) {
                NGT::NGTAQ::build_tier2_lut_fast_m(q_res_init_tl.data(), M_PQ,
                                                    tier2_codebook_T_.data(),
                                                    t2_lut_tl.data());
                maybe_rebuild_adc(initial_cid);
            }
        }
    }

seeds_done:  // Phase 3 graph-only entry jumps here, skipping the IVF seeding block

    AQ_ADD(seed);  // [AQ_PROFILE] region (a): cluster-probe seeding + seed enqueue

    // ── Tech 2: pyglass LinearPool beam loop (BATCH path only) ────────────────────────
    // Flat ef-bounded sorted frontier; termination is purely has_next() (cur_ < ef). No
    // gamma fast-path gate, no two-gate tier-2 recompute, no dk_tracker heap, no ef_gate.
    // Each popped node's neighbors are scored in one vpshufb pass; tombstoned/visited
    // neighbors are skipped at INSERT (no per-pop tombstone check). Every popped node is
    // appended to `results` for the existing exact-L2 rerank below.
    if (use_batch) {
        // Tech 4.1: prefer PackedV2Node neighbor IDs (record + IDs co-located in 5 cache
        // lines, one prefetch covers them) over the CSR offsets_->edge_ids_ indirection.
        // Only present when built at load via AQ_PACKED=1 (see loadV2).
        const bool use_packed = graph_->hasPackedV2();
        // Skip the per-neighbor isTombstone() random gather into state_ (1 MB) when the
        // index has no tombstones (the common case for a freshly built SIFT index).
        const bool has_tomb = graph_->hasTombstones();
        // Task 1 (prefetch auto-tune, pyglass Optimize() analogue): MEASURED a sweep of the
        // frontier-prefetch depth — depth=1 is optimal; depth>1 monotonically regresses
        // (the pool re-sorts on every insert, so nodes past cur_ are NOT the actual
        // next-popped ones → deeper prefetch pollutes cache). Unlike pyglass (which gathers
        // per-neighbor vector data and tunes po), our batch path reads the whole GPQ4 block
        // in one vpshufb shot — there is no per-neighbor data gather to tune. So we keep the
        // single next-frontier prefetch (already optimal); no auto-tune machinery needed.
        // Task 3: QG-style epsilon early-termination. NGTAQ's LinearPool otherwise pops the
        // WHOLE ef frontier (has_next == cur_<ef) with no distance stop, so it over-explores
        // far past where the top-k result set stabilizes (measured: ~56% of hops wasted at
        // r=0.96, ~75% at r=0.99). Mirror QG (QuantizedGraph.h:331): stop when the next
        // frontier candidate's quantized distance exceeds (1+eps) * the k-th best result
        // distance (the pool is sorted ascending, so data_[k-1].dist IS the k-th best, and
        // data_[cur_].dist IS the next candidate). eps=0 disables (default), preserving the
        // exact prior behavior. The exact fp16 rerank below still recovers true k-NN.
        static const float term_eps = [] {
            const char* e = std::getenv("AQ_TERM_EPS");
            return e ? (float)std::atof(e) : -1.0f;  // <0 => disabled (default)
        }();
        const float term_coeff = 1.0f + term_eps;
        // fp16-sharpened early-termination (AQ_TERM_EPS_FP16). The 4-bit gate above failed at
        // iso-recall because it compared a NOISY 4-bit frontier distance against a sharp fp16
        // radius (mismatch: 4-bit underestimates → non-winners look promising → can't stop).
        // Fix: gate the frontier node's EXACT fp16 distance against the EXACT fp16 k-th-best
        // radius — both sharp. The frontier node is reranked anyway when popped, so its fp16
        // distance is computed ~1 hop early and REUSED (stored as the popped node's exact
        // dist). dk_fp16 (max-heap of the k smallest popped exact fp16 dists) is the radius.
        static const float term_eps_fp16 = [] {
            const char* e = std::getenv("AQ_TERM_EPS_FP16");
            return e ? (float)std::atof(e) : -1.0f;  // <0 => disabled (default)
        }();
        const bool use_fp16_gate = (term_eps_fp16 >= 0.0f) && use_batch && !raw_flat_.empty();
        const float term_coeff_fp16 = 1.0f + term_eps_fp16;
        std::priority_queue<float> dk_fp16;  // max-heap of the k smallest popped exact fp16 dists
        auto exact_fp16 = [&](uint32_t id) -> float {
            if ((size_t)id * D + D > raw_flat_.size()) return std::numeric_limits<float>::max();
            return NGT::NGTAQ::l2_sq_f32_fp16(q_ptr, raw_flat_.data() + (size_t)id * D, D);
        };
        while (lp.has_next()) {
            // Capture the frontier node's pool distance BEFORE pop advances the cursor —
            // it IS x's coarse (gpq4) score, so no recompute is needed for `results`.
            float dx = lp.data_[lp.cur_].dist;
            // 4-bit epsilon early-stop (legacy AQ_TERM_EPS): once >= k results, stop if the
            // next candidate is farther than (1+eps)*k-th-best (all in noisy 4-bit space).
            if (term_eps >= 0.0f && lp.size_ >= k_out &&
                dx > term_coeff * lp.data_[k_out - 1].dist)
                break;
            // fp16-sharpened gate: compute the frontier candidate's EXACT fp16 distance; if it
            // exceeds (1+eps)*the exact fp16 k-th-best radius, no later candidate can win →
            // terminate. dxf is reused as the popped node's exact dist (no double-compute).
            uint32_t frontier_id = AQLinearPool::rawid(lp.data_[lp.cur_].id);
            float dxf = 0.f;
            if (use_fp16_gate) {
                dxf = exact_fp16(frontier_id);
                if ((int)dk_fp16.size() >= k_out && dxf > term_coeff_fp16 * dk_fp16.top())
                    break;
            }
            uint32_t x = lp.pop();
            // Prefetch the next frontier node's GPQ4 block + neighbor list while we work.
            if (lp.cur_ < lp.size_) {
                uint32_t nxt = AQLinearPool::rawid(lp.data_[lp.cur_].id);
                graph_->prefetchGPQ4(nxt);
                if (use_packed) graph_->prefetchPackedNode(nxt);
                else            graph_->prefetchNeighbors(nxt);
            }
            results.push_back({dx, x});  // approximate score for the exact-L2 rerank below
            ++n_visits;
            // Maintain the exact fp16 k-th-best radius from popped nodes (gate uses it). dxf
            // was already computed for the frontier == popped node, so no extra work here.
            if (use_fp16_gate) {
                if ((int)dk_fp16.size() < k_out) dk_fp16.push(dxf);
                else if (dxf < dk_fp16.top()) { dk_fp16.pop(); dk_fp16.push(dxf); }
            }

            const uint8_t* blocks = graph_->gpq4Blocks(x);
            const uint32_t nblk   = graph_->gpq4NumBlocks(x);
            if (!blocks || nblk == 0) continue;
            graph_->prefetchGPQ4(x);
            const size_t blk_bytes = graph_->gpq4BlockBytes();
            // Neighbor IDs: from the packed node (co-located) when its degree fits the
            // 64-slot store AND matches the CSR degree the GPQ4 blocks were built from;
            // otherwise fall back to the CSR view so block-position indexing stays aligned.
            const uint32_t* nbr_ids;
            size_t n_nbrs;
            SoAGraph::NeighborView csr_nbrs{nullptr, 0};
            const SoAGraph::PackedV2Node* pn = use_packed ? graph_->getPackedNode(x) : nullptr;
            const uint32_t csr_deg = graph_->neighborCount(x);
            if (pn && csr_deg <= SoAGraph::PACKED_V2_MAX_NBRS && pn->n_nbrs == csr_deg) {
                nbr_ids = pn->nbrs;
                n_nbrs  = pn->n_nbrs;
            } else {
                csr_nbrs = graph_->getNeighbors(x);
                nbr_ids  = csr_nbrs.data;
                n_nbrs   = csr_nbrs.size();
            }
            if (use_sq8) {
                // Tech 1: symmetric SQ8 routing. One signed-int8 dot per neighbor over the
                // co-located int8 code — no vpshufb LUT pass, no recon-norm gather. Prefetch
                // each neighbor's int8 code (sq8_dim_align_ bytes) ahead of the dot.
                for (size_t ni = 0; ni < n_nbrs; ++ni) {
                    uint32_t u = nbr_ids[ni];
                    if (u < N) __builtin_prefetch(
                        sq8_codes_.data() + (size_t)u * sq8_dim_align_, 0, 1);
                }
                for (size_t ni = 0; ni < n_nbrs; ++ni) {
                    uint32_t u = nbr_ids[ni];
                    if (u >= N || graph_->isTombstone(u)) continue;
                    if (is_visited(u)) continue;
                    mark_visited(u);
                    lp.insert(u, sq8_dist1(u));
                }
                continue;
            }
            static thread_local std::vector<float> block_ip_lp;
            block_ip_lp.resize((size_t)nblk * 16);
            for (uint32_t b = 0; b < nblk; ++b)
                NGT::NGTAQ::gpq4_batch_ip(blocks + (size_t)b * blk_bytes,
                                          batch_lut_tl, block_ip_lp.data() + (size_t)b * 16);
            if (use_dist_lut) {
                // Distance-LUT (QG form): the kernel output IS the squared-L2 distance.
                // No per-neighbor norm read, no IP->L2 assembly — the leanest per-hop path.
                for (size_t ni = 0; ni < n_nbrs; ++ni) {
                    uint32_t u = nbr_ids[ni];
                    if (u >= N) continue;
                    if (has_tomb && graph_->isTombstone(u)) continue;
                    if (is_visited(u)) continue;
                    mark_visited(u);
                    const uint32_t b = (uint32_t)(ni / 16), pos = (uint32_t)(ni % 16);
                    lp.insert(u, block_ip_lp[(size_t)b * 16 + pos]);
                }
            } else {
                // IP-LUT (legacy): assemble L2 = ||q||^2 + ||x||^2 - 2*IP per neighbor. The
                // fused bf16 block norm avoids the 4MB gpq4_norm_sq_ gather; legacy fp16-norm
                // indices (gpq4FusedNorm()==false) fall back to the gather.
                const bool fused = graph_->gpq4FusedNorm();
                if (!fused) {
                    for (size_t ni = 0; ni < n_nbrs; ++ni) {
                        uint32_t u = nbr_ids[ni];
                        if (u < N) __builtin_prefetch(&gpq4_norm_sq_[u], 0, 1);
                    }
                }
                for (size_t ni = 0; ni < n_nbrs; ++ni) {
                    uint32_t u = nbr_ids[ni];
                    if (u >= N || graph_->isTombstone(u)) continue;  // skip tombstone at INSERT
                    if (is_visited(u)) continue;
                    mark_visited(u);
                    const uint32_t b = (uint32_t)(ni / 16), pos = (uint32_t)(ni % 16);
                    float ip   = block_ip_lp[(size_t)b * 16 + pos];
                    float nsq  = fused ? graph_->gpq4BlockNorm(blocks + (size_t)b * blk_bytes, pos)
                                       : gpq4_norm_sq_[u];
                    float d_u  = q_ns_batch + nsq - 2.0f * ip;
                    lp.insert(u, d_u);   // pool drops it if worse than the ef-th best
                }
            }
        }
        AQ_ADD(dabs);
        AQ_CNT(hops, n_visits);            // [AQ_PROFILE] hops (popped nodes) per query
        AQ_CNT(npops, results.size());     // [AQ_PROFILE] candidates handed to rerank
        goto post_dabs;  // skip the legacy cand_q loop entirely
    }

    while (!cand_q.empty()) {
        // Visit-budget cap: bound total processed nodes (primary recall-QPS knob).
        // Checked before popping so the budget caps committed work regardless of
        // how many far/tombstone nodes are skipped below.
        if (n_visits >= visit_budget) break;

        auto [dist_x, x] = cand_q.top(); cand_q.pop();

        // Skip tombstoned (hole) nodes: zero-norm train vectors excluded during
        // normalization. Their raw_flat_ entry is all-zeros, so exact L2 reranking
        // gives dist = ||q_norm|| = 1.0, which beats true NNs with dist > 1.0
        // (cos_sim < 0.5) and causes recall collapse on angular datasets (NYTimes).
        if (graph_->isTombstone(x)) continue;

        // Fast-path termination: node is so far that even maximum tier-1 deflation
        // (3σ ≈ 2.32×) of a true near-neighbor can't explain this distance.
        // Avoids rec_x cache miss for clearly-far nodes.
        if (dk_tracker.size() >= static_cast<size_t>(k_beam) &&
            dist_x > (1.f + gamma_term) * d_k * 2.0f) break;

        // Prefetch next-popped node's data while we process current node
        if (!cand_q.empty()) {
            uint32_t nxt = cand_q.top().second;
            // Batch mode reads the GPQ4 block (codes+norms) of the next node, not its
            // V2 record; prefetch that instead to hide its DRAM latency.
            if (use_batch) { graph_->prefetchGPQ4(nxt); graph_->prefetchNeighbors(nxt); }
            else           { graph_->prefetchRecord(nxt); graph_->prefetchNeighbors(nxt); }
        }
        // Prefetch current node's neighbor list (hides CSR access latency)
        graph_->prefetchNeighbors(x);

        auto rec_x = graph_->getRecordConstView(x);

        // Two-gate gamma_term: tier-1 noise (~44% std) can inflate a true near-neighbor's
        // heap estimate above threshold. Verify with tier-2 PQ (lower noise, ~10%) before
        // terminating. Only fires in the "uncertain zone" (1× to 2× threshold): nodes that
        // tier-1 calls too far but tier-2 may recognize as close. Cost: 1 fp16_to_float +
        // 1 tier2_adc_pq (~5ns) per uncertain termination candidate.
        //
        // Angular EXCEPTION: t2_lut_tl uses the initial cluster's residual, but DABS pops
        // nodes from any cluster (after maybe_rebuild_adc). For cross-cluster nodes, this
        // gives a grossly wrong distance (wrong residual basis) → overestimates → false
        // termination → recall collapse for angular. Skip tier-2 for angular; rely solely
        // on tier-1 with a suitably large gamma_term (≥0.50 recommended for angular).
        if (dk_tracker.size() >= static_cast<size_t>(k_beam) &&
            dist_x > (1.f + gamma_term) * d_k) {
            if (use_coarse) {
                break;  // coarse routing: dist_x and d_k share one metric → single gate
            } else if (!is_angular_) {
                float nx2  = NGT::NGTAQ::fp16_to_float(rec_x.norm_fp16());
                float t2ip = NGT::NGTAQ::tier2_adc_pq_m(t2_lut_tl.data(), rec_x.tier2(), M_PQ);
                float d_t2 = q_norm_sq_initial + nx2 * nx2 - 2.0f * t2ip;
                if (d_t2 > (1.f + gamma_term) * d_k) break;
                // tier-2 override: tier-1 overestimated this node — continue processing
            } else {
                break;  // angular: trust tier-1 termination (no 2-gate with wrong LUT)
            }
        }

        float d_approx;
        if (use_coarse) {
            // x's queue priority dist_x IS its coarse distance (set when enqueued by the
            // batch kernel / coarse_dist), so reuse it — no recompute, no gather.
            d_approx = dist_x;
        } else {
            maybe_rebuild_adc(rec_x.centroid_id());
            d_approx = adc_dist(rec_x);
        }

        // Count this as a processed visit (it passed all termination gates and we
        // are about to do its neighbor sweep — the expensive part the cap bounds).
        ++n_visits;

        // Add ALL popped candidates to results for exact reranking.
        // We only use d_k for ROUTING termination, not for result filtering.
        results.push_back({d_approx, x});
        dk_tracker.push(d_approx);
        if (static_cast<int>(dk_tracker.size()) > k_beam) {
            dk_tracker.pop();
            d_k = dk_tracker.top();
        } else if (static_cast<int>(dk_tracker.size()) == k_beam) {
            d_k = dk_tracker.top();
        }

        auto neighbors = graph_->getNeighbors(x);
        const size_t n_nbrs = neighbors.size();

        // ── Stage C-full: QG-style vpshufb BATCH routing ─────────────────────────
        // Score ALL of x's neighbors in one shuffle pass over its CONTIGUOUS block
        // store (codes + recon-norms co-located), then enqueue. No per-neighbor
        // maybe_rebuild_adc, no record-view touch, no gather — the whole point.
        if (use_batch) {
            const uint8_t* blocks = graph_->gpq4Blocks(x);
            const uint32_t nblk   = graph_->gpq4NumBlocks(x);
            if (blocks && nblk > 0) {
                graph_->prefetchGPQ4(x);
                const size_t blk_bytes = graph_->gpq4BlockBytes();
                static thread_local std::vector<float> block_ip_tl;
                // Prefetch the per-neighbor reconstructed-norm gather targets BEFORE the
                // vpshufb pass. nsq lives in gpq4_norm_sq_ (N floats = 4MB), accessed
                // randomly by neighbor id → a near-guaranteed cache miss on the critical
                // path of every scored neighbor. Issuing the loads up front lets the
                // block-IP shuffle below hide their DRAM latency.
                for (size_t ni = 0; ni < n_nbrs; ++ni) {
                    uint32_t u = neighbors[ni];
                    if (u < N) __builtin_prefetch(&gpq4_norm_sq_[u], 0, 1);
                }
                block_ip_tl.resize((size_t)nblk * 16);
                for (uint32_t b = 0; b < nblk; ++b)
                    NGT::NGTAQ::gpq4_batch_ip(blocks + (size_t)b * blk_bytes,
                                              batch_lut_tl, block_ip_tl.data() + (size_t)b * 16);
                for (size_t ni = 0; ni < n_nbrs; ++ni) {
                    uint32_t u = neighbors[ni];
                    if (u >= N || graph_->isTombstone(u)) continue;
                    if (is_visited(u)) continue;
                    mark_visited(u);
                    const uint32_t b = (uint32_t)(ni / 16), pos = (uint32_t)(ni % 16);
                    float ip   = block_ip_tl[(size_t)b * 16 + pos];
                    // Per-neighbor reconstructed-norm^2: the block's fp16 norm OVERFLOWS
                    // (SIFT recon norms ~2e5 >> fp16 max 65504 → +inf → every neighbor
                    // pruned → beam never expands → recall collapse). Read the exact
                    // float norm from gpq4_norm_sq_ instead (single cheap gather).
                    float nsq  = gpq4_norm_sq_[u];
                    float d_u  = q_ns_batch + nsq - 2.0f * ip;
                    if (static_cast<int>(dk_tracker.size()) >= k_beam &&
                        d_u > (1.f + gamma_enq) * d_k)
                        continue;
                    // Bounded-ef frontier: drop candidates that can't enter the ef-best
                    // live set before the visit budget is exhausted, avoiding their heapify.
                    if (ef_gate.size() >= ef_cap) {
                        if (d_u >= ef_gate.top()) continue;  // worse than ef-th best → drop
                        ef_gate.pop();                       // evict current worst
                    }
                    ef_gate.push(d_u);
                    // No prefetchOffset(u) here: the next popped node is already prefetched
                    // at the loop top (prefetchGPQ4(nxt) + prefetchNeighbors(nxt)).
                    cand_q.push({d_u, u});
                }
                continue;  // neighbor sweep done for x
            }
            // No block store for x (shouldn't happen for an active node) → fall through
            // to the generic coarse path below.
        }

        // Sliding-window prefetch (PFDIST=8): issue record prefetch 8 iterations ahead.
        // With 8-slot ADC cache, hit-path cost ~10ns → 8×10=80ns look-ahead covers DRAM
        // (~100ns). Bulk (n_nbrs=20-40 simultaneous) overloads the CPU LSQ/MSHR, competing
        // with centroid and offset prefetches already in-flight, causing P99 spikes.
        constexpr int PFDIST = 8;
        for (size_t pf = 0; pf < std::min((size_t)PFDIST, n_nbrs); ++pf)
            graph_->prefetchRecord(neighbors[pf]);

        // Pass 1: filter (tombstone/visited/OOB) + mark visited + read centroid_id.
        // We bucket the surviving neighbors by centroid_id so maybe_rebuild_adc fires
        // once per DISTINCT cluster instead of once per neighbor (QBG memoizes one LUT
        // per subspace; QuantizedBlobGraph.h:1292-1298). The 8-slot ADC LRU thrashes
        // when neighbors interleave clusters; sorting makes each cluster's run
        // contiguous, so a node touching c clusters pays c rebuilds, not n_nbrs.
        // Correctness: each neighbor is still scored under ITS OWN cluster's ADC state.
        struct NbrCand { uint32_t cid; uint32_t id; };
        static thread_local std::vector<NbrCand> nbr_buf_tl;
        nbr_buf_tl.clear();
        if (nbr_buf_tl.capacity() < n_nbrs) nbr_buf_tl.reserve(n_nbrs);
        for (size_t ni = 0; ni < n_nbrs; ++ni) {
            if (ni + PFDIST < n_nbrs)
                graph_->prefetchRecord(neighbors[ni + PFDIST]);
            uint32_t u = neighbors[ni];
            if (u >= N || graph_->isTombstone(u)) continue;
            if (is_visited(u)) continue;
            mark_visited(u);
            // Coarse routing scores any node with one LUT (no per-cluster ADC), so the
            // centroid_id read (a record-view touch) and the cluster sort below are
            // both unnecessary — store a dummy cid and skip the sort.
            uint32_t cid_u = use_coarse ? 0u
                                         : graph_->getRecordConstView(u).centroid_id();
            nbr_buf_tl.push_back({cid_u, u});
        }
        // Sort by cluster id → contiguous per-cluster runs (small n_nbrs, ~20-64).
        // Coarse routing needs no per-cluster batching, so the sort is pure overhead → skip.
        if (!use_coarse)
            std::sort(nbr_buf_tl.begin(), nbr_buf_tl.end(),
                      [](const NbrCand& a, const NbrCand& b){ return a.cid < b.cid; });

        // Pass 2: score each neighbor under its own cluster's ADC. maybe_rebuild_adc
        // self-skips when cid == active_cid, so contiguous runs cost one rebuild each.
        for (const auto& nc : nbr_buf_tl) {
            uint32_t u = nc.id;
            float d_u;
            if (use_coarse) {
                d_u = coarse_dist(u);
            } else {
                auto rec_u = graph_->getRecordConstView(u);
                maybe_rebuild_adc(nc.cid);
                d_u = adc_dist(rec_u);
            }
            // Skip hopeless candidates: when d_k is initialized, a node with
            // d_u > (1+gamma)*d_k would trigger the outer-loop termination as
            // soon as it's popped — it can never contribute to the top-k result.
            // Skipping the push avoids a wasted heap insertion+extraction.
            if (static_cast<int>(dk_tracker.size()) >= k_beam &&
                d_u > (1.f + gamma_enq) * d_k)
                continue;
            cand_q.push({d_u, u});
            // Prefetch offset for u: when u is eventually popped and
            // prefetchNeighbors(u) is called, offsets_[u] will already be
            // in L1/L2 cache, eliminating the blocking DRAM read that
            // gates the neighbor-list prefetch.
            graph_->prefetchOffset(u);
        }
    }

    AQ_ADD(dabs);  // [AQ_PROFILE] region (b): DABS beam-search loop

post_dabs:  // Tech 2 batch LinearPool loop jumps here (skips the legacy cand_q loop)

    // Skip-rerank fast path (rerank_factor < 0): return top-k by approximate ADC
    // distance, no exact L2 refinement. Cheap low-recall path (QBG/QG essence:
    // trust the quantized distance when recall budget is tight).
    if (rerank_factor < 0) {
        const size_t out_n = std::min(static_cast<size_t>(k_out), results.size());
        std::partial_sort(results.begin(), results.begin() + out_n, results.end());
        std::vector<SearchResult> approx_results;
        approx_results.reserve(out_n);
        for (size_t i = 0; i < out_n; ++i) {
            uint32_t id = results[i].second;
            if (graph_->isTombstone(id)) continue;
            approx_results.push_back({id, results[i].first, results[i].first});
        }
        if (!id_to_external_.empty())
            for (auto& r : approx_results)
                if (r.id < id_to_external_.size()) r.id = id_to_external_[r.id];
        AQ_ADD(refine);
        AQ_ADD(expand);
        AQ_ADD(rerank);
#ifdef AQ_PROFILE
        ++g_aqprof.n;
#endif
        return approx_results;
    }

    // Select top candidates by approximate score for exact L2 reranking.
    // Bounded rerank window (QBG/QG essence): a multiple of the beam width is enough
    // — the old k_beam*100 reranked far more than needed and rerank is <7% of query
    // time, so shrinking it is ~free on QPS.
    //   L2:      k_beam*15 sits at the knee — iso-recall while well-bounded.
    //   Angular: true NNs are scattered much wider by BQ ADC noise (unit vectors have
    //            no magnitude separation), so a tight window drops genuine neighbors
    //            before exact rerank (k_beam*15 cost GloVe gt=0.90 ~4.5 recall pts).
    //            Keep the original k_beam*100; rerank is only ~3% of angular query
    //            time so this is still effectively free, and recall == baseline.
    // Env override AQ_REFINE_MULT (read once) for sweeping the rerank window.
    static const int refine_mult_env = [] {
        const char* e = std::getenv("AQ_REFINE_MULT");
        return e ? std::atoi(e) : 0;
    }();
    const size_t refine_mult = (refine_mult_env > 0)
        ? static_cast<size_t>(refine_mult_env)
        : (is_angular_ ? (size_t)100 : (size_t)15);
    // Global-PQ routing is slightly looser than per-cluster tier-2 (Stage A: exact
    // top-10 sit within global-PQ top-200=92.8%, top-500=98.2%, top-1000=99.4%), so the
    // exact-rerank pool needs a wide floor (~1000) to recover true k-NN. AQ_REFINE_MULT
    // still drives it higher when set (k_beam*mult overrides the floor when larger).
    // Batch (K=16, M=128, GORDER) cascade rerank (Lever 4): the gpq4 pre-ranking is fine
    // enough that the true top-k sit within the top ~64 candidates BY GPQ4 DISTANCE across
    // the whole recall curve (measured: N=64 recall-neutral vs uncapped at r=0.93..0.997,
    // |delta|<=2e-4). So cap the exact-fp16 rerank set (reads the cold 256B/cand raw_flat_)
    // to a tight floor instead of 2000 — ~47% less rerank time, recall-neutral. AQ_REFINE_MULT
    // raises it (k_beam*mult) and AQ_RERANK_N overrides outright for sweeps / safety.
    const size_t refine_floor = use_batch ? (size_t)64
                                          : (use_global_pq ? (size_t)1000 : (size_t)0);
    size_t refine_n;
    if (use_batch && refine_mult_env == 0) {
        // Batch cascade: tight recall-safe cap (top-64 by gpq4 distance holds the true top-k).
        // Stays >= k_out*rerank_factor so we never under-fill the result set.
        refine_n = std::max<size_t>(refine_floor,
            static_cast<size_t>(k_out) * static_cast<size_t>(rerank_factor > 0 ? rerank_factor : 1));
    } else {
        // Legacy / explicitly-tuned path: k_beam*mult with the (wide) floor.
        refine_n = std::max<size_t>(
            std::max<size_t>(static_cast<size_t>(k_beam) * refine_mult, refine_floor),
            static_cast<size_t>(k_out) * static_cast<size_t>(rerank_factor > 0 ? rerank_factor : 1));
    }
    // Phase 4: recall-adaptive rerank window (batch path). AQ_RERANK_N>0 caps the exact-L2
    // rerank to the top-N popped candidates BY QUANTIZED DISTANCE (results is keyed on the
    // gpq4 pool distance), QG-style — rerank only the best N of the explored set instead of
    // ALL popped. Decouples rerank cost from hop count. Finer quant (M=128) orders well
    // enough that a small N may hold recall; sweep to find the knee.
    static const int rerank_n_env = [] {
        const char* e = std::getenv("AQ_RERANK_N");
        return e ? std::atoi(e) : 0;
    }();
    if (use_batch && rerank_n_env > 0) refine_n = static_cast<size_t>(rerank_n_env);
    if (results.size() > refine_n) {
        std::nth_element(results.begin(), results.begin() + refine_n, results.end());
        results.resize(refine_n);
    }

    AQ_ADD(refine);  // [AQ_PROFILE] region (c): nth_element (refine_n = k_beam * {15 L2, 100 angular})

    // 1-hop expansion from top-EXPAND_N candidates.
    // Angular DABS only: ANNG edges bridge inter-cluster gaps; expanding top seeds
    // finds true NNs in neighboring clusters not reached by beam search. For L2 the
    // beam already covers the local frontier (magnitude diversity keeps true NNs in
    // the primary cluster), so expansion is wasted work → disabled (EXPAND_N=0).
    {
        const size_t EXPAND_N = is_angular_ ? 200 : 0;
        const size_t expand_from = std::min(EXPAND_N, results.size());
        for (size_t ei = 0; ei < expand_from; ++ei) {
            uint32_t node = results[ei].second;
            for (uint32_t u : graph_->getNeighbors(node)) {
                if (u >= static_cast<uint32_t>(N) || graph_->isTombstone(u)) continue;
                if (is_visited(u)) continue;
                mark_visited(u);
                float d_u;
                if (use_coarse) {
                    d_u = coarse_dist(u);
                } else {
                    auto rec_u = graph_->getRecordConstView(u);
                    maybe_rebuild_adc(rec_u.centroid_id());
                    d_u = adc_dist(rec_u);
                }
                results.push_back({d_u, u});
            }
        }
    }

    AQ_ADD(expand);  // [AQ_PROFILE] region (d): EXPAND_N 1-hop expansion

    // Exact L2 refinement: l2_sq_avx2 for D=128 ≈ 5ns/vector × 150 = 0.75μs.
    // Store squared distances to avoid redundant sqrts during sort; take sqrt only
    // for the final top-k (10 sqrts instead of 150).
#ifdef AQ_PROFILE
    // Task 1: stabilization-hop. `results` is in pop order (one entry per popped hop), so
    // index i == the hop at which node results[i] was discovered. Rerank ALL by exact L2,
    // take the true top-k, and record the MAX pop-index among those k winners = the last
    // hop that contributed a surviving result. If << total hops, the tail is wasted.
    if (use_batch && !results.empty()) {
        std::vector<std::pair<float,int>> ex; ex.reserve(results.size());  // (exact_sq, hop_idx)
        for (size_t i = 0; i < results.size(); ++i) {
            uint32_t id = results[i].second;
            if ((size_t)id * D + D > raw_flat_.size() || graph_->isTombstone(id)) continue;
            const uint16_t* v = raw_flat_.data() + (size_t)id * D;
            ex.push_back({ NGT::NGTAQ::l2_sq_f32_fp16(q_ptr, v, D), (int)i });
        }
        size_t kk = std::min((size_t)k_out, ex.size());
        std::partial_sort(ex.begin(), ex.begin() + kk, ex.end(),
            [](const auto&a, const auto&b){ return a.first < b.first; });
        int stab = 0; for (size_t i = 0; i < kk; ++i) stab = std::max(stab, ex[i].second);
        g_aqprof.stab_sum += stab; g_aqprof.stab_n += 1; g_aqprof.stab_samples.push_back(stab);
    }
#endif
    std::vector<SearchResult> final_results;
    final_results.reserve(results.size());
    for (auto& [approx_d, id] : results) {
        if (static_cast<size_t>(id) * D + D > raw_flat_.size()) continue;
        // Safety net: tombstoned nodes have raw_flat_=zeros → exact dist=1.0
        // which beats angular NNs with dist>1.0. Filter here as belt-and-suspenders.
        if (graph_->isTombstone(static_cast<uint32_t>(id))) continue;
        // raw_flat_ is fp16-packed; F16C rerank reads half the bytes vs fp32.
        const uint16_t* vec = raw_flat_.data() + static_cast<size_t>(id) * D;
        float exact_sq = NGT::NGTAQ::l2_sq_f32_fp16(q_ptr, vec, D);
        // Store exact_sq in .distance temporarily (sqrt deferred until after sort).
        final_results.push_back({id, exact_sq, approx_d});
    }
    // partial_sort: O(n log k) ≈ 500 comparisons vs std::sort O(n log n) ≈ 1080.
    // We only need the top-k; the rest are discarded.
    const size_t out_n = std::min(static_cast<size_t>(k_out), final_results.size());
    std::partial_sort(final_results.begin(), final_results.begin() + out_n,
        final_results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        });
    final_results.resize(out_n);
    // Now apply sqrt to the top-k distances (deferred from above).
    for (auto& r : final_results) r.distance = std::sqrt(r.distance);
    // Task 2: translate internal ids back to original (insertion-order) ids if reordered.
    if (!id_to_external_.empty())
        for (auto& r : final_results)
            if (r.id < id_to_external_.size()) r.id = id_to_external_[r.id];

    AQ_ADD(rerank);  // [AQ_PROFILE] region (e): exact-L2 rerank + partial_sort
#ifdef AQ_PROFILE
    ++g_aqprof.n;  // count this timed searchV2 call (only reached past early-return guards)
#endif
    return final_results;
}

// ---------------------------------------------------------------------------
// rebuildGraphFromNGT: hot-swap graph edges from a denser NGT source index.
// Reuses existing SRHT/K-means/PCA/PQ/BQ encoding — only rebuilds edges.
// ~50s vs ~400s for a full fromNGTv2 rebuild.
// ---------------------------------------------------------------------------
void NGTAQIndex::rebuildGraphFromNGT(const std::string& ngt_path,
                                      float new_alpha,
                                      int   new_max_edges)
{
    if (new_alpha    > 0.0f) prop_.alpha     = new_alpha;
    if (new_max_edges > 0)   prop_.max_edges = new_max_edges;

    // Use graph_->size() (= state_.size(), includes tombstones) not size()
    // (= activeCount()).  resetEdges() requires adj.size() == state_.size().
    const size_t N    = graph_->size();
    const int    D    = prop_.dimension;
    const int    words = D / 64;

    AlphaCGPruner pruner(prop_.alpha, prop_.kappa);
    const float   tau  = bq_.tau();

    NGT::Index ngt(ngt_path);
    NGT::GraphIndex& gi = static_cast<NGT::GraphIndex&>(ngt.getIndex());

    std::vector<std::vector<uint32_t>> adj(N);
    for (size_t i = 1; i <= N; ++i) {
        uint32_t aq_id = static_cast<uint32_t>(i - 1);
        if (graph_->isTombstone(aq_id)) continue;

        NGT::GraphNode* node = nullptr;
        try { node = gi.getNode(static_cast<NGT::ObjectID>(i)); }
        catch (...) { continue; }
        if (!node || node->empty()) continue;

        // Use getRecordConstView (correct variable stride) for cluster-aware sort.
        uint32_t own_cid = graph_->getRecordConstView(aq_id).centroid_id();

        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(node->size());
        for (auto& edge : *node) {
            if (edge.id == 0 || edge.id > static_cast<unsigned int>(N)) continue;
            uint32_t nbr = static_cast<uint32_t>(edge.id - 1);
            float d_bq = bqDistance(graph_->getNodeBQ(aq_id),
                                     graph_->getNodeBQ(nbr), words, D);
            candidates.push_back({nbr, d_bq});
        }

        // Pure BQ distance sort (no cluster priority — see rebuildGraphSelf comment).
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (static_cast<int>(candidates.size()) > prop_.max_edges)
            candidates.resize(static_cast<size_t>(prop_.max_edges));

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(graph_->getNodeBQ(v), graph_->getNodeBQ(u), words, D);
        };
        adj[aq_id] = pruner.prune(candidates, tau, dist_fn);
    }

    graph_->resetEdges(adj);

    // Re-select entry points from the new graph
    int n_ep = std::min(prop_.n_entry_points, static_cast<int>(N));
    entry_points_   = selectEntryPoints(*graph_, n_ep);
    v2_entry_points_ = entry_points_;

    // Invalidate lazy cluster tables — rebuilt on next searchV2 call
    cluster_members_once_ = std::make_unique<std::once_flag>();
    cluster_members_v2_.clear();
    cluster_neighbors_v2_.clear();
}

// ---------------------------------------------------------------------------
// fixHoleTombstones: post-hoc tombstone repair.
// Scan raw_flat_ for zero-norm vectors (hole nodes created when zero-norm
// train vectors were excluded during normalization but not properly tombstoned
// in the serialized graph). Tombstones them so they cannot appear in search results.
// ---------------------------------------------------------------------------
int NGTAQIndex::fixHoleTombstones() {
    const int D = (d_eff_ > 0) ? d_eff_ : prop_.dimension;
    const size_t N = graph_->size();
    int n_fixed = 0;
    for (size_t i = 0; i < N; ++i) {
        if (graph_->isTombstone(static_cast<uint32_t>(i))) continue;
        const uint16_t* h = raw_flat_.data() + i * static_cast<size_t>(D);
        float norm2 = 0.f;
        for (int d = 0; d < D; ++d) {
            float f = NGT::NGTAQ::fp16_to_float(h[d]);
            norm2 += f * f;
        }
        if (norm2 < 1e-12f) {
            graph_->removeNode(static_cast<uint32_t>(i));
            ++n_fixed;
        }
    }
    fprintf(stderr, "[fixHoleTombstones] tombstoned %d zero-norm (hole) nodes\n", n_fixed);
    return n_fixed;
}

// ---------------------------------------------------------------------------
// rebuildGraphSelf: self-referential graph refinement.
// Runs searchV2 on every node to find high-quality candidate neighbors, then
// re-prunes with AlphaCGPruner. One pass raises recall ceiling by ~5-10%.
// ---------------------------------------------------------------------------
void NGTAQIndex::rebuildGraphSelf(int   k_search,
                                   float gamma,
                                   int   n_threads,
                                   float new_alpha,
                                   int   new_max_edges)
{
    if (new_alpha    > 0.0f) prop_.alpha     = new_alpha;
    if (new_max_edges > 0)   prop_.max_edges = new_max_edges;

    // BUGFIX: use graph_->size() (= state_.size(), includes tombstones) not size()
    // (= activeCount(), excludes tombstones).  resetEdges() requires adj.size() ==
    // state_.size(); if adj is undersized it reads past the end → SIGSEGV.
    const size_t N    = graph_->size();
    const int    D    = prop_.dimension;
    const int    words = D / 64;

    AlphaCGPruner pruner(prop_.alpha, prop_.kappa);
    const float   tau  = bq_.tau();

    // Warm up lazy cluster tables with a dummy query (triggers call_once).
    // This ensures subsequent parallel queries don't race on cluster_members_once_.
    if (!raw_flat_.empty()) {
        std::vector<float> dummy(D);
        const uint16_t* h = raw_flat_.data();
        for (int d = 0; d < D; ++d) dummy[d] = NGT::NGTAQ::fp16_to_float(h[d]);
        searchV2(dummy, 1, gamma, gamma);
    }

    // During rebuild we want maximum neighbor diversity, not maximum QPS.
    // Default: probe ALL K clusters so every node's true k-NN can be discovered
    // across cluster boundaries.  Caller can override via setNProbe() before calling
    // rebuildGraphSelf() (e.g. to cap rebuild time on large datasets).
    // rebuild_max_seeds_ is set to match so all probed centroids are used as seeds.
    int saved_n_probe_override = n_probe_override_;
    if (is_angular_) {
        const int K_clusters = static_cast<int>(kmeans_v2_ ? kmeans_v2_->num_clusters() : 0);
        if (K_clusters > 0) {
            // If caller pre-set via setNProbe(), honour that value (capped at K).
            // Otherwise default to probing all K clusters.
            const int caller_probe = saved_n_probe_override;  // 0 = "not set by caller"
            const int effective_probe = (caller_probe > 0)
                ? std::min(caller_probe, K_clusters)
                : K_clusters;
            n_probe_override_  = effective_probe;
            rebuild_max_seeds_ = effective_probe;
        }
    }

    // Parallel search: for each node i, find its k_search approximate NN.
    // searchV2 uses shared_lock — safe for concurrent reads.
    std::vector<std::vector<uint32_t>> adj(N);

#ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 512)
#endif
    for (size_t i = 0; i < N; ++i) {
        if (graph_->isTombstone(static_cast<uint32_t>(i))) continue;
        std::vector<float> q(D);
        const uint16_t* h = raw_flat_.data() + i * static_cast<size_t>(D);
        for (int d = 0; d < D; ++d) q[d] = NGT::NGTAQ::fp16_to_float(h[d]);
        // Search for k_search+1: self may appear in results; we'll skip it.
        auto results = searchV2(q, k_search + 1, gamma, gamma);

        // Build candidate list sorted by EXACT distance (two-phase approach).
        //
        // Problem: original code sorted/trimmed by BQ distance, which is
        // cluster-relative (residual BQ).  Cross-cluster true NNs have noisy
        // inter-cluster BQ distances and were ranked at positions 50-200+,
        // well outside the max_edges=128 cutoff → never reached alpha-CG.
        //
        // Fix phase 1: sort and trim by EXACT distance so cross-cluster true
        // NNs (small exact_dist) survive the max_edges cutoff.
        // Fix phase 2: replace with BQ distances for alpha-CG input so the
        // pruner retains its original navigability behaviour (same tau).
        //
        // Why BQ for alpha-CG: cross-cluster candidates have high BQ dist →
        // high pruning threshold → hard to be "dominated" → they are KEPT.
        // In-cluster candidates behave as before (accurate BQ distances).
        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(results.size());
        for (auto& r : results) {
            if (r.id == static_cast<uint32_t>(i)) continue;  // skip self
            candidates.push_back({r.id, r.distance});  // exact dist for ranking
        }

        // Phase 1: trim by exact distance (cross-cluster true NNs survive).
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (static_cast<int>(candidates.size()) > prop_.max_edges)
            candidates.resize(static_cast<size_t>(prop_.max_edges));

        // Phase 2: cluster-aware selective alpha-CG.
        //
        // Cross-cluster candidates (confirmed true NNs by Phase 1 exact trim) are
        // added unconditionally to the graph.  Applying BQ alpha-CG to them is
        // counterproductive: inter-cluster BQ distances are meaningless (different
        // residual frames), so the pruning decision is random noise and removes
        // valid cross-cluster edges that are critical for navigability.
        //
        // In-cluster candidates still use BQ alpha-CG which IS meaningful within a
        // cluster (same residual frame) and produces a navigable in-cluster sub-graph.
        const uint32_t cid_i = graph_->getRecordConstView(static_cast<uint32_t>(i)).centroid_id();

        std::vector<std::pair<uint32_t, float>> in_cluster_cands;
        std::vector<uint32_t> cross_cluster_ids;
        in_cluster_cands.reserve(candidates.size());
        // candidates is currently sorted by exact distance (Phase 1 order).
        // We iterate in that order so cross_cluster_ids preserves exact-dist rank.
        for (auto& c : candidates) {
            const uint32_t cid_c = graph_->getRecordConstView(c.first).centroid_id();
            const float bq_d = bqDistance(graph_->getNodeBQ(static_cast<uint32_t>(i)),
                                           graph_->getNodeBQ(c.first), words, D);
            if (cid_c == cid_i) {
                in_cluster_cands.push_back({c.first, bq_d});
            } else {
                cross_cluster_ids.push_back(c.first);  // exact-dist order preserved
            }
        }

        std::sort(in_cluster_cands.begin(), in_cluster_cands.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(graph_->getNodeBQ(v), graph_->getNodeBQ(u), words, D);
        };

        // Alpha-CG on in-cluster candidates → navigable in-cluster sub-graph.
        adj[i] = pruner.prune(in_cluster_cands, tau, dist_fn);

        // Append cross-cluster candidates unconditionally (closest first, exact-dist order).
        const int max_adj = prop_.max_edges;
        for (uint32_t ccid : cross_cluster_ids) {
            if (static_cast<int>(adj[i].size()) >= max_adj) break;
            adj[i].push_back(ccid);
        }
    }

    // Reset overrides before any post-rebuild searches.
    rebuild_max_seeds_ = 0;
    n_probe_override_  = saved_n_probe_override;

    graph_->resetEdges(adj);

    // Re-select entry points from the refined graph
    int n_ep = std::min(prop_.n_entry_points, static_cast<int>(N));
    entry_points_    = selectEntryPoints(*graph_, n_ep);
    v2_entry_points_ = entry_points_;

    // Invalidate lazy cluster tables — rebuilt on next searchV2 call
    cluster_members_once_ = std::make_unique<std::once_flag>();
    cluster_members_v2_.clear();
    cluster_neighbors_v2_.clear();
}

// ---------------------------------------------------------------------------
// saveV2 / loadV2
// ---------------------------------------------------------------------------
void NGTAQIndex::saveV2(const std::string& dir) const {
    if (!is_v2_) return;
    // SoAGraph v2 records
    graph_->saveV2Records(dir + "/v2_records.bin");
    // SRHT diagonal
    {
        std::vector<float> diag;
        srht_v2_->serialize(diag);
        std::ofstream f(dir + "/v2_srht.bin", std::ios::binary);
        uint32_t Dsz = (uint32_t)diag.size();
        f.write(reinterpret_cast<const char*>(&Dsz), sizeof(Dsz));
        f.write(reinterpret_cast<const char*>(diag.data()), Dsz * sizeof(float));
    }
    // K-means centroids
    {
        std::ofstream f(dir + "/v2_kmeans.bin", std::ios::binary);
        uint32_t K = kmeans_v2_->num_clusters();
        uint32_t Dim = (uint32_t)kmeans_v2_->dim();
        f.write(reinterpret_cast<const char*>(&K), sizeof(K));
        f.write(reinterpret_cast<const char*>(&Dim), sizeof(Dim));
        const auto& c = kmeans_v2_->centroids_data();
        f.write(reinterpret_cast<const char*>(c.data()), c.size() * sizeof(float));
    }
    // PCA components + mean + eigenvalues
    {
        std::ofstream f(dir + "/v2_pca.bin", std::ios::binary);
        uint32_t Dim = (uint32_t)pca_v2_->in_dim();
        uint32_t Top = (uint32_t)pca_v2_->out_dim();
        f.write(reinterpret_cast<const char*>(&Dim), sizeof(Dim));
        f.write(reinterpret_cast<const char*>(&Top), sizeof(Top));
        const auto& comp = pca_v2_->components();
        const auto& mean = pca_v2_->mean();
        const auto& eig  = pca_v2_->eigenvalues();
        f.write(reinterpret_cast<const char*>(comp.data()), comp.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(mean.data()), mean.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(eig.data()),  eig.size()  * sizeof(float));
    }
    // Tier-2 codebook
    {
        std::ofstream f(dir + "/v2_codebook.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(tier2_codebook_.data()),
                tier2_codebook_.size() * sizeof(float));
    }
    // GLOBAL PQ tier (Stage A): codebook (row-major) + per-vector codes + recon norms.
    // Written only when present; loadV2 keys on the meta version (slot 3 == 2).
    if (has_global_pq_) {
        {
            std::ofstream f(dir + "/v2_global_pq.bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(global_pq_codebook_.data()),
                    global_pq_codebook_.size() * sizeof(float));
        }
        {
            std::ofstream f(dir + "/v2_global_codes.bin", std::ios::binary);
            uint64_t n = (uint64_t)global_pq_norm_sq_.size();
            f.write(reinterpret_cast<const char*>(&n), sizeof(n));
            f.write(reinterpret_cast<const char*>(global_codes_.data()),
                    global_codes_.size() * sizeof(uint8_t));
            f.write(reinterpret_cast<const char*>(global_pq_norm_sq_.data()),
                    global_pq_norm_sq_.size() * sizeof(float));
        }
    }
    // GLOBAL PQ-16 tier (Stage B/C): K=16 codebook + per-node contiguous neighbor store.
    // Written only when present; loadV2 keys on meta version (slot 3 >= 3).
    const bool save_gpq4 = has_gpq4_ && graph_ && graph_->hasGPQ4();
    if (save_gpq4) {
        {
            std::ofstream f(dir + "/v2_gpq4_codebook.bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(gpq4_codebook_.data()),
                    gpq4_codebook_.size() * sizeof(float));
        }
        {
            // Flat per-node codes + recon norms (single-node seed/expansion scoring).
            std::ofstream f(dir + "/v2_gpq4_codes.bin", std::ios::binary);
            uint64_t n = (uint64_t)gpq4_norm_sq_.size();
            f.write(reinterpret_cast<const char*>(&n), sizeof(n));
            f.write(reinterpret_cast<const char*>(gpq4_codes_.data()),
                    gpq4_codes_.size() * sizeof(uint8_t));
            f.write(reinterpret_cast<const char*>(gpq4_norm_sq_.data()),
                    gpq4_norm_sq_.size() * sizeof(float));
        }
        graph_->saveGPQ4(dir + "/v2_gpq4_store.bin");
    }
    // Tech 1: symmetric SQ8 codes (optional sidecar; loadV2 keys on file presence).
    if (has_sq8_ && !sq8_codes_.empty()) {
        std::ofstream f(dir + "/v2_sq8.bin", std::ios::binary);
        uint64_t n  = (uint64_t)sq8_max_.size();
        uint32_t da = (uint32_t)sq8_dim_align_;
        f.write(reinterpret_cast<const char*>(&n),  sizeof(n));
        f.write(reinterpret_cast<const char*>(&da), sizeof(da));
        f.write(reinterpret_cast<const char*>(sq8_codes_.data()), sq8_codes_.size() * sizeof(int8_t));
        f.write(reinterpret_cast<const char*>(sq8_max_.data()),   sq8_max_.size()   * sizeof(float));
        f.write(reinterpret_cast<const char*>(sq8_norm_.data()),  sq8_norm_.size()  * sizeof(float));
    }
    // Task 2: internal->external id map (present only after BFS reorder; sidecar, file-keyed).
    if (!id_to_external_.empty()) {
        std::ofstream f(dir + "/v2_idmap.bin", std::ios::binary);
        uint64_t n = (uint64_t)id_to_external_.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(id_to_external_.data()), n * sizeof(uint32_t));
    }
    // Metadata: is_angular_, d_eff_, m_pq_, meta_version [, gpq4_m_pq_].
    // meta_version: 1 = no global PQ (legacy), 2 = K=256 global PQ, 3 = K=16 GPQ4 store
    //               (D_sub=8, gpq4_m_pq_ == m_pq_), 4 = GPQ4 with its OWN gpq4_m_pq_ (slot 4).
    {
        std::string meta_path = dir + "/v2_meta.bin";
        FILE* f = fopen(meta_path.c_str(), "wb");
        if (f) {
            const int32_t gm = (gpq4_m_pq_ > 0) ? gpq4_m_pq_ : m_pq_;
            int32_t meta_version = save_gpq4 ? 4 : (has_global_pq_ ? 2 : 1);
            int32_t meta[5] = {(int32_t)is_angular_, d_eff_, m_pq_, meta_version, gm};
            fwrite(meta, sizeof(meta), 1, f);
            fclose(f);
        }
    }
}

void NGTAQIndex::loadV2(const std::string& dir) {
    graph_->loadV2Records(dir + "/v2_records.bin");
    // SRHT
    {
        std::ifstream f(dir + "/v2_srht.bin", std::ios::binary);
        uint32_t Dsz; f.read(reinterpret_cast<char*>(&Dsz), sizeof(Dsz));
        std::vector<float> diag(Dsz);
        f.read(reinterpret_cast<char*>(diag.data()), Dsz * sizeof(float));
        srht_v2_ = std::make_unique<NGT::NGTAQ::SRHT>((int)Dsz, 0);
        srht_v2_->deserialize(diag);
    }
    // K-means
    {
        std::ifstream f(dir + "/v2_kmeans.bin", std::ios::binary);
        uint32_t K, Dim;
        f.read(reinterpret_cast<char*>(&K), sizeof(K));
        f.read(reinterpret_cast<char*>(&Dim), sizeof(Dim));
        kmeans_v2_ = std::make_unique<NGT::NGTAQ::KMeansCentering>(K, (int)Dim, 0);
        std::vector<float> c((size_t)K * Dim);
        f.read(reinterpret_cast<char*>(c.data()), c.size() * sizeof(float));
        kmeans_v2_->set_centroids(std::move(c));
    }
    // PCA
    {
        std::ifstream f(dir + "/v2_pca.bin", std::ios::binary);
        uint32_t Dim, Top;
        f.read(reinterpret_cast<char*>(&Dim), sizeof(Dim));
        f.read(reinterpret_cast<char*>(&Top), sizeof(Top));
        pca_v2_ = std::make_unique<NGT::NGTAQ::PCAProjector>((int)Dim, (int)Top, 0);
        std::vector<float> comp((size_t)Top*Dim), mean(Dim), eig(Top);
        f.read(reinterpret_cast<char*>(comp.data()), comp.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(mean.data()), mean.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(eig.data()),  eig.size()  * sizeof(float));
        pca_v2_->set_state(std::move(comp), std::move(mean), std::move(eig));
    }
    // New metadata: is_angular_, d_eff_, m_pq_, meta_version (optional in old indices).
    // Must be read BEFORE codebook so M_cb and D_sub are known.
    // meta_version: <2 (or absent) = no global PQ; >=2 = global PQ files present.
    int32_t meta_version = 1;
    gpq4_m_pq_ = 0;  // 0 → fall back to m_pq_ (meta_version<4 indices: GPQ4 D_sub=8)
    {
        std::string meta_path = dir + "/v2_meta.bin";
        FILE* f = fopen(meta_path.c_str(), "rb");
        if (f) {
            // Read up to 5 ints; old indices wrote 4. Slot 4 (gpq4_m_pq_) only present
            // for meta_version>=4 — read it conditionally so 4-int files don't underflow.
            int32_t meta[5] = {0, 0, 16, 1, 0};
            size_t got = fread(meta, sizeof(int32_t), 5, f);
            if (got >= 4) {
                is_angular_  = (bool)meta[0];
                d_eff_       = meta[1];
                m_pq_        = meta[2];
                meta_version = meta[3];
                if (got >= 5 && meta_version >= 4 && meta[4] > 0) gpq4_m_pq_ = meta[4];
            }
            fclose(f);
        }
        if (m_pq_ <= 0) m_pq_ = 16;
        if (d_eff_ <= 0) d_eff_ = prop_.dimension;
    }
    // Tier-2 codebook: M_cb sub-spaces × K=256 codes × D_sub=8 dims
    {
        std::ifstream f(dir + "/v2_codebook.bin", std::ios::binary);
        const int M_cb  = m_pq_;   // derived from metadata
        const int D_sub = 8;       // always 8 dims per sub-space
        tier2_codebook_.resize((size_t)M_cb * 256 * D_sub);
        f.read(reinterpret_cast<char*>(tier2_codebook_.data()),
               tier2_codebook_.size() * sizeof(float));
        // Build transposed codebook [M][D_sub][K] for AVX2 FMA LUT build
        tier2_codebook_T_.resize(tier2_codebook_.size());
        NGT::NGTAQ::build_tier2_codebook_T(
            tier2_codebook_.data(), M_cb, 256, D_sub,
            tier2_codebook_T_.data());
    }
    // GLOBAL PQ tier (Stage A): present iff meta_version >= 2. Old indices fall back
    // (has_global_pq_ stays false) instead of crashing.
    has_global_pq_ = false;
    if (meta_version >= 2) {
        const int M_cb  = m_pq_;
        const int D_sub = 8;
        std::ifstream fc(dir + "/v2_global_pq.bin", std::ios::binary);
        std::ifstream fx(dir + "/v2_global_codes.bin", std::ios::binary);
        if (fc && fx) {
            global_pq_codebook_.resize((size_t)M_cb * 256 * D_sub);
            fc.read(reinterpret_cast<char*>(global_pq_codebook_.data()),
                    global_pq_codebook_.size() * sizeof(float));
            uint64_t n = 0;
            fx.read(reinterpret_cast<char*>(&n), sizeof(n));
            global_codes_.resize((size_t)n * M_cb);
            global_pq_norm_sq_.resize((size_t)n);
            fx.read(reinterpret_cast<char*>(global_codes_.data()),
                    global_codes_.size() * sizeof(uint8_t));
            fx.read(reinterpret_cast<char*>(global_pq_norm_sq_.data()),
                    global_pq_norm_sq_.size() * sizeof(float));
            global_pq_codebook_T_.resize(global_pq_codebook_.size());
            NGT::NGTAQ::build_tier2_codebook_T(
                global_pq_codebook_.data(), M_cb, 256, D_sub,
                global_pq_codebook_T_.data());
            has_global_pq_ = (fc.good() || fc.eof()) && (fx.good() || fx.eof());
        }
    }
    // GLOBAL PQ-16 tier (Stage B/C): present iff meta_version >= 3. Old indices fall back
    // (has_gpq4_ stays false → legacy/global routing) instead of crashing.
    has_gpq4_ = false;
    if (meta_version >= 3) {
        // meta_version 3: gpq4_m_pq_ stayed 0 → gpq4MPQ() == m_pq_ (D_sub=8, old layout).
        // meta_version 4: gpq4_m_pq_ was read above (finer M). D_sub derived from it.
        const int M_cb  = gpq4MPQ();
        const int D_sub = dEff() / M_cb;
        std::ifstream fc(dir + "/v2_gpq4_codebook.bin", std::ios::binary);
        if (fc) {
            gpq4_codebook_.resize((size_t)M_cb * NGT::NGTAQ::GPQ4_K * D_sub);
            fc.read(reinterpret_cast<char*>(gpq4_codebook_.data()),
                    gpq4_codebook_.size() * sizeof(float));
            gpq4_codebook_T_.resize(gpq4_codebook_.size());
            NGT::NGTAQ::build_tier2_codebook_T(
                gpq4_codebook_.data(), M_cb, NGT::NGTAQ::GPQ4_K, D_sub,
                gpq4_codebook_T_.data());
            bool codes_ok = false;
            {
                std::ifstream fx(dir + "/v2_gpq4_codes.bin", std::ios::binary);
                if (fx) {
                    uint64_t n = 0;
                    fx.read(reinterpret_cast<char*>(&n), sizeof(n));
                    gpq4_codes_.resize((size_t)n * M_cb);
                    gpq4_norm_sq_.resize((size_t)n);
                    fx.read(reinterpret_cast<char*>(gpq4_codes_.data()),
                            gpq4_codes_.size() * sizeof(uint8_t));
                    fx.read(reinterpret_cast<char*>(gpq4_norm_sq_.data()),
                            gpq4_norm_sq_.size() * sizeof(float));
                    codes_ok = (fx.good() || fx.eof());
                }
            }
            bool store_ok = graph_->loadGPQ4(dir + "/v2_gpq4_store.bin");
            has_gpq4_ = (fc.good() || fc.eof()) && codes_ok && store_ok && graph_->hasGPQ4();
        }
    }
    // Tech 1: load symmetric SQ8 sidecar if present (keyed on file presence, no meta bump).
    {
        std::ifstream f(dir + "/v2_sq8.bin", std::ios::binary);
        if (f) {
            uint64_t n = 0; uint32_t da = 0;
            f.read(reinterpret_cast<char*>(&n),  sizeof(n));
            f.read(reinterpret_cast<char*>(&da), sizeof(da));
            if (f && n > 0 && da > 0) {
                sq8_dim_align_ = (int)da;
                sq8_codes_.resize((size_t)n * da);
                sq8_max_.resize((size_t)n);
                sq8_norm_.resize((size_t)n);
                f.read(reinterpret_cast<char*>(sq8_codes_.data()), sq8_codes_.size() * sizeof(int8_t));
                f.read(reinterpret_cast<char*>(sq8_max_.data()),   sq8_max_.size()   * sizeof(float));
                f.read(reinterpret_cast<char*>(sq8_norm_.data()),  sq8_norm_.size()  * sizeof(float));
                has_sq8_ = (f.good() || f.eof());
            }
        }
    }
    // Task 2: load the internal->external id map sidecar if present (BFS-reordered index).
    {
        std::ifstream f(dir + "/v2_idmap.bin", std::ios::binary);
        if (f) {
            uint64_t n = 0;
            f.read(reinterpret_cast<char*>(&n), sizeof(n));
            if (f && n > 0) {
                id_to_external_.resize((size_t)n);
                f.read(reinterpret_cast<char*>(id_to_external_.data()), n * sizeof(uint32_t));
                if (!(f.good() || f.eof())) id_to_external_.clear();
            }
        }
    }
    // Tech 4.1: optionally build the DiskANN-style PackedV2Node layout (record + up to 64
    // neighbor IDs co-located in 5 cache lines) in-memory. The batch DABS loop can then read
    // neighbor IDs from it instead of the CSR offsets_->edge_ids_ indirection. On SIFT-1M
    // this saves only ~3% of dabs at mid-recall and ~0 at low recall (offsets_/edge_ids_ are
    // largely cache-resident once LinearPool shrank the visit count), at a 320 B/node memory
    // cost. So it is OPT-IN (AQ_PACKED=1) — useful for larger graphs where the CSR round-trip
    // dominates. No format change (rebuilt from CSR at load).
    {
        const char* e = std::getenv("AQ_PACKED");
        if (e && std::atoi(e) != 0) graph_->buildPackedV2();
    }
    is_v2_ = true;
}

} // namespace NGTAQ
