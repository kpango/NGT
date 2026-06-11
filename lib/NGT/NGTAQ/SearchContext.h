// lib/NGT/NGTAQ/SearchContext.h
#pragma once
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <memory>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <utility>
#include "NGT/NGTAQ/ADCTable.h"    // NGT::NGTAQ::ADCQueryState
#include "NGT/NGTAQ/GlobalPQ4.h"   // NGT::NGTAQ::GlobalPQ4LUT

namespace NGTAQ {

// === hoisted from AQIndex.cpp in Step 0 (verbatim definitions) ===

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

// hoisted from AQIndex.cpp searchV2 neighbor-expansion (was a function-local struct)
struct NbrCand { uint32_t cid; uint32_t id; };

// hoisted from AQIndex.cpp searchV2 seeding (was a function-local struct)
struct SeedScore { float score; uint32_t id; };

// ADC slot-cache metadata. Defined locally in AQIndex.cpp's searchV2 today; mirrored
// here so SearchContext is self-contained. ADC_SLOTS must match the searchV2 value.
constexpr int ADC_SLOTS = 8;
struct ADCSlotMeta {
    uint32_t cid       = UINT32_MAX;
    float    q_norm_sq = 0.f;
    float    q_norm    = 0.f;
    int32_t  q_sum     = 0;
};
// searchV2 / maybe_rebuild_adc aggregate-init this as {cid, q_norm_sq, q_norm, q_sum};
// pin the field order + size so a silent reorder can't misassign those slots.
static_assert(offsetof(ADCSlotMeta, cid)       == 0,  "ADCSlotMeta field order changed; aggregate-init relies on {cid,q_norm_sq,q_norm,q_sum}");
static_assert(offsetof(ADCSlotMeta, q_norm_sq) == 4,  "ADCSlotMeta field order changed; aggregate-init relies on {cid,q_norm_sq,q_norm,q_sum}");
static_assert(offsetof(ADCSlotMeta, q_norm)    == 8,  "ADCSlotMeta field order changed; aggregate-init relies on {cid,q_norm_sq,q_norm,q_sum}");
static_assert(offsetof(ADCSlotMeta, q_sum)     == 12, "ADCSlotMeta field order changed; aggregate-init relies on {cid,q_norm_sq,q_norm,q_sum}");
static_assert(sizeof(ADCSlotMeta) == 16, "ADCSlotMeta size changed");

// ---- per-query tuning (immutable during a search). Defaults == current env defaults ----
struct SearchParameters {
    int   k                = 10;
    int   ef               = 0;      // AQ_EF       (0 = derive from max_visits)
    int   max_visits       = 0;      // searchV2 arg(0 = derive)
    float gamma_enq        = 0.2f;   // searchV2 arg
    float gamma_term       = 0.4f;   // searchV2 arg
    float k_prime_factor   = -1.0f;  // searcher_.k_prime_factor (-1 = default)
    int   rerank_factor    = 0;      // searchV2 arg
    int   rerank_n         = 0;      // AQ_RERANK_N
    int   refine_mult      = 0;      // AQ_REFINE_MULT
    int   rerank_sq8       = 0;      // AQ_RERANK_SQ8
    int   cq_probe         = 4;      // AQ_CQ_PROBE
    int   n_probe          = 0;      // clusters to probe (0 = worker derives default)
    int   seeds            = 0;      // AQ_SEEDS
    int   n_cluster_seeds  = 0;      // 0 = use index default (prop_.n_cluster_seeds); falls back in searchV2
    int   seeds_per_cluster= 0;      // 0 = unbounded (legacy full-cluster scan); >0 = cap
    bool  seed_cap_topk    = false;  // AQ_SEED_CAP_TOPK
    float term_eps         = -1.0f;  // AQ_TERM_EPS
    float term_eps_fp16    = -1.0f;  // AQ_TERM_EPS_FP16
};

// ---- per-thread reusable scratch (one alloc, reused forever). 1:1 with the ~30 former thread_locals (several were comma-declared) ----
// NOTE: ADCQueryState/GlobalPQ4LUT live in nested namespace NGT::NGTAQ (legacy); the hoisted types and SearchContext/Pool live in top-level NGTAQ — hence the NGT::NGTAQ:: qualification on those two members.
struct SearchContext {
    std::vector<float>  q_normalized;   // q_normalized_tl
    std::vector<float>  q_padded;       // q_padded_tl
    std::vector<float>  q_rot;          // q_rot_tl
    std::vector<float>  q_res;          // q_res_tl
    std::vector<float>  q_res_init;     // q_res_init_tl
    NGT::NGTAQ::ADCQueryState adc;      // adc_tl
    // --- helper-internal scratch (buildGlobalLUT/buildGlobalLUT16) ---
    std::vector<float>  q_norm_lut;     // q_norm_tl (buildGlobalLUT/16); buildGlobalLUT and buildGlobalLUT16 are mutually exclusive per query (use_global_pq = ... && !use_batch), so sharing is safe
    std::vector<float>  q_padded_lut;   // q_padded_tl (helper-local)
    std::vector<float>  q_rot_lut;      // q_rot_tl (helper-local)
    std::vector<float>  gpq4_ip_scratch; // buildGlobalLUT16 ip_tl (helper-internal)
    std::vector<float>  t2_lut;         // t2_lut_tl
    std::vector<float>  t2_lut_probe;   // t2_lut_probe
    std::vector<float>  global_lut;     // global_lut_tl
    NGT::NGTAQ::GlobalPQ4LUT batch_lut; // batch_lut_tl
    std::vector<float>  batch_ip;       // batch_ip_tl
    std::vector<int8_t>                 adc_int8;  // adc_int8_tl (8 slots × D)
    std::array<ADCSlotMeta, ADC_SLOTS>  adc_meta;  // adc_meta_tl
    std::vector<int8_t> q_sq8;          // q_sq8_tl
    AQLinearPool        lp;             // lp
    std::vector<std::pair<float,uint32_t>> results;  // results_tl
    std::vector<float>  block_ip;       // block_ip_lp / block_ip_tl (unified)
    std::vector<NbrCand> nbr_buf;       // nbr_buf_tl
    std::vector<SeedScore> scored;          // scored
    std::vector<uint32_t>  probe_clusters;  // probe_clusters
    std::vector<SeedScore> clus_buf;        // clus_buf
    std::vector<uint64_t> vis;          // t_vis (bitvector backend)
    std::vector<uint16_t> vis_ver;      // t_vis_ver (versioned backend)
    uint16_t              vis_cur = 0;  // t_vis_cur
    std::atomic<uint64_t> epoch{0};     // P2 placeholder (unused in P0)
};

// ---- context pool (mutex-guarded free list; provably TSan-clean for P0).
// acquire/release run once per search vs the long search body, so uncontended. ----
class SearchContextPool {
public:
    SearchContextPool() = default;
    SearchContextPool(const SearchContextPool&) = delete;
    SearchContextPool& operator=(const SearchContextPool&) = delete;
    SearchContextPool(SearchContextPool&&) = delete;
    SearchContextPool& operator=(SearchContextPool&&) = delete;
    ~SearchContextPool() { for (auto* c : all_) delete c; }
    SearchContext* acquire() {
        std::lock_guard<std::mutex> g(m_);
        if (free_.empty()) {
            auto owned = std::make_unique<SearchContext>();
            all_.push_back(owned.get());   // throws (if any) before ownership transfer
            return owned.release();
        }
        auto* c = free_.back(); free_.pop_back(); return c;
    }
    void release(SearchContext* c) {
        std::lock_guard<std::mutex> g(m_);
        free_.push_back(c);
    }
private:
    std::mutex m_;
    std::vector<SearchContext*> free_;
    std::vector<SearchContext*> all_;
};

// RAII acquire/release
struct SearchContextGuard {
    SearchContextPool& pool; SearchContext* ctx;
    explicit SearchContextGuard(SearchContextPool& p) : pool(p), ctx(p.acquire()) {}
    ~SearchContextGuard() { pool.release(ctx); }
    SearchContextGuard(const SearchContextGuard&) = delete;
    SearchContextGuard& operator=(const SearchContextGuard&) = delete;
    SearchContextGuard(SearchContextGuard&&) = delete;
    SearchContextGuard& operator=(SearchContextGuard&&) = delete;
    SearchContext& operator*()  { return *ctx; }
    SearchContext* operator->() { return ctx; }
};

} // namespace NGTAQ
