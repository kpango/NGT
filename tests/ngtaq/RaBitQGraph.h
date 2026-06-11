// tests/ngtaq/RaBitQGraph.h
// Dynamic SymphonyQG-class graph: online insert (Vamana RobustPrune) + in-place
// delete (IP-DiskANN, merge-free, no reverse list) over the vertex-relative
// RaBitQ-FastScan routing layout (RaBitQFastScan.h) with an implicit-rerank beam.
// Single-threaded correctness milestone (D1). Concurrency (EBR) is a later step.
//
// This is the decisive differentiator vs static QG / Glass / SymphonyQG, all of
// which are build-once-static.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>
#include <immintrin.h>

#include "NGT/NGTAQ/SRHT.h"
#include "RaBitQFastScan.h"

namespace NGTAQ {

struct RaBitQGraph {
    int R = 32;                             // out-degree (FastScan batch alignment; multiple of 32)
    static constexpr float ALPHA = 1.2f;    // RobustPrune occlusion
    int q_dim = 0, D = 0, num_codebook = 0;
    size_t block_bytes = 0;
    NGT::NGTAQ::SRHT srht;

    // per-slot SoA
    std::vector<float>    raw;     // nslot * q_dim  (exact distance + re-encode source)
    std::vector<float>    vrot;    // nslot * D      (rotated; resident for dynamic encode)
    std::vector<uint8_t>  blocks;  // nslot * block_bytes
    std::vector<float>    trx, fdq, fvq;  // nslot * R
    std::vector<uint32_t> nbr;     // nslot * R  (neighbor SLOTS; self-pad)
    std::vector<int>      deg;     // nslot      (real degree <= R)
    std::vector<uint8_t>  alive;   // nslot
    std::vector<uint32_t> slot_id; // nslot -> external id
    std::unordered_map<uint32_t, uint32_t> id2slot;
    std::vector<uint32_t> freelist;
    int nslot = 0, nalive = 0;
    uint32_t entry = 0;            // medoid-ish entry slot
    std::vector<uint32_t> stamp;   // visited epoch (search)
    uint32_t epoch = 0;
    // search scratch (reused across queries; single-thread search path -> zero per-query alloc)
    mutable std::vector<float> s_pad, s_qrot;
    mutable std::vector<uint8_t> s_byteq, s_lut;
    mutable std::vector<std::pair<float, uint32_t>> s_pool, s_res;
    mutable std::vector<uint32_t> s_vis;    // visited set: 2048-slot open-addr hash (expanded nodes)
    mutable std::vector<uint32_t> s_vis2;   // seen set: 2048-slot open-addr hash (pooled nodes, dedup)
    mutable std::vector<std::pair<float, uint32_t>> s_sdists;  // scratch for seed scan (reused)
    bool has_deletions = false;           // true once any remove() is called; gates alive[] checks
    bool angular_ = false;               // absolute-encoding mode for cosine/angular datasets
    std::vector<uint32_t> seeds;         // sqrt(N) random seeds for multi-start navigation (large N)

    static int pow2(int n) { int p = 1; while (p < n) p <<= 1; return p < 64 ? 64 : p; }

    RaBitQGraph(int q_dim_, uint64_t seed, int R_ = 32, bool angular = false)
        : R(R_ < 32 ? 32 : (R_ / 32) * 32),
          q_dim(q_dim_), D(pow2(q_dim_)), num_codebook(D / 4),
          block_bytes((size_t)R * (size_t)pow2(q_dim_) / 8), srht(pow2(q_dim_), seed),
          angular_(angular) {}

    inline float exact_l2(uint32_t slot, const float* q) const {
        const float* x = &raw[(size_t)slot * q_dim];
        int d = 0; float s = 0.f;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; d + 8 <= q_dim; d += 8) {
            __m256 e = _mm256_sub_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(q + d));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(e, e));
        }
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
        s = _mm_cvtss_f32(t);
#endif
        for (; d < q_dim; ++d) { float e = x[d] - q[d]; s += e * e; }
        return s;
    }

    // grow all SoA arrays to hold `cap` slots.
    void reserve_slots(int cap) {
        raw.resize((size_t)cap * q_dim, 0.f);
        vrot.resize((size_t)cap * D, 0.f);
        blocks.resize((size_t)cap * block_bytes, 0);
        trx.resize((size_t)cap * R, 0.f);
        fdq.resize((size_t)cap * R, 0.f);
        fvq.resize((size_t)cap * R, 0.f);
        nbr.resize((size_t)cap * R, 0u);
        deg.resize(cap, 0);
        alive.resize(cap, 0);
        slot_id.resize(cap, 0u);
        stamp.resize(cap, 0u);
    }

    uint32_t alloc_slot(uint32_t id, const float* vec) {
        uint32_t s;
        if (!freelist.empty()) { s = freelist.back(); freelist.pop_back(); }
        else { s = (uint32_t)nslot++; if ((int)alive.size() < nslot) reserve_slots(nslot + nslot / 2 + 64); }
        std::memcpy(&raw[(size_t)s * q_dim], vec, (size_t)q_dim * sizeof(float));
        // rotate — thread_local scratch avoids 1M heap allocs during bulk build
        {
            static thread_local std::vector<float> rot_sc_;
            if ((int)rot_sc_.size() < D) rot_sc_.assign(D, 0.f);
            std::memcpy(rot_sc_.data(), vec, (size_t)q_dim * sizeof(float));
            if (D > q_dim) std::fill(rot_sc_.begin() + q_dim, rot_sc_.begin() + D, 0.f);
            srht.apply(rot_sc_.data(), &vrot[(size_t)s * D]);
        }
        deg[s] = 0; alive[s] = 1; slot_id[s] = id; id2slot[id] = s; ++nalive;
        return s;
    }

    // vertex-relative encode of slot v's current `deg[v]` neighbors (pad to R w/ self).
    void encode_row(uint32_t v) {
        const int words = D / 64;
        const float fac_norm = 1.0f / std::sqrt((float)D);
        const float* vr = &vrot[(size_t)v * D];
        uint32_t* nbv = &nbr[(size_t)v * R];
        for (int j = deg[v]; j < R; ++j) nbv[j] = v;  // self-pad
        static thread_local std::vector<uint64_t> s_bin32_;
        static thread_local std::vector<int> s_bin_;
        s_bin32_.assign((size_t)R * words, 0ull);
        s_bin_.resize(D);
        auto& bin32 = s_bin32_;
        auto& bin   = s_bin_;
        for (int j = 0; j < R; ++j) {
            const float* ur = &vrot[(size_t)nbv[j] * D];
            double nr2 = 0, ipabs = 0, ipc = 0;
            if (angular_) {
                // Absolute encoding: encode sign(u_rot[d]) directly.
                // Distance estimate: ||q-u||² ≈ 2 - 2*(||u_rot||_1/D)*<q_rot, sgn(u_rot)>
                for (int d = 0; d < D; ++d) {
                    float r = ur[d];    // absolute rotated coordinate
                    int b = r > 0.f ? 1 : 0; bin[d] = b;
                    float sgn = 2.f * b - 1.f;
                    nr2 += (double)r * r; ipabs += (double)r * sgn;  // ipc unused
                }
            } else {
                // Vertex-relative encoding: encode sign(u_rot[d] - v_rot[d]).
                for (int d = 0; d < D; ++d) {
                    float r = ur[d] - vr[d];
                    int b = r > 0.f ? 1 : 0; bin[d] = b;
                    float sgn = 2.f * b - 1.f;
                    nr2 += (double)r * r; ipabs += (double)r * sgn; ipc += (double)vr[d] * sgn;
                }
            }
            rbfs::pack_binary(bin.data(), &bin32[(size_t)j * words], D);
            int popc = 0;
            for (int w = 0; w < words; ++w) popc += __builtin_popcountll(bin32[(size_t)j * words + w]);
            if (angular_) {
                // Coefficients for: 2 - (2*ipabs/D)*(width*fsr + vl*(2*popc-D))
                float dq = -2.f * (float)ipabs / (float)D;
                trx[(size_t)v * R + j] = 2.0f;
                fdq[(size_t)v * R + j] = dq;
                fvq[(size_t)v * R + j] = dq * (float)(2 * popc - D);
            } else {
                float nr = (float)std::sqrt(nr2);
                float fac_x0 = (float)(ipabs * fac_norm) / (nr > 1e-12f ? nr : 1e-12f);
                float fac_x1 = (float)(ipc * fac_norm);
                float x_x0 = nr / (fac_x0 != 0.f ? fac_x0 : 1e-12f);
                trx[(size_t)v * R + j] = nr * nr + 2.f * x_x0 * fac_x1;
                fdq[(size_t)v * R + j] = -2.f * x_x0 * fac_norm;
                fvq[(size_t)v * R + j] = fdq[(size_t)v * R + j] * (float)(2 * popc - D);
            }
        }
        rbfs::pack_codes(D, bin32.data(), R, &blocks[(size_t)v * block_bytes]);
    }

    struct Q { float width = 0, vl = 0; int32_t sumq = 0; };  // buffers live in member scratch
    void query_prepare(const float* q_raw, Q& o) const {
        if ((int)s_pad.size() < D) {
            s_pad.assign(D, 0.f); s_qrot.assign(D, 0.f);
            s_byteq.assign(D, 0); s_lut.assign((size_t)num_codebook * 16, 0);
        }
        std::fill(s_pad.begin(), s_pad.begin() + D, 0.f);
        std::memcpy(s_pad.data(), q_raw, (size_t)q_dim * sizeof(float));
        const_cast<NGT::NGTAQ::SRHT&>(srht).apply(s_pad.data(), s_qrot.data());
        float lo, hi; rbfs::data_range(s_qrot.data(), D, lo, hi);
        o.width = (hi - lo) / (float)((1 << rbfs::QG_BQUERY) - 1);
        if (o.width <= 0.f) o.width = 1e-6f;
        o.vl = lo;
        rbfs::quantize_q(s_byteq.data(), s_qrot.data(), D, lo, o.width, o.sumq);
        rbfs::pack_lut(D, s_byteq.data(), s_lut.data());
    }

    // implicit-rerank dual-pool beam -> sorted (exact_dist, slot) result pool (<= ef).
    // Flat ef-bounded cursor pools (Glass LinearPool style): no heap, no per-query alloc,
    // cache-friendly contiguous storage, software prefetch of the next row.
    void beam(const float* q_raw, int ef, std::vector<std::pair<float, uint32_t>>& got) {
        got.clear();
        if (nalive == 0) return;
        static const bool PROF = std::getenv("RBQ_PROF") != nullptr;
        static uint64_t c_prep = 0, c_loop = 0, c_n = 0;
        uint64_t t0 = PROF ? __rdtsc() : 0;
        Q q; query_prepare(q_raw, q);
        uint64_t t1 = PROF ? __rdtsc() : 0;
        const uint8_t* lut = s_lut.data();
        uint16_t fsres[128];  // FastScan results, up to R=128 (four 32-vector blocks)
        if (++epoch == 0) { std::fill(stamp.begin(), stamp.end(), 0u); epoch = 1; }
        auto seen = [&](uint32_t id) { return stamp[id] == epoch; };
        auto& pool = s_pool; auto& res = s_res; pool.clear(); res.clear();
        int cur = 0;
        auto pool_insert = [&](float key, uint32_t id) {           // ascending, capped ef
            if ((int)pool.size() >= ef && key >= pool.back().first) return;
            int lo = 0, hi = (int)pool.size();
            while (lo < hi) { int m = (lo + hi) >> 1; if (pool[m].first < key) lo = m + 1; else hi = m; }
            pool.insert(pool.begin() + lo, {key, id});
            if ((int)pool.size() > ef) pool.pop_back();
            if (lo < cur) cur = lo;   // better candidate appeared before cursor -> revisit
        };
        auto res_insert = [&](float key, uint32_t id) {            // ascending exact, capped ef
            if ((int)res.size() >= ef && key >= res.back().first) return;
            int lo = 0, hi = (int)res.size();
            while (lo < hi) { int m = (lo + hi) >> 1; if (res[m].first < key) lo = m + 1; else hi = m; }
            res.insert(res.begin() + lo, {key, id});
            if ((int)res.size() > ef) res.pop_back();
        };
        uint32_t e = entry;
        if (e >= (uint32_t)nslot || !alive[e]) { for (int s = 0; s < nslot; ++s) if (alive[s]) { e = s; break; } }
        pool_insert(exact_l2(e, q_raw), e);
        while (cur < (int)pool.size()) {
            if ((int)res.size() >= ef && pool[cur].first > res.back().first) break;
            uint32_t v = pool[cur].second; ++cur;
            if (seen(v) || !alive[v]) continue;
            stamp[v] = epoch;
            if (cur < (int)pool.size()) {                          // prefetch next pop's ALL hot streams
                uint32_t nx = pool[cur].second;
                const char* bp = (const char*)&blocks[(size_t)nx * block_bytes];
                _mm_prefetch(bp, _MM_HINT_T0);
                if (block_bytes > 64) _mm_prefetch(bp + 64, _MM_HINT_T0);
                _mm_prefetch((const char*)&raw[(size_t)nx * q_dim], _MM_HINT_T0);
                _mm_prefetch((const char*)&trx[(size_t)nx * R], _MM_HINT_T0);
                _mm_prefetch((const char*)&fdq[(size_t)nx * R], _MM_HINT_T0);
                _mm_prefetch((const char*)&fvq[(size_t)nx * R], _MM_HINT_T0);
                _mm_prefetch((const char*)&nbr[(size_t)nx * R], _MM_HINT_T0);
            }
            float ev = exact_l2(v, q_raw);
            res_insert(ev, v);
            for (int blk = 0; blk < R; blk += 32)   // R/32 FastScan blocks (each 4*D bytes)
                rbfs::accumulate_block(D, &blocks[(size_t)v * block_bytes + (size_t)(blk / 32) * (size_t)D * 4],
                                       lut, fsres + blk);
            const uint32_t* nbv = &nbr[(size_t)v * R];
            const float* tx = &trx[(size_t)v * R];
            const float* dq = &fdq[(size_t)v * R];
            const float* vq = &fvq[(size_t)v * R];
            for (int j = 0; j < R; ++j) {
                uint32_t u = nbv[j];
                if (u == v || !alive[u] || seen(u)) continue;
                float fsr = (float)((int)fsres[j] * 2 - q.sumq);
                // angular: absolute estimate (no vertex anchor); vertex-relative: add ev
                float est = tx[j] + dq[j] * q.width * fsr + vq[j] * q.vl;
                if (!angular_) est += ev;
                pool_insert(est, u);
            }
        }
        if (PROF) { uint64_t t2 = __rdtsc(); c_prep += (t1 - t0); c_loop += (t2 - t1);
            if (++c_n % 2000 == 0) std::fprintf(stderr, "[prof] cyc/q prep=%llu loop=%llu (prep %.1f%%)\n",
                (unsigned long long)(c_prep / c_n), (unsigned long long)(c_loop / c_n),
                100.0 * (double)c_prep / (double)(c_prep + c_loop)); }
        got.assign(res.begin(), res.end());   // already ascending by exact
    }

    void search(const float* q_raw, int k, int ef, std::vector<uint32_t>& out_ids) {
        std::vector<std::pair<float, uint32_t>> got;
        beam(q_raw, ef, got);
        out_ids.clear();
        for (auto& p : got) { out_ids.push_back(slot_id[p.second]); if ((int)out_ids.size() >= k) break; }
    }

    // PURE-ROUTING beam with exact-L2 result collection.
    // Glass LinearPool style sorted-vector cursor: L1-cache-friendly sequential insert+scan.
    // Full prefetch of ALL cache lines for next candidate's data (blocks + raw + neighbors).
    void beam_pure(const float* q_raw, int ef, std::vector<std::pair<float, uint32_t>>& pool_out) {
        pool_out.clear();
        if (nalive == 0) return;
        Q q; query_prepare(q_raw, q);
        const uint8_t* lut = s_lut.data();
        uint16_t fsres[128];  // up to R=128
        // Two VisitedSets (2048-slot open-addr hash each, 8KB each, L1-resident):
        //   s_vis  = expanded (visited) nodes — prevents re-expansion
        //   s_vis2 = pooled (seen) nodes     — prevents duplicate pool insertions
        // Without dedup, multi-seed init causes the same node to appear 10-20x in pool
        // (once per seed that has it as a neighbor), draining pool after ~33 hops instead of ef.
        if (s_vis.empty())  s_vis.assign(2048, ~0u);
        else std::memset(s_vis.data(),  0xFF, 2048 * sizeof(uint32_t));
        if (s_vis2.empty()) s_vis2.assign(2048, ~0u);
        else std::memset(s_vis2.data(), 0xFF, 2048 * sizeof(uint32_t));
        auto vis_contains = [&](uint32_t v) -> bool {
            uint32_t h = v & 2047u;
            while (s_vis[h] != ~0u) { if (s_vis[h] == v) return true; h = (h + 1) & 2047u; }
            return false;
        };
        auto vis_insert = [&](uint32_t v) {
            uint32_t h = v & 2047u;
            while (s_vis[h] != ~0u && s_vis[h] != v) h = (h + 1) & 2047u;
            s_vis[h] = v;
        };
        auto vis2_contains = [&](uint32_t v) -> bool {
            uint32_t h = v & 2047u;
            while (s_vis2[h] != ~0u) { if (s_vis2[h] == v) return true; h = (h + 1) & 2047u; }
            return false;
        };
        auto vis2_insert = [&](uint32_t v) {
            uint32_t h = v & 2047u;
            while (s_vis2[h] != ~0u && s_vis2[h] != v) h = (h + 1) & 2047u;
            s_vis2[h] = v;
        };
        auto& pool = s_pool; pool.clear();
        auto& res  = s_res;  res.clear();
        int cur = 0, n_visited = 0;
        auto pool_insert = [&](float key, uint32_t id) {
            if (vis2_contains(id)) return;  // already pooled — prevent duplicates
            if ((int)pool.size() >= ef && key >= pool.back().first) return;
            vis2_insert(id);
            int lo = 0, hi = (int)pool.size();
            while (lo < hi) { int m = (lo + hi) >> 1; if (pool[m].first < key) lo = m + 1; else hi = m; }
            pool.insert(pool.begin() + lo, {key, id});
            if ((int)pool.size() > ef) pool.pop_back();
            if (lo < cur) cur = lo;
        };
        // Initialization: scan seeds for top-C nearest entry points.
        // Greedy K-center seeds ensure diverse coverage of the data space, so top-C
        // nearest seeds give C independent footholds across the graph. C=5 keeps
        // vis2 load at 5*R/2048 ≈ 15% (safe). More than 5 starts risks vis2 overflow.
        if (!seeds.empty()) {
            s_sdists.clear();
            for (uint32_t s : seeds) {
                if (s < (uint32_t)nslot && alive[s])
                    s_sdists.emplace_back(exact_l2(s, q_raw), s);
            }
            if (!s_sdists.empty()) {
                const int C = std::min(5, (int)s_sdists.size());
                std::partial_sort(s_sdists.begin(), s_sdists.begin() + C, s_sdists.end());
                for (int i = 0; i < C; ++i) {
                    pool_insert(s_sdists[i].first, s_sdists[i].second);
                    res.push_back(s_sdists[i]);
                }
            }
        } else {
            uint32_t e = entry;
            if (e >= (uint32_t)nslot || !alive[e]) { for (int s = 0; s < nslot; ++s) if (alive[s]) { e = s; break; } }
            float d_seed = exact_l2(e, q_raw);
            pool_insert(d_seed, e);
            res.push_back({d_seed, e});
        }
        while (cur < (int)pool.size()) {
            uint32_t v = pool[cur].second;
            ++cur;
            if (vis_contains(v) || (has_deletions && !alive[v])) continue;
            if (n_visited >= ef) break;
            vis_insert(v);
            ++n_visited;
            float ev = exact_l2(v, q_raw);
            res.push_back({ev, v});
            const uint32_t* nbv = &nbr[(size_t)v * R];
            if (angular_) {
                // Angular (unit-sphere): exact L2 routing — 1-bit SNR ~0.24 is insufficient
                // for unit-sphere navigation; exact L2 gives correct routing at the cost of
                // ~R extra distance computations per visited node (still O(ef*R), not brute-force).
                if (cur < (int)pool.size()) {
                    uint32_t nx = pool[cur].second;
                    const char* rp = (const char*)&raw[(size_t)nx * q_dim];
                    for (size_t off = 0; off < (size_t)q_dim * 4; off += 64) _mm_prefetch(rp + off, _MM_HINT_T0);
                    _mm_prefetch((const char*)&nbr[(size_t)nx * R], _MM_HINT_T0);
                }
                for (int j = 0; j < R; ++j) {
                    uint32_t u = nbv[j];
                    if (u == v || vis_contains(u) || (has_deletions && !alive[u])) continue;
                    pool_insert(exact_l2(u, q_raw), u);
                }
            } else {
                // Euclidean: 1-bit vertex-relative routing with ev anchor (high SNR)
                if (cur < (int)pool.size()) {
                    uint32_t nx = pool[cur].second;
                    const char* bp = (const char*)&blocks[(size_t)nx * block_bytes];
                    for (size_t off = 0; off < block_bytes; off += 64) _mm_prefetch(bp + off, _MM_HINT_T0);
                    const char* rp = (const char*)&raw[(size_t)nx * q_dim];
                    for (size_t off = 0; off < (size_t)q_dim * 4; off += 64) _mm_prefetch(rp + off, _MM_HINT_T0);
                    _mm_prefetch((const char*)&trx[(size_t)nx * R], _MM_HINT_T0);
                    _mm_prefetch((const char*)&fdq[(size_t)nx * R], _MM_HINT_T0);
                    _mm_prefetch((const char*)&fvq[(size_t)nx * R], _MM_HINT_T0);
                    _mm_prefetch((const char*)&nbr[(size_t)nx * R], _MM_HINT_T0);
                }
                for (int blk = 0; blk < R; blk += 32)
                    rbfs::accumulate_block(D, &blocks[(size_t)v * block_bytes + (size_t)(blk / 32) * (size_t)D * 4],
                                           lut, fsres + blk);
                const float* tx = &trx[(size_t)v * R];
                const float* dq = &fdq[(size_t)v * R];
                const float* vq = &fvq[(size_t)v * R];
                for (int j = 0; j < R; ++j) {
                    uint32_t u = nbv[j];
                    if (u == v || vis_contains(u) || (has_deletions && !alive[u])) continue;
                    float fsr = (float)((int)fsres[j] * 2 - q.sumq);
                    pool_insert(tx[j] + dq[j] * q.width * fsr + vq[j] * q.vl + ev, u);
                }
            }
        }
        // Frontier: unvisited pool entries → exact L2 (collected, sorted once at end)
        for (auto& p : pool) {
            uint32_t u = p.second;
            if (vis_contains(u) || (has_deletions && !alive[u])) continue;
            vis_insert(u);
            res.push_back({exact_l2(u, q_raw), u});
        }
        std::sort(res.begin(), res.end());
        if ((int)res.size() > ef) res.resize(ef);
        pool_out.assign(res.begin(), res.end());
    }

    // search_pure: pool_out from beam_pure is exact-L2-sorted over all ef candidates (visited+frontier).
    // rerank_factor is kept for API compatibility; ignored since pool is already exact-ranked.
    void search_pure(const float* q_raw, int k, int ef, int /*rerank_factor*/, std::vector<uint32_t>& out_ids) {
        std::vector<std::pair<float, uint32_t>> pool;
        beam_pure(q_raw, ef, pool);
        out_ids.clear();
        for (auto& p : pool) { out_ids.push_back(slot_id[p.second]); if ((int)out_ids.size() >= k) break; }
    }

    // RobustPrune: cand = (dist-to-p, slot) sorted asc; keep <=R occlusion-pruned slots.
    void robust_prune(uint32_t p, std::vector<std::pair<float, uint32_t>>& cand,
                      std::vector<uint32_t>& kept) {
        kept.clear();
        std::sort(cand.begin(), cand.end());
        std::vector<uint8_t> dropped(cand.size(), 0);
        for (size_t i = 0; i < cand.size() && (int)kept.size() < R; ++i) {
            if (dropped[i]) continue;
            uint32_t s = cand[i].second;
            if (s == p || !alive[s]) continue;
            kept.push_back(s);
            for (size_t j = i + 1; j < cand.size(); ++j) {
                if (dropped[j]) continue;
                uint32_t t = cand[j].second;
                float d_st = exact_l2(s, &raw[(size_t)t * q_dim]);
                if (ALPHA * d_st < cand[j].first) dropped[j] = 1;  // s occludes t
            }
        }
    }

    void set_neighbors(uint32_t v, const std::vector<uint32_t>& ns) {
        int d = std::min<int>(R, (int)ns.size());
        for (int j = 0; j < d; ++j) nbr[(size_t)v * R + j] = ns[j];
        deg[v] = d;
        encode_row(v);
    }

    void add_backedge(uint32_t u, uint32_t v) {  // add v to u's neighbor list
        uint32_t* nbu = &nbr[(size_t)u * R];
        for (int j = 0; j < deg[u]; ++j) if (nbu[j] == v) return;  // exists
        if (deg[u] < R) { nbu[deg[u]++] = v; encode_row(u); return; }
        // full -> RobustPrune over existing + v
        std::vector<std::pair<float, uint32_t>> cand;
        for (int j = 0; j < deg[u]; ++j) cand.emplace_back(exact_l2(nbu[j], &raw[(size_t)u * q_dim]), nbu[j]);
        cand.emplace_back(exact_l2(v, &raw[(size_t)u * q_dim]), v);
        std::vector<uint32_t> kept; robust_prune(u, cand, kept);
        set_neighbors(u, kept);
    }

    void insert(uint32_t id, const float* vec) {
        if (id2slot.count(id)) return;
        uint32_t s = alloc_slot(id, vec);
        if (nalive == 1) { entry = s; deg[s] = 0; encode_row(s); return; }
        if (nalive <= R + 1) {  // bootstrap: connect to all existing alive
            std::vector<std::pair<float, uint32_t>> cand;
            for (int t = 0; t < nslot; ++t) if (alive[t] && (uint32_t)t != s)
                cand.emplace_back(exact_l2((uint32_t)t, vec), (uint32_t)t);
            std::vector<uint32_t> kept; robust_prune(s, cand, kept);
            set_neighbors(s, kept);
            for (uint32_t u : kept) add_backedge(u, s);
            return;
        }
        std::vector<std::pair<float, uint32_t>> got;
        beam(vec, 128, got);                 // ef_construction = 128
        std::vector<uint32_t> kept; robust_prune(s, got, kept);
        set_neighbors(s, kept);
        for (uint32_t u : kept) add_backedge(u, s);
    }

    void remove(uint32_t id) {
        auto it = id2slot.find(id);
        if (it == id2slot.end()) return;
        uint32_t p = it->second;
        // out-neighbors of p (for reconnection donors)
        std::vector<uint32_t> outn(nbr.begin() + (size_t)p * R, nbr.begin() + (size_t)p * R + deg[p]);
        // approximate in-neighbors: beam from p's own vector
        std::vector<std::pair<float, uint32_t>> got;
        beam(&raw[(size_t)p * q_dim], 128, got);
        // also include p's out-neighbors as candidate in-neighbors (graphs are ~symmetric here)
        std::vector<uint32_t> innbr;
        for (auto& pr : got) innbr.push_back(pr.second);
        for (uint32_t u : outn) innbr.push_back(u);
        // mark dead BEFORE repair so reconnection never re-points to p
        alive[p] = 0;
        has_deletions = true;
        for (uint32_t w : innbr) {
            if (w == p || !alive[w]) continue;
            uint32_t* nbw = &nbr[(size_t)w * R];
            int has = -1;
            for (int j = 0; j < deg[w]; ++j) if (nbw[j] == p) { has = j; break; }
            if (has < 0) continue;
            // remove p from w
            for (int j = has; j < deg[w] - 1; ++j) nbw[j] = nbw[j + 1];
            deg[w]--;
            // reconnect w to up to c=3 closest of p's out-neighbors not already in w
            std::vector<std::pair<float, uint32_t>> donors;
            for (uint32_t o : outn) {
                if (o == w || !alive[o]) continue;
                bool dup = false;
                for (int j = 0; j < deg[w]; ++j) if (nbw[j] == o) { dup = true; break; }
                if (!dup) donors.emplace_back(exact_l2(o, &raw[(size_t)w * q_dim]), o);
            }
            std::sort(donors.begin(), donors.end());
            for (int c = 0; c < 3 && c < (int)donors.size() && deg[w] < R; ++c)
                nbw[deg[w]++] = donors[c].second;
            encode_row(w);
        }
        // free the slot
        id2slot.erase(it);
        slot_id[p] = 0xffffffffu;
        freelist.push_back(p);
        --nalive;
        if (entry == p) { for (int s = 0; s < nslot; ++s) if (alive[s]) { entry = s; break; } }
    }

    // ---- parallel α-RNG bulk builder (SymphonyQG qg_builder-style, Apache-2.0 adapted) ----
    // ~100x faster than incremental build() at 1M scale + higher recall (proper occlusion
    // over t iterations + reverse edges). Uses std::thread (no -fopenmp needed). The
    // incremental build()/insert()/remove() remain for dynamic churn; build_parallel is the
    // initial bulk build only.
    void set_entry_medoid() {
        std::vector<float> mean(q_dim, 0.f);
        for (int s = 0; s < nslot; ++s) if (alive[s])
            for (int d = 0; d < q_dim; ++d) mean[d] += raw[(size_t)s * q_dim + d];
        for (int d = 0; d < q_dim; ++d) mean[d] /= (float)std::max(1, nalive);
        float best = 1e30f; uint32_t bs = entry;
        for (int s = 0; s < nslot; ++s) if (alive[s]) { float e = exact_l2((uint32_t)s, mean.data()); if (e < best) { best = e; bs = (uint32_t)s; } }
        entry = bs;
    }

    // Build sqrt(N) navigation seeds via greedy K-center (furthest-point) sampling.
    // Random seeds have no coverage guarantee on high-dimensional unit spheres (concentration
    // of measure makes all points ~equidistant). K-center greedily maximizes the minimum
    // distance between seeds, ensuring diverse entry points that cover the data space.
    // O(N * K) time — ~0.2s for N=100K, ~0.8s for N=290K at D=256.
    void build_seeds(int n_seeds = -1) {
        if (n_seeds < 0) n_seeds = (int)std::ceil(std::sqrt((double)nalive));
        seeds.clear();
        if (nalive == 0) return;
        std::vector<uint32_t> alive_slots;
        alive_slots.reserve(nalive);
        for (int s = 0; s < nslot; ++s) if (alive[s]) alive_slots.push_back(s);
        if ((int)alive_slots.size() <= n_seeds) { seeds = alive_slots; return; }
        // Greedy furthest-point sampling: start from medoid, each next seed is the
        // point farthest from any existing seed.
        seeds.reserve(n_seeds);
        uint32_t start = (entry < (uint32_t)nslot && alive[entry]) ? entry : alive_slots[0];
        seeds.push_back(start);
        std::vector<float> min_dist(nslot, std::numeric_limits<float>::max());
        const float* rs0 = &raw[(size_t)start * q_dim];
        for (uint32_t v : alive_slots) min_dist[v] = exact_l2(v, rs0);
        while ((int)seeds.size() < n_seeds) {
            uint32_t farthest = alive_slots[0]; float fd = 0.f;
            for (uint32_t v : alive_slots) { if (min_dist[v] > fd) { fd = min_dist[v]; farthest = v; } }
            seeds.push_back(farthest);
            const float* rs = &raw[(size_t)farthest * q_dim];
            for (uint32_t v : alive_slots) { float d = exact_l2(v, rs); if (d < min_dist[v]) min_dist[v] = d; }
        }
    }

    // exact-distance greedy beam over the CURRENT adjacency; thread-safe via caller scratch
    // (vis sized nslot, vep a per-thread counter). Used only during parallel build.
    // start_entry: optional override for the search entry point (~0u = use global medoid entry).
    void beam_exact(const float* q, int ef, std::vector<uint32_t>& vis, uint32_t& vep,
                    std::vector<std::pair<float, uint32_t>>& got, uint32_t start_entry = ~0u) const {
        got.clear();
        if (nalive == 0) return;
        if (++vep == 0) { std::fill(vis.begin(), vis.end(), 0u); vep = 1; }
        using P = std::pair<float, uint32_t>;
        std::priority_queue<P, std::vector<P>, std::greater<P>> bm;
        std::priority_queue<P> result;
        uint32_t e = (start_entry < (uint32_t)nslot && alive[start_entry]) ? start_entry : entry;
        if (e >= (uint32_t)nslot || !alive[e]) { for (int s = 0; s < nslot; ++s) if (alive[s]) { e = (uint32_t)s; break; } }
        bm.emplace(exact_l2(e, q), e);
        int nvis = 0; const int vcap = ef * 4;  // bound early-iteration over-exploration
        while (!bm.empty()) {
            P t = bm.top(); bm.pop();
            if ((int)result.size() >= ef && t.first > result.top().first) break;
            uint32_t v = t.second;
            if (vis[v] == vep || !alive[v]) continue;
            vis[v] = vep;
            result.emplace(t.first, v);
            if ((int)result.size() > ef) result.pop();
            const uint32_t* nbv = &nbr[(size_t)v * R];
            int dg = deg[v];
            for (int j = 0; j < dg; ++j) {
                uint32_t u = nbv[j];
                if (u == v || u >= (uint32_t)nslot || vis[u] == vep || !alive[u]) continue;
                bm.emplace(exact_l2(u, q), u);
            }
            if (++nvis >= vcap) break;
        }
        while (!result.empty()) { got.push_back(result.top()); result.pop(); }
        std::sort(got.begin(), got.end());
    }

    void build_parallel(const float* data, int n, const uint32_t* ids,
                        int iters = 3, int L_build = 128, int nthreads = 0) {
        if (nthreads <= 0) nthreads = (int)std::max(1u, std::thread::hardware_concurrency());
        std::printf("[build_parallel] N=%d R=%d iters=%d L=%d nthreads=%d\n",
                    n, R, iters, L_build, nthreads);
        std::fflush(stdout);
        reserve_slots(n + 64);
        for (int i = 0; i < n; ++i) alloc_slot(ids[i], data + (size_t)i * q_dim);
        const int NS = nslot;
        auto run = [&](auto fn) {
            std::vector<std::thread> th;
            for (int t = 0; t < nthreads; ++t) th.emplace_back(fn, t);
            for (auto& x : th) x.join();
        };
        // random init: R distinct random neighbors per vertex
        run([&](int tid) {
            std::mt19937 rng(0x9e3779b9u + (uint32_t)tid * 2654435761u);
            for (int v = tid; v < NS; v += nthreads) {
                if (!alive[v]) continue;
                uint32_t* nbv = &nbr[(size_t)v * R];
                int d = 0, guard = 0;
                while (d < R && NS > 1 && guard++ < R * 20) {
                    uint32_t u = rng() % (uint32_t)NS;
                    if (u == (uint32_t)v || !alive[u]) continue;
                    bool dup = false; for (int j = 0; j < d; ++j) if (nbv[j] == u) { dup = true; break; }
                    if (!dup) nbv[d++] = u;
                }
                deg[v] = d;
            }
        });
        set_entry_medoid();
        // flat contiguous newedges arrays — one alloc per build, reused across iters
        std::vector<uint32_t> newedges_flat((size_t)NS * R);
        std::vector<int>      newedges_deg(NS, 0);
        for (int it = 0; it < iters; ++it) {
            std::printf("[build_parallel] iter %d/%d\n", it + 1, iters); std::fflush(stdout);
            std::fill(newedges_deg.begin(), newedges_deg.end(), 0);
            run([&](int tid) {
                std::mt19937 rng_b(0x9e3779b9u ^ ((uint32_t)tid * 2654435761u));
                std::vector<uint32_t> vis((size_t)NS, 0u); uint32_t vep = 0;
                std::vector<std::pair<float, uint32_t>> got; std::vector<uint32_t> kept;
                for (int v = tid; v < NS; v += nthreads) {
                    if (!alive[v]) continue;
                    // Random entry point per vertex — avoids medoid-centric bias on unit-sphere
                    // graphs where all nodes are ~equidistant from medoid, causing beam_exact
                    // to never navigate toward v's true neighbors.
                    uint32_t rstart = (uint32_t)(rng_b() % (uint32_t)NS);
                    while (!alive[rstart]) rstart = (rstart + 1) % (uint32_t)NS;
                    beam_exact(&raw[(size_t)v * q_dim], L_build, vis, vep, got, rstart);
                    if (got.size() > 128) got.resize(128);   // cap RobustPrune O(cand^2)
                    robust_prune((uint32_t)v, got, kept);
                    int d = std::min(R, (int)kept.size());
                    uint32_t* ne = &newedges_flat[(size_t)v * R];
                    for (int j = 0; j < d; ++j) ne[j] = kept[j];
                    newedges_deg[v] = d;
                }
            });
            for (int v = 0; v < NS; ++v) if (alive[v]) {
                int d = newedges_deg[v];
                const uint32_t* ne = &newedges_flat[(size_t)v * R];
                for (int j = 0; j < d; ++j) nbr[(size_t)v * R + j] = ne[j];
                deg[v] = d;
            }
            // reverse edges: each v->u induces a candidate u->v; merge + RobustPrune u to R
            std::vector<std::vector<uint32_t>> rev((size_t)NS);
            for (int v = 0; v < NS; ++v) if (alive[v])
                for (int j = 0; j < deg[v]; ++j) rev[nbr[(size_t)v * R + j]].push_back((uint32_t)v);
            for (int u = 0; u < NS; ++u) {
                if (!alive[u] || rev[u].empty()) continue;
                std::vector<std::pair<float, uint32_t>> cand;
                const float* ru = &raw[(size_t)u * q_dim];
                for (int j = 0; j < deg[u]; ++j) { uint32_t x = nbr[(size_t)u * R + j]; cand.emplace_back(exact_l2(x, ru), x); }
                for (uint32_t v : rev[u]) {
                    if (v == (uint32_t)u || !alive[v]) continue;
                    bool dup = false; for (auto& c : cand) if (c.second == v) { dup = true; break; }
                    if (!dup) cand.emplace_back(exact_l2(v, ru), v);
                }
                std::vector<uint32_t> kept; robust_prune((uint32_t)u, cand, kept);
                int d = std::min<int>(R, (int)kept.size());
                for (int j = 0; j < d; ++j) nbr[(size_t)u * R + j] = kept[j];
                deg[u] = d;
            }
        }
        run([&](int tid) { for (int v = tid; v < NS; v += nthreads) if (alive[v]) encode_row((uint32_t)v); });
    }

    // bulk build via incremental insert (Vamana build = incremental). ids = external ids.
    void build(const float* data, int n, const uint32_t* ids) {
        reserve_slots(n + n / 4 + 64);
        for (int i = 0; i < n; ++i) insert(ids[i], data + (size_t)i * q_dim);
        // set medoid entry = nearest alive to global mean
        std::vector<float> mean(q_dim, 0.f);
        for (int s = 0; s < nslot; ++s) if (alive[s])
            for (int d = 0; d < q_dim; ++d) mean[d] += raw[(size_t)s * q_dim + d];
        for (int d = 0; d < q_dim; ++d) mean[d] /= (float)std::max(1, nalive);
        float best = 1e30f; uint32_t bs = entry;
        for (int s = 0; s < nslot; ++s) if (alive[s]) { float e = exact_l2((uint32_t)s, mean.data()); if (e < best) { best = e; bs = s; } }
        entry = bs;
    }

    size_t bytes_per_vertex() const { return block_bytes + (size_t)R * (3 * 4 + 4) + (size_t)q_dim * 4 + (size_t)D * 4; }

    void save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "[RaBitQGraph::save] cannot open %s\n", path.c_str()); return; }
        auto W = [&](const void* p, size_t n) { std::fwrite(p, 1, n, f); };
        const char magic[8] = {'R','B','Q','G','v','2','\0','\0'};
        W(magic, 8);
        int32_t hdr[6] = {R, q_dim, D, nslot, nalive, (int32_t)angular_};
        W(hdr, sizeof(hdr));
        W(&entry, sizeof(entry));
        // SRHT diagonal
        const auto& diag = srht.diag();
        int32_t dsz = (int32_t)diag.size();
        W(&dsz, sizeof(dsz));
        W(diag.data(), (size_t)dsz * sizeof(float));
        // SoA arrays
        W(raw.data(),      (size_t)nslot * q_dim  * sizeof(float));
        W(vrot.data(),     (size_t)nslot * D       * sizeof(float));
        W(blocks.data(),   (size_t)nslot * block_bytes);
        W(trx.data(),      (size_t)nslot * R       * sizeof(float));
        W(fdq.data(),      (size_t)nslot * R       * sizeof(float));
        W(fvq.data(),      (size_t)nslot * R       * sizeof(float));
        W(nbr.data(),      (size_t)nslot * R       * sizeof(uint32_t));
        W(deg.data(),      (size_t)nslot            * sizeof(int));
        W(alive.data(),    (size_t)nslot            * sizeof(uint8_t));
        W(slot_id.data(),  (size_t)nslot            * sizeof(uint32_t));
        // id2slot map
        uint32_t msz = (uint32_t)id2slot.size();
        W(&msz, sizeof(msz));
        for (auto& kv : id2slot) { W(&kv.first, sizeof(uint32_t)); W(&kv.second, sizeof(uint32_t)); }
        std::fclose(f);
        std::printf("[RaBitQGraph::save] wrote %s\n", path.c_str());
    }

    // Construct a graph object from a saved file (no seed needed — srht diag is stored directly).
    static RaBitQGraph load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "[RaBitQGraph::load] cannot open %s\n", path.c_str()); std::abort(); }
        auto R2 = [&](void* p, size_t n) { if (std::fread(p, 1, n, f) != n) { std::fprintf(stderr,"[load] read error\n"); std::abort(); } };
        char magic[8]; R2(magic, 8);
        int R_, qdim, D_, ns, na; bool ang;
        if (std::memcmp(magic, "RBQGv1\0\0", 8) == 0) {
            int32_t hdr[5]; R2(hdr, sizeof(hdr));
            R_=hdr[0]; qdim=hdr[1]; D_=hdr[2]; ns=hdr[3]; na=hdr[4]; ang=false;
        } else if (std::memcmp(magic, "RBQGv2\0\0", 8) == 0) {
            int32_t hdr[6]; R2(hdr, sizeof(hdr));
            R_=hdr[0]; qdim=hdr[1]; D_=hdr[2]; ns=hdr[3]; na=hdr[4]; ang=(bool)hdr[5];
        } else { std::fprintf(stderr,"[load] bad magic\n"); std::abort(); }
        uint32_t ent; R2(&ent, sizeof(ent));
        int32_t dsz; R2(&dsz, sizeof(dsz));
        std::vector<float> diag_v((size_t)dsz); R2(diag_v.data(), (size_t)dsz * sizeof(float));
        // Reconstruct object with correct SRHT diagonal loaded from file.
        RaBitQGraph g(qdim, 0, R_, ang);
        g.srht = NGT::NGTAQ::SRHT(std::move(diag_v));
        g.nslot = ns; g.nalive = na; g.entry = ent;
        g.reserve_slots(ns);
        R2(g.raw.data(),     (size_t)ns * qdim * sizeof(float));
        R2(g.vrot.data(),    (size_t)ns * D_   * sizeof(float));
        R2(g.blocks.data(),  (size_t)ns * g.block_bytes);
        R2(g.trx.data(),     (size_t)ns * R_   * sizeof(float));
        R2(g.fdq.data(),     (size_t)ns * R_   * sizeof(float));
        R2(g.fvq.data(),     (size_t)ns * R_   * sizeof(float));
        R2(g.nbr.data(),     (size_t)ns * R_   * sizeof(uint32_t));
        R2(g.deg.data(),     (size_t)ns         * sizeof(int));
        R2(g.alive.data(),   (size_t)ns         * sizeof(uint8_t));
        R2(g.slot_id.data(), (size_t)ns         * sizeof(uint32_t));
        uint32_t msz; R2(&msz, sizeof(msz));
        for (uint32_t i = 0; i < msz; ++i) {
            uint32_t k, v; R2(&k, sizeof(k)); R2(&v, sizeof(v));
            g.id2slot[k] = v;
        }
        std::fclose(f);
        std::printf("[RaBitQGraph::load] loaded %s (N=%d R=%d)\n", path.c_str(), ns, R_);
        return g;
    }
};

}  // namespace NGTAQ
