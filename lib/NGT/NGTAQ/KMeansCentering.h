#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#if defined(__AVX2__) || defined(__AVX__)
#  include <immintrin.h>
#endif
#include "SIMDUtils.h"

namespace NGT { namespace NGTAQ {

// K = clamp(N/1000, 256, 4,000,000)
inline uint32_t select_k(uint64_t N) {
    uint64_t k = N / 1000;
    if (k < 256)      k = 256;
    if (k > 4000000)  k = 4000000;
    return (uint32_t)k;
}

class KMeansCentering {
public:
    KMeansCentering(uint32_t K, int D, uint64_t seed)
        : K_(K), D_(D), seed_(seed)
    {
        centroids_.resize((size_t)K * D, 0.f);
    }

    // Train: subsample up to max_train_pts points, run K-means until convergence
    void train(const float* data, size_t N, size_t max_train_pts = 262144, int max_iter = 25) {
        size_t n = std::min(N, max_train_pts);
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        if (n < N) {
            std::mt19937_64 rng(seed_);
            for (size_t i = 0; i < n; ++i) {
                size_t j = i + (rng() % (N - i));
                std::swap(idx[i], idx[j]);
            }
            idx.resize(n);
        }

        // Random centroid initialization
        std::mt19937_64 rng2(seed_ ^ 0xDEADBEEF);
        std::uniform_int_distribution<size_t> unif(0, n - 1);
        for (uint32_t k = 0; k < K_; ++k) {
            size_t src = idx[unif(rng2)];
            memcpy(centroids_.data() + k * D_, data + src * D_, D_ * sizeof(float));
        }

        std::vector<uint32_t> assignments(n, 0);
        std::vector<float>    new_centroids((size_t)K_ * D_);
        std::vector<uint32_t> counts(K_);

        for (int iter = 0; iter < max_iter; ++iter) {
            bool changed = false;
            for (size_t i = 0; i < n; ++i) {
                uint32_t best = nearest(data + idx[i] * D_);
                if (best != assignments[i]) { assignments[i] = best; changed = true; }
            }
            if (!changed && iter > 0) break;

            std::fill(new_centroids.begin(), new_centroids.end(), 0.f);
            std::fill(counts.begin(), counts.end(), 0u);
            for (size_t i = 0; i < n; ++i) {
                uint32_t k = assignments[i];
                float* c = new_centroids.data() + k * D_;
                const float* v = data + idx[i] * D_;
                for (int d = 0; d < D_; ++d) c[d] += v[d];
                ++counts[k];
            }
            for (uint32_t k = 0; k < K_; ++k) {
                if (counts[k] == 0) continue;
                float inv = 1.f / counts[k];
                float* c = new_centroids.data() + k * D_;
                for (int d = 0; d < D_; ++d) c[d] *= inv;
            }
            centroids_.swap(new_centroids);
        }
    }

    // 1-pass assignment: find nearest centroid for each of N vectors
    void assign(const float* data, size_t N, uint32_t* ids) const {
        for (size_t i = 0; i < N; ++i)
            ids[i] = nearest(data + i * D_);
    }

    // Compute residual: out[d] = x[d] - centroids[id][d]
    void get_residual(const float* x, uint32_t id, float* out) const {
        const float* c = centroids_.data() + id * D_;
#if defined(__AVX2__)
        if (D_ % 8 == 0) {
            for (int d = 0; d < D_; d += 8)
                _mm256_storeu_ps(out + d,
                    _mm256_sub_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(c + d)));
            return;
        }
#endif
        for (int d = 0; d < D_; ++d) out[d] = x[d] - c[d];
    }

    const float* centroid(uint32_t id) const {
        return centroids_.data() + id * D_;
    }

    // Public wrapper for query-time nearest centroid lookup
    uint32_t nearest_public(const float* x) const {
        return nearest(x);
    }

    // Approximate nearest centroid via an fp16 centroid cache. The full-fp32 scan over
    // K*D floats streams 2x the bytes (K=1000,D=128 -> 512KB) and is memory-bandwidth
    // bound (~12us/query measured). For SEED-CLUSTER selection the exact nearest is not
    // required — the graph walk corrects any near-tie — so an fp16 scan (256KB, half the
    // bandwidth) is sufficient and ~2x faster. Cache is built lazily on first use.
    // Reuses the F16C l2_sq_f32_fp16 kernel already used by the exact rerank.
    uint32_t nearest_fp16(const float* x) const {
        if (centroids_fp16_.size() != (size_t)K_ * D_) build_fp16_cache();
        float best_dist = std::numeric_limits<float>::max();
        uint32_t best = 0;
        const uint16_t* cp = centroids_fp16_.data();
        constexpr int PREFETCH_DIST = 24;
        for (uint32_t k = 0; k < K_; ++k) {
            if (k + PREFETCH_DIST < K_)
                __builtin_prefetch(cp + (size_t)(k + PREFETCH_DIST) * D_, 0, 1);
            float d = NGT::NGTAQ::l2_sq_f32_fp16(x, cp + (size_t)k * D_, D_);
            if (d < best_dist) { best_dist = d; best = k; }
        }
        return best;
    }

    // ── 2-level coarse quantizer over the K centroids ─────────────────────────────────
    // Replaces the brute-force K-way query->centroid scan (the ~9us setup floor for the
    // batch path) with: scan S super-centroids (S≈sqrt(K)), then exactly scan only the
    // centroids assigned to the best `probe` super-centroids. Cost ≈ S + probe*(K/S)
    // distance evals vs K. For K=1000, S=32, probe=4: ~32+128=160 vs 1000 (~6x fewer).
    //
    // Accuracy: probing the top-`probe` super-centroids (not just 1) makes the true
    // nearest centroid almost always present in the candidate set; the final exact pass
    // over those members guarantees the best among them. Built lazily on first use from
    // the in-memory centroids (no serialization / format change).
    uint32_t nearest_2level(const float* x, int probe = 4) const {
        if (super_centroids_.empty()) build_2level();
        if (S_ == 0) return nearest_fp16(x);  // degenerate (tiny K) → fall back
        // 1. Scan S super-centroids, keep the top-`probe` nearest (small partial select).
        constexpr int MAXP = 32;
        int P = probe < 1 ? 1 : (probe > MAXP ? MAXP : probe);
        if (P > (int)S_) P = (int)S_;
        // best-`P` super-centroids by distance: simple insertion into a tiny array.
        float bestd[MAXP]; uint32_t besti[MAXP];
        for (int i = 0; i < P; ++i) { bestd[i] = std::numeric_limits<float>::max(); besti[i] = 0; }
        for (uint32_t s = 0; s < S_; ++s) {
            float d = NGT::NGTAQ::l2_sq(x, super_centroids_.data() + (size_t)s * D_, D_);
            if (d < bestd[P - 1]) {
                int j = P - 1;
                while (j > 0 && bestd[j - 1] > d) { bestd[j] = bestd[j - 1]; besti[j] = besti[j - 1]; --j; }
                bestd[j] = d; besti[j] = s;
            }
        }
        // 2. Exact scan over the members of the top-`probe` super-centroids.
        float best_dist = std::numeric_limits<float>::max();
        uint32_t best = 0;
        for (int i = 0; i < P; ++i) {
            uint32_t s = besti[i];
            const uint32_t lo = super_off_[s], hi = super_off_[s + 1];
            for (uint32_t m = lo; m < hi; ++m) {
                uint32_t c = super_members_[m];
                float d = NGT::NGTAQ::l2_sq(x, centroids_.data() + (size_t)c * D_, D_);
                if (d < best_dist) { best_dist = d; best = c; }
            }
        }
        return best;
    }

    // Eagerly build the 2-level accelerator (call once at index init, under a guard, so
    // the lazy in-nearest_2level build never races across query threads).
    void buildCoarseQuantizer() const { if (super_centroids_.empty()) build_2level(); }

    uint32_t num_clusters() const { return K_; }
    int dim() const { return D_; }

    const std::vector<float>& centroids_data() const { return centroids_; }
    void set_centroids(std::vector<float>&& c) { centroids_ = std::move(c); }

    // AVX-512F/AVX2/scalar unified L2² — dispatched via SIMDUtils.h
    static float l2sq(const float* __restrict__ a, const float* __restrict__ b, int D) {
        return NGT::NGTAQ::l2_sq(a, b, D);
    }

private:
    uint32_t K_;
    int D_;
    uint64_t seed_;
    std::vector<float> centroids_;
    mutable std::vector<uint16_t> centroids_fp16_;  // lazy fp16 copy for nearest_fp16

    // 2-level coarse quantizer state (lazy, in-memory only).
    mutable uint32_t              S_ = 0;            // number of super-centroids
    mutable std::vector<float>    super_centroids_; // [S_*D_] super-centroid vectors
    mutable std::vector<uint32_t> super_off_;       // [S_+1] CSR offsets into super_members_
    mutable std::vector<uint32_t> super_members_;   // [K_] centroid ids grouped by super-centroid

    // Build the fp16 centroid cache from the fp32 centroids (one-time, query-thread safe:
    // idempotent full overwrite; concurrent builders write identical bytes).
    void build_fp16_cache() const {
        std::vector<uint16_t> tmp((size_t)K_ * D_);
        for (size_t i = 0; i < tmp.size(); ++i)
            tmp[i] = NGT::NGTAQ::float_to_fp16(centroids_[i]);
        centroids_fp16_.swap(tmp);
    }

    // Build the 2-level coarse quantizer: a small k-means over the K centroids producing
    // S≈sqrt(K) super-centroids, plus a CSR grouping of centroid ids by super-centroid.
    // Idempotent full overwrite; the in-memory structure is rebuilt from centroids_ only.
    void build_2level() const {
        if (K_ < 64) { S_ = 0; return; }  // too few centroids to bother → caller falls back
        uint32_t S = (uint32_t)std::lround(std::sqrt((double)K_));
        if (S < 8) S = 8; if (S > 256) S = 256;
        // k-means++ style init via stride sampling (deterministic), then a few Lloyd iters.
        std::vector<float> sc((size_t)S * D_);
        for (uint32_t s = 0; s < S; ++s) {
            uint32_t src = (uint32_t)((uint64_t)s * K_ / S);
            std::memcpy(sc.data() + (size_t)s * D_, centroids_.data() + (size_t)src * D_,
                        (size_t)D_ * sizeof(float));
        }
        std::vector<uint32_t> assign(K_, 0);
        std::vector<float> newc((size_t)S * D_);
        std::vector<uint32_t> cnt(S);
        for (int iter = 0; iter < 8; ++iter) {
            bool changed = false;
            for (uint32_t c = 0; c < K_; ++c) {
                const float* cv = centroids_.data() + (size_t)c * D_;
                float bd = std::numeric_limits<float>::max(); uint32_t bs = 0;
                for (uint32_t s = 0; s < S; ++s) {
                    float d = NGT::NGTAQ::l2_sq(cv, sc.data() + (size_t)s * D_, D_);
                    if (d < bd) { bd = d; bs = s; }
                }
                if (assign[c] != bs) { assign[c] = bs; changed = true; }
            }
            if (!changed && iter > 0) break;
            std::fill(newc.begin(), newc.end(), 0.f);
            std::fill(cnt.begin(), cnt.end(), 0u);
            for (uint32_t c = 0; c < K_; ++c) {
                float* acc = newc.data() + (size_t)assign[c] * D_;
                const float* cv = centroids_.data() + (size_t)c * D_;
                for (int d = 0; d < D_; ++d) acc[d] += cv[d];
                ++cnt[assign[c]];
            }
            for (uint32_t s = 0; s < S; ++s) {
                if (cnt[s] == 0) {  // empty super-cluster: reseat on a far centroid
                    std::memcpy(newc.data() + (size_t)s * D_,
                                centroids_.data() + (size_t)(s % K_) * D_,
                                (size_t)D_ * sizeof(float));
                    continue;
                }
                float inv = 1.f / cnt[s];
                float* acc = newc.data() + (size_t)s * D_;
                for (int d = 0; d < D_; ++d) acc[d] *= inv;
            }
            sc.swap(newc);
        }
        // CSR grouping by super-centroid (final assignment).
        std::vector<uint32_t> off(S + 1, 0);
        for (uint32_t c = 0; c < K_; ++c) ++off[assign[c] + 1];
        for (uint32_t s = 0; s < S; ++s) off[s + 1] += off[s];
        std::vector<uint32_t> mem(K_);
        std::vector<uint32_t> cur(off.begin(), off.end() - 1);
        for (uint32_t c = 0; c < K_; ++c) mem[cur[assign[c]]++] = c;
        super_centroids_.swap(sc);
        super_off_.swap(off);
        super_members_.swap(mem);
        S_ = S;
    }

    uint32_t nearest(const float* x) const {
        float best_dist = std::numeric_limits<float>::max();
        uint32_t best = 0;
        const float* cp = centroids_.data();
        // Prefetch next centroid while computing current one to hide L3 latency.
        // PREFETCH_DIST=24: AVX2 compute per centroid ≈ 8 cycles @ 3GHz ≈ 2.7ns.
        // L3 latency ≈ 90 cycles ≈ 30ns → need DIST=90/8=11 centroids ahead minimum.
        // Use 24 for margin. Issue 2 cache lines (128B) to cover the start of each
        // centroid; the hardware stride prefetcher handles the remaining 6 lines.
        constexpr int PREFETCH_DIST = 24;
        for (uint32_t k = 0; k < K_; ++k) {
            if (k + PREFETCH_DIST < K_) {
                const float* pp = cp + (k + PREFETCH_DIST) * D_;
                __builtin_prefetch(pp,      0, 1);  // cache line 0 (bytes   0- 63)
                __builtin_prefetch(pp + 16, 0, 1);  // cache line 1 (bytes  64-127)
            }
            float d = l2sq(x, cp + k * D_, D_);
            if (d < best_dist) { best_dist = d; best = k; }
        }
        return best;
    }
};

}} // NGT::NGTAQ
