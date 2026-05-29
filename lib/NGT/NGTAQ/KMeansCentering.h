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
