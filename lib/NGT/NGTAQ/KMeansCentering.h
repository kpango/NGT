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

    // AVX2-accelerated L2 squared distance — public so call sites can reuse it.
    // Falls back to scalar for the tail when D % 8 != 0.
    static float l2sq(const float* __restrict__ a, const float* __restrict__ b, int D) {
#if defined(__AVX2__)
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 16 <= D; i += 16) {
            __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a+i),   _mm256_loadu_ps(b+i));
            __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a+i+8), _mm256_loadu_ps(b+i+8));
            s0 = _mm256_fmadd_ps(d0, d0, s0);
            s1 = _mm256_fmadd_ps(d1, d1, s1);
        }
        for (; i + 8 <= D; i += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i));
            s0 = _mm256_fmadd_ps(d, d, s0);
        }
        __m256 acc = _mm256_add_ps(s0, s1);
        __m128 lo  = _mm256_castps256_ps128(acc);
        __m128 hi  = _mm256_extractf128_ps(acc, 1);
        __m128 s   = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        float r = _mm_cvtss_f32(s);
        for (; i < D; ++i) { float d = a[i]-b[i]; r += d*d; }
        return r;
#else
        float d = 0.f;
        for (int i = 0; i < D; ++i) { float diff = a[i]-b[i]; d += diff*diff; }
        return d;
#endif
    }

private:
    uint32_t K_;
    int D_;
    uint64_t seed_;
    std::vector<float> centroids_;

    uint32_t nearest(const float* x) const {
        float best_dist = std::numeric_limits<float>::max();
        uint32_t best = 0;
        for (uint32_t k = 0; k < K_; ++k) {
            const float* c = centroids_.data() + k * D_;
            float d = l2sq(x, c, D_);
            if (d < best_dist) { best_dist = d; best = k; }
        }
        return best;
    }
};

}} // NGT::NGTAQ
