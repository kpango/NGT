#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <random>

namespace NGT { namespace NGTAQ {

// Top-K PCA via covariance matrix (D×D) + deflated power iteration eigensolver
// Suitable for D ≤ 512. For D=128, K=32: covariance is 128×128 = 65KB.
class PCAProjector {
public:
    PCAProjector(int D, int top_k, uint64_t seed, bool whiten = false)
        : D_(D), top_k_(top_k), seed_(seed), whiten_(whiten), fitted_(false)
    {
        components_.resize((size_t)top_k * D, 0.f);
        mean_.resize(D, 0.f);
        eigenvalues_.resize(top_k, 0.f);
    }

    void fit(const float* data, size_t N) {
        assert(N > (size_t)D_);

        // Compute mean
        std::fill(mean_.begin(), mean_.end(), 0.f);
        for (size_t i = 0; i < N; ++i)
            for (int d = 0; d < D_; ++d)
                mean_[d] += data[i*D_+d];
        for (int d = 0; d < D_; ++d) mean_[d] /= (float)N;

        // Compute covariance matrix D×D (upper triangle, then mirror)
        std::vector<double> cov((size_t)D_ * D_, 0.0);
        for (size_t i = 0; i < N; ++i) {
            for (int a = 0; a < D_; ++a) {
                float xa = data[i*D_+a] - mean_[a];
                for (int b = a; b < D_; ++b) {
                    cov[a*D_+b] += xa * (double)(data[i*D_+b] - mean_[b]);
                }
            }
        }
        double inv_N = 1.0 / N;
        for (int a = 0; a < D_; ++a)
            for (int b = a; b < D_; ++b) {
                cov[a*D_+b] *= inv_N;
                cov[b*D_+a]  = cov[a*D_+b];
            }

        // Extract top_k eigenvectors via deflated power iteration
        std::mt19937_64 rng(seed_);
        std::vector<float> v(D_), Av(D_);

        for (int k = 0; k < top_k_; ++k) {
            // Random init
            for (int d = 0; d < D_; ++d)
                v[d] = (float)((int)(rng() % 1000) - 500);
            normalize(v);

            float prev_lambda = -1.f;
            for (int iter = 0; iter < 300; ++iter) {
                // Av = cov * v
                for (int d = 0; d < D_; ++d) {
                    double s = 0.0;
                    for (int j = 0; j < D_; ++j) s += cov[d*D_+j] * v[j];
                    Av[d] = (float)s;
                }
                // Gram-Schmidt deflation: subtract projections onto already-found eigenvecs
                for (int j = 0; j < k; ++j) {
                    float dot = 0.f;
                    const float* ej = components_.data() + j * D_;
                    for (int d = 0; d < D_; ++d) dot += Av[d] * ej[d];
                    for (int d = 0; d < D_; ++d) Av[d] -= dot * ej[d];
                }
                float lambda = norm_vec(Av);
                if (lambda < 1e-12f) break;
                for (int d = 0; d < D_; ++d) v[d] = Av[d] / lambda;
                // Check convergence
                if (std::abs(lambda - prev_lambda) < 1e-6f * lambda) break;
                prev_lambda = lambda;
            }
            eigenvalues_[k] = norm_vec(Av);
            normalize(v);
            memcpy(components_.data() + k * D_, v.data(), D_ * sizeof(float));
        }
        fitted_ = true;
    }

    // Project x (D-dim) to top_k-dim output
    void project(const float* x, float* out) const {
        assert(fitted_);
        for (int k = 0; k < top_k_; ++k) {
            float dot = 0.f;
            const float* comp = components_.data() + k * D_;
            for (int d = 0; d < D_; ++d) dot += (x[d] - mean_[d]) * comp[d];
            if (whiten_ && eigenvalues_[k] > 1e-10f)
                dot /= std::sqrt(eigenvalues_[k]);
            out[k] = dot;
        }
    }

    float variance_ratio(int k) const {
        float total = 0.f;
        for (float v : eigenvalues_) total += v;
        return (total > 0.f) ? eigenvalues_[k] / total : 0.f;
    }

    int out_dim() const { return top_k_; }
    int in_dim()  const { return D_; }
    const std::vector<float>& components()   const { return components_; }
    const std::vector<float>& mean()         const { return mean_; }
    const std::vector<float>& eigenvalues()  const { return eigenvalues_; }

    void set_state(std::vector<float>&& comp, std::vector<float>&& mean,
                   std::vector<float>&& eigenvals) {
        components_  = std::move(comp);
        mean_        = std::move(mean);
        eigenvalues_ = std::move(eigenvals);
        fitted_      = true;
    }

private:
    int D_, top_k_;
    uint64_t seed_;
    bool whiten_, fitted_;
    std::vector<float> components_, mean_, eigenvalues_;

    static float norm_vec(const std::vector<float>& v) {
        float s = 0.f; for (float x : v) s += x*x; return std::sqrt(s);
    }
    static void normalize(std::vector<float>& v) {
        float n = norm_vec(v);
        if (n > 1e-10f) for (float& x : v) x /= n;
    }
};

}} // NGT::NGTAQ
