#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>

namespace NGT { namespace NGTAQ {

// Subsampled Randomized Hadamard Transform
// y = (1/sqrt(D)) * H * D_rand * x
// D_rand: random ±1 diagonal; H: Walsh-Hadamard transform
// Preserves L2 norm (orthogonal transform)
class SRHT {
public:
    // skip_diagonal=true: test helper to apply only WHT (diagonal = all +1)
    SRHT(int D, uint64_t seed, bool skip_diagonal = false)
        : D_(D), skip_diagonal_(skip_diagonal)
    {
        assert((D & (D - 1)) == 0 && D >= 2);
        diag_.resize(D);
        if (!skip_diagonal) {
            std::mt19937_64 rng(seed);
            for (int i = 0; i < D; ++i)
                diag_[i] = (rng() & 1) ? 1.f : -1.f;
        } else {
            for (int i = 0; i < D; ++i) diag_[i] = 1.f;
        }
        inv_sqrt_D_ = 1.f / std::sqrt((float)D);
    }

    // Full SRHT: apply diagonal then WHT then scale by 1/sqrt(D)
    void apply(const float* __restrict__ x, float* __restrict__ y) const {
        std::vector<float> tmp(D_);
        for (int i = 0; i < D_; ++i) tmp[i] = diag_[i] * x[i];
        wht(tmp.data(), D_);
        for (int i = 0; i < D_; ++i) y[i] = tmp[i] * inv_sqrt_D_;
    }

    // Test helper: only WHT + 1/sqrt(D) scale (diagonal = +1)
    void apply_hadamard_only(const float* __restrict__ x, float* __restrict__ y) const {
        std::vector<float> tmp(x, x + D_);
        wht(tmp.data(), D_);
        for (int i = 0; i < D_; ++i) y[i] = tmp[i] * inv_sqrt_D_;
    }

    int dim() const { return D_; }
    const std::vector<float>& diag() const { return diag_; }

    void serialize(std::vector<float>& out) const { out = diag_; }
    void deserialize(const std::vector<float>& in) {
        assert((int)in.size() == D_);
        diag_ = in;
        // Recompute inv_sqrt_D_ in case D changed
        inv_sqrt_D_ = 1.f / std::sqrt((float)D_);
    }

private:
    int D_;
    bool skip_diagonal_;
    float inv_sqrt_D_;
    std::vector<float> diag_;

    // In-place Fast Walsh-Hadamard Transform, O(D log D)
    static void wht(float* x, int n) {
        for (int len = n >> 1; len >= 1; len >>= 1) {
            for (int i = 0; i < n; i += len << 1) {
                for (int j = 0; j < len; ++j) {
                    float a = x[i + j];
                    float b = x[i + j + len];
                    x[i + j]       = a + b;
                    x[i + j + len] = a - b;
                }
            }
        }
    }
};

}} // NGT::NGTAQ
