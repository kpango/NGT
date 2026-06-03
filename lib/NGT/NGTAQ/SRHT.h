#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#  include <immintrin.h>
#endif

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
#if defined(__AVX512F__)
        if (D_ == 128) { apply_avx512_d128(x, y); return; }
#elif defined(__AVX2__)
        if (D_ == 128) { apply_avx2_d128(x, y); return; }
#endif
        // Scalar fallback: thread_local to avoid heap alloc
        static thread_local std::vector<float> tmp;
        tmp.resize(static_cast<size_t>(D_));
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

#if defined(__AVX512F__)
    // AVX-512F fully-unrolled D=128 SRHT: 7 stages, all in zmm registers.
    // 16 floats/iter (zmm) vs 8 (ymm) → ~2× fewer iterations per stage.
    // Stages 1-4: standard butterfly; stages 5-7: cross-lane permutes.
    //
    // Permute constants (derived & verified):
    // Stage 4 (len=8): shuffle_f32x4(0x4E) swaps 256-bit halves, blend 0xFF00
    // Stage 5 (len=4): shuffle_f32x4(0xB1) swaps lanes 0↔1, 2↔3, blend 0xF0F0
    // Stage 6 (len=2): permute_ps(0x44/0xEE) dup pairs, blend 0xCCCC
    // Stage 7 (len=1) + scale: permute_ps(0xA0/0xF5) dup even/odd, blend 0xAAAA
    void apply_avx512_d128(const float* __restrict__ x, float* __restrict__ y) const {
        const float* __restrict__ diag = diag_.data();

        // Stage 1 (len=64): diagonal mul fused with butterfly, 4 zmm iterations
        for (int i = 0; i < 64; i += 16) {
            __m512 da = _mm512_loadu_ps(diag + i);
            __m512 db = _mm512_loadu_ps(diag + i + 64);
            __m512 xa = _mm512_loadu_ps(x + i);
            __m512 xb = _mm512_loadu_ps(x + i + 64);
            __m512 a  = _mm512_mul_ps(da, xa);
            __m512 b  = _mm512_mul_ps(db, xb);
            _mm512_storeu_ps(y + i,      _mm512_add_ps(a, b));
            _mm512_storeu_ps(y + i + 64, _mm512_sub_ps(a, b));
        }

        // Stage 2 (len=32): 2 blocks, 2 zmm iterations each
        for (int blk = 0; blk < 128; blk += 64) {
            for (int i = 0; i < 32; i += 16) {
                __m512 a = _mm512_loadu_ps(y + blk + i);
                __m512 b = _mm512_loadu_ps(y + blk + i + 32);
                _mm512_storeu_ps(y + blk + i,      _mm512_add_ps(a, b));
                _mm512_storeu_ps(y + blk + i + 32, _mm512_sub_ps(a, b));
            }
        }

        // Stage 3 (len=16): 4 blocks, 1 zmm iteration each
        for (int blk = 0; blk < 128; blk += 32) {
            __m512 a = _mm512_loadu_ps(y + blk);
            __m512 b = _mm512_loadu_ps(y + blk + 16);
            _mm512_storeu_ps(y + blk,      _mm512_add_ps(a, b));
            _mm512_storeu_ps(y + blk + 16, _mm512_sub_ps(a, b));
        }

        // Stage 4 (len=8): 8 blocks, 1 zmm/block (16 floats spans both halves)
        // shuffle_f32x4(0x4E): swaps 256-bit halves
        // blend 0xFF00: lo 8 from sum (a+b), hi 8 from diff (perm-v = a-b for positions 8..15)
        for (int blk = 0; blk < 128; blk += 16) {
            __m512 v    = _mm512_loadu_ps(y + blk);
            __m512 perm = _mm512_shuffle_f32x4(v, v, 0x4E);
            __m512 s    = _mm512_add_ps(v, perm);
            __m512 d    = _mm512_sub_ps(perm, v);  // positions 8..15 give a-b
            _mm512_storeu_ps(y + blk, _mm512_mask_blend_ps(0xFF00, s, d));
        }

        // Stage 5 (len=4): 8 zmm iterations
        // shuffle_f32x4(0xB1): [lane1,lane0,lane3,lane2]; perm[0..3]=b, perm[4..7]=a
        // blend 0xF0F0: lo 4 from sum (a+b), hi 4 from diff (perm-v = a-b for positions 4..7)
        for (int i = 0; i < 128; i += 16) {
            __m512 v    = _mm512_loadu_ps(y + i);
            __m512 perm = _mm512_shuffle_f32x4(v, v, 0xB1);
            __m512 s    = _mm512_add_ps(v, perm);
            __m512 d    = _mm512_sub_ps(perm, v);
            _mm512_storeu_ps(y + i, _mm512_mask_blend_ps(0xF0F0, s, d));
        }

        // Stage 6 (len=2): 8 zmm iterations
        // permute_ps(0x44): [a,b,a,b,...] per lane (dup first pair)
        // permute_ps(0xEE): [c,d,c,d,...] per lane (dup second pair)
        // blend 0xCCCC: positions 0,1 per lane from sum, positions 2,3 from diff
        for (int i = 0; i < 128; i += 16) {
            __m512 v  = _mm512_loadu_ps(y + i);
            __m512 ga = _mm512_permute_ps(v, 0x44);
            __m512 gb = _mm512_permute_ps(v, 0xEE);
            __m512 s  = _mm512_add_ps(ga, gb);
            __m512 d  = _mm512_sub_ps(ga, gb);
            _mm512_storeu_ps(y + i, _mm512_mask_blend_ps(0xCCCC, s, d));
        }

        // Stage 7 (len=1) + scale fold: 8 zmm iterations
        // permute_ps(0xA0): [a,a,c,c,...] per lane (dup even positions)
        // permute_ps(0xF5): [b,b,d,d,...] per lane (dup odd positions)
        // blend 0xAAAA: even positions from (a+b)*scale, odd from (a-b)*scale
        const __m512 scale_v = _mm512_set1_ps(inv_sqrt_D_);
        for (int i = 0; i < 128; i += 16) {
            __m512 v  = _mm512_loadu_ps(y + i);
            __m512 ga = _mm512_permute_ps(v, 0xA0);
            __m512 gb = _mm512_permute_ps(v, 0xF5);
            __m512 s  = _mm512_mul_ps(_mm512_add_ps(ga, gb), scale_v);
            __m512 d  = _mm512_mul_ps(_mm512_sub_ps(ga, gb), scale_v);
            _mm512_storeu_ps(y + i, _mm512_mask_blend_ps(0xAAAA, s, d));
        }
    }
#endif // __AVX512F__

#if defined(__AVX2__)
    // AVX2 fully-unrolled D=128 SRHT:
    // - Folds diagonal multiply into WHT stage 1 (len=64)
    // - Folds inv_sqrt_D_ scale into WHT stage 7 (len=1)
    // - No heap allocation; uses output buffer y directly as working space
    void apply_avx2_d128(const float* __restrict__ x, float* __restrict__ y) const {
        const float* __restrict__ diag = diag_.data();

        // Stage 1 (len=64): fold diagonal mul + butterfly
        // For i in [0..56 step 8]: a=diag[i]*x[i], b=diag[i+64]*x[i+64]
        //                           y[i]=a+b, y[i+64]=a-b
        for (int i = 0; i < 64; i += 8) {
            __m256 da = _mm256_loadu_ps(diag + i);
            __m256 db = _mm256_loadu_ps(diag + i + 64);
            __m256 xa = _mm256_loadu_ps(x + i);
            __m256 xb = _mm256_loadu_ps(x + i + 64);
            __m256 a  = _mm256_mul_ps(da, xa);
            __m256 b  = _mm256_mul_ps(db, xb);
            _mm256_storeu_ps(y + i,      _mm256_add_ps(a, b));
            _mm256_storeu_ps(y + i + 64, _mm256_sub_ps(a, b));
        }

        // Stage 2 (len=32): 2 blocks at {0, 64}
        for (int blk = 0; blk < 128; blk += 64) {
            for (int i = 0; i < 32; i += 8) {
                __m256 a = _mm256_loadu_ps(y + blk + i);
                __m256 b = _mm256_loadu_ps(y + blk + i + 32);
                _mm256_storeu_ps(y + blk + i,      _mm256_add_ps(a, b));
                _mm256_storeu_ps(y + blk + i + 32, _mm256_sub_ps(a, b));
            }
        }

        // Stage 3 (len=16): 4 blocks at {0, 32, 64, 96}
        for (int blk = 0; blk < 128; blk += 32) {
            for (int i = 0; i < 16; i += 8) {
                __m256 a = _mm256_loadu_ps(y + blk + i);
                __m256 b = _mm256_loadu_ps(y + blk + i + 16);
                _mm256_storeu_ps(y + blk + i,      _mm256_add_ps(a, b));
                _mm256_storeu_ps(y + blk + i + 16, _mm256_sub_ps(a, b));
            }
        }

        // Stage 4 (len=8): 8 blocks at {0, 16, 32, ..., 112}
        for (int blk = 0; blk < 128; blk += 16) {
            __m256 a = _mm256_loadu_ps(y + blk);
            __m256 b = _mm256_loadu_ps(y + blk + 8);
            _mm256_storeu_ps(y + blk,     _mm256_add_ps(a, b));
            _mm256_storeu_ps(y + blk + 8, _mm256_sub_ps(a, b));
        }

        // Stage 5 (len=4): 16 blocks of 8 floats each
        // permute2f128 technique: swap 128-bit halves to pair the right elements
        for (int i = 0; i < 128; i += 8) {
            __m256 v    = _mm256_loadu_ps(y + i);
            __m256 perm = _mm256_permute2f128_ps(v, v, 1); // swap 128-bit halves
            __m256 s    = _mm256_add_ps(v, perm);
            __m256 d    = _mm256_sub_ps(perm, v);  // perm-v (not v-perm)
            _mm256_storeu_ps(y + i, _mm256_blend_ps(s, d, 0xF0)); // lo=s[a+b], hi=d[a-b]
        }

        // Stage 6 (len=2): 16 AVX2 iterations (2 blocks of 4 each = 8 floats)
        for (int i = 0; i < 128; i += 8) {
            __m256 v  = _mm256_loadu_ps(y + i);
            __m256 ga = _mm256_permute_ps(v, 0x44); // dup first pair:  [c0,c1,c0,c1, c4,c5,c4,c5]
            __m256 gb = _mm256_permute_ps(v, 0xEE); // dup second pair: [c2,c3,c2,c3, c6,c7,c6,c7]
            __m256 s  = _mm256_add_ps(ga, gb);
            __m256 d  = _mm256_sub_ps(ga, gb);
            _mm256_storeu_ps(y + i, _mm256_blend_ps(s, d, 0xCC)); // 0b11001100: pos 2,3,6,7 from d
        }

        // Stage 7 (len=1) WITH scale fold: 16 AVX2 iterations
        const __m256 scale_v = _mm256_set1_ps(inv_sqrt_D_);
        for (int i = 0; i < 128; i += 8) {
            __m256 v  = _mm256_loadu_ps(y + i);
            __m256 ga = _mm256_permute_ps(v, 0xA0); // dup even positions: [c0,c0,c2,c2, c4,c4,c6,c6]
            __m256 gb = _mm256_permute_ps(v, 0xF5); // dup odd positions:  [c1,c1,c3,c3, c5,c5,c7,c7]
            __m256 s  = _mm256_mul_ps(_mm256_add_ps(ga, gb), scale_v);
            __m256 d  = _mm256_mul_ps(_mm256_sub_ps(ga, gb), scale_v);
            _mm256_storeu_ps(y + i, _mm256_blend_ps(s, d, 0xAA)); // 0b10101010: odd positions from d
        }
    }
#endif // __AVX2__
};

}} // NGT::NGTAQ
