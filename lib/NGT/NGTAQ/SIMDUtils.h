#pragma once
#include <cstddef>
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#  include <immintrin.h>
#endif

namespace NGT { namespace NGTAQ {

// Squared L2 distance between two float vectors of dimension D.
// ISA dispatch: AVX-512F > AVX2+FMA > scalar (compile-time selection).
// AVX-512F: 2-accumulator 32-floats/iter FMA
// AVX2:     4-accumulator 32-floats/iter FMA  (fixes missing fmadd in old l2_sq_avx2)
// scalar:   reference path
inline float l2_sq(const float* __restrict__ a, const float* __restrict__ b, int D) {
#if defined(__AVX512F__)
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= D; i += 32) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        s0 = _mm512_fmadd_ps(d0, d0, s0);
        s1 = _mm512_fmadd_ps(d1, d1, s1);
    }
    for (; i + 16 <= D; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        s0 = _mm512_fmadd_ps(d, d, s0);
    }
    float r = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
    for (; i < D; ++i) { float d = a[i] - b[i]; r += d * d; }
    return r;
#elif defined(__AVX2__)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= D; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
    }
    for (; i + 8 <= D; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        s0 = _mm256_fmadd_ps(d, d, s0);
    }
    float tail = 0.f;
    for (; i < D; ++i) { float d = a[i] - b[i]; tail += d * d; }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 s   = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s) + tail;
#else
    float r = 0.f;
    for (int i = 0; i < D; ++i) { float d = a[i] - b[i]; r += d * d; }
    return r;
#endif
}

}} // NGT::NGTAQ
