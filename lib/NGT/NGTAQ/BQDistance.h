// lib/NGT/NGTAQ/BQDistance.h
// Binary Quantization distance for AQ-DABS — word-interleaved layout.
//
// δ_BQ(p, q) = popcount(XOR(pA, qA) AND OR(pB, qB)) / D
//
// Each pointer points to 2*words uint64_t in interleaved format:
//   [s₀, m₀, s₁, m₁, …, s_{words-1}, m_{words-1}]
// where sᵢ = sign-plane word i, mᵢ = magnitude-plane word i.
//
// D=128 → words=2 → each node = 4×uint64_t = 32 bytes = 1 cache line.
//
// Three SIMD paths (selected at compile time via preprocessor macros):
//   NGT_AVX512  — AVX-512 popcnt, 8 logical words per iteration
//   NGT_AVX2    — 256-bit deinterleave + scalar popcount, 4 words/iter
//   generic     — scalar __builtin_popcountll loop (used for D=128)
#pragma once

#include <cassert>
#include <cstdint>

#if defined(NGT_AVX512) || defined(NGT_AVX2)
#  include <immintrin.h>
#endif

namespace NGTAQ {

// ---------------------------------------------------------------------------
// AVX-512 path — requires AVX512F + AVX512VPOPCNTDQ
// Processes 8 logical words (16 uint64_t) per iteration.
// ---------------------------------------------------------------------------
#if defined(NGT_AVX512) && defined(__AVX512VPOPCNTDQ__)

inline float bqDistance(const uint64_t* __restrict__ p,
                        const uint64_t* __restrict__ q,
                        int words,
                        int D) noexcept {
  assert(D > 0 && "bqDistance: D must be positive");
  __m512i acc = _mm512_setzero_si512();
  int i = 0;
  for (; i + 8 <= words; i += 8) {
    __m512i vp0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(p + i * 2));
    __m512i vp1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(p + i * 2 + 8));
    __m512i vq0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q + i * 2));
    __m512i vq1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q + i * 2 + 8));
    __m512i sign_p = _mm512_unpacklo_epi64(vp0, vp1);
    __m512i mag_p  = _mm512_unpackhi_epi64(vp0, vp1);
    __m512i sign_q = _mm512_unpacklo_epi64(vq0, vq1);
    __m512i mag_q  = _mm512_unpackhi_epi64(vq0, vq1);
    __m512i xr  = _mm512_xor_si512(sign_p, sign_q);
    __m512i ors = _mm512_or_si512(mag_p,   mag_q);
    __m512i msk = _mm512_and_si512(xr, ors);
    acc         = _mm512_add_epi64(acc, _mm512_popcnt_epi64(msk));
  }
  uint64_t total = static_cast<uint64_t>(_mm512_reduce_add_epi64(acc));
  for (; i < words; ++i) {
    uint64_t pA = p[i*2], pB = p[i*2+1], qA = q[i*2], qB = q[i*2+1];
    total += static_cast<uint64_t>(__builtin_popcountll((pA ^ qA) & (pB | qB)));
  }
  return static_cast<float>(total) / static_cast<float>(D);
}

// ---------------------------------------------------------------------------
// AVX2 path — deinterleave with unpacklo/hi_epi64, then scalar popcount.
// Processes 4 logical words (8 uint64_t) per iteration.
// AVX2 path: also activated when NGT_AVX512 is set but __AVX512VPOPCNTDQ__ is unavailable
// (i.e. CPU has AVX-512 base but not the population-count extension).
// ---------------------------------------------------------------------------
#elif defined(NGT_AVX512) || defined(NGT_AVX2)

inline float bqDistance(const uint64_t* __restrict__ p,
                        const uint64_t* __restrict__ q,
                        int words,
                        int D) noexcept {
  assert(D > 0 && "bqDistance: D must be positive");
  uint64_t total = 0;
  int i = 0;
  for (; i + 4 <= words; i += 4) {
    __m256i vp_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i * 2));
    __m256i vp_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i * 2 + 4));
    __m256i vq_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i * 2));
    __m256i vq_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i * 2 + 4));
    __m256i sign_p = _mm256_unpacklo_epi64(vp_lo, vp_hi);
    __m256i mag_p  = _mm256_unpackhi_epi64(vp_lo, vp_hi);
    __m256i sign_q = _mm256_unpacklo_epi64(vq_lo, vq_hi);
    __m256i mag_q  = _mm256_unpackhi_epi64(vq_lo, vq_hi);
    __m256i xr  = _mm256_xor_si256(sign_p, sign_q);
    __m256i ors = _mm256_or_si256(mag_p,   mag_q);
    __m256i msk = _mm256_and_si256(xr, ors);
    alignas(32) uint64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), msk);
    total += __builtin_popcountll(tmp[0]) + __builtin_popcountll(tmp[1])
           + __builtin_popcountll(tmp[2]) + __builtin_popcountll(tmp[3]);
  }
  for (; i < words; ++i) {
    uint64_t pA = p[i*2], pB = p[i*2+1], qA = q[i*2], qB = q[i*2+1];
    total += static_cast<uint64_t>(__builtin_popcountll((pA ^ qA) & (pB | qB)));
  }
  return static_cast<float>(total) / static_cast<float>(D);
}

// ---------------------------------------------------------------------------
// Generic (scalar) path — always used for D=128 (words=2).
// ---------------------------------------------------------------------------
#else

inline float bqDistance(const uint64_t* __restrict__ p,
                        const uint64_t* __restrict__ q,
                        int words,
                        int D) noexcept {
  assert(D > 0 && "bqDistance: D must be positive");
  uint64_t total = 0;
  for (int i = 0; i < words; ++i) {
    uint64_t pA = p[i * 2], pB = p[i * 2 + 1];
    uint64_t qA = q[i * 2], qB = q[i * 2 + 1];
    total += static_cast<uint64_t>(__builtin_popcountll((pA ^ qA) & (pB | qB)));
  }
  return static_cast<float>(total) / static_cast<float>(D);
}

#endif  // NGT_AVX512 / NGT_AVX2 / generic

}  // namespace NGTAQ
