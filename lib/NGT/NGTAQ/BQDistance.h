// lib/NGT/NGTAQ/BQDistance.h
// Binary Quantization distance for AQ-DABS.
//
// δ_BQ(p, q) = popcount(XOR(pA, qA) AND OR(pB, qB)) / D
//
// pA = sign plane of p    (1 bit per dimension, packed into uint64_t words)
// pB = magnitude plane of p
// qA = sign plane of q
// qB = magnitude plane of q
// words = number of 64-bit words (D must equal words * 64)
//
// Three SIMD paths:
//   NGT_AVX512  – _mm512_popcnt_epi64,  8 words per iteration
//   NGT_AVX2    – 256-bit XOR/OR/AND,   4 words per iteration
//   generic     – scalar __builtin_popcountll loop

#pragma once

#include <cassert>
#include <cstdint>

#if defined(NGT_AVX512) || defined(NGT_AVX2)
#  include <immintrin.h>
#endif

namespace NGTAQ {

// ---------------------------------------------------------------------------
// AVX-512 path — requires AVX512F + AVX512VPOPCNTDQ
// ---------------------------------------------------------------------------
#if defined(NGT_AVX512) && defined(__AVX512VPOPCNTDQ__)

inline float bqDistance(const uint64_t* __restrict__ pA,
                        const uint64_t* __restrict__ pB,
                        const uint64_t* __restrict__ qA,
                        const uint64_t* __restrict__ qB,
                        int words,
                        int D) noexcept {
  // Requires D > 0 (D == words * 64; caller must ensure non-empty input)
  assert(D > 0 && "bqDistance: D must be positive");
  __m512i acc = _mm512_setzero_si512();
  int i       = 0;
  // Process 8 words (512 bits) per iteration
  for (; i + 8 <= words; i += 8) {
    __m512i va  = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pA + i));
    __m512i qa  = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(qA + i));
    __m512i vb  = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pB + i));
    __m512i qb  = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(qB + i));
    __m512i xr  = _mm512_xor_si512(va, qa);
    __m512i ors = _mm512_or_si512(vb, qb);
    __m512i msk = _mm512_and_si512(xr, ors);
    acc         = _mm512_add_epi64(acc, _mm512_popcnt_epi64(msk));
  }
  // Horizontal sum of 8 x 64-bit lanes
  uint64_t total = static_cast<uint64_t>(_mm512_reduce_add_epi64(acc));
  // Scalar tail
  for (; i < words; ++i) {
    total += static_cast<uint64_t>(__builtin_popcountll((pA[i] ^ qA[i]) & (pB[i] | qB[i])));
  }
  return static_cast<float>(total) / static_cast<float>(D);
}

// ---------------------------------------------------------------------------
// AVX2 path — 256-bit XOR/OR/AND, then scalar popcount per 64-bit word
// ---------------------------------------------------------------------------
#elif defined(NGT_AVX2)

inline float bqDistance(const uint64_t* __restrict__ pA,
                        const uint64_t* __restrict__ pB,
                        const uint64_t* __restrict__ qA,
                        const uint64_t* __restrict__ qB,
                        int words,
                        int D) noexcept {
  // Requires D > 0 (D == words * 64; caller must ensure non-empty input)
  assert(D > 0 && "bqDistance: D must be positive");
  uint64_t total = 0;
  int i          = 0;
  // Process 4 words (256 bits) per iteration
  for (; i + 4 <= words; i += 4) {
    __m256i va  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pA + i));
    __m256i qa  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qA + i));
    __m256i vb  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pB + i));
    __m256i qb  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qB + i));
    __m256i xr  = _mm256_xor_si256(va, qa);
    __m256i ors = _mm256_or_si256(vb, qb);
    __m256i msk = _mm256_and_si256(xr, ors);
    // Store to memory and popcount 4 x 64-bit words
    alignas(32) uint64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), msk);
    total += __builtin_popcountll(tmp[0]) + __builtin_popcountll(tmp[1])
           + __builtin_popcountll(tmp[2]) + __builtin_popcountll(tmp[3]);
  }
  // Scalar tail
  for (; i < words; ++i) {
    total += static_cast<uint64_t>(__builtin_popcountll((pA[i] ^ qA[i]) & (pB[i] | qB[i])));
  }
  return static_cast<float>(total) / static_cast<float>(D);
}

// ---------------------------------------------------------------------------
// Generic (scalar) path
// ---------------------------------------------------------------------------
#else

inline float bqDistance(const uint64_t* __restrict__ pA,
                        const uint64_t* __restrict__ pB,
                        const uint64_t* __restrict__ qA,
                        const uint64_t* __restrict__ qB,
                        int words,
                        int D) noexcept {
  // Requires D > 0 (D == words * 64; caller must ensure non-empty input)
  assert(D > 0 && "bqDistance: D must be positive");
  uint64_t total = 0;
  for (int i = 0; i < words; ++i) {
    total += static_cast<uint64_t>(__builtin_popcountll((pA[i] ^ qA[i]) & (pB[i] | qB[i])));
  }
  return static_cast<float>(total) / static_cast<float>(D);
}

#endif  // NGT_AVX512 / NGT_AVX2 / generic

}  // namespace NGTAQ
