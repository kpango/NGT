#pragma once
#include "ADCTable.h"
#include "SIMDUtils.h"
#include "VectorRecord.h"
#include <cstdint>
#include <cmath>
#include <cstring>

// SIMD includes — compile with appropriate flags
#if defined(__AVX512VNNI__) || defined(__AVX512F__) || defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace NGT { namespace NGTAQ {

// RaBitQ scaling constant: sqrt(pi/2)
static constexpr float RABITQ_SCALE = 1.2533141373f;

// ============================================================
// Tier-1 ADC (sign bits, 128-dim, 16 bytes)
// result = sum_i q_int8[i] * (sign_bit_i ? +1 : -1)
// ============================================================

inline float tier1_adc_scalar(const int8_t* q_int8, const uint8_t* tier1, int D = 128) {
    int32_t acc = 0;
    for (int i = 0; i < D; ++i) {
        int bit = (tier1[i>>3] >> (i&7)) & 1;
        acc += (int32_t)q_int8[i] * (bit ? 1 : -1);
    }
    return (float)acc;
}

#if defined(__AVX512VNNI__)
// AVX-512VNNI tier-1 ADC (D=128, 2 zmm iterations):
//   Expand 64 tier1 bits → 64 uint8 (0 or 1) via _mm512_maskz_set1_epi8
//   Accumulate: acc[i] += expanded[i]*q_int8[i] via _mm512_dpbusd_epi32
//   sum_pos = horizontal_reduce(acc); result = 2*sum_pos - q_sum
// Requires: __AVX512BW__ (implied by __AVX512VNNI__ on all practical hardware)
inline float tier1_adc_vnni(const int8_t* __restrict__ q_int8,
                             const uint8_t* __restrict__ tier1,
                             int32_t q_sum) {
    // Load all 128 tier1 bits (16 bytes) as two 64-bit masks
    uint64_t lo64, hi64;
    memcpy(&lo64, tier1,     8);
    memcpy(&hi64, tier1 + 8, 8);

    // Expand: bit=1 → 0x01, bit=0 → 0x00 (unsigned int8 for dpbusd)
    __m512i exp0 = _mm512_maskz_set1_epi8((__mmask64)lo64, (int8_t)1);
    __m512i exp1 = _mm512_maskz_set1_epi8((__mmask64)hi64, (int8_t)1);

    // VNNI dot product: acc[i] += a_unsigned[4i..4i+3] * b_signed[4i..4i+3]
    __m512i acc = _mm512_setzero_si512();
    acc = _mm512_dpbusd_epi32(acc, exp0, _mm512_loadu_si512(q_int8));
    acc = _mm512_dpbusd_epi32(acc, exp1, _mm512_loadu_si512(q_int8 + 64));

    int32_t sum_pos = _mm512_reduce_add_epi32(acc);
    return (float)(2 * sum_pos - q_sum);
}
#endif // __AVX512VNNI__

#if defined(__AVX2__)
// AVX2 tier-1 ADC using masked-sum decomposition:
//   t1 = sum(q[i]*sign_i) = 2*sum_pos - q_sum
//   where sum_pos = sum(q[i] where bit_i=1), q_sum = sum(q[i]) precomputed per query
//
// Bit expansion: 4 bytes → 32 byte masks (0xFF where bit=1, 0x00 where bit=0)
// via vpbroadcastd → vpshufb → vpand → vpcmpeqb → vpxor
inline float tier1_adc_avx2(const int8_t* __restrict__ q_int8,
                              const uint8_t* __restrict__ tier1,
                              int32_t q_sum) {
    // Repeating bit-position mask: position i selects bit (i%8) via AND
    static const __m256i BIT_MASK = _mm256_set_epi8(
        (int8_t)0x80,(int8_t)0x40,(int8_t)0x20,(int8_t)0x10,
        (int8_t)0x08,(int8_t)0x04,(int8_t)0x02,(int8_t)0x01,
        (int8_t)0x80,(int8_t)0x40,(int8_t)0x20,(int8_t)0x10,
        (int8_t)0x08,(int8_t)0x04,(int8_t)0x02,(int8_t)0x01,
        (int8_t)0x80,(int8_t)0x40,(int8_t)0x20,(int8_t)0x10,
        (int8_t)0x08,(int8_t)0x04,(int8_t)0x02,(int8_t)0x01,
        (int8_t)0x80,(int8_t)0x40,(int8_t)0x20,(int8_t)0x10,
        (int8_t)0x08,(int8_t)0x04,(int8_t)0x02,(int8_t)0x01
    );
    // Shuffle: route byte j of bits32 to all 8 positions [j*8 .. j*8+7]
    // _mm256_set_epi8 args: byte[31]..byte[0]
    static const __m256i BYTE_SHUF = _mm256_set_epi8(
        3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2,
        1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0
    );
    const __m256i ALLFF = _mm256_set1_epi8((int8_t)0xFF);
    const __m256i ZERO  = _mm256_setzero_si256();

    __m256i acc_lo = ZERO, acc_hi = ZERO;

    // 4 blocks × 32 dims = 128 dims
    for (int blk = 0; blk < 4; ++blk) {
        // Load 32 int8 query components
        __m256i q = _mm256_loadu_si256((const __m256i*)(q_int8 + blk * 32));

        // Expand 4 sign-bit bytes to 32 byte masks (0xFF = bit set, 0x00 = bit clear)
        uint32_t bits32;
        memcpy(&bits32, tier1 + blk * 4, 4);
        __m256i b = _mm256_set1_epi32((int)bits32);
        b = _mm256_shuffle_epi8(b, BYTE_SHUF);           // route bytes to lanes
        b = _mm256_and_si256(b, BIT_MASK);               // isolate bit (0 or mask value)
        b = _mm256_xor_si256(_mm256_cmpeq_epi8(b, ZERO), ALLFF);
        // b[i] = 0xFF if bit_i=1, 0x00 if bit_i=0

        // q_masked[i] = q[i] if bit_i=1, else 0
        __m256i q_masked = _mm256_and_si256(q, b);

        // Sign-extend int8 → int16 and accumulate
        acc_lo = _mm256_add_epi16(acc_lo,
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q_masked, 0)));
        acc_hi = _mm256_add_epi16(acc_hi,
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q_masked, 1)));
    }

    // Horizontal sum: int16 → int32 → scalar
    __m256i acc16 = _mm256_add_epi16(acc_lo, acc_hi);
    __m256i acc32 = _mm256_add_epi32(
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc16, 0)),
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc16, 1)));
    __m128i s = _mm_add_epi32(
        _mm256_extracti128_si256(acc32, 0),
        _mm256_extracti128_si256(acc32, 1));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    int32_t sum_pos = _mm_cvtsi128_si32(s);

    return (float)(2 * sum_pos - q_sum);
}
#endif // __AVX2__

// Runtime-dispatched tier-1 ADC (D=128 optimized path)
// q_sum = sum(q_int8[i]), precomputed by build_tier1_query; used by VNNI and AVX2 paths
inline float tier1_adc_fast(const int8_t* q_int8, const uint8_t* tier1, int32_t q_sum = 0) {
#if defined(__AVX512VNNI__)
    return tier1_adc_vnni(q_int8, tier1, q_sum);
#elif defined(__AVX2__)
    return tier1_adc_avx2(q_int8, tier1, q_sum);
#else
    (void)q_sum;
    return tier1_adc_scalar(q_int8, tier1, 128);
#endif
}

/// Dispatch tier-1 ADC for arbitrary D (power-of-2, divisible by 64).
/// For D=128 uses the optimized VNNI/AVX2 path; other D uses the scalar loop.
inline float tier1_adc_fast_d(const int8_t* q_int8, const uint8_t* tier1,
                               int32_t q_sum, int D) {
    if (D == 128) {
        return tier1_adc_fast(q_int8, tier1, q_sum);
    }
    (void)q_sum;
    return tier1_adc_scalar(q_int8, tier1, D);
}

// ============================================================
// Tier-2 ADC (4-bit nibbles, 32 dims, 16 bytes)
// result = sum_{d=0}^{31} lut[nibble_d][d/2]
// ============================================================

inline float tier2_adc_scalar(const int8_t lut[16][16], const uint8_t* tier2) {
    int32_t acc = 0;
    for (int d = 0; d < 32; ++d) {
        uint8_t nibble;
        if (d & 1) nibble = (tier2[d>>1] >> 4) & 0xF;
        else        nibble =  tier2[d>>1]        & 0xF;
        acc += (int32_t)lut[nibble][d >> 1];
    }
    return (float)acc;
}

inline float tier2_adc_fast(const int8_t lut[16][16], const uint8_t* tier2) {
    return tier2_adc_scalar(lut, tier2);
}

// ============================================================
// Tier-2 PQ ADC: float LUT, 16 sub-spaces × 8-bit codes (16 bytes, K=256)
// lut[sub][code] = dot(q_res[D_sub*sub:D_sub*(sub+1)], sub_centroid[sub][code])
// Covers full D-dimensional SRHT residual — standard asymmetric ADC.
// M=16, K=256, D_sub=8: 8× better quantization than M=32 K=16.
// ============================================================

#if defined(__AVX512F__)
// AVX-512F version: all 16 subs in one _mm512_i32gather_ps.
// codes[0..15] each 0..255, STRIDE512[sub] = sub*256 → index = sub*256 + code[sub]
inline float tier2_adc_pq_avx512(const float lut[16][256], const uint8_t* tier2) {
    const float* base = &lut[0][0];
    static const __m512i STRIDE512 = _mm512_set_epi32(
        15*256, 14*256, 13*256, 12*256, 11*256, 10*256, 9*256, 8*256,
         7*256,  6*256,  5*256,  4*256,  3*256,  2*256,   256,    0);
    __m512i codes = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)tier2));
    __m512i idx   = _mm512_add_epi32(STRIDE512, codes);
    __m512  v     = _mm512_i32gather_ps(idx, base, 4);
    return _mm512_reduce_add_ps(v);
}
#endif // __AVX512F__

#if defined(__AVX2__)
// AVX2 version: 8-wide gather over K=256 LUT.
// tier2[0..7] → 8 codes → gather from lut[0..7][code]
// tier2[8..15] → 8 codes → gather from lut[8..15][code]
// Index for gather: sub*256 + code (float offset, scale=4)
inline float tier2_adc_pq_avx2(const float lut[16][256], const uint8_t* tier2) {
    const float* base = &lut[0][0];
    // Sub-space base offsets (in float units): sub*256 for sub=0..7
    static const __m256i STRIDE256 = _mm256_set_epi32(7*256, 6*256, 5*256, 4*256,
                                                        3*256, 2*256,   256,    0);
    // Process first 8 sub-spaces
    __m256i codes0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)tier2));
    __m256 v0 = _mm256_i32gather_ps(base, _mm256_add_epi32(STRIDE256, codes0), 4);
    // Process next 8 sub-spaces (offset base by 8*256 floats)
    __m256i codes1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(tier2 + 8)));
    __m256 v1 = _mm256_i32gather_ps(base + 8*256,
                                     _mm256_add_epi32(STRIDE256, codes1), 4);

    __m256 acc = _mm256_add_ps(v0, v1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
#endif // __AVX2__

inline float tier2_adc_pq(const float lut[16][256], const uint8_t* tier2) {
#if defined(__AVX512F__)
    return tier2_adc_pq_avx512(lut, tier2);
#elif defined(__AVX2__)
    return tier2_adc_pq_avx2(lut, tier2);
#else
    float acc = 0.f;
    for (int sub = 0; sub < 16; ++sub)
        acc += lut[sub][tier2[sub]];
    return acc;
#endif
}

// ============================================================
// Squared L2 distance for exact reranking — delegates to SIMDUtils
// Kept as thin wrapper so existing call sites compile without change.
// AQIndex.cpp should call NGT::NGTAQ::l2_sq() directly (Task 6).
// ============================================================
inline float l2_sq_avx2(const float* __restrict__ a, const float* __restrict__ b, int D) {
    return NGT::NGTAQ::l2_sq(a, b, D);
}

// ============================================================
// Full RaBitQ-style distance
// dist ≈ q_norm_sq + norm_x² - 2*sqrt(π/2)*norm_x*adc_score/√D
// ============================================================
struct RaBitQDistance {
    static float compute(
        float q_norm_sq,
        float norm_x,
        float adc_score,
        float inv_sqrt_D,
        float adc_scale = RABITQ_SCALE)
    {
        return q_norm_sq + norm_x * norm_x
               - 2.0f * norm_x * adc_scale * adc_score * inv_sqrt_D;
    }
};

}} // NGT::NGTAQ
