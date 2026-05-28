#pragma once
#include "ADCTable.h"
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

inline float tier1_adc_scalar(const int8_t* q_int8, const uint8_t* tier1) {
    int32_t acc = 0;
    for (int i = 0; i < 128; ++i) {
        int bit = (tier1[i>>3] >> (i&7)) & 1;
        acc += (int32_t)q_int8[i] * (bit ? 1 : -1);
    }
    return (float)acc;
}

// Runtime-dispatched tier-1 ADC
// Currently uses scalar (VNNI sign-bit expansion is non-trivial; scalar is correct baseline)
inline float tier1_adc_fast(const int8_t* q_int8, const uint8_t* tier1) {
    return tier1_adc_scalar(q_int8, tier1);
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
