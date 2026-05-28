#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

namespace NGT { namespace NGTAQ {

// Per-query ADC state, rebuilt when active centroid changes
struct ADCQueryState {
    int8_t  q_int8[128];        // SRHT-rotated query as ±127 int8 (tier-1 encoding)
    float   q_norm_sq;          // ||query - centroid||^2
    int8_t  tier2_lut[16][16];  // 256 bytes: 16 codewords × 16 half-dim LUT
};

// Build tier-1 query: positive → +127, negative → -127
inline void build_tier1_query(const float* residual, int D, int8_t* q_int8) {
    for (int i = 0; i < D; ++i)
        q_int8[i] = (residual[i] >= 0.f) ? 127 : -127;
}

// Build tier-2 LUT from float PCA-projected query (32 dims)
// pca_query[32]: query projected into PCA space
// codebook: shape [16][32], flat storage (row-major)
// lut[k][half]: inner product of pca_query[2*half:2*half+2] with codebook[k][2*half:2*half+2]
inline void build_tier2_lut(const float* pca_query,
                             const float* codebook,  // [16][32] flat
                             int8_t lut[16][16])
{
    for (int k = 0; k < 16; ++k) {
        const float* cb = codebook + k * 32;
        for (int half = 0; half < 16; ++half) {
            float dot = pca_query[2*half]   * cb[2*half]
                      + pca_query[2*half+1] * cb[2*half+1];
            int ival = (int)(dot + (dot >= 0.f ? 0.5f : -0.5f));
            if (ival > 127)  ival = 127;
            if (ival < -128) ival = -128;
            lut[k][half] = (int8_t)ival;
        }
    }
}

}} // NGT::NGTAQ
