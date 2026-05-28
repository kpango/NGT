#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

namespace NGT { namespace NGTAQ {

// Per-query ADC state, rebuilt when active centroid changes
struct ADCQueryState {
    int8_t  q_int8[128];        // unit query residual as int8 (scaled by 127)
    float   q_norm_sq;          // ||query - centroid||^2
    float   q_norm;             // sqrt(q_norm_sq) — needed for RaBitQ formula
    int32_t q_sum;              // sum(q_int8[i]) — precomputed for AVX2 masked-sum formula
    int8_t  tier2_lut[16][16];  // kept for structural compat, not used in hot path
};

// Build tier-1 query: store proportional int8 of unit residual vector (asymmetric ADC)
// q_int8[i] ≈ 127 * q_res[i] / ||q_res||
// Returns sum(q_int8[i]) for use with AVX2 formula: t1 = 2*sum_pos - q_sum
// RaBitQ: <q_res, x_res> ≈ ||x_res|| * sqrt(π/2) * sum_i q_int8[i]*sign(x_i) / (127*sqrt(D))
inline int32_t build_tier1_query(const float* residual, int D, int8_t* q_int8) {
    float norm_sq = 0.f;
    for (int i = 0; i < D; ++i) norm_sq += residual[i] * residual[i];
    float scale = (norm_sq > 1e-10f) ? 127.f / sqrtf(norm_sq) : 0.f;
    int32_t qsum = 0;
    for (int i = 0; i < D; ++i) {
        float v = residual[i] * scale;
        int vi = (int)(v + (v >= 0.f ? 0.5f : -0.5f));
        int8_t qv = (int8_t)(vi < -127 ? -127 : vi > 127 ? 127 : vi);
        q_int8[i] = qv;
        qsum += qv;
    }
    return qsum;
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
