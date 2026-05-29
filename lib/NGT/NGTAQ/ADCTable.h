#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

namespace NGT { namespace NGTAQ {

// Per-query ADC state, rebuilt when active centroid changes
struct ADCQueryState {
    int8_t  q_int8[128];         // unit query residual as int8 (scaled by 127)
    float   q_norm_sq;           // ||query - centroid||^2
    float   q_norm;              // sqrt(q_norm_sq) — needed for RaBitQ formula
    int32_t q_sum;               // sum(q_int8[i]) — precomputed for AVX2 masked-sum formula
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

// Build tier-2 PQ LUT from SRHT-rotated query residual (D dims).
// M=16 sub-spaces × D/16 dims each (D/16=8 for D=128). K=256 centroids (8 bits/sub).
// Codebook layout (original): codebook[(sub * 256 + code) * D_sub + dim]
// lut[sub][code] = dot(q_res[D_sub*sub : D_sub*(sub+1)], codebook[sub][code])
// Standard asymmetric ADC over all D dims (no PCA projection needed).
inline void build_tier2_lut(const float* q_res, int D,
                             const float* codebook,
                             float lut[16][256])
{
    const int D_sub = D / 16;  // = 8 for D=128
    for (int sub = 0; sub < 16; ++sub) {
        const float* q = q_res + sub * D_sub;
        for (int code = 0; code < 256; ++code) {
            const float* c = codebook + (sub * 256 + code) * D_sub;
            float dot = 0.f;
            for (int d = 0; d < D_sub; ++d) dot += q[d] * c[d];
            lut[sub][code] = dot;
        }
    }
}

// Transpose tier-2 codebook from [M][K][D_sub] to [M][D_sub][K].
// Call once at index load time; result stored in tier2_codebook_T_.
// Enables AVX2 FMA LUT build: vectorize over K (SIMD width 8), iterate over D_sub.
inline void build_tier2_codebook_T(const float* cb, int M, int K, int D_sub,
                                    float* cb_T)
{
    for (int sub = 0; sub < M; ++sub)
        for (int d = 0; d < D_sub; ++d)
            for (int code = 0; code < K; ++code)
                cb_T[(sub * D_sub + d) * K + code] = cb[(sub * K + code) * D_sub + d];
}

// AVX2 FMA fast tier-2 LUT build using transposed codebook [M][D_sub][K].
// Strategy: broadcast each query dim scalar, FMA against K consecutive centroid values.
// For M=16, D_sub=8, K=256: 16 sub-spaces × 8 dims × 32 float8-groups × FMA = ~1.4μs.
// Requires: defined(__AVX2__) && defined(__FMA__)
inline void build_tier2_lut_fast(const float* q_res, int D,
                                  const float* cb_T,    // [M][D_sub][K], K must be multiple of 8
                                  float lut[16][256])
{
    const int M    = 16;
    const int K    = 256;
    const int D_sub = D / M;  // 8 for D=128

#if defined(__AVX2__) && defined(__FMA__)
    const int K8 = K / 8;  // 32 groups of 8 codes
    for (int sub = 0; sub < M; ++sub) {
        const float* q_sub  = q_res + sub * D_sub;
        const float* cb_sub = cb_T + sub * D_sub * K;  // D_sub × K floats
        float*       out    = lut[sub];                 // K floats

        // Initialize K/8 AVX2 accumulators
        __m256 acc[32];
        for (int g = 0; g < K8; ++g) acc[g] = _mm256_setzero_ps();

        // FMA: for each dim d, accumulate qd * cb_row[d*K + g*8 .. g*8+7]
        for (int d = 0; d < D_sub; ++d) {
            __m256 qd = _mm256_set1_ps(q_sub[d]);
            const float* c_row = cb_sub + d * K;
            for (int g = 0; g < K8; ++g)
                acc[g] = _mm256_fmadd_ps(qd, _mm256_loadu_ps(c_row + g * 8), acc[g]);
        }
        for (int g = 0; g < K8; ++g) _mm256_storeu_ps(out + g * 8, acc[g]);
    }
#else
    // Scalar fallback (same as build_tier2_lut but using transposed layout)
    const int K_CB = K;
    for (int sub = 0; sub < M; ++sub) {
        const float* q_sub  = q_res + sub * D_sub;
        const float* cb_sub = cb_T + sub * D_sub * K_CB;
        float*       out    = lut[sub];
        for (int code = 0; code < K_CB; ++code) {
            float dot = 0.f;
            for (int d = 0; d < D_sub; ++d)
                dot += q_sub[d] * cb_sub[d * K_CB + code];
            out[code] = dot;
        }
    }
#endif
}

}} // NGT::NGTAQ
