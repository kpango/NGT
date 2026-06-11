#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#  include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

namespace NGT { namespace ArcFlare {

// Per-query ADC state, rebuilt when active centroid changes
struct ADCQueryState {
    std::vector<int8_t> q_int8;  // unit query residual as int8 (scaled by 127)
    float   q_norm_sq = 0.f;     // ||query - centroid||^2
    float   q_norm    = 0.f;     // sqrt(q_norm_sq) — needed for RaBitQ formula
    int32_t q_sum     = 0;       // sum(q_int8[i]) — precomputed for AVX2 masked-sum formula

    explicit ADCQueryState(int D = 128) : q_int8(D, 0) {}

    int8_t*       data()       { return q_int8.data(); }
    const int8_t* data() const { return q_int8.data(); }
    int dim() const { return (int)q_int8.size(); }
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

// Compute residual and tier-1 query state in a single fused operation.
// Replaces the 3-step sequence: get_residual + norm_sq loop + build_tier1_query.
// AVX-512F, AVX2, and scalar paths — dispatched at compile time.
inline void compute_residual_and_tier1(
    const float* __restrict__ x,
    const float* __restrict__ centroid,
    int D,
    float*   __restrict__ out_residual,
    float&   out_norm_sq,
    int8_t*  __restrict__ out_q_int8,
    int32_t& out_q_sum
) {
#if defined(__AVX512F__)
    // ---- AVX-512F path ----
    // Pass 1: residual + norm_sq accumulation (16 floats/iter)
    __m512 norm_acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= D; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vc = _mm512_loadu_ps(centroid + i);
        __m512 r  = _mm512_sub_ps(vx, vc);
        _mm512_storeu_ps(out_residual + i, r);
        norm_acc = _mm512_fmadd_ps(r, r, norm_acc);
    }
    float norm_sq = _mm512_reduce_add_ps(norm_acc);
    // Handle tail (D not multiple of 16)
    for (; i < D; ++i) {
        float r = x[i] - centroid[i];
        out_residual[i] = r;
        norm_sq += r * r;
    }
    out_norm_sq = norm_sq;

    const float scale = (norm_sq > 1e-10f) ? 127.f / sqrtf(norm_sq) : 0.f;
    const __m512 scale16 = _mm512_set1_ps(scale);
    const __m512i hi32   = _mm512_set1_epi32(127);
    const __m512i lo32   = _mm512_set1_epi32(-127);

    // Pass 2: quantize to int8 (16 floats/iter)
    __m512i qsum_acc = _mm512_setzero_epi32();
    i = 0;
    for (; i + 16 <= D; i += 16) {
        __m512  r  = _mm512_loadu_ps(out_residual + i);
        __m512  v  = _mm512_mul_ps(r, scale16);
        // round-half-away-from-zero: add 0.5 if >=0, else -0.5, then truncate
        const __m512  half     = _mm512_set1_ps(0.5f);
        const __m512  neg_half = _mm512_set1_ps(-0.5f);
        __mmask16 pos_mask = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GE_OQ);
        __m512 offset = _mm512_mask_blend_ps(pos_mask, neg_half, half);
        __m512i vi = _mm512_cvttps_epi32(_mm512_add_ps(v, offset));
        vi = _mm512_max_epi32(_mm512_min_epi32(vi, hi32), lo32);
        qsum_acc = _mm512_add_epi32(qsum_acc, vi);
        // Convert int32 → int8 using AVX-512F saturating conversion
        __m128i vi8 = _mm512_cvtsepi32_epi8(vi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out_q_int8 + i), vi8);
    }
    int32_t qsum = _mm512_reduce_add_epi32(qsum_acc);
    // Scalar tail
    for (; i < D; ++i) {
        float v  = out_residual[i] * scale;
        int   vi = (int)(v + (v >= 0.f ? 0.5f : -0.5f));
        int8_t qv = (int8_t)(vi < -127 ? -127 : vi > 127 ? 127 : vi);
        out_q_int8[i] = qv;
        qsum += qv;
    }
    out_q_sum = qsum;

#elif defined(__AVX2__)
    // ---- AVX2 path ----
    // Pass 1: residual + FMA norm_sq accumulation (8 floats/iter)
    __m256 norm_acc0 = _mm256_setzero_ps();
    __m256 norm_acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= D; i += 16) {
        __m256 vx0 = _mm256_loadu_ps(x + i);
        __m256 vc0 = _mm256_loadu_ps(centroid + i);
        __m256 r0  = _mm256_sub_ps(vx0, vc0);
        _mm256_storeu_ps(out_residual + i, r0);
        norm_acc0 = _mm256_fmadd_ps(r0, r0, norm_acc0);

        __m256 vx1 = _mm256_loadu_ps(x + i + 8);
        __m256 vc1 = _mm256_loadu_ps(centroid + i + 8);
        __m256 r1  = _mm256_sub_ps(vx1, vc1);
        _mm256_storeu_ps(out_residual + i + 8, r1);
        norm_acc1 = _mm256_fmadd_ps(r1, r1, norm_acc1);
    }
    for (; i + 8 <= D; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vc = _mm256_loadu_ps(centroid + i);
        __m256 r  = _mm256_sub_ps(vx, vc);
        _mm256_storeu_ps(out_residual + i, r);
        norm_acc0 = _mm256_fmadd_ps(r, r, norm_acc0);
    }

    // Horizontal reduce norm_sq: merge accumulators then reduce
    __m256 norm_acc = _mm256_add_ps(norm_acc0, norm_acc1);
    __m128 lo  = _mm256_castps256_ps128(norm_acc);
    __m128 hi  = _mm256_extractf128_ps(norm_acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float norm_sq = _mm_cvtss_f32(sum);
    // Scalar tail for norm_sq
    for (; i < D; ++i) {
        float r = x[i] - centroid[i];
        out_residual[i] = r;
        norm_sq += r * r;
    }
    out_norm_sq = norm_sq;

    const float scale = (norm_sq > 1e-10f) ? 127.f / sqrtf(norm_sq) : 0.f;
    const __m256  scale8 = _mm256_set1_ps(scale);
    const __m256  pos_half = _mm256_set1_ps(0.5f);
    const __m256  neg_half = _mm256_set1_ps(-0.5f);
    const __m256  zero8    = _mm256_setzero_ps();
    const __m256i hi32 = _mm256_set1_epi32(127);
    const __m256i lo32 = _mm256_set1_epi32(-127);

    // Pass 2: quantize to int8 — 32 floats/iter (4 groups of 8)
    __m256i qsum_acc = _mm256_setzero_si256();
    i = 0;
    for (; i + 32 <= D; i += 32) {
        // Group a: floats [i .. i+7]
        __m256 va = _mm256_mul_ps(_mm256_loadu_ps(out_residual + i),      scale8);
        __m256 vb = _mm256_mul_ps(_mm256_loadu_ps(out_residual + i + 8),  scale8);
        __m256 vc = _mm256_mul_ps(_mm256_loadu_ps(out_residual + i + 16), scale8);
        __m256 vd = _mm256_mul_ps(_mm256_loadu_ps(out_residual + i + 24), scale8);

        // round-half-away-from-zero: add ±0.5 based on sign then truncate
        __m256 oa = _mm256_blendv_ps(neg_half, pos_half, _mm256_cmp_ps(va, zero8, _CMP_GE_OQ));
        __m256 ob = _mm256_blendv_ps(neg_half, pos_half, _mm256_cmp_ps(vb, zero8, _CMP_GE_OQ));
        __m256 oc = _mm256_blendv_ps(neg_half, pos_half, _mm256_cmp_ps(vc, zero8, _CMP_GE_OQ));
        __m256 od = _mm256_blendv_ps(neg_half, pos_half, _mm256_cmp_ps(vd, zero8, _CMP_GE_OQ));

        __m256i ia = _mm256_cvttps_epi32(_mm256_add_ps(va, oa));
        __m256i ib = _mm256_cvttps_epi32(_mm256_add_ps(vb, ob));
        __m256i ic = _mm256_cvttps_epi32(_mm256_add_ps(vc, oc));
        __m256i id = _mm256_cvttps_epi32(_mm256_add_ps(vd, od));

        // Clamp to [-127, 127]
        ia = _mm256_max_epi32(_mm256_min_epi32(ia, hi32), lo32);
        ib = _mm256_max_epi32(_mm256_min_epi32(ib, hi32), lo32);
        ic = _mm256_max_epi32(_mm256_min_epi32(ic, hi32), lo32);
        id = _mm256_max_epi32(_mm256_min_epi32(id, hi32), lo32);

        // Accumulate qsum
        qsum_acc = _mm256_add_epi32(qsum_acc,
            _mm256_add_epi32(_mm256_add_epi32(ia, ib), _mm256_add_epi32(ic, id)));

        // Pack int32→int16
        __m256i ab = _mm256_packs_epi32(ia, ib);
        __m256i cd = _mm256_packs_epi32(ic, id);
        // Fix lane interleave from packs_epi32 (AVX2 operates within 128-bit lanes)
        __m256i ab_fixed = _mm256_permute4x64_epi64(ab, 0xD8);
        __m256i cd_fixed = _mm256_permute4x64_epi64(cd, 0xD8);

        // Pack int16→int8
        __m256i abcd = _mm256_packs_epi16(ab_fixed, cd_fixed);
        // Fix second lane interleave
        abcd = _mm256_permute4x64_epi64(abcd, 0xD8);

        // Store 32 int8s
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_q_int8 + i), abcd);
    }

    // Reduce qsum_acc: lo128 + hi128, then two shuffles+adds
    __m128i qlo = _mm256_castsi256_si128(qsum_acc);
    __m128i qhi = _mm256_extracti128_si256(qsum_acc, 1);
    __m128i qs  = _mm_add_epi32(qlo, qhi);
    qs = _mm_add_epi32(qs, _mm_shuffle_epi32(qs, 0x4E)); // 0b01001110
    qs = _mm_add_epi32(qs, _mm_shuffle_epi32(qs, 0xB1)); // 0b10110001
    int32_t qsum = _mm_cvtsi128_si32(qs);

    // Scalar tail
    for (; i < D; ++i) {
        float v  = out_residual[i] * scale;
        int   vi = (int)(v + (v >= 0.f ? 0.5f : -0.5f));
        int8_t qv = (int8_t)(vi < -127 ? -127 : vi > 127 ? 127 : vi);
        out_q_int8[i] = qv;
        qsum += qv;
    }
    out_q_sum = qsum;

#else
    // ---- Scalar path (used for ARM NEON and other architectures) ----
    // TODO: NEON path
    {
        float norm_sq = 0.f;
        for (int i = 0; i < D; ++i) {
            float r = x[i] - centroid[i];
            out_residual[i] = r;
            norm_sq += r * r;
        }
        out_norm_sq = norm_sq;
        const float scale = (norm_sq > 1e-10f) ? 127.f / sqrtf(norm_sq) : 0.f;
        int32_t qsum = 0;
        for (int i = 0; i < D; ++i) {
            float v  = out_residual[i] * scale;
            int   vi = (int)(v + (v >= 0.f ? 0.5f : -0.5f));
            int8_t qv = (int8_t)(vi < -127 ? -127 : vi > 127 ? 127 : vi);
            out_q_int8[i] = qv;
            qsum += qv;
        }
        out_q_sum = qsum;
    }
#endif
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

#if defined(__AVX512F__)
    // AVX-512F path: 16 × __m512 accumulators (K=256 → K/16=16 groups × 16 codes).
    // Uses 32-wide zmm register file; halves accumulator count vs AVX2 (16 vs 32 regs).
    const int K16 = K / 16;  // 16 groups of 16 codes
    for (int sub = 0; sub < M; ++sub) {
        const float* q_sub  = q_res + sub * D_sub;
        const float* cb_sub = cb_T + sub * D_sub * K;
        float*       out    = lut[sub];

        __m512 acc[16];
        for (int g = 0; g < K16; ++g) acc[g] = _mm512_setzero_ps();

        for (int d = 0; d < D_sub; ++d) {
            __m512 qd = _mm512_set1_ps(q_sub[d]);
            const float* c_row = cb_sub + d * K;
            for (int g = 0; g < K16; ++g)
                acc[g] = _mm512_fmadd_ps(qd, _mm512_loadu_ps(c_row + g * 16), acc[g]);
        }
        for (int g = 0; g < K16; ++g) _mm512_storeu_ps(out + g * 16, acc[g]);
    }
#elif defined(__AVX2__) && defined(__FMA__)
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

/// Variable-M tier-2 LUT build. M = D_eff/8, D_sub = 8 (fixed), K = 256.
/// lut must point to M * 256 floats pre-allocated by the caller.
/// cb_T layout: [M][D_sub][K] (transposed, same convention as build_tier2_codebook_T).
inline void build_tier2_lut_fast_m(const float* q_res, int M,
                                    const float* cb_T,
                                    float* lut)
{
    const int K     = 256;
    const int D_sub = 8;

#if defined(__AVX512F__)
    const int K16 = K / 16;
    for (int sub = 0; sub < M; ++sub) {
        const float* q_sub  = q_res + sub * D_sub;
        const float* cb_sub = cb_T + sub * D_sub * K;
        float*       out    = lut + sub * K;
        __m512 acc[16];
        for (int g = 0; g < K16; ++g) acc[g] = _mm512_setzero_ps();
        for (int d = 0; d < D_sub; ++d) {
            __m512 qd = _mm512_set1_ps(q_sub[d]);
            const float* c = cb_sub + d * K;
            for (int g = 0; g < K16; ++g)
                acc[g] = _mm512_fmadd_ps(qd, _mm512_loadu_ps(c + g * 16), acc[g]);
        }
        for (int g = 0; g < K16; ++g) _mm512_storeu_ps(out + g * 16, acc[g]);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    const int K8 = K / 8;
    for (int sub = 0; sub < M; ++sub) {
        const float* q_sub  = q_res + sub * D_sub;
        const float* cb_sub = cb_T + sub * D_sub * K;
        float*       out    = lut + sub * K;
        __m256 acc[32];
        for (int g = 0; g < K8; ++g) acc[g] = _mm256_setzero_ps();
        for (int d = 0; d < D_sub; ++d) {
            __m256 qd = _mm256_set1_ps(q_sub[d]);
            const float* c = cb_sub + d * K;
            for (int g = 0; g < K8; ++g)
                acc[g] = _mm256_fmadd_ps(qd, _mm256_loadu_ps(c + g * 8), acc[g]);
        }
        for (int g = 0; g < K8; ++g) _mm256_storeu_ps(out + g * 8, acc[g]);
    }
#else
    for (int sub = 0; sub < M; ++sub) {
        const float* q_sub  = q_res + sub * D_sub;
        const float* cb_sub = cb_T + sub * D_sub * K;
        float*       out    = lut + sub * K;
        for (int code = 0; code < K; ++code) {
            float dot = 0.f;
            for (int d = 0; d < D_sub; ++d) dot += q_sub[d] * cb_sub[d * K + code];
            out[code] = dot;
        }
    }
#endif
}

/// Variable-M tier-2 PQ ADC score. lut[M * 256], tier2[M].
inline float tier2_adc_pq_m(const float* lut, const uint8_t* tier2, int M) {
    float sum = 0.f;
    for (int sub = 0; sub < M; ++sub)
        sum += lut[(size_t)sub * 256 + tier2[sub]];
    return sum;
}

}} // NGT::ArcFlare
