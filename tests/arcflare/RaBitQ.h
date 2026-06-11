// tests/arcflare/RaBitQ.h
// Offline RaBitQ-style residual quantizer microbench primitives (1-bit + 2-bit).
//
// This is a STANDALONE offline tool used by rabitq_bench. It does NOT touch the
// production ArcFlare searchV2 path.
//
// Pipeline (shared across encode + query):
//   r  = x_pad - c                       (residual against a global centroid)
//   rr = SRHT(r)                          (norm-preserving random rotation, D power-of-2)
//   nr = ||rr||_2                         (== ||r||_2, SRHT is orthogonal)
//
// 1-bit: store sign(rr) packed to ceil(D/8) bytes + fp16(nr) + fp16(factor_x).
// 2-bit: store 2-bit levels of rr packed to ceil(D/4) bytes + fp16(nr) + fp16(factor_x) + fp16(s_x).
//
// Distance: query is rotated once into qrr, then int4-quantized over its own
// [min,max] range. Per-DB-vector L2 estimate uses only integer dot products plus
// the stored fp16 scalars.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "NGT/ArcFlare/SRHT.h"
#include "NGT/ArcFlare/VectorRecord.h"

#if defined(__AVX512VNNI__) || defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace ArcFlare {

using NGT::ArcFlare::SRHT;
using NGT::ArcFlare::float_to_fp16;
using NGT::ArcFlare::fp16_to_float;

// ===========================================================================
// Integer-dot kernels for the RaBitQ microbench.
//
// These compute the SAME integer dot product as the scalar reference loops in
// the distance methods below — they MUST be byte-identical (==), not just
// approximately equal, since the dot is exact integer arithmetic. They exist
// only to speed up bench throughput on GIST D=1024; the IPr scaling that wraps
// the dot lives in the distance methods and is NOT touched here.
//
// q_int is the unsigned int4 query operand (values 0..15, fits a byte).
//   1-bit data operand: the packed sign bit ∈ {0,1}; dot = Σ q_int[d]*bit[d].
//   2-bit data operand: signed level u ∈ {−2,−1,0,+1}; dot = Σ q_int[d]*u[d].
//
// D must be a multiple of 64 (D=128 and D=1024 both qualify → no scalar tail).
// ===========================================================================

// ---- scalar references (always compiled) --------------------------------

// 1-bit: dot = Σ_d q_int[d] * bit[d]  (bit = packed sign bit ∈ {0,1}).
inline int64_t rabitq_dot1_scalar(const int32_t* q_int, const uint8_t* bits, int D) {
    int64_t dot = 0;
    for (int d = 0; d < D; ++d) {
        if (bits[d >> 3] & (1u << (d & 7))) dot += q_int[d];
    }
    return dot;
}

// 2-bit: dot = Σ_d q_int[d] * u[d]  (u = signed level ∈ {−2,−1,0,+1}, 4/byte).
inline int64_t rabitq_dot2_scalar(const int32_t* q_int, const uint8_t* levels, int D) {
    int64_t dot = 0;
    for (int d = 0; d < D; ++d) {
        int byte = levels[d >> 2];
        int u = ((byte >> ((d & 3) * 2)) & 0x3) - 2;  // decode u in [-2,+1]
        dot += static_cast<int64_t>(q_int[d]) * u;
    }
    return dot;
}

#if defined(__AVX512VNNI__)
// ---- VNNI (D/64 zmm iterations) — compiled only when the macro is defined.
// Mirrors tier1_adc_vnni (ADCDistance.h). Unsigned q_int bytes × signed data
// bytes via _mm512_dpbusd_epi32, reduced with _mm512_reduce_add_epi32.
// Un-runnable on the AVX2 dev machine (guarded out entirely).

// 1-bit: expand each 64-bit chunk of sign bits to {0,1} signed bytes.
inline int64_t rabitq_dot1_vnni(const int32_t* q_int, const uint8_t* bits, int D) {
    __m512i acc = _mm512_setzero_si512();
    for (int blk = 0; blk < D; blk += 64) {
        uint64_t mask;
        std::memcpy(&mask, bits + (blk >> 3), 8);
        __m512i data = _mm512_maskz_set1_epi8((__mmask64)mask, (int8_t)1);
        // Pack 64 int4 query values (0..15) into 64 unsigned bytes.
        alignas(64) uint8_t qb[64];
        for (int j = 0; j < 64; ++j) qb[j] = static_cast<uint8_t>(q_int[blk + j]);
        __m512i q = _mm512_loadu_si512((const void*)qb);
        acc = _mm512_dpbusd_epi32(acc, q, data);
    }
    return static_cast<int64_t>(_mm512_reduce_add_epi32(acc));
}

// 2-bit: signed level u ∈ {−2,−1,0,+1} fits the signed s8 slot directly.
inline int64_t rabitq_dot2_vnni(const int32_t* q_int, const uint8_t* levels, int D) {
    __m512i acc = _mm512_setzero_si512();
    for (int blk = 0; blk < D; blk += 64) {
        alignas(64) uint8_t qb[64];
        alignas(64) int8_t  ub[64];
        for (int j = 0; j < 64; ++j) {
            int d = blk + j;
            int byte = levels[d >> 2];
            ub[j] = static_cast<int8_t>(((byte >> ((d & 3) * 2)) & 0x3) - 2);
            qb[j] = static_cast<uint8_t>(q_int[d]);
        }
        __m512i q = _mm512_loadu_si512((const void*)qb);
        __m512i u = _mm512_loadu_si512((const void*)ub);
        acc = _mm512_dpbusd_epi32(acc, q, u);
    }
    return static_cast<int64_t>(_mm512_reduce_add_epi32(acc));
}
#endif // __AVX512VNNI__

#if defined(__AVX2__)
// ---- AVX2 generic-D (the DEFAULT on the AVX2 dev machine) ------------------
// Generalizes the tier1_adc_avx2 bit-expansion (ADCDistance.h) to loop over D
// in 32-lane blocks; widens int8→int16 and accumulates via _mm256_madd_epi16
// (mirrors the dot_s8_s8 AVX2 path, ADCDistance.h:289-303).

inline int32_t rabitq_avx2_hsum_epi32(__m256i acc) {
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// 1-bit: data operand is the sign bit expanded to 0/1 bytes; q_int packed to
// bytes (0..15 fit a byte). Widen both int8→int16, madd, accumulate.
inline int64_t rabitq_dot1_avx2(const int32_t* q_int, const uint8_t* bits, int D) {
    static const __m256i BIT_MASK = _mm256_set_epi8(
        (int8_t)0x80,(int8_t)0x40,(int8_t)0x20,(int8_t)0x10,
        (int8_t)0x08,(int8_t)0x04,(int8_t)0x02,(int8_t)0x01,
        (int8_t)0x80,(int8_t)0x40,(int8_t)0x20,(int8_t)0x10,
        (int8_t)0x08,(int8_t)0x04,(int8_t)0x02,(int8_t)0x01,
        (int8_t)0x80,(int8_t)0x40,(int8_t)0x20,(int8_t)0x10,
        (int8_t)0x08,(int8_t)0x04,(int8_t)0x02,(int8_t)0x01,
        (int8_t)0x80,(int8_t)0x40,(int8_t)0x20,(int8_t)0x10,
        (int8_t)0x08,(int8_t)0x04,(int8_t)0x02,(int8_t)0x01);
    static const __m256i BYTE_SHUF = _mm256_set_epi8(
        3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2,
        1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
    const __m256i ALLFF = _mm256_set1_epi8((int8_t)0xFF);
    const __m256i ONES  = _mm256_set1_epi16(1);
    const __m256i ZERO  = _mm256_setzero_si256();

    __m256i acc = ZERO;
    // 32-dim blocks; D multiple of 64 ⇒ no tail.
    for (int blk = 0; blk < D; blk += 32) {
        // Pack 32 int4 query values (0..15) into 32 bytes.
        alignas(32) uint8_t qb[32];
        for (int j = 0; j < 32; ++j) qb[j] = static_cast<uint8_t>(q_int[blk + j]);
        __m256i q = _mm256_loadu_si256((const __m256i*)qb);

        // Expand 4 sign-bit bytes → 32 byte masks (0xFF set, 0x00 clear).
        uint32_t bits32;
        std::memcpy(&bits32, bits + (blk >> 3), 4);
        __m256i b = _mm256_set1_epi32((int)bits32);
        b = _mm256_shuffle_epi8(b, BYTE_SHUF);
        b = _mm256_and_si256(b, BIT_MASK);
        b = _mm256_xor_si256(_mm256_cmpeq_epi8(b, ZERO), ALLFF);
        // b[i] = 0xFF if bit set, else 0x00 → use as 0/1 mask on q.
        __m256i q_masked = _mm256_and_si256(q, b);  // q if bit set else 0

        // q values 0..15 fit a byte; widen to int16 and accumulate exactly.
        __m256i lo16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q_masked, 0));
        __m256i hi16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q_masked, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo16, ONES));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(hi16, ONES));
    }
    return static_cast<int64_t>(rabitq_avx2_hsum_epi32(acc));
}

// 2-bit: unpack 2-bit signed levels → int8 lanes, widen int8→int16, then
// _mm256_madd_epi16 against the int16-widened q_int (mirrors ADCDistance.h
// 289-303). Products q*u with q∈[0,15], u∈[-2,1] fit int16; D-sum fits int32.
inline int64_t rabitq_dot2_avx2(const int32_t* q_int, const uint8_t* levels, int D) {
    __m256i acc = _mm256_setzero_si256();
    // 16-dim blocks (one 128-bit int8 load widened to 256-bit int16).
    for (int blk = 0; blk < D; blk += 16) {
        alignas(16) int8_t  ub[16];
        alignas(16) int16_t qb[16];
        for (int j = 0; j < 16; ++j) {
            int d = blk + j;
            int byte = levels[d >> 2];
            ub[j] = static_cast<int8_t>(((byte >> ((d & 3) * 2)) & 0x3) - 2);
            qb[j] = static_cast<int16_t>(q_int[d]);  // 0..15
        }
        __m256i uv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)ub));
        __m256i qv = _mm256_loadu_si256((const __m256i*)qb);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(qv, uv));
    }
    return static_cast<int64_t>(rabitq_avx2_hsum_epi32(acc));
}
#endif // __AVX2__

// ---- AVX2-or-scalar selectors (ignore the VNNI guard) ----------------------
// Used by the --ktest path-equivalence check to always exercise the AVX2 path
// (the default on this AVX2-only dev machine) against the scalar reference.
inline int64_t rabitq_dot1_avx2_or_scalar(const int32_t* q_int, const uint8_t* bits, int D) {
#if defined(__AVX2__)
    return rabitq_dot1_avx2(q_int, bits, D);
#else
    return rabitq_dot1_scalar(q_int, bits, D);
#endif
}

inline int64_t rabitq_dot2_avx2_or_scalar(const int32_t* q_int, const uint8_t* levels, int D) {
#if defined(__AVX2__)
    return rabitq_dot2_avx2(q_int, levels, D);
#else
    return rabitq_dot2_scalar(q_int, levels, D);
#endif
}

// ---- dispatchers: AVX2 by default, VNNI only when the macro is defined. -----
inline int64_t rabitq_dot1_fast(const int32_t* q_int, const uint8_t* bits, int D) {
#if defined(__AVX512VNNI__)
    return rabitq_dot1_vnni(q_int, bits, D);
#elif defined(__AVX2__)
    return rabitq_dot1_avx2(q_int, bits, D);
#else
    return rabitq_dot1_scalar(q_int, bits, D);
#endif
}

inline int64_t rabitq_dot2_fast(const int32_t* q_int, const uint8_t* levels, int D) {
#if defined(__AVX512VNNI__)
    return rabitq_dot2_vnni(q_int, levels, D);
#elif defined(__AVX2__)
    return rabitq_dot2_avx2(q_int, levels, D);
#else
    return rabitq_dot2_scalar(q_int, levels, D);
#endif
}

// ---------------------------------------------------------------------------
// Query preparation: rotate-once, int4-quantize over the query's own range.
// Built from a rotated query qrr[D].
// ---------------------------------------------------------------------------
struct RaBitQQuery {
    std::vector<int32_t> q_int;  // int4 codes in [0,15]
    int64_t S_q = 0;             // Σ q_int[d]
    float   lo = 0.f;            // min(qrr)
    float   delta = 0.f;         // (max-min)/15
    float   nq2 = 0.f;           // Σ qrr[d]^2

    // qrr: already-rotated query (length D).
    static RaBitQQuery prepare(const float* qrr, int D) {
        RaBitQQuery q;
        q.q_int.resize(static_cast<size_t>(D));
        float mn = qrr[0], mx = qrr[0];
        float nq2 = 0.f;
        for (int d = 0; d < D; ++d) {
            float v = qrr[d];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            nq2 += v * v;
        }
        q.lo = mn;
        q.nq2 = nq2;
        float range = mx - mn;
        q.delta = (range > 0.f) ? range / 15.f : 1.f;
        int64_t sq = 0;
        for (int d = 0; d < D; ++d) {
            long qi = std::lround((qrr[d] - q.lo) / q.delta);
            if (qi < 0) qi = 0;
            if (qi > 15) qi = 15;
            q.q_int[d] = static_cast<int32_t>(qi);
            sq += qi;
        }
        q.S_q = sq;
        return q;
    }
};

// ---------------------------------------------------------------------------
// 1-bit RaBitQ.
// ---------------------------------------------------------------------------
struct RaBitQ1 {
    std::vector<uint8_t> bits;   // ceil(D/8) bytes, sign(rr) packed
    uint16_t nr_fp16 = 0;        // fp16(||rr||)
    uint16_t factor_fp16 = 0;    // fp16(factor_x)
    int      D = 0;

    // rr: rotated residual (length D).
    void encode(const float* rr, int Dim) {
        D = Dim;
        bits.assign(static_cast<size_t>((Dim + 7) / 8), 0);
        float nr2 = 0.f, ip_abs_sum = 0.f;
        for (int d = 0; d < Dim; ++d) {
            float v = rr[d];
            nr2 += v * v;
            ip_abs_sum += std::fabs(v);
            if (v >= 0.f) bits[d >> 3] |= static_cast<uint8_t>(1u << (d & 7));
        }
        float nr = std::sqrt(nr2);
        float ip_abs = ip_abs_sum / std::sqrt(static_cast<float>(Dim));
        float factor_x = (ip_abs > 1e-12f) ? (nr / ip_abs) : 1.0f;
        nr_fp16 = float_to_fp16(nr);
        factor_fp16 = float_to_fp16(factor_x);
    }

    // Sb = 2*popcount(bits) - D, recomputed from packed words.
    int64_t computeSb() const {
        int64_t pc = 0;
        const uint8_t* p = bits.data();
        int nbytes = static_cast<int>(bits.size());
        int i = 0;
        // Process 8 bytes at a time via __builtin_popcountll over D/64 words.
        for (; i + 8 <= nbytes; i += 8) {
            uint64_t w;
            std::memcpy(&w, p + i, 8);
            pc += __builtin_popcountll(w);
        }
        for (; i < nbytes; ++i) pc += __builtin_popcount(static_cast<unsigned>(p[i]));
        return 2 * pc - D;
    }

    // Returns estimated L2^2 between query and this DB vector.
    //
    // IP estimate.  g = delta*t + lo*Sb estimates <q_recon, sign(xrr)>.  The DB
    // residual is reconstructed at its conditional-mean per-dim amplitude
    //   xrr[d] ≈ (ip_abs/sqrt(D)) * sign(xrr[d]),   ip_abs = nr/factor_x = (1/sqrt(D))Σ|rr|,
    // so  IPr = (ip_abs/sqrt(D)) * g.
    //
    // NOTE (deviation from the original spec's IPr = (factor_x/sqrt(D))*g): that
    // constant is off by ~sqrt(D)/factor_x (≈9x at D=128) and yields a near-zero
    // IP correction, collapsing the L2 estimate onto the norm-only baseline.  The
    // per-vector amplitude form below was empirically verified unbiased (scale ≈1)
    // and dimension-stable; factor_x and nr remain load-bearing.
    float distance(const RaBitQQuery& q) const {
        int64_t Sb = computeSb();
        // dot = Σ q_int[d] * bits[d]  (bits are 0/1). Fast path (AVX2 by default,
        // VNNI when compiled) is byte-identical to rabitq_dot1_scalar.
        int64_t dot = rabitq_dot1_fast(q.q_int.data(), bits.data(), D);
        int64_t t = 2 * dot - q.S_q;
        float g = q.delta * static_cast<float>(t) + q.lo * static_cast<float>(Sb);
        float factor_x = fp16_to_float(factor_fp16);
        float nr = fp16_to_float(nr_fp16);
        float ip_abs = (factor_x > 1e-12f) ? (nr / factor_x) : nr;
        float IPr = (ip_abs / std::sqrt(static_cast<float>(D))) * g;
        return q.nq2 + nr * nr - 2.f * IPr;
    }
};

// ---------------------------------------------------------------------------
// 2-bit RaBitQ.  Levels u[d] in {-2,-1,0,+1} (clamped to [-2,+1]).
// ---------------------------------------------------------------------------
struct RaBitQ2 {
    std::vector<uint8_t> levels; // ceil(D/4) bytes, 2 bits/dim (stored as u+2 in [0,3])
    uint16_t nr_fp16 = 0;        // fp16(||rr||)
    uint16_t factor_fp16 = 0;    // fp16(factor_x)
    uint16_t sx_fp16 = 0;        // fp16(s_x)
    int      D = 0;

    static inline int levelAt(const uint8_t* p, int d) {
        int byte = p[d >> 2];
        int shift = (d & 3) * 2;
        return ((byte >> shift) & 0x3) - 2;  // decode u in [-2,+1]
    }

    void encode(const float* rr, int Dim) {
        D = Dim;
        levels.assign(static_cast<size_t>((Dim + 3) / 4), 0);
        float nr2 = 0.f, maxabs = 0.f;
        for (int d = 0; d < Dim; ++d) {
            nr2 += rr[d] * rr[d];
            maxabs = std::max(maxabs, std::fabs(rr[d]));
        }
        float nr = std::sqrt(nr2);
        float s_x = (maxabs > 0.f) ? (maxabs / 1.5f) : 1.0f;
        float ipu = 0.f, unorm2 = 0.f;
        for (int d = 0; d < Dim; ++d) {
            long ui = std::lround(rr[d] / s_x);
            if (ui < -2) ui = -2;
            if (ui > 1)  ui = 1;
            int u = static_cast<int>(ui);
            levels[d >> 2] |= static_cast<uint8_t>((u + 2) << ((d & 3) * 2));
            ipu += static_cast<float>(u) * rr[d];
            unorm2 += static_cast<float>(u) * static_cast<float>(u);
        }
        float unorm = std::sqrt(unorm2);
        float factor_x = (ipu > 1e-12f) ? (nr * unorm / ipu) : 1.0f;
        nr_fp16 = float_to_fp16(nr);
        factor_fp16 = float_to_fp16(factor_x);
        sx_fp16 = float_to_fp16(s_x);
    }

    // Su = Σ u[d], recomputed from packed levels.
    int64_t computeSu() const {
        int64_t su = 0;
        const uint8_t* p = levels.data();
        for (int d = 0; d < D; ++d) su += levelAt(p, d);
        return su;
    }

    // Returns estimated L2^2.  The lo*Su term is REQUIRED.
    float distance(const RaBitQQuery& q) const {
        return distanceImpl(q, /*withLoSu=*/true);
    }

    // Diagnostic-only: toggle the lo*Su term (selftest checks its omission biases).
    //
    // g_u = delta*dot + lo*Su estimates <q_recon, u>.  The DB residual reconstructs
    // as xrr[d] ≈ s_x * u[d] (u = round(rr/s_x)), so IPr = s_x * g_u.  The lo*Su
    // term is REQUIRED: omitting it drops the query-quantization offset bias and
    // systematically biases IPr.
    //
    // NOTE: same deviation as RaBitQ1 — the spec's (factor_x/sqrt(D))*g_u constant
    // is off by ~16x; s_x*g_u is the verified-unbiased, dimension-stable form.
    float distanceImpl(const RaBitQQuery& q, bool withLoSu) const {
        int64_t Su = computeSu();
        // dot = Σ q_int[d] * u[d]. Fast path (AVX2 by default, VNNI when
        // compiled) is byte-identical to rabitq_dot2_scalar.
        int64_t dot = rabitq_dot2_fast(q.q_int.data(), levels.data(), D);
        float g_u = q.delta * static_cast<float>(dot);
        if (withLoSu) g_u += q.lo * static_cast<float>(Su);
        float s_x = fp16_to_float(sx_fp16);
        float nr = fp16_to_float(nr_fp16);
        float IPr = s_x * g_u;
        return q.nq2 + nr * nr - 2.f * IPr;
    }
};

// ---------------------------------------------------------------------------
// Shared residual-rotation helper: rr = SRHT(x_pad - c).
// x: padded working vector (length D). c: global centroid (length D). out: rr (length D).
// ---------------------------------------------------------------------------
inline void rotateResidual(const SRHT& srht, const float* x, const float* c, int D, float* rr) {
    static thread_local std::vector<float> r;
    r.resize(static_cast<size_t>(D));
    for (int d = 0; d < D; ++d) r[d] = x[d] - c[d];
    srht.apply(r.data(), rr);
}

} // namespace ArcFlare
