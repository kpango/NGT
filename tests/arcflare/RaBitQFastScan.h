// tests/arcflare/RaBitQFastScan.h
// FastScan-over-RaBitQ-bit block distance kernel + 3-factor RaBitQ estimator.
// The make-or-break core of a SymphonyQG-class quantized graph: 1-bit-per-dim RaBitQ
// codes packed into Faiss-style 32-vector FastScan blocks (4 sign-bits -> one 4-bit
// sub-segment index into a 16-entry LUT), accumulated with vpshufb (AVX2; AVX-512
// optional). Distance is the RaBitQ unbiased estimator folded into a 3-factor
// per-vector post-process.
//
// The FastScan packing/accumulate/LUT and the 3-factor estimator are ported (adapted,
// de-Eigen'd) from SymphonyQG (SIGMOD'25), Apache-2.0:
//   github.com/gouyt13/SymphonyQG  symqglib/{quantization/fastscan_impl.hpp,
//   quantization/rabitq.hpp, qg/qg_scanner.hpp, utils/scalar_quantize.hpp}
// Rotation uses our own SRHT (orthonormal, == SymphonyQG's FHTRotator role).
#pragma once

#include <immintrin.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "NGT/ArcFlare/SRHT.h"

namespace ArcFlare {
namespace rbfs {

inline constexpr int QG_BQUERY = 6;  // query scalar-quant bits (matches SymphonyQG)
#define RBFS_LOWBIT(x) ((x) & (-(x)))

inline constexpr std::array<int, 16> kPos = {3, 3, 2, 3, 1, 3, 2, 3,
                                             0, 3, 2, 3, 1, 3, 2, 3};
inline constexpr std::array<int, 16> kPerm0 = {0, 8,  1, 9,  2, 10, 3, 11,
                                               4, 12, 5, 13, 6, 14, 7, 15};

// bin (0/1 ints) -> uint64 words, MSB-first within each 64-group (SymphonyQG convention).
inline void pack_binary(const int* bin, uint64_t* out, size_t length) {
    for (size_t i = 0; i < length; i += 64) {
        uint64_t cur = 0;
        for (size_t j = 0; j < 64; ++j) cur |= (static_cast<uint64_t>(bin[i + j]) << (63 - j));
        *out++ = cur;
    }
}

template <typename T, class TA>
inline void get_column(const T* src, size_t rows, size_t cols, size_t row, size_t col, TA& dest) {
    size_t k = 0, max_k = std::min(rows - row, dest.size());
    for (; k < max_k; ++k) dest[k] = src[((k + row) * cols) + col];
    if (k < dest.size()) std::fill(dest.begin() + k, dest.end(), 0);
}

inline void pack_codes_helper(size_t padded_dim, const uint8_t* codes, size_t ncode,
                              uint8_t* blocks) {
    size_t ncode_pad = (ncode + 31) & ~size_t(31);
    size_t num_codebook = padded_dim / 4;
    std::memset(blocks, 0, ncode_pad * num_codebook / 2);
    uint8_t* codes2 = blocks;
    for (size_t blk = 0; blk < ncode_pad; blk += 32) {
        for (size_t i = 0; i < num_codebook; i += 2) {
            std::array<uint8_t, 32> col, col_lo, col_hi;
            get_column(codes, ncode, num_codebook / 2, blk, i / 2, col);
            for (int j = 0; j < 32; ++j) { col_lo[j] = col[j] & 15; col_hi[j] = col[j] >> 4; }
            for (int j = 0; j < 16; ++j) {
                codes2[j]      = col_lo[kPerm0[j]] | (col_lo[kPerm0[j] + 16] << 4);
                codes2[j + 16] = col_hi[kPerm0[j]] | (col_hi[kPerm0[j] + 16] << 4);
            }
            codes2 += 32;
        }
    }
}

// binary_code: ncode * (padded_dim/64) uint64 words -> packed FastScan blocks.
inline void pack_codes(size_t padded_dim, const uint64_t* binary_code, size_t ncode,
                       uint8_t* blocks) {
    size_t ncode_pad = (ncode + 31) & ~size_t(31);
    std::vector<uint8_t> b8(ncode_pad * padded_dim / 8, 0);
    std::memcpy(b8.data(), binary_code, ncode * padded_dim / 64 * sizeof(uint64_t));
    for (size_t i = 0; i < ncode; ++i)
        for (size_t j = 0; j < padded_dim / 64; ++j)
            for (size_t k = 0; k < 4; ++k)
                std::swap(b8[(i * padded_dim / 8) + (8 * j) + k],
                          b8[(i * padded_dim / 8) + (8 * j) + 8 - k - 1]);
    for (size_t i = 0; i < ncode * padded_dim / 8; ++i) {
        uint8_t v = b8[i];
        b8[i] = ((v & 15) << 4) | (v >> 4);
    }
    pack_codes_helper(padded_dim, b8.data(), ncode, blocks);
}

// per-segment 16-entry LUT from the int-quantized query (Gray-code accumulation).
inline void pack_lut(size_t padded_dim, const uint8_t* byte_query, uint8_t* LUT) {
    size_t num_codebook = padded_dim >> 2;
    for (size_t i = 0; i < num_codebook; ++i) {
        LUT[0] = 0;
        for (int j = 1; j < 16; ++j) LUT[j] = LUT[j - RBFS_LOWBIT(j)] + byte_query[kPos[j]];
        LUT += 16;
        byte_query += 4;
    }
}

// accumulate ONE 32-vector block -> result[32] uint16 (= sum_d byte_query[d]*bit[d] per vec).
inline void accumulate_block(size_t padded_dim, const uint8_t* __restrict__ codes,
                             const uint8_t* __restrict__ LUT, uint16_t* __restrict__ result) {
    size_t code_length = padded_dim << 2;
#if defined(__AVX512F__)
    const __m512i lo_mask = _mm512_set1_epi8(0x0f);
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
    for (size_t i = 0; i < code_length; i += 64) {
        __m512i c = _mm512_loadu_si512(&codes[i]);
        __m512i lut = _mm512_loadu_si512(&LUT[i]);
        __m512i lo = _mm512_and_si512(c, lo_mask);
        __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);
        __m512i rl = _mm512_shuffle_epi8(lut, lo), rh = _mm512_shuffle_epi8(lut, hi);
        a0 = _mm512_add_epi16(a0, rl); a1 = _mm512_add_epi16(a1, _mm512_srli_epi16(rl, 8));
        a2 = _mm512_add_epi16(a2, rh); a3 = _mm512_add_epi16(a3, _mm512_srli_epi16(rh, 8));
    }
    a0 = _mm512_sub_epi16(a0, _mm512_slli_epi16(a1, 8));
    a2 = _mm512_sub_epi16(a2, _mm512_slli_epi16(a3, 8));
    __m512i r1 = _mm512_add_epi16(_mm512_mask_blend_epi64(0b11110000, a0, a1),
                                  _mm512_shuffle_i64x2(a0, a1, 0b01001110));
    __m512i r2 = _mm512_add_epi16(_mm512_mask_blend_epi64(0b11110000, a2, a3),
                                  _mm512_shuffle_i64x2(a2, a3, 0b01001110));
    __m512i r = _mm512_add_epi16(_mm512_shuffle_i64x2(r1, r2, 0b10001000),
                                 _mm512_shuffle_i64x2(r1, r2, 0b11011101));
    _mm512_storeu_si512(result, r);
#elif defined(__AVX2__)
    const __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    for (size_t i = 0; i < code_length; i += 64) {
        for (size_t off = 0; off < 64; off += 32) {
            __m256i c = _mm256_loadu_si256((const __m256i*)&codes[i + off]);
            __m256i lut = _mm256_loadu_si256((const __m256i*)&LUT[i + off]);
            __m256i lo = _mm256_and_si256(c, low_mask);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
            __m256i rl = _mm256_shuffle_epi8(lut, lo), rh = _mm256_shuffle_epi8(lut, hi);
            a0 = _mm256_add_epi16(a0, rl); a1 = _mm256_add_epi16(a1, _mm256_srli_epi16(rl, 8));
            a2 = _mm256_add_epi16(a2, rh); a3 = _mm256_add_epi16(a3, _mm256_srli_epi16(rh, 8));
        }
    }
    a0 = _mm256_sub_epi16(a0, _mm256_slli_epi16(a1, 8));
    __m256i d0 = _mm256_add_epi16(_mm256_permute2f128_si256(a0, a1, 0x21),
                                  _mm256_blend_epi32(a0, a1, 0xF0));
    _mm256_storeu_si256((__m256i*)result, d0);
    a2 = _mm256_sub_epi16(a2, _mm256_slli_epi16(a3, 8));
    __m256i d1 = _mm256_add_epi16(_mm256_permute2f128_si256(a2, a3, 0x21),
                                  _mm256_blend_epi32(a2, a3, 0xF0));
    _mm256_storeu_si256((__m256i*)&result[16], d1);
#else
#error "RaBitQFastScan requires AVX2 or AVX-512"
#endif
}

inline void data_range(const float* v, size_t d, float& lo, float& hi) {
    lo = v[0]; hi = v[0];
    for (size_t i = 1; i < d; ++i) { lo = v[i] < lo ? v[i] : lo; hi = v[i] > hi ? v[i] : hi; }
}
inline void quantize_q(uint8_t* out, const float* v, size_t d, float lo, float width,
                       int32_t& sumq) {
    float inv = 1.0f / width;
    int32_t s = 0;
    const int maxv = (1 << QG_BQUERY) - 1;
    for (size_t i = 0; i < d; ++i) {
        int c = (int)std::lround(((v[i] - lo) * inv) + 0.5f);
        if (c < 0) c = 0; if (c > maxv) c = maxv;
        out[i] = (uint8_t)c;
        s += c;
    }
    sumq = s;
}

// --- standalone single-centroid index (for validation; on-graph uses vertex-relative c) ---
struct FSIndex {
    int raw_dim = 0, D = 0, N = 0, Npad = 0, num_codebook = 0;
    NGT::ArcFlare::SRHT srht;
    std::vector<float> c_raw;   // raw-space centroid (mean), [D]
    std::vector<float> c_rot;   // rotated centroid, [D]
    std::vector<uint8_t> blocks;             // packed FastScan codes
    std::vector<float> triple_x, fac_dq, fac_vq;  // [Npad]
    std::vector<uint64_t> binary;            // [N * D/64] logical bits (for scalar ref)

    explicit FSIndex(int raw_dim_, uint64_t seed)
        : raw_dim(raw_dim_),
          D([&] { int p = 1; while (p < raw_dim_) p <<= 1; return p < 64 ? 64 : p; }()),
          num_codebook(D / 4),
          srht(D, seed) {}

    static int popcnt_bits(const uint64_t* w, int words) {
        int s = 0;
        for (int i = 0; i < words; ++i) s += __builtin_popcountll(w[i]);
        return s;
    }

    // encode N raw vectors (row-major raw_dim). centroid = mean (rotated).
    void encode(const float* data, int n) {
        N = n;
        Npad = (N + 31) & ~31;
        const int words = D / 64;
        std::vector<float> mean(D, 0.f);
        std::vector<float> xpad(D), xrot(D);
        for (int i = 0; i < N; ++i) {
            const float* x = data + (size_t)i * raw_dim;
            for (int d = 0; d < raw_dim; ++d) mean[d] += x[d];
        }
        for (int d = 0; d < D; ++d) mean[d] /= (float)N;
        c_raw = mean;
        c_rot.assign(D, 0.f);
        srht.apply(mean.data(), c_rot.data());

        binary.assign((size_t)N * words, 0);
        triple_x.assign(Npad, 0.f); fac_dq.assign(Npad, 0.f); fac_vq.assign(Npad, 0.f);
        std::vector<uint8_t> code8;  // not used; we go bits->pack_codes per all
        const float fac_norm = 1.0f / std::sqrt((float)D);
        std::vector<int> bin(D);
        for (int i = 0; i < N; ++i) {
            const float* x = data + (size_t)i * raw_dim;
            std::fill(xpad.begin(), xpad.end(), 0.f);
            std::memcpy(xpad.data(), x, raw_dim * sizeof(float));
            srht.apply(xpad.data(), xrot.data());
            // residual to rotated centroid
            double nr2 = 0, ipabs = 0, ipc = 0;
            for (int d = 0; d < D; ++d) {
                float r = xrot[d] - c_rot[d];
                int b = r > 0.f ? 1 : 0;
                bin[d] = b;
                float sgn = 2.f * b - 1.f;
                nr2 += (double)r * r;
                ipabs += (double)r * sgn;      // = sum |r|
                ipc += (double)c_rot[d] * sgn; // <c, signed>
            }
            pack_binary(bin.data(), &binary[(size_t)i * words], D);
            float nr = (float)std::sqrt(nr2);
            float fac_x0 = (float)(ipabs * fac_norm) / (nr > 1e-12f ? nr : 1e-12f);
            float fac_x1 = (float)(ipc * fac_norm);
            float x_x0 = nr / (fac_x0 != 0.f ? fac_x0 : 1e-12f);
            int popc = popcnt_bits(&binary[(size_t)i * words], words);
            triple_x[i] = nr * nr + 2.f * x_x0 * fac_x1;
            fac_dq[i]   = -2.f * x_x0 * fac_norm;
            fac_vq[i]   = fac_dq[i] * (float)(2 * popc - D);
        }
        // pack all codes into FastScan blocks
        blocks.assign((size_t)Npad * num_codebook / 2, 0);
        pack_codes(D, binary.data(), N, blocks.data());
    }

    // per-query scratch
    struct Query {
        std::vector<uint8_t> lut;
        std::vector<uint8_t> byte_q;   // [D] for scalar ref
        float width = 0, vl = 0;
        int32_t sumq = 0;
        float sqr_y_to_c = 0;          // ||q - c||^2 (raw)
        std::vector<float> q_raw_pad;  // [D]
    };

    void query_prepare(const float* q, Query& out) const {
        out.q_raw_pad.assign(D, 0.f);
        std::memcpy(out.q_raw_pad.data(), q, raw_dim * sizeof(float));
        std::vector<float> qrot(D);
        srht.apply(out.q_raw_pad.data(), qrot.data());
        float lo, hi; data_range(qrot.data(), D, lo, hi);
        out.width = (hi - lo) / (float)((1 << QG_BQUERY) - 1);
        if (out.width <= 0.f) out.width = 1e-6f;
        out.vl = lo;
        out.byte_q.assign(D, 0);
        quantize_q(out.byte_q.data(), qrot.data(), D, lo, out.width, out.sumq);
        out.lut.assign((size_t)num_codebook * 16, 0);
        pack_lut(D, out.byte_q.data(), out.lut.data());
        double s = 0;
        for (int d = 0; d < D; ++d) { float e = out.q_raw_pad[d] - c_raw[d]; s += (double)e * e; }
        out.sqr_y_to_c = (float)s;
    }

    // estimate appro_dist for all N via FastScan (AVX2). out[N].
    void estimate(const Query& q, std::vector<float>& out) const {
        out.assign(Npad, 0.f);
        std::vector<uint16_t> res(32);
        const size_t block_bytes = (size_t)D * 4;  // padded_dim<<2
        for (int blk = 0; blk < Npad; blk += 32) {
            accumulate_block(D, blocks.data() + (size_t)(blk / 32) * block_bytes,
                             q.lut.data(), res.data());
            for (int j = 0; j < 32; ++j) {
                int idx = blk + j;
                float fsr = (float)((int)res[j] * 2 - q.sumq);
                out[idx] = triple_x[idx] + q.sqr_y_to_c + fac_dq[idx] * q.width * fsr +
                           fac_vq[idx] * q.vl;
            }
        }
        out.resize(N);
    }

    // scalar reference FastScan result for vector i = sum_d byte_q[d]*bit_i[d].
    int scalar_fs_result(int i, const uint8_t* byte_q) const {
        const int words = D / 64;
        const uint64_t* w = &binary[(size_t)i * words];
        int s = 0;
        for (int d = 0; d < D; ++d) {
            int b = (int)((w[d >> 6] >> (63 - (d & 63))) & 1ULL);
            s += b * byte_q[d];
        }
        return s;
    }
};

}  // namespace rbfs
}  // namespace ArcFlare
