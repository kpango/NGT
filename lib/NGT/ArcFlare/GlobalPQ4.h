// lib/NGT/ArcFlare/GlobalPQ4.h
//
// Stage B/C: QG-style 16-wide batch SIMD routing for ArcFlare.
//
// A 16-centroid (4-bit) GLOBAL product quantizer trained on SRHT-rotated vectors,
// plus a contiguous, block-16 transposed, uint4-packed per-NODE neighbor-code store
// and an AVX2 `vpshufb` batch kernel that scores ALL ~16 neighbors of a popped node
// in a few SIMD instructions over one shared per-query LUT (no per-neighbor gather,
// no per-cluster ADC rebuild).
//
// Mirrors NGTQ's QuantizedGraph kernel (lib/NGT/NGTQ/Quantizer.h:2335-2470) and layout
// (arrangeQuantizedObject + compressIntoUint4), but for a GLOBAL codebook so a single
// LUT scores any node.
//
// Distance model (squared L2):
//   ||q_rot - x_pq||^2 = ||q_rot||^2 + ||x_pq||^2 - 2 <q_rot, x_pq>
// The kernel computes <q_rot, x_pq> via the LUT; ||x_pq||^2 is stored per neighbor and
// ||q_rot||^2 is a per-query scalar (both added in the caller).

#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace NGT { namespace ArcFlare {

// 16 neighbors per SIMD block (matches NGTQ_SIMD_BLOCK_SIZE and the 16-lane uint16 acc).
static constexpr int GPQ4_BLOCK = 16;
// 16 centroids per subspace → 4-bit codes (single _mm256_shuffle_epi8 lookup).
static constexpr int GPQ4_K     = 16;

// ---------------------------------------------------------------------------
// Encode one rotated sub-vector to its nearest 4-bit code (K=16, D_sub dims).
// cb_sub layout: code*D_sub + dim  (16 centroids of one subspace, row-major).
// Returns the code and adds ||centroid||^2 of the chosen code into recon_norm_sq.
// ---------------------------------------------------------------------------
inline uint8_t gpq4_encode_sub(const float* sv, const float* cb_sub /* [16][D_sub] */,
                               int D_sub, float& recon_norm_sq) {
    float best = std::numeric_limits<float>::max();
    uint8_t bc = 0;
    for (int code = 0; code < GPQ4_K; ++code) {
        const float* c = cb_sub + (size_t)code * D_sub;
        float d = 0.f;
        for (int dd = 0; dd < D_sub; ++dd) { float df = sv[dd] - c[dd]; d += df * df; }
        if (d < best) { best = d; bc = (uint8_t)code; }
    }
    const float* bcptr = cb_sub + (size_t)bc * D_sub;
    for (int dd = 0; dd < D_sub; ++dd) recon_norm_sq += bcptr[dd] * bcptr[dd];
    return bc;
}

// ---------------------------------------------------------------------------
// ScaNN anisotropic (noise-shaped) encoding of ONE rotated vector across all M subspaces.
// Port of ScaNN IndexDatapointNoiseShaped (asymmetric_hashing_impl.cc:282-503). The plain
// per-subspace L2-argmin minimizes reconstruction MSE; anisotropic instead minimizes
//   eta * ||r_parallel||^2 + ||r_perp||^2   (r = x - x_recon, split by x's direction),
// penalizing the PARALLEL residual (the ranking-critical component) by eta>1. The codebook
// stays plain k-means; only the per-vector code SELECTION changes. eta==1 reproduces the
// L2-argmin bit-for-bit. Codes stay 0..15 (no LUT/kernel/storage change).
//
//   xr        : the rotated full vector [M*D_sub]
//   cb        : codebook [M][16][D_sub] (sub-major, code-major) — same layout as gpq4_cb
//   M, D_sub  : subspace count, subspace dim
//   eta       : parallel cost multiplier (>=1)
//   codes_out : [M] selected codes
//   recon_norm_sq (out) : sum_s ||centroid[s][code_s]||^2 (feeds gpq4_norm_sq_)
inline void gpq4_encode_anisotropic(const float* xr, const float* cb, int M, int D_sub,
                                    float eta, uint8_t* codes_out, float& recon_norm_sq) {
    // ||x||: norm of the rotated vector (over all M*D_sub dims). inv_norm scales the
    // per-coordinate parallel projection r_i * x_i / ||x||.
    double nrm2 = 0.0;
    for (int d = 0; d < M * D_sub; ++d) nrm2 += (double)xr[d] * xr[d];
    const double inv_norm = (nrm2 > 1e-30) ? 1.0 / std::sqrt(nrm2) : 0.0;

    // Per-subspace, per-code stats: residual_norm[s][k] = ||sv_s - c_{s,k}||^2,
    // par[s][k] = <sv_s - c_{s,k}, x_s> * inv_norm  (1-D projection contribution).
    static thread_local std::vector<float> rn_tl, par_tl;
    rn_tl.assign((size_t)M * GPQ4_K, 0.f);
    par_tl.assign((size_t)M * GPQ4_K, 0.f);
    for (int s = 0; s < M; ++s) {
        const float* sv = xr + (size_t)s * D_sub;
        const float* cbs = cb + (size_t)s * GPQ4_K * D_sub;
        for (int k = 0; k < GPQ4_K; ++k) {
            const float* c = cbs + (size_t)k * D_sub;
            double rn = 0.0, par = 0.0;
            for (int d = 0; d < D_sub; ++d) {
                double rc = (double)sv[d] - (double)c[d];
                rn  += rc * rc;
                par += rc * (double)sv[d] * inv_norm;
            }
            rn_tl[(size_t)s * GPQ4_K + k]  = (float)rn;
            par_tl[(size_t)s * GPQ4_K + k] = (float)par;
        }
    }

    // Init each subspace to argmin residual_norm (== eta=1 L2 baseline).
    static thread_local std::vector<uint8_t> code_tl;
    code_tl.assign(M, 0);
    double total_par = 0.0;
    for (int s = 0; s < M; ++s) {
        const float* rn = &rn_tl[(size_t)s * GPQ4_K];
        int best = 0; float bv = rn[0];
        for (int k = 1; k < GPQ4_K; ++k) if (rn[k] < bv) { bv = rn[k]; best = k; }
        code_tl[s] = (uint8_t)best;
        total_par += par_tl[(size_t)s * GPQ4_K + best];
    }

    // eta==1 (or no-op): the anisotropic cost reduces to residual_norm, so the argmin
    // init IS the optimum — skip coordinate descent (guarantees bit-identical to L2-argmin).
    if (eta > 1.0f) {
        // Process subspaces in descending init-residual-norm order (ScaNN), recomputed once.
        static thread_local std::vector<int> order_tl;
        order_tl.assign(M, 0);
        for (int s = 0; s < M; ++s) order_tl[s] = s;
        std::sort(order_tl.begin(), order_tl.end(), [&](int a, int b){
            return rn_tl[(size_t)a * GPQ4_K + code_tl[a]] > rn_tl[(size_t)b * GPQ4_K + code_tl[b]];
        });
        const double pcm = (double)eta;  // parallel_cost_multiplier
        for (int round = 0, changed = 1; changed && round < 10; ++round) {
            changed = 0;
            for (int oi = 0; oi < M; ++oi) {
                const int s = order_tl[oi];
                const float* rn = &rn_tl[(size_t)s * GPQ4_K];
                const float* pr = &par_tl[(size_t)s * GPQ4_K];
                const uint8_t cur = code_tl[s];
                const double old_rn = rn[cur], old_par = pr[cur];
                double best_delta = 0.0; int best_k = cur; double best_total_par = total_par;
                for (int k = 0; k < GPQ4_K; ++k) {
                    if (k == (int)cur) continue;
                    const double new_total_par = total_par - old_par + pr[k];
                    const double par_norm_delta = new_total_par * new_total_par - total_par * total_par;
                    if (par_norm_delta > 0.0) continue;                 // ScaNN: skip if parallel grows
                    const double rn_delta = (double)rn[k] - old_rn;
                    const double perp_norm_delta = rn_delta - par_norm_delta;
                    const double cost_delta = pcm * par_norm_delta + perp_norm_delta;
                    if (cost_delta < best_delta) {
                        best_delta = cost_delta; best_k = k; best_total_par = new_total_par;
                    }
                }
                if (best_k != (int)cur) {
                    code_tl[s] = (uint8_t)best_k; total_par = best_total_par; changed = 1;
                }
            }
        }
    }

    // Emit codes + recon_norm_sq (||centroid||^2 summed), exactly as the L2 path does.
    for (int s = 0; s < M; ++s) {
        const uint8_t c = code_tl[s];
        codes_out[s] = c;
        const float* cv = cb + ((size_t)s * GPQ4_K + c) * D_sub;
        for (int d = 0; d < D_sub; ++d) recon_norm_sq += cv[d] * cv[d];
    }
}

// ---------------------------------------------------------------------------
// Per-query uint8 LUT for the batch kernel.
//   IP ~= acc * scale + total_offset
// with a single global `scale` and per-subspace offsets folded into `total_offset`,
// so the kernel rescales once. `planes` is 32-byte interleaved: plane p (subspace pair)
// is [sub 2p : 16 entries][sub 2p+1 : 16 entries] — consumed directly by the AVX2 kernel.
// ---------------------------------------------------------------------------
struct GlobalPQ4LUT {
    std::vector<uint8_t> planes;   // [((M+1)/2) * 32]
    float scale        = 1.f;
    float total_offset = 0.f;
    int   M            = 0;        // number of subspaces

    void resize(int m) {
        M = m;
        planes.assign((size_t)((m + 1) / 2) * 32, 0);
    }
};

// Build the per-subspace float inner-product table ip[sub*16 + code] =
//   <q_rot_sub, centroid_{sub,code}> from the transposed codebook [M][D_sub][16].
// D_sub = 8 fixed (matches the rest of the pipeline). K = 16.
inline void gpq4_ip_table(const float* q_rot, int M, const float* cb_T /* [M][8][16] */,
                          int D_sub, float* ip /* [M*16] */) {
    for (int s = 0; s < M; ++s) {
        const float* q  = q_rot + (size_t)s * D_sub;
        const float* cb = cb_T  + (size_t)s * D_sub * GPQ4_K;
        float* out = ip + (size_t)s * GPQ4_K;
#if defined(__AVX2__) && defined(__FMA__)
        // 16 codes = two __m256 accumulators; FMA over D_sub dims.
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        for (int d = 0; d < D_sub; ++d) {
            __m256 qd = _mm256_set1_ps(q[d]);
            const float* c = cb + (size_t)d * GPQ4_K;
            a0 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(c),     a0);
            a1 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(c + 8), a1);
        }
        _mm256_storeu_ps(out,     a0);
        _mm256_storeu_ps(out + 8, a1);
#else
        for (int c = 0; c < GPQ4_K; ++c) {
            float dot = 0.f;
            for (int d = 0; d < D_sub; ++d) dot += q[d] * cb[(size_t)d * GPQ4_K + c];
            out[c] = dot;
        }
#endif
    }
}

// Build the per-subspace SQUARED-DISTANCE table dist[sub*16 + code] =
//   ||q_rot_sub - centroid_{sub,code}||^2, from the transposed codebook [M][D_sub][16].
// This is QG's createFloatL2DistanceLookup form: the kernel then accumulates
//   sum_s dist[s][code_s] = ||q_rot - x_pq||^2  (L2 directly), with NO per-neighbor norm
// read and NO IP->L2 assembly in the loop (vs the IP table which needs q_ns + nsq - 2*ip).
inline void gpq4_dist_table(const float* q_rot, int M, const float* cb_T /* [M][D_sub][16] */,
                            int D_sub, float* dist /* [M*16] */) {
    for (int s = 0; s < M; ++s) {
        const float* q  = q_rot + (size_t)s * D_sub;
        const float* cb = cb_T  + (size_t)s * D_sub * GPQ4_K;
        float* out = dist + (size_t)s * GPQ4_K;
#if defined(__AVX2__) && defined(__FMA__)
        // 16 codes = two __m256 accumulators of (q[d]-c[d])^2 summed over D_sub dims.
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        for (int d = 0; d < D_sub; ++d) {
            __m256 qd = _mm256_set1_ps(q[d]);
            const float* c = cb + (size_t)d * GPQ4_K;
            __m256 df0 = _mm256_sub_ps(qd, _mm256_loadu_ps(c));
            __m256 df1 = _mm256_sub_ps(qd, _mm256_loadu_ps(c + 8));
            a0 = _mm256_fmadd_ps(df0, df0, a0);
            a1 = _mm256_fmadd_ps(df1, df1, a1);
        }
        _mm256_storeu_ps(out,     a0);
        _mm256_storeu_ps(out + 8, a1);
#else
        for (int c = 0; c < GPQ4_K; ++c) {
            float dsum = 0.f;
            for (int d = 0; d < D_sub; ++d) {
                float df = q[d] - cb[(size_t)d * GPQ4_K + c];
                dsum += df * df;
            }
            out[c] = dsum;
        }
#endif
    }
}

// Build the per-query LUT from the per-subspace float inner-product table.
//   ip[sub*16 + code] = <q_rot_sub, centroid_{sub,code}>
// Global scale = max_sub((max-min)/255); per-subspace offset = min_sub.
inline void gpq4_build_lut(const float* ip /* [M*16] */, int M, GlobalPQ4LUT& out) {
    out.resize(M);
    float gmax_range = 0.f;
    static thread_local std::vector<float> mins_tl, maxs_tl;
    mins_tl.resize(M); maxs_tl.resize(M);
    for (int s = 0; s < M; ++s) {
        float lo = ip[s * 16 + 0], hi = lo;
        for (int c = 1; c < GPQ4_K; ++c) {
            float v = ip[s * 16 + c];
            lo = std::min(lo, v); hi = std::max(hi, v);
        }
        mins_tl[s] = lo; maxs_tl[s] = hi;
        gmax_range = std::max(gmax_range, hi - lo);
    }
    const float scale = (gmax_range > 1e-20f) ? (gmax_range / 255.f) : 1.f;
    const float inv   = 1.f / scale;
    out.scale = scale;
    float toff = 0.f;
    for (int s = 0; s < M; ++s) {
        const int plane = s / 2;
        const int half  = (s & 1) * 16;
        uint8_t* dst = out.planes.data() + (size_t)plane * 32 + half;
        const float off = mins_tl[s];
        toff += off;
        for (int c = 0; c < GPQ4_K; ++c) {
            float q = (ip[s * 16 + c] - off) * inv;
            int qi = (int)(q + 0.5f);
            dst[c] = (uint8_t)(qi < 0 ? 0 : qi > 255 ? 255 : qi);
        }
    }
    out.total_offset = toff;
}

// ---------------------------------------------------------------------------
// Block layout (mirrors NGTQ arrangeQuantizedObject + compressIntoUint4):
//   block = [((M+1)/2) planes][16 bytes].  Plane p = subspace pair (2p, 2p+1).
//     bytes 0..7  = subspace 2p   codes for 16 neighbors, packed in pairs:
//                   byte b = (code_{2p}[2b+1] << 4) | code_{2p}[2b]
//     bytes 8..15 = subspace 2p+1 codes for 16 neighbors, packed the same way.
// One block thus holds 16 neighbors × M subspaces in (M/2)*16 bytes (= M*8).
// ---------------------------------------------------------------------------

// Pack 16 neighbors' codes (codes[n*M + s], 4-bit) into the block layout above.
// `n_real` (<=16) is the number of real neighbors; the rest are padded with code 0.
inline void gpq4_pack_block(const uint8_t* codes /* [16*M] */, int M, int n_real,
                            uint8_t* block /* [((M+1)/2)*16] */) {
    const int planes = (M + 1) / 2;
    std::memset(block, 0, (size_t)planes * 16);
    for (int n = 0; n < n_real && n < GPQ4_BLOCK; ++n) {
        for (int s = 0; s < M; ++s) {
            const int plane = s / 2;
            const int half  = (s & 1) * 8;          // even subspace → bytes 0..7, odd → 8..15
            const uint8_t code = codes[(size_t)n * M + s] & 0x0f;
            uint8_t* dst = block + (size_t)plane * 16 + half + (n / 2);
            if (n & 1) *dst |= (code << 4);
            else       *dst |= code;
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar reference batch kernel (correctness + non-AVX2 fallback).
// Writes 16 inner products into out_ip.
// ---------------------------------------------------------------------------
inline void gpq4_batch_ip_scalar(const uint8_t* block, const GlobalPQ4LUT& lut,
                                 float* out_ip /* [16] */) {
    const int planes = (lut.M + 1) / 2;
    const float scale = lut.scale, toff = lut.total_offset;
    for (int n = 0; n < GPQ4_BLOCK; ++n) {
        uint32_t acc = 0;
        for (int p = 0; p < planes; ++p) {
            const uint8_t* le = lut.planes.data() + (size_t)p * 32;
            // even subspace 2p: byte (n/2) of bytes 0..7, nibble by parity
            uint8_t be = block[(size_t)p * 16 + (n / 2)];
            uint8_t ce = (n & 1) ? (be >> 4) : (be & 0x0f);
            acc += le[ce];
            // odd subspace 2p+1: byte 8 + (n/2)
            uint8_t bo = block[(size_t)p * 16 + 8 + (n / 2)];
            uint8_t co = (n & 1) ? (bo >> 4) : (bo & 0x0f);
            acc += le[16 + co];
        }
        out_ip[n] = (float)acc * scale + toff;
    }
}

#if defined(__AVX2__)
// ---------------------------------------------------------------------------
// AVX2 batch kernel: <q, x_pq> for 16 neighbors of `block` in a few vpshufb ops.
// Replicates the NGTQG AVX2 inner loop (Quantizer.h:2382-2399 + reduce 2432-2437).
// ---------------------------------------------------------------------------
inline void gpq4_batch_ip_avx2(const uint8_t* __restrict block,
                               const GlobalPQ4LUT& lut,
                               float* __restrict out_ip /* [16] */) {
    const int planes = (lut.M + 1) / 2;
    const __m256i mask0F = _mm256_set1_epi16(0x000f);
    const __m256i maskF0 = _mm256_set1_epi16(0x00f0);
    // After one plane's shuffle, v's low 128 = 16 bytes = even-subspace contributions
    // for neighbors 0..15; v's high 128 = odd-subspace contributions for neighbors 0..15.
    // cvtepu8_epi16 widens each 128-bit half to 16 uint16 (= the 16 neighbors), so the two
    // accumulators hold neighbor n's even / odd subspace running sums respectively.
    __m256i acc_even = _mm256_setzero_si256();  // 16 uint16: neighbor n even-subspace sum
    __m256i acc_odd  = _mm256_setzero_si256();  // 16 uint16: neighbor n odd-subspace sum
    const uint8_t* lp = lut.planes.data();
    for (int p = 0; p < planes; ++p) {
        __m256i lookup = _mm256_loadu_si256((const __m256i*)(lp + (size_t)p * 32));
        __m256i packed = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(block + (size_t)p * 16)));
        __m256i lo  = _mm256_and_si256(packed, mask0F);
        __m256i hi  = _mm256_slli_epi16(_mm256_and_si256(packed, maskF0), 4);
        __m256i obj = _mm256_or_si256(lo, hi);
        __m256i v   = _mm256_shuffle_epi8(lookup, obj);
        acc_even = _mm256_adds_epu16(acc_even, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 0)));
        acc_odd  = _mm256_adds_epu16(acc_odd,  _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1)));
    }
    // total[n] = acc_even[n] + acc_odd[n]; widen to int32 to avoid overflow on the sum.
    __m256i evlo = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(acc_even, 0)); // n0..7
    __m256i evhi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(acc_even, 1)); // n8..15
    __m256i odlo = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(acc_odd, 0));
    __m256i odhi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(acc_odd, 1));
    __m256i tlo  = _mm256_add_epi32(evlo, odlo);  // neighbors 0..7 totals
    __m256i thi  = _mm256_add_epi32(evhi, odhi);  // neighbors 8..15 totals
    const __m256 vscale = _mm256_set1_ps(lut.scale);
    const __m256 voff   = _mm256_set1_ps(lut.total_offset);
    __m256 flo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tlo), vscale, voff);
    __m256 fhi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(thi), vscale, voff);
    _mm256_storeu_ps(out_ip,     flo);
    _mm256_storeu_ps(out_ip + 8, fhi);
}
#endif // __AVX2__

// Dispatch wrapper.
inline void gpq4_batch_ip(const uint8_t* block, const GlobalPQ4LUT& lut, float* out_ip) {
#if defined(__AVX2__)
    gpq4_batch_ip_avx2(block, lut, out_ip);
#else
    gpq4_batch_ip_scalar(block, lut, out_ip);
#endif
}

}} // NGT::ArcFlare
