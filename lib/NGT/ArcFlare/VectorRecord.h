// lib/NGT/ArcFlare/VectorRecord.h
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <bit>     // std::bit_cast (C++20/23): zero-overhead type-punning, replaces memcpy

namespace NGT { namespace ArcFlare {

// Per-vector storage: 38 bytes total
// Layout: [tier1 16B][tier2 16B][norm_fp16 2B][centroid_id 4B]
#pragma pack(push,1)
struct VectorRecord {
    uint8_t  tier1[16];      // SRHT 1-bit quantization: 128 sign bits
    uint8_t  tier2[16];      // PCA top-32 dims × 4-bit: 32 nibbles
    uint16_t norm_fp16;      // ||x - c_k||_2 as float16
    uint32_t centroid_id;    // K-means cluster index
};
#pragma pack(pop)

static_assert(sizeof(VectorRecord) == 38, "VectorRecord must be 38 bytes");
static_assert(offsetof(VectorRecord, tier1) == 0);
static_assert(offsetof(VectorRecord, tier2) == 16);
static_assert(offsetof(VectorRecord, norm_fp16) == 32);
static_assert(offsetof(VectorRecord, centroid_id) == 34);

// ---------- fp16 utilities ----------

inline uint16_t float_to_fp16(float f) {
    uint32_t u = std::bit_cast<uint32_t>(f);
    uint32_t sign     = (u >> 31) & 0x1;
    uint32_t exp32    = (u >> 23) & 0xFF;
    uint32_t mant32   = u & 0x7FFFFF;

    if (exp32 == 0xFF) {
        uint16_t mant16 = mant32 ? 0x200 : 0;
        return (uint16_t)((sign << 15) | 0x7C00 | mant16);
    }
    int exp16 = (int)exp32 - 127 + 15;
    if (exp16 >= 31) {
        return (uint16_t)((sign << 15) | 0x7C00);
    }
    if (exp16 <= 0) {
        return (uint16_t)(sign << 15);
    }
    uint16_t mant16 = (uint16_t)(mant32 >> 13);
    if ((mant32 >> 12) & 1) mant16++;
    return (uint16_t)((sign << 15) | ((uint16_t)exp16 << 10) | (mant16 & 0x3FF));
}

// ---------- bf16 utilities ----------
// bf16 = the top 16 bits of fp32: same 8-bit exponent, so the full fp32 dynamic range
// is preserved (no overflow for SIFT reconstructed norms ~2e5, which exceed fp16 max
// 65504). Used for the fused per-neighbor norm in the gpq4 block. Round-to-nearest-even.
inline uint16_t float_to_bf16(float f) {
    uint32_t u = std::bit_cast<uint32_t>(f);
    // NaN: force a quiet NaN that survives truncation.
    if (((u >> 23) & 0xFF) == 0xFF && (u & 0x7FFFFF)) return (uint16_t)((u >> 16) | 0x40);
    uint32_t rounding_bias = 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)((u + rounding_bias) >> 16);
}

inline float bf16_to_float(uint16_t b) {
    return std::bit_cast<float>((uint32_t)b << 16);
}

inline float fp16_to_float(uint16_t h) {
    uint32_t sign  = (h >> 15) & 0x1;
    uint32_t exp16 = (h >> 10) & 0x1F;
    uint32_t mant  = h & 0x3FF;

    uint32_t exp32, mant32;
    if (exp16 == 0) {
        if (mant == 0) { exp32 = 0; mant32 = 0; }
        else {
            exp32 = 127 - 14;
            mant32 = mant << (23 - 10 + 1);
            while (!(mant32 & 0x800000)) { mant32 <<= 1; exp32--; }
            mant32 &= ~0x800000;
        }
    } else if (exp16 == 31) {
        exp32 = 0xFF;
        mant32 = mant ? (mant << 13) : 0;
    } else {
        exp32 = exp16 + 127 - 15;
        mant32 = mant << 13;
    }
    uint32_t u2 = (sign << 31) | (exp32 << 23) | mant32;
    return std::bit_cast<float>(u2);
}

// ---------- tier-1 bit access (128 sign bits) ----------

inline void set_tier1_bit(VectorRecord& rec, int i, bool v) {
    if (v) rec.tier1[i >> 3] |=  (uint8_t)(1u << (i & 7));
    else   rec.tier1[i >> 3] &= ~(uint8_t)(1u << (i & 7));
}

inline bool get_tier1_bit(const VectorRecord& rec, int i) {
    return (rec.tier1[i >> 3] >> (i & 7)) & 1;
}

// ---------- tier-2 byte access (16 bytes, 8-bit each, M=16 K=256) ----------
// tier2[16] stores 16 PQ codes, one per sub-space (0-255).

inline void set_tier2_byte(VectorRecord& rec, int i, uint8_t v) {
    rec.tier2[i] = v;
}

inline uint8_t get_tier2_byte(const VectorRecord& rec, int i) {
    return rec.tier2[i];
}

// ============================================================
// Variable-stride VectorRecord view (for D_eff != 128)
// ============================================================

/// Stride in bytes for a VectorRecord with D_eff dimensions.
/// tier1 = D_eff/8 bytes,  tier2 = D_eff/8 bytes,  norm = 2B,  centroid = 4B.
constexpr int vrec_stride(int D_eff) { return D_eff / 4 + 6; }

/// Mutable view into a variable-stride record
struct VectorRecordView {
    uint8_t* ptr;
    int tier1_n;  ///< = D_eff / 8
    int tier2_n;  ///< = D_eff / 8

    uint8_t*       tier1()  const { return ptr; }
    uint8_t*       tier2()  const { return ptr + tier1_n; }

    uint16_t norm_fp16()    const { uint16_t v; memcpy(&v, ptr+tier1_n+tier2_n,   2); return v; }
    uint32_t centroid_id()  const { uint32_t v; memcpy(&v, ptr+tier1_n+tier2_n+2, 4); return v; }
    void set_norm_fp16 (uint16_t v) { memcpy(ptr+tier1_n+tier2_n,   &v, 2); }
    void set_centroid_id(uint32_t v){ memcpy(ptr+tier1_n+tier2_n+2, &v, 4); }

    void set_tier1_bit(int i, bool v) const {
        if (v) tier1()[i>>3] |=  (uint8_t)(1u<<(i&7));
        else   tier1()[i>>3] &= ~(uint8_t)(1u<<(i&7));
    }
    bool get_tier1_bit(int i)         const { return (tier1()[i>>3]>>(i&7))&1; }
    void set_tier2_byte(int i, uint8_t v) const { tier2()[i] = v; }
    uint8_t get_tier2_byte(int i)          const { return tier2()[i]; }
};

/// Const view
struct VectorRecordConstView {
    const uint8_t* ptr;
    int tier1_n;
    int tier2_n;

    const uint8_t* tier1()  const { return ptr; }
    const uint8_t* tier2()  const { return ptr + tier1_n; }

    uint16_t norm_fp16()   const { uint16_t v; memcpy(&v, ptr+tier1_n+tier2_n,   2); return v; }
    uint32_t centroid_id() const { uint32_t v; memcpy(&v, ptr+tier1_n+tier2_n+2, 4); return v; }
    bool get_tier1_bit(int i)          const { return (tier1()[i>>3]>>(i&7))&1; }
    uint8_t get_tier2_byte(int i)      const { return tier2()[i]; }
};

inline VectorRecordView vrec_view(uint8_t* base, uint32_t id, int tier1_n, int tier2_n) {
    return {base + (size_t)id * (size_t)(tier1_n + tier2_n + 6), tier1_n, tier2_n};
}
inline VectorRecordConstView vrec_const_view(const uint8_t* base, uint32_t id, int t1n, int t2n) {
    return {base + (size_t)id * (size_t)(t1n + t2n + 6), t1n, t2n};
}

/// Helper for reading fixed D=128 VectorRecord as VectorRecordConstView
inline VectorRecordConstView vrec_from_fixed(const VectorRecord& r) {
    return {reinterpret_cast<const uint8_t*>(&r), 16, 16};
}

}} // NGT::ArcFlare
