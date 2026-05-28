// lib/NGT/NGTAQ/VectorRecord.h
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>

namespace NGT { namespace NGTAQ {

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
    uint32_t u;
    memcpy(&u, &f, 4);
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
    float result;
    memcpy(&result, &u2, 4);
    return result;
}

// ---------- tier-1 bit access (128 sign bits) ----------

inline void set_tier1_bit(VectorRecord& rec, int i, bool v) {
    if (v) rec.tier1[i >> 3] |=  (uint8_t)(1u << (i & 7));
    else   rec.tier1[i >> 3] &= ~(uint8_t)(1u << (i & 7));
}

inline bool get_tier1_bit(const VectorRecord& rec, int i) {
    return (rec.tier1[i >> 3] >> (i & 7)) & 1;
}

// ---------- tier-2 nibble access (32 nibbles, 4-bit each) ----------

inline void set_tier2_nibble(VectorRecord& rec, int i, uint8_t v) {
    if (i & 1) rec.tier2[i >> 1] = (rec.tier2[i >> 1] & 0x0F) | (uint8_t)((v & 0xF) << 4);
    else        rec.tier2[i >> 1] = (rec.tier2[i >> 1] & 0xF0) | (uint8_t)(v & 0xF);
}

inline uint8_t get_tier2_nibble(const VectorRecord& rec, int i) {
    if (i & 1) return (rec.tier2[i >> 1] >> 4) & 0xF;
    else        return  rec.tier2[i >> 1]        & 0xF;
}

}} // NGT::NGTAQ
