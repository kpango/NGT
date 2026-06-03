// tests/ngtaq/test_vector_record.cpp
#include "../../lib/NGT/NGTAQ/VectorRecord.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>

static int failures = 0;
#define EXPECT_EQ(a,b) do { if ((a)!=(b)) { fprintf(stderr,"FAIL %s:%d: %lld != %lld\n",__FILE__,__LINE__,(long long)(a),(long long)(b)); ++failures; } } while(0)
#define EXPECT_NEAR(a,b,eps) do { if (std::abs((double)(a)-(double)(b))>(eps)) { fprintf(stderr,"FAIL %s:%d: %f not near %f (eps=%f)\n",__FILE__,__LINE__,(double)(a),(double)(b),(double)(eps)); ++failures; } } while(0)
#define EXPECT_TRUE(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: condition false\n",__FILE__,__LINE__); ++failures; } } while(0)

static void test_fp16_roundtrip() {
    float values[] = {0.0f, 1.0f, -1.0f, 3.14159f, 100.0f, -100.0f, 0.001f};
    for (float v : values) {
        uint16_t h = NGT::NGTAQ::float_to_fp16(v);
        float back = NGT::NGTAQ::fp16_to_float(h);
        float rel_err = (v != 0.0f) ? std::abs(back - v) / std::abs(v) : std::abs(back);
        EXPECT_TRUE(rel_err < 0.005f);
    }
}

static void test_fp16_special() {
    EXPECT_EQ(NGT::NGTAQ::float_to_fp16(0.0f), 0);
    EXPECT_NEAR(NGT::NGTAQ::fp16_to_float(0), 0.0f, 1e-6f);
}

static void test_vector_record_size() {
    EXPECT_EQ(sizeof(NGT::NGTAQ::VectorRecord), 38u);
}

static void test_vector_record_layout() {
    EXPECT_EQ(offsetof(NGT::NGTAQ::VectorRecord, tier1), 0u);
    EXPECT_EQ(offsetof(NGT::NGTAQ::VectorRecord, tier2), 16u);
    EXPECT_EQ(offsetof(NGT::NGTAQ::VectorRecord, norm_fp16), 32u);
    EXPECT_EQ(offsetof(NGT::NGTAQ::VectorRecord, centroid_id), 34u);
}

static void test_tier1_bit_set() {
    NGT::NGTAQ::VectorRecord rec = {};
    NGT::NGTAQ::set_tier1_bit(rec, 5, true);
    EXPECT_TRUE(NGT::NGTAQ::get_tier1_bit(rec, 5));
    NGT::NGTAQ::set_tier1_bit(rec, 5, false);
    EXPECT_TRUE(!NGT::NGTAQ::get_tier1_bit(rec, 5));
}

static void test_tier2_nibble() {
    NGT::NGTAQ::VectorRecord rec = {};
    for (int i = 0; i < 32; ++i) {
        uint8_t val = (uint8_t)(i % 16);
        NGT::NGTAQ::set_tier2_nibble(rec, i, val);
        EXPECT_EQ(NGT::NGTAQ::get_tier2_nibble(rec, i), val);
    }
}

int main() {
    test_fp16_roundtrip();
    test_fp16_special();
    test_vector_record_size();
    test_vector_record_layout();
    test_tier1_bit_set();
    test_tier2_nibble();
    if (failures == 0) { printf("OK (6 tests)\n"); return 0; }
    fprintf(stderr, "%d test(s) FAILED\n", failures); return 1;
}
