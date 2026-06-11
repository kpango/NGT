#include "../../lib/NGT/ArcFlare/ADCDistance.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>

static int failures = 0;
#define EXPECT_NEAR(a,b,eps) do { if (std::abs((double)(a)-(double)(b))>(eps)) { \
    fprintf(stderr,"FAIL %s:%d: %f not near %f\n",__FILE__,__LINE__,(double)(a),(double)(b)); ++failures; } } while(0)
#define EXPECT_TRUE(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL %s:%d: false\n",__FILE__,__LINE__); ++failures; } } while(0)

static void test_tier1_adc_all_same() {
    int8_t q[128]; memset(q, 127, 128);
    uint8_t tier1[16]; memset(tier1, 0xFF, 16);
    float got = NGT::ArcFlare::tier1_adc_scalar(q, tier1);
    EXPECT_NEAR(got, 128.f * 127.f, 0.1f);
}

static void test_tier1_adc_all_neg() {
    int8_t q[128]; memset(q, 127, 128);
    uint8_t tier1[16]; memset(tier1, 0x00, 16);
    float got = NGT::ArcFlare::tier1_adc_scalar(q, tier1);
    EXPECT_NEAR(got, -128.f * 127.f, 0.1f);
}

static void test_tier1_adc_random_consistency() {
    std::mt19937 rng(42);
    int8_t q[128];
    uint8_t tier1[16];
    for (int trial = 0; trial < 100; ++trial) {
        int32_t q_sum = 0;
        for (int i = 0; i < 128; ++i) {
            q[i] = (int8_t)((rng() % 254) - 127);
            q_sum += q[i];
        }
        for (int i = 0; i < 16; ++i) tier1[i] = (uint8_t)(rng() & 0xFF);
        float scalar_val = NGT::ArcFlare::tier1_adc_scalar(q, tier1);
        // tier1_adc_fast requires precomputed q_sum = sum(q[i]) for VNNI/AVX2 paths
        float fast_val   = NGT::ArcFlare::tier1_adc_fast(q, tier1, q_sum);
        EXPECT_NEAR(scalar_val, fast_val, 2.f);
    }
}

static void test_tier2_nibble_lookup_scalar() {
    int8_t lut[16][16] = {};
    for (int k = 0; k < 16; ++k)
        for (int d = 0; d < 16; ++d)
            lut[k][d] = (int8_t)((k + d) % 127);
    uint8_t tier2[16] = {};
    for (int i = 0; i < 32; ++i) {
        uint8_t v = (uint8_t)(i % 16);
        if (i & 1) tier2[i>>1] = (tier2[i>>1] & 0x0F) | (uint8_t)(v << 4);
        else        tier2[i>>1] = (tier2[i>>1] & 0xF0) | v;
    }
    float got = NGT::ArcFlare::tier2_adc_scalar(lut, tier2);
    float expected = 0.f;
    for (int d = 0; d < 32; ++d) expected += lut[d % 16][d >> 1];
    EXPECT_NEAR(got, expected, 1.f);
}

int main() {
    test_tier1_adc_all_same();
    test_tier1_adc_all_neg();
    test_tier1_adc_random_consistency();
    test_tier2_nibble_lookup_scalar();
    if (failures == 0) { printf("OK (4 tests)\n"); return 0; }
    fprintf(stderr, "%d test(s) FAILED\n", failures); return 1;
}
