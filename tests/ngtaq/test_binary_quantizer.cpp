// tests/ngtaq/test_binary_quantizer.cpp
#include "NGT/NGTAQ/BinaryQuantizer.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

static int failures = 0;

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { std::cerr << __FILE__ << ":" << __LINE__ << " FAIL: " #cond "\n"; ++failures; } \
} while(0)

#define EXPECT_NEAR(a, b, eps) do { \
    if (std::abs((double)(a) - (double)(b)) > (eps)) { \
        std::cerr << __FILE__ << ":" << __LINE__ << " FAIL: |" #a " - " #b "| > " #eps \
                  << " (diff=" << std::abs((double)(a)-(double)(b)) << ")\n"; ++failures; \
    } \
} while(0)

// Generate random unit vectors
std::vector<std::vector<float>> randomUnitVectors(int N, int D, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> vecs(N, std::vector<float>(D));
    for (auto& v : vecs) {
        float norm = 0;
        for (auto& x : v) { x = dist(rng); norm += x * x; }
        norm = std::sqrt(norm);
        for (auto& x : v) x /= norm;
    }
    return vecs;
}

void testEncodeDecodeDimension() {
    // Encoded BQ buffer must have words*2 entries (interleaved sign+mag)
    const int D = 128;
    NGTAQ::BinaryQuantizer bq;
    bq.init(D);
    auto vecs = randomUnitVectors(10, D);
    bq.setIdentityRotation();

    std::vector<uint64_t> bq_buf(bq.words() * 2);
    bq.encode(vecs[0].data(), bq_buf.data());
    EXPECT_TRUE(static_cast<int>(bq_buf.size()) == bq.words() * 2);
}

void testSignBitCorrectness() {
    // With identity rotation, sign plane bit i = (vec[i] < 0) ? 1 : 0
    // In interleaved layout: sign word 0 = bq_buf[0], mag word 0 = bq_buf[1]
    const int D = 64;
    NGTAQ::BinaryQuantizer bq;
    bq.init(D);
    bq.setIdentityRotation();
    bq.setTau(0.5f);

    std::vector<float> v(D, 0.0f);
    v[0] = -1.0f; // bit 0 of sign plane should be 1
    v[1] =  1.0f; // bit 1 of sign plane should be 0
    float norm = std::sqrt(2.0f);
    for (auto& x : v) x /= norm;

    // words=1 for D=64: bq_buf[0]=sign_word0, bq_buf[1]=mag_word0
    std::vector<uint64_t> bq_buf(bq.words() * 2);
    bq.encode(v.data(), bq_buf.data());

    bool bit0 = (bq_buf[0] >> 0) & 1;  // sign word 0, bit 0 — should be 1 (negative)
    bool bit1 = (bq_buf[0] >> 1) & 1;  // sign word 0, bit 1 — should be 0 (positive)
    EXPECT_TRUE(bit0 == true);
    EXPECT_TRUE(bit1 == false);
}

void testTauCalibration() {
    // τ should be positive and less than 0.5 for random unit vectors with 128 dims
    const int D = 128;
    NGTAQ::BinaryQuantizer bq;
    bq.init(D);
    bq.setIdentityRotation();

    auto vecs = randomUnitVectors(1000, D, 123);
    std::vector<const float*> ptrs;
    for (auto& v : vecs) ptrs.push_back(v.data());

    bq.calibrateTau(ptrs, 500, NGT::ObjectSpace::DistanceTypeInnerProduct);

    EXPECT_TRUE(bq.tau() > 0.0f);
    EXPECT_TRUE(bq.tau() < 0.5f);
}

void testSelfDistanceIsZero() {
    // bqDistance(encode(v), encode(v)) == 0
    const int D = 128;
    NGTAQ::BinaryQuantizer bq;
    bq.init(D);
    bq.setIdentityRotation();
    bq.setTau(0.1f);

    auto vecs = randomUnitVectors(5, D, 7);
    std::vector<uint64_t> bq_buf(bq.words() * 2);
    bq.encode(vecs[0].data(), bq_buf.data());

    float dist = NGTAQ::bqDistance(bq_buf.data(), bq_buf.data(), bq.words(), D);
    EXPECT_NEAR(dist, 0.0f, 1e-6f);
}

int main() {
    testEncodeDecodeDimension();
    testSignBitCorrectness();
    testTauCalibration();
    testSelfDistanceIsZero();
    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cout << "All BinaryQuantizer tests PASSED\n";
    return 0;
}
