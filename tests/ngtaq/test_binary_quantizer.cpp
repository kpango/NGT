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
    // Encoded BQ signature must have correct number of words
    const int D = 128;
    NGTAQ::BinaryQuantizer bq;
    bq.init(D);
    auto vecs = randomUnitVectors(10, D);
    // Set identity rotation (no rotation)
    bq.setIdentityRotation();

    std::vector<uint64_t> sign(bq.words()), mag(bq.words());
    bq.encode(vecs[0].data(), sign.data(), mag.data());
    EXPECT_TRUE(static_cast<int>(sign.size()) == bq.words());
    EXPECT_TRUE(static_cast<int>(mag.size())  == bq.words());
}

void testSignBitCorrectness() {
    // With identity rotation, sign plane bit i = (vec[i] < 0) ? 1 : 0
    const int D = 64;
    NGTAQ::BinaryQuantizer bq;
    bq.init(D);
    bq.setIdentityRotation();
    bq.setTau(0.5f);  // tau = 0.5 (normalized), magnitude bit for |x| > 0.5

    std::vector<float> v(D, 0.0f);
    v[0] = -1.0f; // bit 0 of sign plane should be 1
    v[1] =  1.0f; // bit 1 of sign plane should be 0
    // Normalize
    float norm = std::sqrt(2.0f);
    for (auto& x : v) x /= norm;

    std::vector<uint64_t> sign(bq.words()), mag(bq.words());
    bq.encode(v.data(), sign.data(), mag.data());

    bool bit0 = (sign[0] >> 0) & 1;  // should be 1 (negative)
    bool bit1 = (sign[0] >> 1) & 1;  // should be 0 (positive)
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
    // Flatten for calibration
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
    std::vector<uint64_t> sign(bq.words()), mag(bq.words());
    bq.encode(vecs[0].data(), sign.data(), mag.data());

    float dist = NGTAQ::bqDistance(
        sign.data(), mag.data(), sign.data(), mag.data(), bq.words(), D);
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
