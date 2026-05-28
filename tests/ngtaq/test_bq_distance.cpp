// tests/ngtaq/test_bq_distance.cpp
// TDD tests for NGTAQ::bqDistance (interleaved 2-pointer API)
// Run: ./test_bq_distance (returns 0 on success, 1 on failure)

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "NGT/NGTAQ/BQDistance.h"

static int failures = 0;

#define EXPECT_NEAR(a, b, eps) \
  do { \
    if (std::abs((double)(a) - (double)(b)) > (eps)) { \
      std::cerr << __FILE__ << ":" << __LINE__ \
                << " FAIL: |" #a " - " #b "| > " #eps \
                << " (got=" << (double)(a) << ", expected=" << (double)(b) \
                << ", diff=" << std::abs((double)(a) - (double)(b)) << ")\n"; \
      ++failures; \
    } \
  } while (0)

#define EXPECT_TRUE(c) \
  do { \
    if (!(c)) { \
      std::cerr << __FILE__ << ":" << __LINE__ << " FAIL: " #c "\n"; \
      ++failures; \
    } \
  } while (0)

// Test 1: δ_BQ(p, p) == 0.0 — identical vectors
static void testIdenticalVectors() {
  constexpr int words = 4;
  constexpr int D     = words * 64;  // 256

  // Interleaved: p[i*2]=sign_i, p[i*2+1]=mag_i
  uint64_t p[words * 2], q[words * 2];
  for (int i = 0; i < words; ++i) {
    p[i*2]   = 0xDEADBEEFCAFEBABEULL + (uint64_t)i * 0x123456789ABCULL;
    p[i*2+1] = 0xFEEDFACEDEADC0DEULL ^ (uint64_t)i;
    q[i*2]   = p[i*2];
    q[i*2+1] = p[i*2+1];
  }

  float d = NGTAQ::bqDistance(p, q, words, D);
  EXPECT_NEAR(d, 0.0f, 1e-9f);
}

// Test 2: All sign bits flipped + all magnitude bits set → distance == 1.0
static void testOppositeSignMaxMagnitude() {
  constexpr int words = 2;
  constexpr int D     = words * 64;  // 128

  uint64_t p[words * 2], q[words * 2];
  for (int i = 0; i < words; ++i) {
    p[i*2]   = 0x5555555555555555ULL;
    p[i*2+1] = 0xFFFFFFFFFFFFFFFFULL;
    q[i*2]   = ~0x5555555555555555ULL;
    q[i*2+1] = 0xFFFFFFFFFFFFFFFFULL;
  }
  float d = NGTAQ::bqDistance(p, q, words, D);
  EXPECT_NEAR(d, 1.0f, 1e-9f);
}

// Test 3: Both magnitude planes all-zero → distance == 0.0
static void testZeroMagnitudeVectors() {
  constexpr int words = 3;
  constexpr int D     = words * 64;  // 192

  uint64_t p[words * 2], q[words * 2];
  for (int i = 0; i < words; ++i) {
    p[i*2]   = 0xFFFFFFFFFFFFFFFFULL;
    p[i*2+1] = 0ULL;
    q[i*2]   = 0x0000000000000000ULL;
    q[i*2+1] = 0ULL;
  }
  float d = NGTAQ::bqDistance(p, q, words, D);
  EXPECT_NEAR(d, 0.0f, 1e-9f);
}

// Test 4: Symmetry δ_BQ(p, q) == δ_BQ(q, p)
static void testSymmetry() {
  constexpr int words = 4;
  constexpr int D     = words * 64;

  uint64_t p[words * 2], q[words * 2];
  for (int i = 0; i < words; ++i) {
    p[i*2]   = 0xA5A5A5A5A5A5A5A5ULL * (uint64_t)(i + 1);
    p[i*2+1] = 0x3C3C3C3C3C3C3C3CULL | (uint64_t)i;
    q[i*2]   = 0x5A5A5A5A5A5A5A5AULL + (uint64_t)(i * 7);
    q[i*2+1] = 0xC3C3C3C3C3C3C3C3ULL ^ (uint64_t)(i * 3);
  }

  float dpq = NGTAQ::bqDistance(p, q, words, D);
  float dqp = NGTAQ::bqDistance(q, p, words, D);
  EXPECT_NEAR(dpq, dqp, 1e-9f);
}

// Test 5: Result always in [0.0, 1.0]
static void testNormalization() {
  constexpr int words = 8;
  constexpr int D     = words * 64;

  static const uint64_t patterns[][4] = {
    {0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL},
    {0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL, 0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL},
    {0x1234567890ABCDEFULL, 0xFEDCBA9876543210ULL, 0xDEADBEEFCAFEBABEULL, 0xBAADF00DBAADF00DULL},
  };

  for (auto& pat : patterns) {
    uint64_t p[words * 2], q[words * 2];
    for (int i = 0; i < words; ++i) {
      p[i*2]   = pat[0] * (uint64_t)(i + 1);
      p[i*2+1] = pat[1] ^ (uint64_t)i;
      q[i*2]   = pat[2] + (uint64_t)i;
      q[i*2+1] = pat[3] | (uint64_t)(i * 17);
    }
    float d = NGTAQ::bqDistance(p, q, words, D);
    EXPECT_TRUE(d >= 0.0f);
    EXPECT_TRUE(d <= 1.0f);
  }
}

// Test 6: Known-value test with D=64 (1 word)
static void testKnownValue() {
  constexpr int words = 1;
  constexpr int D     = 64;

  // [sign, mag] interleaved
  uint64_t p[2] = {0xFF00FF00FF00FF00ULL, 0xFFFFFFFFFFFFFFFFULL};
  uint64_t q[2] = {0x00FF00FF00FF00FFULL, 0xFFFFFFFFFFFFFFFFULL};
  float d = NGTAQ::bqDistance(p, q, words, D);
  EXPECT_NEAR(d, 1.0f, 1e-9f);

  p[0] = 0xF0F0F0F0F0F0F0F0ULL; p[1] = 0xFFFF0000FFFF0000ULL;
  q[0] = 0x0F0F0F0F0F0F0F0FULL; q[1] = 0x0000FFFF0000FFFFULL;
  d = NGTAQ::bqDistance(p, q, words, D);
  EXPECT_NEAR(d, 1.0f, 1e-9f);

  p[0] = 0x000000000000000FULL; p[1] = 0xFFFFFFFFFFFFFFFFULL;
  q[0] = 0x0000000000000000ULL; q[1] = 0x0000000000000000ULL;
  d = NGTAQ::bqDistance(p, q, words, D);
  EXPECT_NEAR(d, 4.0f / 64.0f, 1e-9f);

  p[0] = 0xFFFFFFFFFFFFFFFFULL; p[1] = 0x0000000000000000ULL;
  q[0] = 0x0000000000000000ULL; q[1] = 0x0000000000000000ULL;
  d = NGTAQ::bqDistance(p, q, words, D);
  EXPECT_NEAR(d, 0.0f, 1e-9f);
}

// Test 7: D=576 (=9*64) tests the AVX-512 remainder loop
static void testNonMultipleOf8Words() {
    const int words = 9, D = words * 64;
    std::vector<uint64_t> p(words * 2), q(words * 2);
    for (int i = 0; i < words; ++i) {
        p[i*2]   = 0xAAAAAAAAAAAAAAAAULL;
        p[i*2+1] = 0xFFFFFFFFFFFFFFFFULL;
        q[i*2]   = 0x5555555555555555ULL;
        q[i*2+1] = 0xFFFFFFFFFFFFFFFFULL;
    }
    // Ground truth via scalar
    uint64_t expected_count = 0;
    for (int i = 0; i < words; i++)
        expected_count += __builtin_popcountll((p[i*2] ^ q[i*2]) & (p[i*2+1] | q[i*2+1]));
    float expected = static_cast<float>(expected_count) / D;

    float got = NGTAQ::bqDistance(p.data(), q.data(), words, D);
    EXPECT_NEAR(got, expected, 1e-6f);
}

int main() {
  testIdenticalVectors();
  testOppositeSignMaxMagnitude();
  testZeroMagnitudeVectors();
  testSymmetry();
  testNormalization();
  testKnownValue();
  testNonMultipleOf8Words();

  if (failures == 0) {
    std::cout << "All tests PASSED\n";
    return 0;
  } else {
    std::cerr << failures << " test(s) FAILED\n";
    return 1;
  }
}
