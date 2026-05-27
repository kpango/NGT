// tests/ngtaq/test_bq_distance.cpp
// TDD tests for NGTAQ::bqDistance
// Run: ./test_bq_distance (returns 0 on success, 1 on failure)

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "NGT/NGTAQ/BQDistance.h"

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Helper: allocate aligned buffer of `words` uint64_t values
// ---------------------------------------------------------------------------
static void fillWords(uint64_t* buf, int words, uint64_t val) {
  for (int i = 0; i < words; ++i) buf[i] = val;
}

// ---------------------------------------------------------------------------
// Test 1: δ_BQ(p, p) == 0.0 for any input
//   XOR(pA, pA) = 0, so popcount(0 & anything) = 0 → distance = 0.0
// ---------------------------------------------------------------------------
static void testIdenticalVectors() {
  constexpr int words = 4;
  constexpr int D     = words * 64;  // 256

  uint64_t pA[words], pB[words];
  // arbitrary non-trivial values
  for (int i = 0; i < words; ++i) {
    pA[i] = 0xDEADBEEFCAFEBABEULL + (uint64_t)i * 0x123456789ABCULL;
    pB[i] = 0xFEEDFACEDEADC0DEULL ^ (uint64_t)i;
  }

  float d = NGTAQ::bqDistance(pA, pB, pA, pB, words, D);
  EXPECT_NEAR(d, 0.0f, 1e-9f);
}

// ---------------------------------------------------------------------------
// Test 2: All sign bits flipped + all magnitude bits set → distance == 1.0
//   XOR(pA, ~pA) = 0xFFFF… all ones, OR(pB, pB) where pB = 0xFFFF… = 0xFFFF…
//   AND = 0xFFFF… → popcount = words*64 = D → distance = D/D = 1.0
// ---------------------------------------------------------------------------
static void testOppositeSignMaxMagnitude() {
  constexpr int words = 2;
  constexpr int D     = words * 64;  // 128

  uint64_t pA[words], pB[words];
  uint64_t qA[words], qB[words];
  fillWords(pA, words, 0x5555555555555555ULL);
  fillWords(pB, words, 0xFFFFFFFFFFFFFFFFULL);  // all magnitude bits set
  fillWords(qA, words, ~0x5555555555555555ULL);  // all sign bits flipped
  fillWords(qB, words, 0xFFFFFFFFFFFFFFFFULL);

  float d = NGTAQ::bqDistance(pA, pB, qA, qB, words, D);
  EXPECT_NEAR(d, 1.0f, 1e-9f);
}

// ---------------------------------------------------------------------------
// Test 3: Both magnitude planes all-zero → distance == 0.0 regardless of signs
//   AND(anything, OR(0, 0)) = AND(anything, 0) = 0 → popcount = 0
// ---------------------------------------------------------------------------
static void testZeroMagnitudeVectors() {
  constexpr int words = 3;
  constexpr int D     = words * 64;  // 192

  uint64_t pA[words], pB[words];
  uint64_t qA[words], qB[words];
  // signs are maximally different
  fillWords(pA, words, 0xFFFFFFFFFFFFFFFFULL);
  fillWords(qA, words, 0x0000000000000000ULL);
  // magnitudes are zero
  fillWords(pB, words, 0ULL);
  fillWords(qB, words, 0ULL);

  float d = NGTAQ::bqDistance(pA, pB, qA, qB, words, D);
  EXPECT_NEAR(d, 0.0f, 1e-9f);
}

// ---------------------------------------------------------------------------
// Test 4: Symmetry δ_BQ(p, q) == δ_BQ(q, p)
//   Formula: popcount(XOR(pA,qA) & OR(pB,qB)) is symmetric since
//   XOR is symmetric and OR is symmetric.
// ---------------------------------------------------------------------------
static void testSymmetry() {
  constexpr int words = 4;
  constexpr int D     = words * 64;

  uint64_t pA[words], pB[words];
  uint64_t qA[words], qB[words];
  for (int i = 0; i < words; ++i) {
    pA[i] = 0xA5A5A5A5A5A5A5A5ULL * (uint64_t)(i + 1);
    pB[i] = 0x3C3C3C3C3C3C3C3CULL | (uint64_t)i;
    qA[i] = 0x5A5A5A5A5A5A5A5AULL + (uint64_t)(i * 7);
    qB[i] = 0xC3C3C3C3C3C3C3C3ULL ^ (uint64_t)(i * 3);
  }

  float dpq = NGTAQ::bqDistance(pA, pB, qA, qB, words, D);
  float dqp = NGTAQ::bqDistance(qA, qB, pA, pB, words, D);
  EXPECT_NEAR(dpq, dqp, 1e-9f);
}

// ---------------------------------------------------------------------------
// Test 5: Result always in [0.0, 1.0]
// ---------------------------------------------------------------------------
static void testNormalization() {
  constexpr int words = 8;
  constexpr int D     = words * 64;

  uint64_t pA[words], pB[words];
  uint64_t qA[words], qB[words];

  // Test several random-ish patterns
  static const uint64_t patterns[][4] = {
    {0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL},
    {0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL, 0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL},
    {0x1234567890ABCDEFULL, 0xFEDCBA9876543210ULL, 0xDEADBEEFCAFEBABEULL, 0xBAADF00DBAADF00DULL},
  };

  for (auto& pat : patterns) {
    for (int i = 0; i < words; ++i) {
      pA[i] = pat[0] * (uint64_t)(i + 1);
      pB[i] = pat[1] ^ (uint64_t)i;
      qA[i] = pat[2] + (uint64_t)i;
      qB[i] = pat[3] | (uint64_t)(i * 17);
    }
    float d = NGTAQ::bqDistance(pA, pB, qA, qB, words, D);
    EXPECT_TRUE(d >= 0.0f);
    EXPECT_TRUE(d <= 1.0f);
  }
}

// ---------------------------------------------------------------------------
// Test 6: Known-value test with D=64 (1 word), exact float equality
//   pA = 0xFF00FF00FF00FF00  (32 ones in specific positions)
//   pB = 0xFFFFFFFFFFFFFFFF  (all magnitude set)
//   qA = 0x00FF00FF00FF00FF  (complement of pA)
//   qB = 0xFFFFFFFFFFFFFFFF
//
//   XOR(pA, qA) = 0xFFFFFFFFFFFFFFFF (all 64 bits differ)
//   OR(pB, qB)  = 0xFFFFFFFFFFFFFFFF
//   AND         = 0xFFFFFFFFFFFFFFFF
//   popcount    = 64
//   distance    = 64/64 = 1.0
// ---------------------------------------------------------------------------
static void testKnownValue() {
  constexpr int words = 1;
  constexpr int D     = 64;

  uint64_t pA[1] = {0xFF00FF00FF00FF00ULL};
  uint64_t pB[1] = {0xFFFFFFFFFFFFFFFFULL};
  uint64_t qA[1] = {0x00FF00FF00FF00FFULL};
  uint64_t qB[1] = {0xFFFFFFFFFFFFFFFFULL};

  float d = NGTAQ::bqDistance(pA, pB, qA, qB, words, D);
  EXPECT_NEAR(d, 1.0f, 1e-9f);

  // Another known case: 32 active bits in XOR, all magnitude set
  // pA=0xF0F0F0F0F0F0F0F0, qA=0x0F0F0F0F0F0F0F0F → XOR=0xFFFFFFFFFFFFFFFF
  // pB=0xFFFF0000FFFF0000, qB=0x0000FFFF0000FFFF → OR=0xFFFFFFFFFFFFFFFF
  // AND=0xFFFFFFFFFFFFFFFF, popcount=64 → d=1.0
  pA[0] = 0xF0F0F0F0F0F0F0F0ULL;
  pB[0] = 0xFFFF0000FFFF0000ULL;
  qA[0] = 0x0F0F0F0F0F0F0F0FULL;
  qB[0] = 0x0000FFFF0000FFFFULL;
  d     = NGTAQ::bqDistance(pA, pB, qA, qB, words, D);
  EXPECT_NEAR(d, 1.0f, 1e-9f);

  // Case: only 4 bits differ and magnitude covers them
  // pA=0x000000000000000F, qA=0x0000000000000000 → XOR=0x000000000000000F (4 bits)
  // pB=0xFFFFFFFFFFFFFFFF, qB=0x0000000000000000 → OR=0xFFFFFFFFFFFFFFFF
  // AND = 0x000000000000000F → popcount=4 → d=4/64=0.0625
  pA[0] = 0x000000000000000FULL;
  pB[0] = 0xFFFFFFFFFFFFFFFFULL;
  qA[0] = 0x0000000000000000ULL;
  qB[0] = 0x0000000000000000ULL;
  d     = NGTAQ::bqDistance(pA, pB, qA, qB, words, D);
  EXPECT_NEAR(d, 4.0f / 64.0f, 1e-9f);

  // Case: magnitude mask zeros out differing bits → distance=0
  // pA=0xFFFFFFFFFFFFFFFF, qA=0x0000000000000000 → XOR=all ones
  // pB=0, qB=0 → OR=0 → AND=0 → popcount=0 → d=0.0
  pA[0] = 0xFFFFFFFFFFFFFFFFULL;
  pB[0] = 0x0000000000000000ULL;
  qA[0] = 0x0000000000000000ULL;
  qB[0] = 0x0000000000000000ULL;
  d     = NGTAQ::bqDistance(pA, pB, qA, qB, words, D);
  EXPECT_NEAR(d, 0.0f, 1e-9f);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  testIdenticalVectors();
  testOppositeSignMaxMagnitude();
  testZeroMagnitudeVectors();
  testSymmetry();
  testNormalization();
  testKnownValue();

  if (failures == 0) {
    std::cout << "All tests PASSED\n";
    return 0;
  } else {
    std::cerr << failures << " test(s) FAILED\n";
    return 1;
  }
}
