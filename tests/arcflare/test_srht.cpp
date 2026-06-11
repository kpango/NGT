#include "../../lib/NGT/ArcFlare/SRHT.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <numeric>
#include <random>

static int failures = 0;
#define EXPECT_NEAR(a,b,eps) do { if (std::abs((double)(a)-(double)(b))>(eps)) { \
    fprintf(stderr,"FAIL %s:%d: %f not near %f\n",__FILE__,__LINE__,(double)(a),(double)(b)); ++failures; } } while(0)
#define EXPECT_TRUE(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: false\n",__FILE__,__LINE__); ++failures; } } while(0)

static void test_norm_preservation() {
    const int D = 128;
    NGT::ArcFlare::SRHT srht(D, 42);
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (int trial = 0; trial < 10; ++trial) {
        std::vector<float> x(D);
        for (auto& v : x) v = dist(rng);
        double norm_before = 0;
        for (float v : x) norm_before += v * v;
        norm_before = std::sqrt(norm_before);
        std::vector<float> y(D);
        srht.apply(x.data(), y.data());
        double norm_after = 0;
        for (float v : y) norm_after += v * v;
        norm_after = std::sqrt(norm_after);
        EXPECT_NEAR(norm_after, norm_before, norm_before * 0.001);
    }
}

static void test_deterministic_with_same_seed() {
    const int D = 128;
    NGT::ArcFlare::SRHT a(D, 99), b(D, 99);
    std::vector<float> x(D, 1.0f), ya(D), yb(D);
    a.apply(x.data(), ya.data());
    b.apply(x.data(), yb.data());
    for (int i = 0; i < D; ++i) EXPECT_NEAR(ya[i], yb[i], 1e-6f);
}

static void test_different_seeds_differ() {
    const int D = 128;
    NGT::ArcFlare::SRHT a(D, 1), b(D, 2);
    std::vector<float> x(D, 1.0f), ya(D), yb(D);
    a.apply(x.data(), ya.data());
    b.apply(x.data(), yb.data());
    int matches = 0;
    for (int i = 0; i < D; ++i) if (std::abs(ya[i] - yb[i]) < 1e-6f) ++matches;
    EXPECT_TRUE(matches < D);
}

static void test_walshhadamard_d4() {
    // For D=4, WHT of [1,0,0,0] should be [1,1,1,1]/2
    const int D = 4;
    NGT::ArcFlare::SRHT srht(D, 0, /*skip_diagonal=*/true);
    std::vector<float> x = {1.f, 0.f, 0.f, 0.f};
    std::vector<float> y(D);
    srht.apply_hadamard_only(x.data(), y.data());
    for (int i = 0; i < D; ++i) EXPECT_NEAR(y[i], 0.5f, 1e-5f);
}

int main() {
    test_norm_preservation();
    test_deterministic_with_same_seed();
    test_different_seeds_differ();
    test_walshhadamard_d4();
    if (failures == 0) { printf("OK (4 tests)\n"); return 0; }
    fprintf(stderr, "%d test(s) FAILED\n", failures); return 1;
}
