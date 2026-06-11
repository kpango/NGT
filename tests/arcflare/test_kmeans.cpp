#include "../../lib/NGT/ArcFlare/KMeansCentering.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static int failures = 0;
#define EXPECT_TRUE(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL %s:%d: false\n",__FILE__,__LINE__); ++failures; } } while(0)
#define EXPECT_NEAR(a,b,eps) do { if (std::abs((double)(a)-(double)(b))>(eps)) { \
    fprintf(stderr,"FAIL %s:%d: %f != %f\n",__FILE__,__LINE__,(double)(a),(double)(b)); ++failures; } } while(0)
#define EXPECT_EQ(a,b) do { if ((a)!=(b)) { \
    fprintf(stderr,"FAIL %s:%d: %lld != %lld\n",__FILE__,__LINE__,(long long)(a),(long long)(b)); ++failures; } } while(0)

static void test_k_selection() {
    EXPECT_EQ(NGT::ArcFlare::select_k(100000), 256u);
    EXPECT_EQ(NGT::ArcFlare::select_k(1000000), 1000u);
    EXPECT_EQ(NGT::ArcFlare::select_k(10000000), 10000u);
    EXPECT_EQ(NGT::ArcFlare::select_k(5000000000ull), 4000000u);
}

static void test_separable_clusters() {
    const int D = 2, N = 200, K = 2;
    std::vector<float> data(N * D);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.1f);
    for (int i = 0; i < N / 2; ++i) {
        data[i * D + 0] = 0.f + nd(rng);
        data[i * D + 1] = 0.f + nd(rng);
    }
    for (int i = N / 2; i < N; ++i) {
        data[i * D + 0] = 10.f + nd(rng);
        data[i * D + 1] = 10.f + nd(rng);
    }
    NGT::ArcFlare::KMeansCentering km(K, D, 42);
    km.train(data.data(), N);
    std::vector<uint32_t> ids(N);
    km.assign(data.data(), N, ids.data());

    uint32_t label0 = ids[0];
    bool first_ok = true;
    for (int i = 0; i < N / 2; ++i) if (ids[i] != label0) { first_ok = false; break; }
    uint32_t label1 = ids[N / 2];
    bool second_ok = (label1 != label0);
    for (int i = N / 2; i < N; ++i) if (ids[i] != label1) { second_ok = false; break; }
    EXPECT_TRUE(first_ok && second_ok);
}

static void test_centroid_count() {
    const int D = 4, N = 1000, K = 8;
    std::vector<float> data(N * D, 0.f);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : data) v = nd(rng);
    NGT::ArcFlare::KMeansCentering km(K, D, 1);
    km.train(data.data(), N);
    EXPECT_EQ(km.num_clusters(), (uint32_t)K);
}

static void test_residual_subtraction() {
    const int D = 2, N = 4, K = 2;
    std::vector<float> data = {0.f,0.f, 0.f,1.f, 10.f,0.f, 10.f,1.f};
    NGT::ArcFlare::KMeansCentering km(K, D, 42);
    km.train(data.data(), N);
    std::vector<uint32_t> ids(N);
    km.assign(data.data(), N, ids.data());
    std::vector<float> residual(D);
    km.get_residual(data.data(), ids[0], residual.data());
    float r2 = residual[0]*residual[0] + residual[1]*residual[1];
    EXPECT_NEAR(std::sqrt(r2), 0.f, 0.5f);
}

int main() {
    test_k_selection();
    test_separable_clusters();
    test_centroid_count();
    test_residual_subtraction();
    if (failures == 0) { printf("OK (4 tests)\n"); return 0; }
    fprintf(stderr, "%d test(s) FAILED\n", failures); return 1;
}
