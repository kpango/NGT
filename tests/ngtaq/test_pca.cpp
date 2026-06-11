#include "../../lib/NGT/NGTAQ/PCAProjector.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

static int failures = 0;
#define EXPECT_TRUE(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d\n",__FILE__,__LINE__); ++failures; } } while(0)
#define EXPECT_NEAR(a,b,eps) do { if (std::abs((double)(a)-(double)(b))>(eps)) { \
    fprintf(stderr,"FAIL %s:%d: %f != %f\n",__FILE__,__LINE__,(double)(a),(double)(b)); ++failures; } } while(0)
#define EXPECT_EQ(a,b) do { if ((a)!=(b)) { \
    fprintf(stderr,"FAIL %s:%d: %lld != %lld\n",__FILE__,__LINE__,(long long)(a),(long long)(b)); ++failures; } } while(0)

static void test_output_dim() {
    const int D = 128, TOP = 32;
    NGT::NGTAQ::PCAProjector pca(D, TOP, 42);
    std::vector<float> data(1000 * D);
    std::mt19937 rng(0); std::normal_distribution<float> nd;
    for (auto& v : data) v = nd(rng);
    pca.fit(data.data(), 1000);
    std::vector<float> out(TOP);
    pca.project(data.data(), out.data());
    EXPECT_EQ((int)out.size(), TOP);
}

static void test_variance_captured() {
    // 1D signal embedded in 4D space: first PC should capture > 90% variance
    const int D = 4, TOP = 1;
    const int N = 500;
    std::vector<float> data(N * D, 0.f);
    std::mt19937 rng(10); std::normal_distribution<float> nd;
    for (int i = 0; i < N; ++i) {
        data[i*D+0] = nd(rng) * 10.f;
        for (int d = 1; d < D; ++d) data[i*D+d] = nd(rng) * 0.1f;
    }
    NGT::NGTAQ::PCAProjector pca(D, TOP, 0);
    pca.fit(data.data(), N);
    float var_explained = pca.variance_ratio(0);
    EXPECT_TRUE(var_explained > 0.9f);
}

static void test_whitened_unit_variance() {
    const int D = 8, TOP = 4;
    const int N = 2000;
    std::vector<float> data(N * D);
    std::mt19937 rng(55); std::normal_distribution<float> nd;
    for (auto& v : data) v = nd(rng);
    NGT::NGTAQ::PCAProjector pca(D, TOP, 0, /*whiten=*/true);
    pca.fit(data.data(), N);
    std::vector<float> projected(N * TOP);
    for (int i = 0; i < N; ++i)
        pca.project(data.data() + i*D, projected.data() + i*TOP);
    for (int j = 0; j < TOP; ++j) {
        float mean = 0, var = 0;
        for (int i = 0; i < N; ++i) mean += projected[i*TOP+j];
        mean /= N;
        for (int i = 0; i < N; ++i) { float d = projected[i*TOP+j]-mean; var += d*d; }
        var /= N;
        EXPECT_NEAR(var, 1.f, 0.15f);
    }
}

int main() {
    test_output_dim();
    test_variance_captured();
    test_whitened_unit_variance();
    if (failures == 0) { printf("OK (3 tests)\n"); return 0; }
    fprintf(stderr, "%d test(s) FAILED\n", failures); return 1;
}
