// tests/arcflare/test_l2_fp16.cpp — fp32-query vs fp16-stored squared-L2 distance.
#include "NGT/ArcFlare/SIMDUtils.h"
#include "NGT/ArcFlare/VectorRecord.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace NGT::ArcFlare;
int main() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (int D : {1, 7, 13, 64, 128, 256, 784, 960, 1024}) {
        std::vector<float> a(D), b(D); std::vector<uint16_t> bh(D);
        for (int i = 0; i < D; ++i){ a[i]=U(rng); b[i]=U(rng); bh[i]=float_to_fp16(b[i]); }
        float ref = 0.f;
        for (int i = 0; i < D; ++i){ float d=a[i]-fp16_to_float(bh[i]); ref+=d*d; }
        float got = l2_sq_f32_fp16(a.data(), bh.data(), D);
        float rel = std::fabs(got-ref)/(ref+1e-9f);
        printf("D=%4d ref=%.6f got=%.6f rel=%.2e\n", D, ref, got, rel);
        assert(rel < 1e-4f && "l2_sq_f32_fp16 mismatch vs scalar reference");
    }
    printf("test_l2_fp16: PASS\n"); return 0;
}
