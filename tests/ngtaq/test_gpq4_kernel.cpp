// Standalone correctness test for the GlobalPQ4 vpshufb batch kernel.
// Verifies: (1) AVX2 kernel == scalar reference, (2) both approximate the true
// float inner products computed from the quantized LUT (not the codebook — the
// LUT IS the ground truth here, since quantization error is what we measure).
#include "NGT/NGTAQ/GlobalPQ4.h"
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>

using namespace NGT::NGTAQ;

// Pack 16 neighbors' codes via the library packer.
static void pack_block(const uint8_t* codes, int M, std::vector<uint8_t>& block) {
    const int planes = (M + 1) / 2;
    block.assign((size_t)planes * 16, 0);
    gpq4_pack_block(codes, M, 16, block.data());
}

int main() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> uf(-2.f, 2.f);
    std::uniform_int_distribution<int> uc(0, 15);

    int fails = 0;
    for (int M : {16, 32, 64, 128 /* QG-fine D_sub=1 */, 18 /* odd */}) {
        for (int trial = 0; trial < 200; ++trial) {
            // Random float IP table [M*16].
            std::vector<float> ip((size_t)M * 16);
            for (auto& x : ip) x = uf(rng);

            GlobalPQ4LUT lut;
            gpq4_build_lut(ip.data(), M, lut);

            // Random 16 neighbors' codes.
            std::vector<uint8_t> codes((size_t)16 * M);
            for (auto& c : codes) c = (uint8_t)uc(rng);
            std::vector<uint8_t> block;
            pack_block(codes.data(), M, block);

            float ip_scalar[16], ip_avx[16];
            gpq4_batch_ip_scalar(block.data(), lut, ip_scalar);
#if defined(__AVX2__)
            gpq4_batch_ip_avx2(block.data(), lut, ip_avx);
#else
            gpq4_batch_ip_scalar(block.data(), lut, ip_avx);
#endif
            // True (dequantized) IP from the LUT: sum of per-subspace LUT entries.
            for (int n = 0; n < 16; ++n) {
                float truth = 0.f;
                for (int s = 0; s < M; ++s)
                    truth += ip[(size_t)s * 16 + codes[n * M + s]];

                // scalar vs avx must match exactly (same integer accumulation).
                if (std::fabs(ip_scalar[n] - ip_avx[n]) > 1e-3f) {
                    if (fails < 10)
                        printf("MISMATCH M=%d trial=%d n=%d scalar=%.4f avx=%.4f\n",
                               M, trial, n, ip_scalar[n], ip_avx[n]);
                    ++fails;
                }
                // quantized approx vs truth: bounded by per-subspace step (scale) × M.
                float tol = lut.scale * M * 0.6f + 1e-3f;
                if (std::fabs(ip_scalar[n] - truth) > tol) {
                    if (fails < 20)
                        printf("QUANT-ERR M=%d trial=%d n=%d approx=%.4f truth=%.4f tol=%.4f\n",
                               M, trial, n, ip_scalar[n], truth, tol);
                    ++fails;
                }
            }
        }
    }
    if (fails == 0) { printf("ALL GPQ4 KERNEL TESTS PASSED\n"); return 0; }
    printf("GPQ4 KERNEL TESTS FAILED: %d\n", fails);
    return 1;
}
