// Round-trip test for SoAGraph::buildGPQ4 + gpq4Blocks + the batch kernel.
// Builds a small graph, a random 4-bit code table, fills the contiguous neighbor
// store, then verifies the batch kernel over the stored blocks reproduces the
// inner products computed by direct per-neighbor LUT gather. Also tests save/load.
#include "NGT/NGTAQ/SoAGraph.h"
#include "NGT/NGTAQ/GlobalPQ4.h"
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>

using namespace NGTAQ;
using namespace NGT::NGTAQ;

int main() {
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> uc(0, 15);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);

    const int words = 2;          // D=128 BQ words (unused here but ctor needs it)
    const int M = 16, N = 137;
    SoAGraph g(words);
    std::vector<uint64_t> bq(words * 2, 0);
    for (int i = 0; i < N; ++i) g.addNode(bq.data());
    g.finalizeCSR();

    // Random adjacency (variable degree 0..40) + global codes + norms.
    std::vector<std::vector<uint32_t>> adj(N);
    std::uniform_int_distribution<int> ud(0, 40), un(0, N - 1);
    for (int i = 0; i < N; ++i) {
        int deg = ud(rng);
        for (int j = 0; j < deg; ++j) adj[i].push_back(un(rng));
    }
    g.resetEdges(adj);

    std::vector<uint8_t> codes((size_t)N * M);
    std::vector<float> norm((size_t)N);
    for (auto& c : codes) c = (uint8_t)uc(rng);
    for (auto& x : norm) x = std::fabs(uf(rng)) * 10.f;

    g.buildGPQ4(M, codes.data(), norm.data());

    // Random query LUT.
    std::vector<float> ip((size_t)M * 16);
    for (auto& x : ip) x = uf(rng);
    GlobalPQ4LUT lut; gpq4_build_lut(ip.data(), M, lut);

    int fails = 0;
    auto verify = [&](SoAGraph& gg, const char* tag) {
        for (int i = 0; i < N; ++i) {
            auto nbrs = gg.getNeighbors(i);
            const uint8_t* blocks = gg.gpq4Blocks(i);
            uint32_t nblk = gg.gpq4NumBlocks(i);
            if (nbrs.size() == 0) { if (nblk != 0) { printf("[%s] node %d expected 0 blocks\n", tag, i); ++fails; } continue; }
            const size_t blk_bytes = gg.gpq4BlockBytes();
            const int planes = (M + 1) / 2;
            for (uint32_t b = 0; b < nblk; ++b) {
                const uint8_t* blkp = blocks + (size_t)b * blk_bytes;
                float out[16];
                gpq4_batch_ip(blkp, lut, out);
                const uint16_t* normp = reinterpret_cast<const uint16_t*>(blkp + (size_t)planes * 16);
                int n_real = (int)std::min<size_t>(16, nbrs.size() - (size_t)b * 16);
                for (int n = 0; n < n_real; ++n) {
                    uint32_t nbr = nbrs[(size_t)b * 16 + n];
                    // direct gather IP from the float LUT
                    float truth = 0.f;
                    for (int s = 0; s < M; ++s) truth += ip[(size_t)s * 16 + codes[(size_t)nbr * M + s]];
                    float tol = lut.scale * M * 0.6f + 1e-3f;
                    if (std::fabs(out[n] - truth) > tol) {
                        if (fails < 10) printf("[%s] IP mismatch node=%d blk=%u n=%d got=%.4f truth=%.4f\n",
                                               tag, i, b, n, out[n], truth);
                        ++fails;
                    }
                    // norm round-trip (fp16 tolerance)
                    float nh = fp16_to_float(normp[n]);
                    if (std::fabs(nh - norm[nbr]) > 0.05f * norm[nbr] + 1e-2f) {
                        if (fails < 10) printf("[%s] norm mismatch node=%d n=%d got=%.4f truth=%.4f\n",
                                               tag, i, n, nh, norm[nbr]);
                        ++fails;
                    }
                }
            }
        }
    };
    verify(g, "build");

    // Save/load round-trip.
    g.saveGPQ4("/tmp/test_gpq4_store.bin");
    SoAGraph g2(words);
    for (int i = 0; i < N; ++i) g2.addNode(bq.data());
    g2.resetEdges(adj);
    g2.finalizeCSR();
    if (!g2.loadGPQ4("/tmp/test_gpq4_store.bin")) { printf("loadGPQ4 failed\n"); ++fails; }
    verify(g2, "load");

    if (fails == 0) { printf("ALL GPQ4 STORE TESTS PASSED (N=%d M=%d)\n", N, M); return 0; }
    printf("GPQ4 STORE TESTS FAILED: %d\n", fails);
    return 1;
}
