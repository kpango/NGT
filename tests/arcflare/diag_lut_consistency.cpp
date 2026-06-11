// Load the real GPQ4 codebook + flat codes, pick a query (a stored vector's rotation
// is not available, so use a random query), and compare the three IP computations
// for a few nodes: exact-from-codebook, ip_table-sum, batch-kernel. They must agree
// to within uint8 quantization tolerance. Isolates whether the LUT/kernel is the bug.
#include "NGT/ArcFlare/GlobalPQ4.h"
#include "NGT/ArcFlare/ADCTable.h"
#include <cstdio>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <cstdint>

using namespace NGT::ArcFlare;

int main(int argc, char** argv) {
    const std::string dir = argv[1];
    const int M = 16, K = 16, Dsub = 8, D = 128;

    // Load codebook [M*16*Dsub] and transpose.
    std::vector<float> cb((size_t)M*K*Dsub);
    { std::ifstream f(dir + "/v2_gpq4_codebook.bin", std::ios::binary);
      f.read((char*)cb.data(), cb.size()*sizeof(float)); }
    std::vector<float> cbT((size_t)M*Dsub*K);
    build_tier2_codebook_T(cb.data(), M, K, Dsub, cbT.data());

    // Load a few node codes.
    std::vector<uint8_t> codes; std::vector<float> norms;
    { std::ifstream f(dir + "/v2_gpq4_codes.bin", std::ios::binary);
      uint64_t n=0; f.read((char*)&n, sizeof(n));
      codes.resize((size_t)n*M); norms.resize(n);
      f.read((char*)codes.data(), codes.size());
      f.read((char*)norms.data(), norms.size()*sizeof(float));
      fprintf(stderr, "loaded %llu node codes\n", (unsigned long long)n); }

    std::mt19937 rng(3); std::normal_distribution<float> g(0,30);
    std::vector<float> qrot(D); for (auto&x:qrot) x=g(rng);

    // (2) ip_table from cbT
    std::vector<float> ip((size_t)M*K);
    gpq4_ip_table(qrot.data(), M, cbT.data(), Dsub, ip.data());
    // verify (2) vs exact dot from cb
    double max_ip_err = 0;
    for (int s=0;s<M;++s) for (int c=0;c<K;++c){
        double dot=0; for(int d=0;d<Dsub;++d) dot += qrot[s*Dsub+d]*cb[((size_t)s*K+c)*Dsub+d];
        max_ip_err = std::max(max_ip_err, std::fabs(dot - ip[(size_t)s*K+c]));
    }
    printf("ip_table vs exact-dot max err = %.5f\n", max_ip_err);

    GlobalPQ4LUT lut; gpq4_build_lut(ip.data(), M, lut);

    // For 8 nodes, compare batch-kernel IP (pack a block of just these as block-of-16)
    // vs ip_table sum over their codes.
    uint8_t cc16[16*M]; for (int i=0;i<16*M;++i) cc16[i]=0;
    for (int n=0;n<8;++n)
        for (int s=0;s<M;++s) cc16[n*M+s] = codes[(size_t)n*M+s];
    uint8_t block[(M+1)/2*16]; gpq4_pack_block(cc16, M, 8, block);
    float out[16]; gpq4_batch_ip(block, lut, out);

    double max_kern_err = 0;
    for (int n=0;n<8;++n){
        float truth=0; for(int s=0;s<M;++s) truth += ip[(size_t)s*K + codes[(size_t)n*M+s]];
        printf("node %d: batch_ip=%.3f ip_sum=%.3f diff=%.3f  norm=%.1f\n",
               n, out[n], truth, out[n]-truth, norms[n]);
        max_kern_err = std::max(max_kern_err, (double)std::fabs(out[n]-truth));
    }
    printf("batch-kernel vs ip_sum max err = %.5f (scale=%.4f, M*scale*0.6=%.3f)\n",
           max_kern_err, lut.scale, lut.scale*M*0.6);
    return 0;
}
