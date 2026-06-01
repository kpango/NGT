// Compare BATCH routing distance vs TRUE L2 for the gt neighbors and for the
// low-ID nodes the batch beam actually returns. Reveals whether the approximate
// routing distance ranks true NNs correctly (ordering), independent of traversal.
#include "NGT/NGTAQ/AQIndex.h"
#include "NGT/NGTAQ/GlobalPQ4.h"
#include "hdf5_io.h"
#include <cstdio>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

using namespace NGT::NGTAQ;

int main(int argc, char** argv) {
    const std::string dir = argv[1];
    const char* h5 = argv[2];
    ::NGTAQ::NGTAQIndex idx = ::NGTAQ::NGTAQIndex::load(dir + "/aqindex");
    idx.loadV2(dir);
    const int M = idx.mPQ(), Deff = idx.dEff();

    // Load flat gpq4 codes + recon norms directly (private in index).
    std::ifstream f(dir + "/v2_gpq4_codes.bin", std::ios::binary);
    uint64_t n=0; f.read((char*)&n,8);
    std::vector<uint8_t> codes((size_t)n*M); std::vector<float> norms(n);
    f.read((char*)codes.data(), codes.size());
    f.read((char*)norms.data(), norms.size()*4);

    H5FloatDataset test = h5_read_float(h5, "test");
    H5FloatDataset train = h5_read_float(h5, "train");
    H5IntDataset   gt   = h5_read_int(h5, "neighbors");
    const int D = test.n_cols;

    auto trueL2 = [&](const float* q, uint64_t id)->double{
        const float* x = train.data.data() + (size_t)id*D;
        double s=0; for(int d=0;d<D;d++){double df=q[d]-x[d]; s+=df*df;} return s;
    };

    GlobalPQ4LUT lut;
    std::vector<float> ipt((size_t)M*GPQ4_K);
    for (int q=0; q<3; ++q) {
        std::vector<float> query(test.data.data()+(size_t)q*D, test.data.data()+(size_t)q*D+D);
        float q_ns = idx.buildGlobalLUT16(query, lut, ipt.data());
        auto rdist = [&](uint64_t id)->double{
            const uint8_t* c = codes.data()+(size_t)id*M;
            double ip=0; for(int s=0;s<M;s++) ip += ipt[(size_t)s*GPQ4_K + c[s]];
            return (double)q_ns + norms[id] - 2.0*ip;
        };
        printf("=== q=%d  q_ns=%.1f ===\n", q, q_ns);
        // gt neighbors: route dist vs true L2
        for (int j=0;j<5;j++){
            uint64_t id = gt.data[(size_t)q*gt.n_cols+j];
            printf("  gt[%d]=%llu  route=%.1f  trueL2=%.1f\n", j,(unsigned long long)id, rdist(id), trueL2(query.data(),id));
        }
        // a few low-ID nodes the batch beam liked
        for (uint64_t id : {13612ull, 14218ull, 5164ull, 11061ull}) {
            printf("  low id=%llu  route=%.1f  trueL2=%.1f\n",(unsigned long long)id, rdist(id), trueL2(query.data(),id));
        }
        // global min route over a 50k sample
        double best=1e30; uint64_t bid=0;
        for (uint64_t id=0; id<n; id+=20){ double r=rdist(id); if(r<best){best=r;bid=id;} }
        printf("  sampled-min route id=%llu route=%.1f trueL2=%.1f (gt0 trueL2=%.1f)\n",
               (unsigned long long)bid, best, trueL2(query.data(),bid),
               trueL2(query.data(), gt.data[(size_t)q*gt.n_cols+0]));
    }
    return 0;
}
