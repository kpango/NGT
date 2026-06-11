// Diagnose batch-routing distance correctness on the real SIFT GPQ4 index.
// For a few queries, compare: (a) gpq4 single-node dist to node id, (b) true L2
// (fp16 raw) to id, for the index's own raw vectors used as queries. If gpq4 dist
// ranks self ~0 and near-neighbors small, routing is sound; else the codebook/LUT
// or rotation is mismatched.
#include "NGT/ArcFlare/ArcFlareIndex.h"
#include "NGT/ArcFlare/GlobalPQ4.h"
#include "NGT/ArcFlare/VectorRecord.h"
#include "hdf5_io.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

int main(int argc, char** argv) {
    const char* dir = argv[1];
    const char* h5  = argv[2];
    ArcFlare::ArcFlareIndex idx = ArcFlare::ArcFlareIndex::load(std::string(dir) + "/aqindex");
    idx.loadV2(dir);
    fprintf(stderr, "hasGPQ4=%d dEff=%d mPQ=%d\n", (int)idx.hasGPQ4(), idx.dEff(), idx.mPQ());

    H5FloatDataset test = h5_read_float(h5, "test");
    H5IntDataset   gt   = h5_read_int(h5, "neighbors");
    const int D = test.n_cols, Deff = idx.dEff();

    // Use the public batch search vs legacy to localize.  But here we just sanity the
    // single-node gpq4 distance via buildGlobalLUT16 + gpq4Dist is NOT public; instead
    // we run searchV2 with batch ON and OFF for 5 queries and print the returned ids/dists
    // plus the ground-truth id, to see whether batch returns garbage ids or wrong order.
    // Mode chosen by env BEFORE first searchV2 (flag is read once).
    const char* mode = (argc > 3) ? argv[3] : "0";
    setenv("AQ_BATCH_ROUTING", mode, 1);
    fprintf(stderr, "AQ_BATCH_ROUTING=%s\n", mode);

    for (int q = 0; q < 5; ++q) {
        std::vector<float> query(Deff, 0.f);
        std::copy(test.data.data() + (size_t)q * D, test.data.data() + (size_t)q * D + D, query.begin());
        auto res = idx.searchV2(query, 10, 0.2f, 0.4f, 3, 200);
        int gt0 = gt.data[(size_t)q * gt.n_cols + 0];
        // Is gt0 anywhere in res?
        bool found = false; for (auto& r : res) if ((int)r.id == gt0) found = true;
        printf("q=%d gt0=%d found=%d ids:", q, gt0, (int)found);
        for (auto& r : res) printf(" %u(%.0f)", r.id, r.distance);
        printf("\n");
    }
    return 0;
}
