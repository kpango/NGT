// diag_2level: validate the 2-level coarse quantizer accuracy + speed vs the exact and
// fp16 brute-force centroid scans, on the real SIFT centroids + SRHT-rotated queries.
// Usage: diag_2level <aq_index_dir> <hdf5_path> [nq=1000]
#include "NGT/ArcFlare/ArcFlareIndex.h"
#include "NGT/ArcFlare/KMeansCentering.h"
#include "hdf5_io.h"
#include <cstdio>
#include <chrono>
#include <vector>

using namespace NGT::ArcFlare;

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <idx> <hdf5> [nq]\n", argv[0]); return 1; }
    const std::string dir = argv[1];
    const int NQ = (argc > 3) ? std::atoi(argv[3]) : 1000;

    ::ArcFlare::ArcFlareIndex idx = ::ArcFlare::ArcFlareIndex::load(dir + "/aqindex");
    idx.loadV2(dir);
    auto qs = h5_read_float(argv[2], "test");
    int nq = std::min(NQ, qs.n_rows), dq = qs.n_cols;
    const KMeansCentering* km = idx.kmeansForDiag();
    const int D = km->dim();
    fprintf(stderr, "K=%u D=%d nq=%d (dq=%d)\n", km->num_clusters(), D, nq, dq);

    std::vector<std::vector<float>> qrot(nq, std::vector<float>(D, 0.f));
    for (int i = 0; i < nq; ++i) idx.rotateForDiag(qs.row(i), dq, qrot[i].data());

    // warm the 2-level + fp16 caches (lazy build) before timing
    (void)km->nearest_2level(qrot[0].data(), 4);
    (void)km->nearest_fp16(qrot[0].data());

    // exact ground truth assignment
    std::vector<uint32_t> exact(nq);
    for (int i = 0; i < nq; ++i) exact[i] = km->nearest_public(qrot[i].data());

    int probes[] = {1, 2, 3, 4, 6, 8};
    for (int probe : probes) {
        int agree = 0; long long t = 0;
        for (int i = 0; i < nq; ++i) {
            auto a = std::chrono::steady_clock::now();
            uint32_t ap = km->nearest_2level(qrot[i].data(), probe);
            auto b = std::chrono::steady_clock::now();
            t += std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
            if (ap == exact[i]) ++agree;
        }
        printf("probe=%d  agree=%.4f  2level=%.3fus/q\n", probe, (double)agree / nq, t / 1000.0 / nq);
    }
    // reference timings
    long long te = 0, tf = 0;
    for (int i = 0; i < nq; ++i) {
        auto a = std::chrono::steady_clock::now();
        volatile uint32_t e = km->nearest_public(qrot[i].data()); (void)e;
        auto b = std::chrono::steady_clock::now();
        te += std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
        a = std::chrono::steady_clock::now();
        volatile uint32_t fp = km->nearest_fp16(qrot[i].data()); (void)fp;
        b = std::chrono::steady_clock::now();
        tf += std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    }
    printf("ref  exact=%.3fus/q  fp16=%.3fus/q\n", te / 1000.0 / nq, tf / 1000.0 / nq);
    return 0;
}
