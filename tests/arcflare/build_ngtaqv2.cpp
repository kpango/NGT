// build_arcflarev2.cpp
// Build ArcFlare index from an NGT index with configurable K-means clusters.
// Usage: ./build_arcflarev2 <ngt_path> <out_dir> [k_clusters=0] [max_edges=64] [alpha=1.2] [metric=l2]
//
// k_clusters=0 uses select_k(N) = N/1000 (default).
// k_clusters=2000 overrides to K=2000 explicitly.
// alpha controls AlphaCGPruner aggressiveness: larger = denser graph (less pruning).
//   alpha=1.2: default (aggressive pruning, sparse graph)
//   alpha=2.0: moderate, ~30-50% more edges
//   alpha=3.0: light pruning, denser graph, higher recall ceiling
// metric: l2 (default) | angular | cosine — l2 uses DistanceTypeL2; angular/cosine use DistanceTypeCosine + L2-normalization
//
// Output directory layout:
//   <out_dir>/aqindex        — base ArcFlareIndex binary
//   <out_dir>/v2_srht.bin    — SRHT rotation matrix
//   <out_dir>/v2_kmeans.bin  — K-means centroids
//   <out_dir>/v2_codebook.bin— PQ sub-codebooks
//   <out_dir>/v2_records.bin — per-vector tier1/tier2/norm/cid
//   <out_dir>/v2_pca.bin     — PCA projection (if used)
#include "NGT/ArcFlare/ArcFlareIndex.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <ngt_path> <out_dir> [k_clusters=0] [max_edges=64] [alpha=1.2] [metric=l2]\n";
        return 1;
    }
    const std::string ngt_path   = argv[1];
    const std::string out_dir    = argv[2];
    const int   k_clusters  = argc > 3 ? std::stoi(argv[3]) : 0;
    const int   max_edges   = argc > 4 ? std::stoi(argv[4]) : 64;
    const float alpha       = argc > 5 ? std::stof(argv[5]) : 1.2f;
    const std::string metric_str = argc > 6 ? argv[6] : "l2";

    ArcFlare::ArcFlareIndex::Property prop;
    prop.dimension      = 128;
    prop.k_clusters     = k_clusters;
    prop.max_edges      = max_edges;
    prop.alpha          = alpha;
    prop.n_tau_samples  = 10000;
    if (metric_str == "angular" || metric_str == "cosine") {
        prop.metric = NGT::ObjectSpace::DistanceTypeCosine;
    } else {
        prop.metric = NGT::ObjectSpace::DistanceTypeL2;
    }

    std::cout << "[build] ngt_path  = " << ngt_path  << "\n";
    std::cout << "[build] out_dir   = " << out_dir   << "\n";
    std::cout << "[build] k_clusters= " << (k_clusters > 0 ? k_clusters : -1)
              << (k_clusters == 0 ? " (auto = N/1000)" : "") << "\n";
    std::cout << "[build] max_edges = " << max_edges << "\n";
    std::cout << "[build] alpha     = " << alpha << "\n";
    std::cout << "[build] metric    = " << metric_str << "\n";
    std::cout.flush();

    const auto t0 = std::chrono::steady_clock::now();

    ArcFlare::ArcFlareIndex idx = ArcFlare::ArcFlareIndex::fromNGTv2(ngt_path, prop);

    const auto t1 = std::chrono::steady_clock::now();
    const double build_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[build] fromNGTv2 done in " << (int)build_s << "s"
              << "  size=" << idx.size() << "\n";
    std::cout.flush();

    std::filesystem::create_directories(out_dir);
    idx.save(out_dir + "/aqindex");
    idx.saveV2(out_dir);

    const auto t2 = std::chrono::steady_clock::now();
    const double save_s = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "[build] saved to " << out_dir << " (" << (int)save_s << "s)\n";
    return 0;
}
