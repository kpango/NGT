// tests/ngtaq/build_ngt_sift1m.cpp
// Build a persistent NGT index from SIFT-1M sift_base.fvecs.
//
// Usage: ./build_ngt_sift1m <sift_base.fvecs> <output_ngt_path> [batch_size=50000]
//
// Example:
//   ./build_ngt_sift1m data/sift1m/sift/sift_base.fvecs /tmp/sift1m_ngt
#include "NGT/Index.h"
#include "fvecs_io.h"
#include <chrono>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <sift_base.fvecs> <output_ngt_path> [batch_size=50000]\n";
        return 1;
    }
    const std::string fvecs_path = argv[1];
    const std::string ngt_path   = argv[2];
    const size_t batch_size = (argc > 3) ? static_cast<size_t>(std::stoi(argv[3])) : 50000;

    std::cout << "Loading " << fvecs_path << " ...\n";
    auto t0 = std::chrono::steady_clock::now();
    auto vecs = NGTAQ::loadFvecs(fvecs_path);
    if (vecs.empty()) { std::cerr << "No vectors loaded\n"; return 1; }
    const int D = static_cast<int>(vecs[0].size());
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Loaded " << vecs.size() << " vectors, D=" << D
              << " (" << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count()
              << "s)\n";

    if (D % 64 != 0) {
        std::cerr << "Warning: D=" << D << " is not a multiple of 64 — "
                  << "NGTAQ fromNGT() will reject this index\n";
    }

    // Create NGT index
    NGT::Property prop;
    prop.dimension    = D;
    prop.objectType   = NGT::ObjectSpace::ObjectType::Float;
    prop.distanceType = NGT::ObjectSpace::DistanceType::DistanceTypeL2;

    std::cout << "Creating NGT index at " << ngt_path << " ...\n";
    NGT::Index::create(ngt_path, prop);
    NGT::Index ngt(ngt_path);

    // Insert in batches: each batch appends vectors then builds graph
    for (size_t i = 0; i < vecs.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, vecs.size());
        for (size_t j = i; j < end; ++j)
            ngt.append(vecs[j].data(), D);
        ngt.createIndex(/*threads=*/8);
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t1).count();
        std::cout << "  " << end << "/" << vecs.size()
                  << " indexed (" << elapsed << "s elapsed)\n";
    }

    ngt.save();
    std::cout << "NGT index saved to " << ngt_path << "\n";
    return 0;
}
