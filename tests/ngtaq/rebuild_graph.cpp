// tests/ngtaq/rebuild_graph.cpp
// Rebuild the graph edges of an existing AQv2 index from a denser NGT source.
// Reuses SRHT/K-means/PQ quantization from the existing index; only re-runs
// graph construction + entry-point selection (~50s vs ~400s for full rebuild).
//
// Usage: rebuild_graph <aqv2_dir> <ngt_dir> <out_dir> [alpha=1.2] [max_edges=64]
#include "NGT/NGTAQ/AQIndex.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

static double elapsed_s(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <aqv2_dir> <ngt_dir> <out_dir> [alpha=1.2] [max_edges=64]\n",
            argv[0]);
        return 1;
    }
    const std::string aqv2_dir   = argv[1];
    const std::string ngt_dir    = argv[2];
    const std::string out_dir    = argv[3];
    const float       new_alpha  = (argc > 4) ? std::stof(argv[4]) : -1.0f;
    const int         new_edges  = (argc > 5) ? std::stoi(argv[5]) : -1;

    const auto t0 = std::chrono::steady_clock::now();
    fprintf(stderr, "[rebuild_graph] aqv2_dir   = %s\n", aqv2_dir.c_str());
    fprintf(stderr, "[rebuild_graph] ngt_dir    = %s\n", ngt_dir.c_str());
    fprintf(stderr, "[rebuild_graph] out_dir    = %s\n", out_dir.c_str());
    fprintf(stderr, "[rebuild_graph] alpha      = %.2f\n", new_alpha);
    fprintf(stderr, "[rebuild_graph] max_edges  = %d\n", new_edges);

    // Load existing AQv2 index (keeps quantization)
    fprintf(stderr, "[rebuild_graph] Loading index...\n");
    NGTAQ::NGTAQIndex idx = NGTAQ::NGTAQIndex::load(aqv2_dir + "/aqindex");
    idx.loadV2(aqv2_dir);
    fprintf(stderr, "[rebuild_graph] Loaded N=%zu D=%d\n",
        idx.size(), idx.dim());

    // Rebuild graph from denser NGT source
    fprintf(stderr, "[rebuild_graph] Rebuilding graph from NGT (alpha=%.2f max_edges=%d)...\n",
        new_alpha, new_edges);
    idx.rebuildGraphFromNGT(ngt_dir, new_alpha, new_edges);
    fprintf(stderr, "[rebuild_graph] Done in %.0fs\n", elapsed_s(t0));

    // Save to output dir
    std::filesystem::create_directories(out_dir);
    idx.save(out_dir + "/aqindex");
    idx.saveV2(out_dir);
    fprintf(stderr, "[rebuild_graph] Saved to %s (%.0fs)\n", out_dir.c_str(), elapsed_s(t0));
    return 0;
}
