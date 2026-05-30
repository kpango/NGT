// tests/ngtaq/refine_graph.cpp
// Refine an existing AQv2 index using rebuildGraphSelf (self-referential).
// Does NOT need a denser NGT source — uses the index's own searchV2.
// One pass raises the recall ceiling by ~5-10%.
//
// Usage: refine_graph <aqv2_dir> <out_dir> [k_search=40] [gamma=0.30] [threads=16] [max_edges=-1] [rebuild_n_probe=0]
//   rebuild_n_probe=0 → auto (uses all K clusters); >0 → cap at that value
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
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <aqv2_dir> <out_dir> [k_search=40] [gamma=0.30] [threads=16] [max_edges=-1] [rebuild_n_probe=0]\n",
            argv[0]);
        return 1;
    }
    const std::string in_dir       = argv[1];
    const std::string out_dir      = argv[2];
    const int    k_search          = (argc > 3) ? std::stoi(argv[3]) : 40;
    const float  gamma             = (argc > 4) ? std::stof(argv[4]) : 0.30f;
    const int    n_threads         = (argc > 5) ? std::stoi(argv[5]) : 16;
    const int    max_edges         = (argc > 6) ? std::stoi(argv[6]) : -1;
    const int    rebuild_n_probe   = (argc > 7) ? std::stoi(argv[7]) : 0;

    fprintf(stderr, "Loading AQv2 from: %s\n", in_dir.c_str());
    auto t0 = std::chrono::steady_clock::now();

    NGTAQ::NGTAQIndex idx = NGTAQ::NGTAQIndex::load(in_dir + "/aqindex");
    idx.loadV2(in_dir);
    fprintf(stderr, "Loaded (%.1fs)\n", elapsed_s(t0));

    // Pre-set n_probe for rebuild phase.  0 means "use all K clusters" (handled
    // inside rebuildGraphSelf).  >0 overrides the auto-K logic.
    if (rebuild_n_probe > 0) {
        idx.setNProbe(rebuild_n_probe);
        fprintf(stderr, "rebuild_n_probe override: %d\n", rebuild_n_probe);
    }

    fprintf(stderr, "Running rebuildGraphSelf (k=%d gamma=%.2f threads=%d max_edges=%d)...\n",
            k_search, gamma, n_threads, max_edges);
    idx.rebuildGraphSelf(k_search, gamma, n_threads, -1.0f, max_edges);
    fprintf(stderr, "rebuildGraphSelf done (%.1fs)\n", elapsed_s(t0));

    namespace fs = std::filesystem;
    if (out_dir != in_dir) {
        if (!fs::exists(out_dir))
            fs::create_directories(out_dir);
        // Copy v2 support files first
        for (const auto& entry : fs::directory_iterator(in_dir)) {
            const std::string fname = entry.path().filename().string();
            if (fname == "aqindex") continue;
            fs::copy_file(entry.path(), out_dir + "/" + fname,
                          fs::copy_options::overwrite_existing);
        }
    }

    fprintf(stderr, "Saving to: %s\n", out_dir.c_str());
    idx.save(out_dir + "/aqindex");
    idx.saveV2(out_dir);
    fprintf(stderr, "Total: %.1fs\n", elapsed_s(t0));
    return 0;
}
