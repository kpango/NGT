// reorder_index: node-ID reorder of an AQv2 index for walk cache-locality (Task 2).
// Reports the graph ID-locality gap before/after, then saves the reordered copy.
// Usage: reorder_index <in_index_dir> <out_index_dir> [mode=bfs|rcm|gorder]
//   default mode = gorder (window co-occurrence; measured +14-15% QPS low-recall,
//   +4-5% high-recall vs bfs on SIFT-1M, recall-neutral). Override via arg3 or AQ_REORDER.
#include "NGT/NGTAQ/AQIndex.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <in_dir> <out_dir> [mode=bfs|rcm|gorder]\n", argv[0]); return 1; }
    const std::string in = argv[1], out = argv[2];
    const char* env = std::getenv("AQ_REORDER");
    const std::string mode_s = (argc > 3) ? argv[3] : (env ? std::string(env) : "gorder");
    auto mode = ::NGTAQ::NGTAQIndex::ReorderMode::BFS;
    if      (mode_s == "rcm")    mode = ::NGTAQ::NGTAQIndex::ReorderMode::RCM;
    else if (mode_s == "gorder") mode = ::NGTAQ::NGTAQIndex::ReorderMode::GORDER;
    ::NGTAQ::NGTAQIndex idx = ::NGTAQ::NGTAQIndex::load(in + "/aqindex");
    idx.loadV2(in);

    auto before = idx.graphLocalityStats();
    fprintf(stderr, "[reorder mode=%s] before: mean |nbr-id - node-id| gap = %.0f  (mean degree %.1f)\n",
            mode_s.c_str(), before.first, before.second);

    idx.reorderForLocality(mode);

    auto after = idx.graphLocalityStats();
    fprintf(stderr, "[reorder] after:  mean |nbr-id - node-id| gap = %.0f  (mean degree %.1f)\n",
            after.first, after.second);
    fprintf(stderr, "[reorder] gap reduction = %.1fx\n",
            after.first > 0 ? before.first / after.first : 0.0);

    std::filesystem::create_directories(out);
    idx.save(out + "/aqindex");
    idx.saveV2(out);
    fprintf(stderr, "[reorder] saved reordered index to %s\n", out.c_str());
    return 0;
}
