// reorder_index: BFS node-ID reorder of an AQv2 index for walk cache-locality (Task 2).
// Reports the graph ID-locality gap before/after, then saves the reordered copy.
// Usage: reorder_index <in_index_dir> <out_index_dir>
#include "NGT/NGTAQ/AQIndex.h"
#include <cstdio>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <in_dir> <out_dir>\n", argv[0]); return 1; }
    const std::string in = argv[1], out = argv[2];
    ::NGTAQ::NGTAQIndex idx = ::NGTAQ::NGTAQIndex::load(in + "/aqindex");
    idx.loadV2(in);

    auto before = idx.graphLocalityStats();
    fprintf(stderr, "[reorder] before: mean |nbr-id - node-id| gap = %.0f  (mean degree %.1f)\n",
            before.first, before.second);

    idx.reorderForLocality();

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
