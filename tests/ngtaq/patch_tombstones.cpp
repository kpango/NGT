// tests/ngtaq/patch_tombstones.cpp
// Patch an existing NGTAQv2 index to tombstone zero-norm (hole) nodes.
// Use to repair indices built with code that did not properly tombstone holes.
//
// Usage: patch_tombstones <idx_dir>
#include "NGT/NGTAQ/AQIndex.h"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <aq_index_dir>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];

    fprintf(stderr, "Loading index from %s...\n", dir.c_str());
    NGTAQ::NGTAQIndex idx = NGTAQ::NGTAQIndex::load(dir + "/aqindex");
    idx.loadV2(dir);

    const int n_fixed = idx.fixHoleTombstones();
    if (n_fixed == 0) {
        fprintf(stderr, "No hole nodes found — index is already clean.\n");
        return 0;
    }

    fprintf(stderr, "Saving patched index...\n");
    idx.save(dir + "/aqindex");
    idx.saveV2(dir);
    fprintf(stderr, "Done. Tombstoned %d hole nodes.\n", n_fixed);
    return 0;
}
