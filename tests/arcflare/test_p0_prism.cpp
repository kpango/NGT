// tests/arcflare/test_p0_prism.cpp
#include "NGT/ArcFlare/ArcFlareIndex.h"   // namespace ArcFlare; SearchResult in DABSSearcher.h
#include "hdf5_io.h"             // h5_read_float, H5FloatDataset (mirror ann_bench.cpp)
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <cassert>

using Stream = std::vector<std::pair<uint32_t, float>>;

// Build the ordered (id,distance) stream for all queries. Pure read of the
// shared const index; each call writes into its own local result vector.
static Stream run_stream(const ArcFlare::ArcFlareIndex& index,
                         const std::vector<std::vector<float>>& queries, int k) {
    Stream stream;
    for (auto& q : queries) {
        auto res = index.searchV2(q, k);
        for (auto& r : res) stream.emplace_back(r.id, r.distance);
    }
    return stream;
}

static Stream load_golden(const char* golden) {
    FILE* f = std::fopen(golden, "rb");
    assert(f && "golden file missing — run --capture first");
    uint64_t n = 0; (void)std::fread(&n, sizeof(n), 1, f);
    Stream gold(n);
    (void)std::fread(gold.data(), sizeof(gold[0]), n, f);
    std::fclose(f);
    return gold;
}

// Byte-for-byte stream equality: id equality + memcmp of the float bits.
static bool stream_eq(const Stream& a, const Stream& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].first != b[i].first ||
            std::memcmp(&a[i].second, &b[i].second, sizeof(float)) != 0)
            return false;
    return true;
}

// Usage: test_p0_prism <index_dir> <hdf5_path> <golden.bin> [--capture|--verify|--concurrent]
int main(int argc, char** argv) {
    assert(argc >= 4);
    const char* idx_dir = argv[1];
    const char* hdf5    = argv[2];
    const char* golden  = argv[3];
    bool capture    = (argc >= 5 && std::strcmp(argv[4], "--capture") == 0);
    bool concurrent = (argc >= 5 && std::strcmp(argv[4], "--concurrent") == 0);
    bool pure       = (argc >= 5 && std::strcmp(argv[4], "--pure") == 0);
    const int k = 10, NQ = 200;

    ArcFlare::ArcFlareIndex index = ArcFlare::ArcFlareIndex::load(std::string(idx_dir) + "/aqindex");
    index.loadV2(idx_dir);
    const int D_eff = index.dEff();

    H5FloatDataset test_ds = h5_read_float(hdf5, "test");
    const int nq = std::min(NQ, test_ds.n_rows), D = test_ds.n_cols;
    std::vector<std::vector<float>> queries(nq, std::vector<float>(D_eff, 0.f));
    for (int i = 0; i < nq; ++i) {
        const float* src = test_ds.data.data() + (size_t)i * D;
        std::copy(src, src + D, queries[i].begin());
    }

    if (pure) {
        // Prove the stateless overload with DEFAULT params (n_cluster_seeds == 0)
        // no longer collapses to zero seeding. With FIX 1 the worker falls back to
        // prop_.n_cluster_seeds, so every query must return k results.
        ArcFlare::SearchParameters p;  // everything default; crucially p.n_cluster_seeds == 0
        p.k = k;
        ArcFlare::SearchContext ctx;
        int ok = 0, collapsed = 0;
        for (auto& q : queries) {
            auto res = index.searchV2(q, p, ctx);
            if ((int)res.size() < k) ++collapsed; else ++ok;
        }
        if (collapsed == 0) {
            std::printf("PASS pure-overload default seeds %d/%d queries returned k results\n", ok, nq);
            return 0;
        }
        std::printf("FAIL pure-overload collapse: %d/%d queries returned <k results\n", collapsed, nq);
        return 1;
    }

    if (concurrent) {
        Stream gold = load_golden(golden);
        const int NT = 16;
        std::vector<Stream> results(NT);
        std::vector<std::thread> threads;
        threads.reserve(NT);
        for (int t = 0; t < NT; ++t)
            threads.emplace_back([&, t] { results[t] = run_stream(index, queries, k); });
        for (auto& th : threads) th.join();

        int matched = 0, diverged = 0;
        for (int t = 0; t < NT; ++t) {
            if (stream_eq(results[t], gold)) ++matched;
            else ++diverged;
        }
        if (diverged == 0) {
            std::printf("PASS concurrent %d threads race-free (%d threads matched golden)\n", NT, matched);
            return 0;
        }
        std::printf("FAIL concurrent (%d threads diverged)\n", diverged);
        return 1;
    }

    Stream stream = run_stream(index, queries, k);

    if (capture) {
        FILE* f = std::fopen(golden, "wb");
        uint64_t n = stream.size();
        std::fwrite(&n, sizeof(n), 1, f);
        std::fwrite(stream.data(), sizeof(stream[0]), n, f);
        std::fclose(f);
        std::printf("captured %llu tuples\n", (unsigned long long)n);
        return 0;
    }
    Stream gold = load_golden(golden);
    if (stream.size() != gold.size()) { std::printf("FAIL size %zu != %zu\n", stream.size(), gold.size()); return 1; }
    if (!stream_eq(stream, gold)) {
        for (size_t i = 0; i < gold.size(); ++i)
            if (stream[i].first != gold[i].first ||
                std::memcmp(&stream[i].second, &gold[i].second, sizeof(float)) != 0) {
                std::printf("FAIL at %zu\n", i); return 1;
            }
    }
    std::printf("PASS byte-identical (%llu tuples)\n", (unsigned long long)gold.size());
    return 0;
}
