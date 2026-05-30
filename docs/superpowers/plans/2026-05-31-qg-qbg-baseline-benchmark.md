# QG/QSG/QBG Baseline Benchmark vs NGTAQv2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build QG, QSG, QBG benchmark tools to measure their recall-QPS curves on SIFT-128 and NYTimes-256, compare against NGTAQv2 at every recall level (0.5–0.99), and improve NGTAQv2 where it loses.

**Architecture:** Six new C++ tools in `tests/ngtaq/` sharing a common header. QG/QSG use `NGTQG::Index` (graph with quantized edges, built from ANNG). QBG uses `QBG::Index` (hierarchical blob clusters + PQ graph). Benchmarks run epsilon sweeps; results are compared in a tabular script.

**Tech Stack:** C++17, NGT (NGTQG + QBG APIs), HDF5, OpenMP-style std::thread parallelism.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `tests/ngtaq/bench_common.hpp` | Shared: L2-norm, `bc_now_us()`, `BenchResult`; includes `hdf5_io.h` |
| Create | `tests/ngtaq/qg_build.cpp` | Read HDF5 → ANNG → QG(D_sub=4) + QSG(D_sub=1,2) index dirs |
| Create | `tests/ngtaq/qg_bench.cpp` | Load `NGTQG::Index`, epsilon sweep, output recall@k + QPS |
| Create | `tests/ngtaq/qbg_build.cpp` | Read HDF5 → TSV → QBG::Index create + build |
| Create | `tests/ngtaq/qbg_bench.cpp` | Load `QBG::Index`, blob-epsilon sweep, output recall@k + QPS |
| Modify | `tests/ngtaq/CMakeLists.txt` | Add 4 new HDF5-dependent targets |

`ann_bench.cpp` and `hdf5_io.h` are NOT modified. `bench_common.hpp` is a new header that `#include`s `hdf5_io.h` and adds helpers.

---

## Epsilon Grid (both QG/QSG and QBG sweeps)

```
static const double EPSILON_GRID[] = {
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0
};
```

## D_sub Mapping

- **QG**: D_sub = 4 (PQ, 4D sub-vectors)
- **QSG-1**: D_sub = 1 (scalar quantization, coarsest, fastest)
- **QSG-2**: D_sub = 2 (2D sub-vectors, middle)

D_sub=1,2,4 evenly divide all four dataset dimensions (128, 256, 100, 960).

---

## Task 1: Create `bench_common.hpp`

**Files:**
- Create: `tests/ngtaq/bench_common.hpp`

- [ ] **Step 1: Write `bench_common.hpp`**

```cpp
// tests/ngtaq/bench_common.hpp
// Shared utilities for all ANN-Benchmarks benchmark tools.
// Includes hdf5_io.h; adds timing, L2-norm, BenchResult.
#pragma once
#include "hdf5_io.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

static inline double bc_now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(
        steady_clock::now().time_since_epoch()).count();
}

struct BenchResult {
    double recall = 0.0;
    double qps    = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
};

/// L2-normalize vector v[0..D) in place. No-op if norm < 1e-12.
static inline void l2_normalize(float* v, int D) {
    float norm2 = 0.f;
    for (int d = 0; d < D; ++d) norm2 += v[d] * v[d];
    if (norm2 > 1e-12f) {
        float inv = 1.f / std::sqrtf(norm2);
        for (int d = 0; d < D; ++d) v[d] *= inv;
    }
}

static const double EPSILON_GRID[] = {
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0
};
static const int EPSILON_GRID_SIZE = 13;
```

- [ ] **Step 2: Verify it compiles as part of ann_bench (unchanged)**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT
# ann_bench.cpp still uses hdf5_io.h directly — no change needed
# Just confirm bench_common.hpp parses correctly
g++ -std=c++17 -c -I build_ngtaq/lib -I tests/ngtaq \
    -I /usr/include/hdf5/serial \
    tests/ngtaq/bench_common.hpp -x c++-header -o /dev/null 2>&1 | head -5
```

Expected: no error output.

- [ ] **Step 3: Commit**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT
git add tests/ngtaq/bench_common.hpp
git commit -m "bench: add bench_common.hpp shared header for QG/QSG/QBG tools

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Create `qg_build.cpp`

**Files:**
- Create: `tests/ngtaq/qg_build.cpp`

**What it does:** Reads HDF5 train set → builds ANNG → copies ANNG 3× → quantizes each copy with D_sub=4,1,2 → produces 4 index directories.

**Usage:**
```
./qg_build <hdf5_path> <out_base> [metric=l2|angular] [edge_size=100] [threads=8]
# Produces:
#   <out_base>_anng       (raw ANNG, reusable)
#   <out_base>_qg4        (NGTQG, D_sub=4)
#   <out_base>_qsg1       (NGTQG, D_sub=1)
#   <out_base>_qsg2       (NGTQG, D_sub=2)
```

- [ ] **Step 1: Write `qg_build.cpp`**

```cpp
// tests/ngtaq/qg_build.cpp
// Build ANNG + quantize to QG (D_sub=4) and QSG (D_sub=1, D_sub=2).
// Usage: qg_build <hdf5_path> <out_base> [metric=l2|angular] [edge_size=100] [threads=8]
#include "bench_common.hpp"
#include "NGT/Index.h"
#include "NGT/NGTQ/QuantizedGraph.h"
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
namespace fs = std::filesystem;

static double elapsed_s(double t0) { return (bc_now_us() - t0) / 1e6; }

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <hdf5_path> <out_base> [metric=l2|angular] [edge_size=100] [threads=8]\n",
            argv[0]);
        return 1;
    }
    const char* hdf5_path  = argv[1];
    const char* out_base   = argv[2];
    const char* metric_str = (argc > 3) ? argv[3] : "l2";
    int edge_size          = (argc > 4) ? std::stoi(argv[4]) : 100;
    int n_threads          = (argc > 5) ? std::stoi(argv[5]) : 8;
    const bool is_angular  = (strcmp(metric_str, "angular") == 0 ||
                               strcmp(metric_str, "cosine")  == 0);

    const std::string anng_path = std::string(out_base) + "_anng";

    // ── Step 1: Read train vectors ─────────────────────────────────────────
    fprintf(stderr, "[Load] HDF5 train from: %s\n", hdf5_path);
    H5FloatDataset train = h5_read_float(hdf5_path, "train");
    const int N = train.n_rows, D = train.n_cols;
    fprintf(stderr, "  N=%d  D=%d  metric=%s  edge_size=%d\n",
            N, D, metric_str, edge_size);

    // ── Step 2: Normalize if angular ──────────────────────────────────────
    if (is_angular) {
        for (int i = 0; i < N; ++i)
            l2_normalize(train.data.data() + (size_t)i * D, D);
        fprintf(stderr, "[Prep] L2-normalized %d vectors\n", N);
    }

    // ── Step 3: Build ANNG (skip if already exists) ───────────────────────
    if (fs::exists(anng_path)) {
        fprintf(stderr, "[Skip] ANNG already exists: %s\n", anng_path.c_str());
    } else {
        const double t0 = bc_now_us();
        NGT::Property prop;
        prop.dimension            = D;
        prop.objectType           = NGT::ObjectSpace::ObjectType::Float;
        prop.distanceType         = NGT::Index::Property::DistanceType::DistanceTypeL2;
        prop.edgeSizeForCreation  = edge_size;

        fprintf(stderr, "[NGT] Creating ANNG (D=%d, edge_size=%d) ...\n", D, edge_size);
        NGT::Index::create(anng_path, prop);
        {
            NGT::Index ngt(anng_path);
            for (int i = 0; i < N; ++i) {
                std::vector<float> v(train.data.data() + (size_t)i * D,
                                     train.data.data() + (size_t)(i+1) * D);
                ngt.insert(v);
                if ((i+1) % 100000 == 0)
                    fprintf(stderr, "  inserted %d/%d (%.1fs)\n", i+1, N, elapsed_s(t0));
            }
            fprintf(stderr, "[NGT] Building graph (threads=%d)...\n", n_threads);
            ngt.createIndex(n_threads);
            fprintf(stderr, "[NGT] ANNG done (%.1fs)\n", elapsed_s(t0));
            ngt.save();
        }
        fprintf(stderr, "[Done] Saved ANNG to: %s\n", anng_path.c_str());
    }

    // ── Step 4: Quantize to QG and QSG variants ───────────────────────────
    // Each quantization modifies a COPY of the ANNG (adds a qg/ subdir).
    // D_sub=4 → QG, D_sub=1 → QSG-1 (scalar), D_sub=2 → QSG-2
    struct Variant { int d_sub; const char* suffix; };
    static const Variant VARIANTS[] = {
        {4, "_qg4"}, {1, "_qsg1"}, {2, "_qsg2"}
    };
    for (auto& v : VARIANTS) {
        std::string dst = std::string(out_base) + v.suffix;
        if (fs::exists(dst)) {
            fprintf(stderr, "[Skip] %s already exists\n", dst.c_str());
            continue;
        }
        fprintf(stderr, "[Copy] %s -> %s\n", anng_path.c_str(), dst.c_str());
        fs::copy(anng_path, dst, fs::copy_options::recursive);

        fprintf(stderr, "[QG] Quantizing D_sub=%d -> %s\n", v.d_sub, dst.c_str());
        const double tq = bc_now_us();
        NGTQG::Index::quantize(dst, (size_t)v.d_sub, (size_t)edge_size, /*verbose=*/true);
        fprintf(stderr, "[QG] Done (%.1fs) -> %s\n", elapsed_s(tq), dst.c_str());
    }

    fprintf(stderr, "[All done]\n");
    return 0;
}
```

- [ ] **Step 2: Commit (before build test)**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT
git add tests/ngtaq/qg_build.cpp
git commit -m "bench: add qg_build ANNG+QG/QSG builder

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Create `qg_bench.cpp`

**Files:**
- Create: `tests/ngtaq/qg_bench.cpp`

**Usage:**
```
./qg_bench <qg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]
# Sweeps all 13 epsilon values; prints one result block per epsilon.
```

- [ ] **Step 1: Write `qg_bench.cpp`**

```cpp
// tests/ngtaq/qg_bench.cpp
// ANN-Benchmarks benchmark for NGTQG (QG / QSG) with epsilon sweep.
// Usage: qg_bench <qg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]
#include "bench_common.hpp"
#include "NGT/NGTQ/QuantizedGraph.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <qg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]\n",
            argv[0]);
        return 1;
    }
    const char* idx_dir    = argv[1];
    const char* hdf5_path  = argv[2];
    int   k                = (argc > 3) ? std::stoi(argv[3]) : 10;
    int   n_threads        = (argc > 4) ? std::stoi(argv[4]) : 4;
    const char* metric_str = (argc > 5) ? argv[5] : "l2";
    const bool is_angular  = (strcmp(metric_str, "angular") == 0 ||
                               strcmp(metric_str, "cosine")  == 0);

    fprintf(stderr, "[Load] NGTQG from: %s\n", idx_dir);
    NGTQG::Index index(idx_dir);

    fprintf(stderr, "[Load] HDF5 from: %s\n", hdf5_path);
    H5FloatDataset test_ds = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt_ds   = h5_read_int(hdf5_path, "neighbors");
    const int nq = test_ds.n_rows, D = test_ds.n_cols;
    fprintf(stderr, "  nq=%d  D=%d  k=%d  threads=%d  metric=%s\n",
            nq, D, k, n_threads, metric_str);

    // Prepare queries (normalize if angular)
    std::vector<std::vector<float>> queries(nq, std::vector<float>(D));
    for (int qi = 0; qi < nq; ++qi) {
        const float* src = test_ds.data.data() + (size_t)qi * D;
        std::copy(src, src + D, queries[qi].begin());
        if (is_angular)
            l2_normalize(queries[qi].data(), D);
    }

    // Warmup with middle epsilon
    const int WARMUP = std::min(200, nq);
    for (int wi = 0; wi < WARMUP; ++wi) {
        NGTQG::SearchQuery sq(queries[wi % nq]);
        NGT::ObjectDistances objs;
        sq.setResults(&objs); sq.setSize(k); sq.setEpsilon(0.1f);
        index.search(sq);
    }

    // Sweep
    for (int gi = 0; gi < EPSILON_GRID_SIZE; ++gi) {
        const float eps = (float)EPSILON_GRID[gi];

        std::vector<std::vector<int>> results(nq);
        std::vector<double> latencies(nq, 0.0);
        std::atomic<int> next_qi{0};
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        const double t_start = bc_now_us();
        for (int t = 0; t < n_threads; ++t) {
            threads.emplace_back([&, eps]() {
                for (;;) {
                    int qi = next_qi.fetch_add(1, std::memory_order_relaxed);
                    if (qi >= nq) break;
                    NGTQG::SearchQuery sq(queries[qi]);
                    NGT::ObjectDistances objs;
                    sq.setResults(&objs); sq.setSize(k); sq.setEpsilon(eps);
                    const double t0 = bc_now_us();
                    index.search(sq);
                    const double t1 = bc_now_us();
                    latencies[qi] = t1 - t0;
                    results[qi].resize(objs.size());
                    for (int i = 0; i < (int)objs.size(); ++i)
                        results[qi][i] = (int)objs[i].id - 1; // NGTQG IDs are 1-based
                }
            });
        }
        for (auto& th : threads) th.join();
        const double elapsed_s = (bc_now_us() - t_start) / 1e6;

        const double qps    = nq / elapsed_s;
        const double recall = compute_recall_k(gt_ds, results, k);
        std::vector<double> lats = latencies;
        std::sort(lats.begin(), lats.end());
        const double p50 = lats[(size_t)(nq * 0.50)];
        const double p99 = lats[(size_t)(nq * 0.99)];

        printf("=== NGTQG Benchmark Result ===\n");
        printf("Index   : %s\n", idx_dir);
        printf("Dataset : %s  nq=%d  D=%d\n", hdf5_path, nq, D);
        printf("Config  : k=%d  epsilon=%.3f  threads=%d\n", k, eps, n_threads);
        printf("recall@%d  = %.4f\n", k, recall);
        printf("agg_QPS   = %.0f\n", qps);
        printf("P50(us)   = %.1f\n", p50);
        printf("P99(us)   = %.1f\n\n", p99);
        fflush(stdout);
    }
    return 0;
}
```

- [ ] **Step 2: Commit**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT
git add tests/ngtaq/qg_bench.cpp
git commit -m "bench: add qg_bench NGTQG epsilon-sweep benchmark

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Create `qbg_build.cpp` and `qbg_bench.cpp`

**Files:**
- Create: `tests/ngtaq/qbg_build.cpp`
- Create: `tests/ngtaq/qbg_bench.cpp`

**qbg_build usage:**
```
./qbg_build <hdf5_path> <out_dir> [metric=l2|angular] [D_sub=4] [max_edges=100]
```

**qbg_bench usage:**
```
./qbg_bench <qbg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]
# Sweeps blob-epsilon; each result block shows recall@k + QPS.
```

- [ ] **Step 1: Write `qbg_build.cpp`**

```cpp
// tests/ngtaq/qbg_build.cpp
// Build QBG::Index from HDF5 training data.
// Creates a TSV tmpfile, then calls QBG::Index::create + build + buildQBG.
// Usage: qbg_build <hdf5_path> <out_dir> [metric=l2|angular] [D_sub=4] [max_edges=100]
#include "bench_common.hpp"
#include "NGT/NGTQ/QuantizedBlobGraph.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;

static double elapsed_s2(double t0) { return (bc_now_us() - t0) / 1e6; }

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <hdf5_path> <out_dir> [metric=l2|angular] [D_sub=4] [max_edges=100]\n",
            argv[0]);
        return 1;
    }
    const char* hdf5_path  = argv[1];
    const char* out_dir    = argv[2];
    const char* metric_str = (argc > 3) ? argv[3] : "l2";
    int D_sub              = (argc > 4) ? std::stoi(argv[4]) : 4;
    int max_edges          = (argc > 5) ? std::stoi(argv[5]) : 100;
    const bool is_angular  = (strcmp(metric_str, "angular") == 0 ||
                               strcmp(metric_str, "cosine")  == 0);

    if (fs::exists(out_dir)) {
        fprintf(stderr, "[Skip] QBG index already exists: %s\n", out_dir);
        return 0;
    }

    // ── Read train ─────────────────────────────────────────────────────────
    fprintf(stderr, "[Load] HDF5 train from: %s\n", hdf5_path);
    H5FloatDataset train = h5_read_float(hdf5_path, "train");
    const int N = train.n_rows, D = train.n_cols;
    fprintf(stderr, "  N=%d  D=%d  metric=%s  D_sub=%d\n", N, D, metric_str, D_sub);

    if (is_angular)
        for (int i = 0; i < N; ++i)
            l2_normalize(train.data.data() + (size_t)i * D, D);

    // ── Write TSV tmpfile ──────────────────────────────────────────────────
    const std::string tsv_path = std::string(out_dir) + "_train.tsv";
    if (!fs::exists(tsv_path)) {
        fprintf(stderr, "[TSV] Writing %d vectors to %s ...\n", N, tsv_path.c_str());
        const double tt = bc_now_us();
        std::ofstream ofs(tsv_path);
        for (int i = 0; i < N; ++i) {
            const float* v = train.data.data() + (size_t)i * D;
            for (int d = 0; d < D; ++d) {
                if (d > 0) ofs << '\t';
                ofs << v[d];
            }
            ofs << '\n';
        }
        fprintf(stderr, "[TSV] Done (%.1fs)\n", elapsed_s2(tt));
    }

    // ── Create QBG index ───────────────────────────────────────────────────
    fprintf(stderr, "[QBG] Creating index at %s\n", out_dir);
    const double t0 = bc_now_us();
    {
        QBG::BuildParameters params;
        params.creation.dimension    = D;
        params.creation.numOfSubvectors = D_sub;
        params.creation.globalProperty.distanceType =
            is_angular
                ? NGT::Index::Property::DistanceType::DistanceTypeCosine
                : NGT::Index::Property::DistanceType::DistanceTypeL2;
        QBG::Index::create(out_dir, params, nullptr, tsv_path);
    }
    fprintf(stderr, "[QBG] create done (%.1fs)\n", elapsed_s2(t0));

    // ── Build NGTQ (inverted index) ────────────────────────────────────────
    fprintf(stderr, "[QBG] buildNGTQ ...\n");
    QBG::Index::buildNGTQ(out_dir, /*verbose=*/true);
    fprintf(stderr, "[QBG] buildNGTQ done (%.1fs)\n", elapsed_s2(t0));

    // ── Build QBG graph ───────────────────────────────────────────────────
    fprintf(stderr, "[QBG] buildQBG ...\n");
    QBG::Index::buildQBG(out_dir, /*verbose=*/true);
    fprintf(stderr, "[QBG] buildQBG done (%.1fs)\n", elapsed_s2(t0));

    fprintf(stderr, "[Done] QBG index saved to: %s (total %.1fs)\n", out_dir, elapsed_s2(t0));
    return 0;
}
```

- [ ] **Step 2: Write `qbg_bench.cpp`**

```cpp
// tests/ngtaq/qbg_bench.cpp
// ANN-Benchmarks benchmark for QBG::Index with blob-epsilon sweep.
// Usage: qbg_bench <qbg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]
#include "bench_common.hpp"
#include "NGT/NGTQ/QuantizedBlobGraph.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <qbg_idx_dir> <hdf5_path> [k=10] [threads=4] [metric=l2|angular]\n",
            argv[0]);
        return 1;
    }
    const char* idx_dir    = argv[1];
    const char* hdf5_path  = argv[2];
    int   k                = (argc > 3) ? std::stoi(argv[3]) : 10;
    int   n_threads        = (argc > 4) ? std::stoi(argv[4]) : 4;
    const char* metric_str = (argc > 5) ? argv[5] : "l2";
    const bool is_angular  = (strcmp(metric_str, "angular") == 0 ||
                               strcmp(metric_str, "cosine")  == 0);

    fprintf(stderr, "[Load] QBG from: %s\n", idx_dir);
    QBG::Index qbg(idx_dir);

    fprintf(stderr, "[Load] HDF5 from: %s\n", hdf5_path);
    H5FloatDataset test_ds = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt_ds   = h5_read_int(hdf5_path, "neighbors");
    const int nq = test_ds.n_rows, D = test_ds.n_cols;
    fprintf(stderr, "  nq=%d  D=%d  k=%d  threads=%d\n", nq, D, k, n_threads);

    // Prepare queries
    std::vector<std::vector<float>> queries(nq, std::vector<float>(D));
    for (int qi = 0; qi < nq; ++qi) {
        const float* src = test_ds.data.data() + (size_t)qi * D;
        std::copy(src, src + D, queries[qi].begin());
        if (is_angular)
            l2_normalize(queries[qi].data(), D);
    }

    // Warmup
    const int WARMUP = std::min(200, nq);
    for (int wi = 0; wi < WARMUP; ++wi) {
        QBG::SearchContainer sc;
        sc.setObjectVector(queries[wi % nq]);
        sc.setSize(k);
        sc.setEpsilon(0.1f);
        sc.setBlobEpsilon(0.05f);
        NGT::ObjectDistances objs;
        sc.setResults(&objs);
        qbg.searchInTwoSteps(sc);
    }

    // Sweep blob-epsilon (main recall knob for QBG)
    for (int gi = 0; gi < EPSILON_GRID_SIZE; ++gi) {
        const float blob_eps = (float)EPSILON_GRID[gi];

        std::vector<std::vector<int>> results(nq);
        std::vector<double> latencies(nq, 0.0);
        std::atomic<int> next_qi{0};
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        const double t_start = bc_now_us();
        for (int t = 0; t < n_threads; ++t) {
            threads.emplace_back([&, blob_eps]() {
                for (;;) {
                    int qi = next_qi.fetch_add(1, std::memory_order_relaxed);
                    if (qi >= nq) break;
                    QBG::SearchContainer sc;
                    sc.setObjectVector(queries[qi]);
                    sc.setSize(k);
                    sc.setEpsilon(0.1f);       // graph epsilon fixed
                    sc.setBlobEpsilon(blob_eps); // blob expansion is the main knob
                    NGT::ObjectDistances objs;
                    sc.setResults(&objs);
                    const double t0 = bc_now_us();
                    qbg.searchInTwoSteps(sc);
                    const double t1 = bc_now_us();
                    latencies[qi] = t1 - t0;
                    results[qi].resize(objs.size());
                    for (int i = 0; i < (int)objs.size(); ++i)
                        results[qi][i] = (int)objs[i].id - 1; // QBG IDs are 1-based
                }
            });
        }
        for (auto& th : threads) th.join();
        const double elapsed_s = (bc_now_us() - t_start) / 1e6;

        const double qps    = nq / elapsed_s;
        const double recall = compute_recall_k(gt_ds, results, k);
        std::vector<double> lats = latencies;
        std::sort(lats.begin(), lats.end());
        const double p50 = lats[(size_t)(nq * 0.50)];
        const double p99 = lats[(size_t)(nq * 0.99)];

        printf("=== QBG Benchmark Result ===\n");
        printf("Index   : %s\n", idx_dir);
        printf("Dataset : %s  nq=%d  D=%d\n", hdf5_path, nq, D);
        printf("Config  : k=%d  blob_eps=%.3f  threads=%d\n", k, blob_eps, n_threads);
        printf("recall@%d  = %.4f\n", k, recall);
        printf("agg_QPS   = %.0f\n", qps);
        printf("P50(us)   = %.1f\n", p50);
        printf("P99(us)   = %.1f\n\n", p99);
        fflush(stdout);
    }
    return 0;
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT
git add tests/ngtaq/qbg_build.cpp tests/ngtaq/qbg_bench.cpp
git commit -m "bench: add qbg_build and qbg_bench QBG benchmark tools

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Update CMakeLists.txt

**Files:**
- Modify: `tests/ngtaq/CMakeLists.txt`

- [ ] **Step 1: Read current CMakeLists.txt state**

```bash
cat /home/kpango/go/src/github.com/kpango/NGT/tests/ngtaq/CMakeLists.txt
```

- [ ] **Step 2: Add new targets to the HDF5-dependent section**

In `tests/ngtaq/CMakeLists.txt`, find the `foreach(HDF5_SRC ann_bench ...)` block and extend it:

```cmake
# Change this line:
    foreach(HDF5_SRC ann_bench build_ngtaqv2_ann build_ngt_ann diag_ids)
# To:
    foreach(HDF5_SRC ann_bench build_ngtaqv2_ann build_ngt_ann diag_ids
                     qg_build qg_bench qbg_build qbg_bench)
```

The new targets all use the same `target_link_libraries(${HDF5_SRC} ngt ${HDF5_C_LIBRARIES})` and `cxx_std_17` that the `foreach` loop already applies.

- [ ] **Step 3: Commit**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT
git add tests/ngtaq/CMakeLists.txt
git commit -m "bench: add qg/qbg bench targets to CMakeLists

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Build and Smoke Test

**Files:** None (build only)

- [ ] **Step 1: Build all new targets**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq
export LD_LIBRARY_PATH=/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
cmake --build . --target qg_build qg_bench qbg_build qbg_bench -j$(nproc) 2>&1 | tail -20
```

Expected:
```
[100%] Built target qg_build
[100%] Built target qg_bench
[100%] Built target qbg_build
[100%] Built target qbg_bench
```

- [ ] **Step 2: Smoke test qg_build with no args (usage message)**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/tests/ngtaq
./qg_build 2>&1 | head -3
```

Expected:
```
Usage: ./qg_build <hdf5_path> <out_base> [metric=l2|angular] [edge_size=100] [threads=8]
```

- [ ] **Step 3: Smoke test qg_bench and qbg_bench with no args**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/tests/ngtaq
./qg_bench 2>&1 | head -2
./qbg_bench 2>&1 | head -2
./qbg_build 2>&1 | head -2
```

Each should print its usage line.

- [ ] **Step 4: Commit build verification**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT
git commit --allow-empty -m "bench: confirmed qg_build/qg_bench/qbg_build/qbg_bench compile clean

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Build QG/QSG Indices — SIFT-128 and NYTimes-256 in Parallel

**Files:**
- Create: `/tmp/build_qg_sift128.sh`
- Create: `/tmp/build_qg_nytimes.sh`

- [ ] **Step 1: Write SIFT-128 build script**

```bash
cat > /tmp/build_qg_sift128.sh << 'EOF'
#!/bin/bash
set -e
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/tests/ngtaq
export LD_LIBRARY_PATH=/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
HDF5=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/sift-128-euclidean.hdf5
echo "=== SIFT-128 QG build started: $(date) ===" | tee /tmp/build_qg_sift128.log
./qg_build "$HDF5" /tmp/qg_sift128 l2 100 16 2>&1 | tee -a /tmp/build_qg_sift128.log
echo "=== Done: $(date) ===" | tee -a /tmp/build_qg_sift128.log
EOF
chmod +x /tmp/build_qg_sift128.sh
```

- [ ] **Step 2: Write NYTimes-256 build script**

```bash
cat > /tmp/build_qg_nytimes.sh << 'EOF'
#!/bin/bash
set -e
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/tests/ngtaq
export LD_LIBRARY_PATH=/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
HDF5=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/nytimes-256-angular.hdf5
echo "=== NYTimes-256 QG build started: $(date) ===" | tee /tmp/build_qg_nytimes.log
./qg_build "$HDF5" /tmp/qg_nytimes angular 100 16 2>&1 | tee -a /tmp/build_qg_nytimes.log
echo "=== Done: $(date) ===" | tee -a /tmp/build_qg_nytimes.log
EOF
chmod +x /tmp/build_qg_nytimes.sh
```

- [ ] **Step 3: Launch both builds in background (parallel)**

```bash
nohup /tmp/build_qg_sift128.sh > /tmp/build_qg_sift128.log 2>&1 &
echo "SIFT-128 QG build PID: $!"
nohup /tmp/build_qg_nytimes.sh > /tmp/build_qg_nytimes.log 2>&1 &
echo "NYTimes QG build PID: $!"
```

- [ ] **Step 4: Monitor progress**

```bash
# Check periodically:
tail -5 /tmp/build_qg_sift128.log
tail -5 /tmp/build_qg_nytimes.log
# Verify outputs when done:
ls -la /tmp/qg_sift128_anng/ /tmp/qg_sift128_qg4/ /tmp/qg_sift128_qsg1/ /tmp/qg_sift128_qsg2/ 2>/dev/null
ls -la /tmp/qg_nytimes_anng/ /tmp/qg_nytimes_qg4/ /tmp/qg_nytimes_qsg1/ /tmp/qg_nytimes_qsg2/ 2>/dev/null
```

Expected: each directory contains `grp`, `obj`, `prf`, `tre` subdirs (standard NGT structure) plus `qg/` after quantization.

---

## Task 8: Run QG/QSG Benchmarks and Compare with NGTAQv2

**Files:**
- Create: `/tmp/bench_qg_sift128.sh`
- Create: `/tmp/bench_qg_nytimes.sh`
- Create: `/tmp/compare_bench.py`

- [ ] **Step 1: Write SIFT-128 QG benchmark script**

```bash
cat > /tmp/bench_qg_sift128.sh << 'EOF'
#!/bin/bash
set -e
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/tests/ngtaq
export LD_LIBRARY_PATH=/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
HDF5=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/sift-128-euclidean.hdf5
LOG=/tmp/bench_qg_sift128.log

echo "=== QG/QSG SIFT-128 benchmark: $(date) ===" | tee "$LOG"
for IDX in /tmp/qg_sift128_qg4 /tmp/qg_sift128_qsg1 /tmp/qg_sift128_qsg2; do
    echo "--- $IDX ---" | tee -a "$LOG"
    ./qg_bench "$IDX" "$HDF5" 10 4 l2 2>/dev/null | tee -a "$LOG"
done
echo "=== Done: $(date) ===" | tee -a "$LOG"
EOF
chmod +x /tmp/bench_qg_sift128.sh
```

- [ ] **Step 2: Write NYTimes-256 QG benchmark script**

```bash
cat > /tmp/bench_qg_nytimes.sh << 'EOF'
#!/bin/bash
set -e
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/tests/ngtaq
export LD_LIBRARY_PATH=/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
HDF5=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/nytimes-256-angular.hdf5
LOG=/tmp/bench_qg_nytimes.log

echo "=== QG/QSG NYTimes-256 benchmark: $(date) ===" | tee "$LOG"
for IDX in /tmp/qg_nytimes_qg4 /tmp/qg_nytimes_qsg1 /tmp/qg_nytimes_qsg2; do
    echo "--- $IDX ---" | tee -a "$LOG"
    ./qg_bench "$IDX" "$HDF5" 10 4 angular 2>/dev/null | tee -a "$LOG"
done
echo "=== Done: $(date) ===" | tee -a "$LOG"
EOF
chmod +x /tmp/bench_qg_nytimes.sh
```

- [ ] **Step 3: Write comparison summary script**

```python
# /tmp/compare_bench.py
# Parse ann_bench and qg_bench output logs and print a comparison table.
# Usage: python3 /tmp/compare_bench.py <ngtaq_log> <qg_log>
import sys, re

def parse_blocks(text):
    """Return list of (recall, qps) pairs from benchmark output."""
    results = []
    blocks = re.split(r'===.*(?:Benchmark|Result).*===', text)
    for block in blocks:
        recall = re.search(r'recall@\d+\s*=\s*([\d.]+)', block)
        qps    = re.search(r'agg_QPS\s*=\s*([\d.]+)', block)
        if recall and qps:
            results.append((float(recall.group(1)), float(qps.group(1))))
    return sorted(results)

if len(sys.argv) < 3:
    print("Usage: python3 compare_bench.py <ngtaq_log> <qg_log>")
    sys.exit(1)

with open(sys.argv[1]) as f: ngtaq = parse_blocks(f.read())
with open(sys.argv[2]) as f: qg    = parse_blocks(f.read())

TARGETS = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]

def find_qps_at_recall(data, target, tol=0.03):
    """Return QPS of the point with recall closest to target (within tol)."""
    best = min(data, key=lambda x: abs(x[0] - target), default=None)
    if best and abs(best[0] - target) < tol:
        return best[1]
    return None

print(f"{'Recall':>8} | {'NGTAQv2 QPS':>12} | {'QG/QSG QPS':>12} | {'Ratio':>8}")
print("-" * 50)
for t in TARGETS:
    aq = find_qps_at_recall(ngtaq, t)
    qg_qps = find_qps_at_recall(qg, t)
    ratio = (aq / qg_qps) if (aq and qg_qps and qg_qps > 0) else None
    aq_s    = f"{aq:.0f}" if aq else "N/A"
    qg_s    = f"{qg_qps:.0f}" if qg_qps else "N/A"
    ratio_s = f"{ratio:.2f}x" if ratio else "N/A"
    print(f"{t:>8.2f} | {aq_s:>12} | {qg_s:>12} | {ratio_s:>8}")
```

```bash
# Save the script
cat > /tmp/compare_bench.py << 'PYEOF'
# (paste above Python content)
PYEOF
```

- [ ] **Step 4: Run benchmarks after builds complete**

```bash
# Verify build complete first:
ls /tmp/qg_sift128_qg4/qg/ && ls /tmp/qg_nytimes_qg4/qg/

# Run QG benchmarks:
/tmp/bench_qg_sift128.sh
/tmp/bench_qg_nytimes.sh

# Compare with NGTAQv2 SIFT-128 baseline (from ann_bench sweeps):
# NGTAQv2 SIFT-128 gamma_term sweep output is in the grill-me session logs.
# Run a quick sweep to get matching data points:
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/tests/ngtaq
export LD_LIBRARY_PATH=/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
HDF5=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/sift-128-euclidean.hdf5
for gt in 0.03 0.05 0.08 0.10 0.12 0.15 0.20 0.25 0.30 0.40; do
    ./ann_bench /tmp/ngtaq_sift128 "$HDF5" 10 0.20 $gt 1 4 0 2>/dev/null
done | tee /tmp/ngtaq_sift128_sweep.log

# Run comparison:
python3 /tmp/compare_bench.py /tmp/ngtaq_sift128_sweep.log /tmp/bench_qg_sift128.log
```

Expected output format:
```
  Recall |  NGTAQv2 QPS |   QG/QSG QPS |    Ratio
--------------------------------------------------
    0.50 |        37000 |         XXXX |     X.XXx
    0.70 |        36000 |         XXXX |     X.XXx
    ...
    0.99 |         3698 |         XXXX |     X.XXx
```

**Interpretation:** Ratio > 1.0 means NGTAQv2 wins. Ratio < 1.0 triggers fallback.

---

## Task 9: Build and Benchmark QBG (SIFT-128 first)

**Files:**
- Create: `/tmp/build_bench_qbg_sift128.sh`

- [ ] **Step 1: Write QBG build+bench script for SIFT-128**

```bash
cat > /tmp/build_bench_qbg_sift128.sh << 'EOF'
#!/bin/bash
set -e
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/tests/ngtaq
export LD_LIBRARY_PATH=/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
HDF5=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/sift-128-euclidean.hdf5
LOG=/tmp/bench_qbg_sift128.log

echo "=== QBG SIFT-128: $(date) ===" | tee "$LOG"
echo "--- Build ---" | tee -a "$LOG"
./qbg_build "$HDF5" /tmp/qbg_sift128 l2 4 100 2>&1 | tee -a "$LOG"
echo "--- Bench ---" | tee -a "$LOG"
./qbg_bench /tmp/qbg_sift128 "$HDF5" 10 4 l2 2>/dev/null | tee -a "$LOG"
echo "=== Done: $(date) ===" | tee -a "$LOG"
EOF
chmod +x /tmp/build_bench_qbg_sift128.sh
nohup /tmp/build_bench_qbg_sift128.sh >> /tmp/bench_qbg_sift128.log 2>&1 &
echo "QBG SIFT-128 PID: $!"
```

- [ ] **Step 2: Run comparison once QBG bench completes**

```bash
python3 /tmp/compare_bench.py /tmp/ngtaq_sift128_sweep.log /tmp/bench_qbg_sift128.log
```

---

## Task 10: Fallback B — Push gamma_term Lower (if NGTAQv2 loses at low recall)

**Trigger:** Ratio < 1.0 at recall ≤ 0.80 for SIFT-128.

**Files:**
- Create: `/tmp/bench_ngtaq_low_gamma.sh`

- [ ] **Step 1: Check where NGTAQv2 loses**

```bash
python3 /tmp/compare_bench.py /tmp/ngtaq_sift128_sweep.log /tmp/bench_qg_sift128.log | grep -E "0\.50|0\.70|0\.80"
```

If any ratio < 1.0, proceed with this task. Otherwise skip.

- [ ] **Step 2: Sweep gamma_term 0.01–0.04 (below current minimum of 0.05)**

```bash
cat > /tmp/bench_ngtaq_low_gamma.sh << 'EOF'
#!/bin/bash
set -e
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/tests/ngtaq
export LD_LIBRARY_PATH=/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
HDF5=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/sift-128-euclidean.hdf5
LOG=/tmp/bench_ngtaq_low_gamma.log
echo "=== Low gamma_term sweep: $(date) ===" | tee "$LOG"
for gt in 0.01 0.02 0.03 0.04; do
    echo "--- gamma_term=$gt ---" | tee -a "$LOG"
    ./ann_bench /tmp/ngtaq_sift128 "$HDF5" 10 0.20 $gt 1 4 0 2>/dev/null | tee -a "$LOG"
done
echo "=== Done: $(date) ===" | tee -a "$LOG"
EOF
chmod +x /tmp/bench_ngtaq_low_gamma.sh
/tmp/bench_ngtaq_low_gamma.sh
```

Expected: gamma_term=0.01–0.04 may push QPS above 50K, potentially above QG at low recall.

- [ ] **Step 3: Re-run comparison with extended sweep**

```bash
cat /tmp/ngtaq_sift128_sweep.log /tmp/bench_ngtaq_low_gamma.log > /tmp/ngtaq_sift128_full.log
python3 /tmp/compare_bench.py /tmp/ngtaq_sift128_full.log /tmp/bench_qg_sift128.log
```

---

## Task 11: Fallback A — BQ-Only Rerank-Free Path (if Task 10 insufficient)

**Trigger:** After Task 10, still ratio < 1.0 at recall ≤ 0.70 for SIFT-128.

**What it does:** When `rerank_factor=0`, skip ADC reranking entirely — return raw BQ-distance top-k results. This eliminates per-candidate ADC computation, reducing latency significantly for the lowest recall operating points.

**Files:**
- Modify: `lib/NGT/NGTAQ/AQIndex.h` or `AQIndex.cpp` — add `rerank_factor=0` fast path in `searchV2()`

- [ ] **Step 1: Find the reranking code path in searchV2**

```bash
grep -n "rerank\|ADC\|rerank_factor" /home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/AQIndex.h | head -20
grep -n "rerank\|ADC\|rerank_factor" /home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/AQIndex.cpp | head -20
```

- [ ] **Step 2: Implement BQ-only fast path**

Locate the `searchV2` function. Find the section that performs ADC reranking (typically loops over `rerank_factor * k` candidates and recomputes distances with ADC). Add a guard:

```cpp
// In searchV2(), before the ADC reranking section:
if (rerank_factor == 0) {
    // Return BQ-distance candidates directly, no ADC
    // Convert beam search results to output format and return
    // (exact code depends on current searchV2 structure — read before editing)
    return bq_results; // top-k from DABS beam search, BQ distances
}
// ... existing ADC reranking code ...
```

**IMPORTANT:** Read the actual `searchV2` source first. The exact variable names and return type must be determined from the code. Do NOT edit blindly.

- [ ] **Step 3: Build and benchmark BQ-only mode**

```bash
cd /home/kpango/go/src/github.com/kpango/NGT/build_ngtaq
cmake --build . --target ngt ann_bench -j$(nproc)

# Run BQ-only sweep
cd tests/ngtaq
export LD_LIBRARY_PATH=/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
HDF5=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/sift-128-euclidean.hdf5
for gt in 0.01 0.02 0.03 0.05 0.10 0.15 0.20; do
    ./ann_bench /tmp/ngtaq_sift128 "$HDF5" 10 0.20 $gt 0 4 0 2>/dev/null  # rf=0 = BQ-only
done | tee /tmp/ngtaq_bqonly_sweep.log

python3 /tmp/compare_bench.py /tmp/ngtaq_bqonly_sweep.log /tmp/bench_qg_sift128.log
```

- [ ] **Step 4: If BQ-only wins, extend to all remaining datasets**

If the BQ-only path gives ratio > 1.0 at recall ≤ 0.70, run the same benchmark on GloVe-100, GIST-960, and NYTimes-256. Commit the AQIndex change.

```bash
cd /home/kpango/go/src/github.com/kpango/NGT
git add lib/NGT/NGTAQ/AQIndex.h lib/NGT/NGTAQ/AQIndex.cpp  # whichever was modified
git commit -m "feat: add BQ-only fast path for rerank_factor=0 in searchV2

When rerank_factor=0, skip ADC reranking and return BQ-distance results.
Enables 2-10x QPS at low recall vs QG/QSG.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

### 1. Spec Coverage

| Requirement | Task |
|-------------|------|
| QG/QSG benchmark tool | Tasks 2, 3, 5, 6 |
| QBG benchmark tool | Tasks 4, 5, 6 |
| Shared HDF5/recall infra | Task 1 |
| SIFT-128 + NYTimes-256 parallel first | Task 7 |
| Epsilon sweep 0.001–1.0 incl. 0.8, 0.9 | bench_common.hpp |
| QG D_sub=4 + QSG D_sub=1,2 | qg_build.cpp |
| Comparison table | Task 8 |
| Fallback gamma_term | Task 10 |
| Fallback BQ-only | Task 11 |

### 2. Placeholder Check

- Task 11 Step 2 says "read before editing" — this is correct caution, NOT a placeholder. The actual edit depends on the current `searchV2` code structure that must be read first.

### 3. Type Consistency

- `NGTQG::Index::quantize(path, size_t D_sub, size_t maxEdges, bool verbose)` — consistent with NGTQ_QBG=defined branch in `QuantizedGraph.h:669`
- `QBG::SearchContainer::setObjectVector(std::vector<float>&)` — consistent with `QuantizedBlobGraph.h:315`
- `QBG::SearchContainer::setBlobEpsilon(float c)` sets `blobExplorationCoefficient = c + 1.0` — consistent with `QuantizedBlobGraph.h:313`
- `NGTQG::Index(path)` constructor — consistent with `QuantizedGraph.h:263`
- All ID conversions: NGTQG and QBG return 1-based IDs; `compute_recall_k` in `hdf5_io.h` uses 0-based IDs from HDF5 neighbors — subtract 1 in result extraction. ✓ (done in qg_bench.cpp and qbg_bench.cpp)

---

Plan complete and saved to `docs/superpowers/plans/2026-05-31-qg-qbg-baseline-benchmark.md`.
