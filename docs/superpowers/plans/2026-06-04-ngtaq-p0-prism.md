# NGTAQ P0 PRISM — Injected SearchContext / SearchParameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all global (`getenv`/`aq_cfg()` singleton) and `static thread_local` search state in `searchV2` with an injected `SearchContext` + `SearchParameters`, served from a `SearchContextPool`, with **recall byte-identical** to the current code, making the search path safe for concurrent multi-thread / CGO-goroutine reuse.

**Architecture:** `searchV2` becomes a pure function of `(query, const SearchParameters&, SearchContext&)`. Per-query tuning lives in an immutable `SearchParameters` (defaults reproduce today's env-defaults exactly). Per-thread scratch (~25 buffers) lives in a pooled, reused `SearchContext`. Structural code-path/layout flags that never vary per query (`use_global_routing`, `batch_routing`, `dist_lut`, `use_sq8`, `graph_entry`, `versioned_vis`) are resolved from env **once at index construction** into an immutable `IndexRuntimeConfig` member. No algorithm change. The `std::shared_mutex` locking stays exactly as-is in P0 (lock-free EBR is P2).

**Tech Stack:** C++26 (`-std=c++2c`, bump from C++23), header-only new struct file, no new dependencies. Existing types reused: `ADCQueryState`, `GlobalPQ4LUT`, `ADCSlotMeta`, `NbrCand`, `SeedScore`, `AQLinearPool`.

---

## ⚠️ Build & Verification Notes (read before any task)

- **NEVER build the `all` target** — `test_vector_record` is broken and aborts the build. Always build a **specific** target.
- All recall/bench runs use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for determinism.
- Worktree root: `/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/ngtaq-beat-qbg` (branch `worktree-ngtaq-beat-qbg` = `origin/feat/ngtaq-speedup`).
- The current code has a **known unrelated AVX-512 fresh-build crash** (reserve overflow). To avoid it during P0 verification, build/test on the **Zen2 (AVX2) path** (the dev machine default) or reuse a **pre-built index**; P0 changes no build path, so the byte-identical test must use a **prebuilt index loaded from disk**, not a fresh build.
- "Byte-identical" gate: for a fixed prebuilt index + fixed query set + fixed `SearchParameters`, the ordered `(id, distance)` result list must match the captured golden file **exactly** (bitwise on ids, and distances equal under `memcmp` of the `float` bytes).

**Test fixtures (concrete paths — `<idx> <q> /tmp/p0_golden.bin` in every task below maps to these):**
```bash
IDX=/home/kpango/go/src/github.com/kpango/NGT/data/indices/ngtaq_sift-128-euclidean
Q=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/sift-128-euclidean.hdf5
GOLDEN=/tmp/p0_golden.bin
```
- **Index load API (verbatim from `ann_bench.cpp:62-63`, two-step):** `auto idx = NGTAQ::NGTAQIndex::load(std::string(IDX)+"/aqindex"); idx.loadV2(IDX);`
- **Namespace is `NGTAQ`** (top-level, confirmed `AQIndex.h:23`; NOT `NGT::NGTAQ`). `SearchResult { uint32_t id; float distance; float bq_distance; }` is in `DABSSearcher.h:32`. `dEff()`/`loadV2`/static `load` at `AQIndex.h:121/117/96`.
- **Queries are hdf5** (`tests/ngtaq/hdf5_io.h`: `H5FloatDataset h5_read_float(path,"test")`), zero-padded to `idx.dEff()` (mirror `ann_bench.cpp:65-72`). The repo has NO fvecs path — do not use fvecs.
- **Other prebuilt indices** (cross-metric spot-checks): `ngtaq_gist-960-euclidean`, `ngtaq_glove-100-angular`, `ngtaq_nytimes-256-angular`, `ngtaq_fashion-mnist-784-euclidean` under `.../data/indices/`, with matching hdf5 under `.../data/ann-benchmarks/`.

**Runner recipe (REQUIRED for every run — verified working in Task 1).** The test binary has no rpath; a stale `/usr/local/lib/libngt.so.2` shadows the freshly built lib and causes `symbol lookup error` for `NGTAQIndex::load`/`loadV2` without this. BLAS/LAPACK also live under `/tmp/blas-local` (not system-wide):
```bash
W=/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/ngtaq-beat-qbg
RUN="LD_LIBRARY_PATH=$W/build_ngtaq/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:/tmp/blas-local/usr/lib/x86_64-linux-gnu/lapack OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1"
# e.g.:  eval "$RUN ./build_ngtaq/tests/ngtaq/test_p0_prism $IDX $Q $GOLDEN --verify"
```
- **New test targets** go into the `if(HDF5_FOUND)` → `foreach(HDF5_SRC ...)` loop in `tests/ngtaq/CMakeLists.txt` (where `ann_bench` is), NOT a standalone `add_executable` — that loop wires `${HDF5_C_LIBRARIES}`/`${HDF5_INCLUDE_DIRS}`/`ngt`/`cxx_std_23`. (Diagnostics from the editor's clangd showing "AQIndex.h not found" / `std::is_same` errors are false positives — clangd lacks the project `-Ilib` + HDF5 flags; the CMake build is the source of truth.)

---

## File Structure

- **Create:** `lib/NGT/NGTAQ/SearchContext.h` — `SearchParameters`, `SearchContext`, `SearchContextPool`, `SearchContextGuard` (RAII).
- **Modify:** `lib/NGT/NGTAQ/AQIndex.h` — new `searchV2` overload, `IndexRuntimeConfig` member, `SearchContextPool` member, remove 4 setters, add `#include "SearchContext.h"`.
- **Modify:** `lib/NGT/NGTAQ/AQIndex.cpp` — resolve env in ctor; migrate `searchV2` (+ helpers `buildGlobalLUT`/`buildGlobalLUT16`) from `thread_local`/`aq_cfg()` to `ctx`/`params`; update internal callers (`rebuildGraphSelf` @2894, `searchBatch`).
- **Create:** `tests/ngtaq/test_p0_prism.cpp` — golden recall byte-identical test + concurrent-safety (TSan) test.
- **Modify:** `tests/ngtaq/CMakeLists.txt` (or the `tests/ngtaq/Makefile` target list) — add `test_p0_prism` target.

---

## Task 1: Capture the golden baseline (lock current behavior BEFORE refactor)

**Files:**
- Create: `tests/ngtaq/test_p0_prism.cpp`
- Create (build target): add `test_p0_prism` to `tests/ngtaq/CMakeLists.txt`

This test is written FIRST against the **unmodified** `searchV2`. It runs a fixed query set against a prebuilt index and writes the ordered `(id, distance)` results to a golden file. Every later task re-runs it and asserts byte-identical.

- [ ] **Step 1: Write the golden-capture test**

```cpp
// tests/ngtaq/test_p0_prism.cpp
#include "NGT/NGTAQ/AQIndex.h"   // namespace NGTAQ; SearchResult in DABSSearcher.h
#include "hdf5_io.h"             // h5_read_float, H5FloatDataset (mirror ann_bench.cpp)
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <cassert>

// Usage: test_p0_prism <index_dir> <hdf5_path> <golden.bin> [--capture|--verify]
// Two-step load (same as ann_bench.cpp): NGTAQIndex::load(dir+"/aqindex") + loadV2(dir).
// Runs searchV2 over the first 200 "test" queries (k=10), zero-padded to dEff(),
// then writes (capture) or compares (verify) the ordered (id, distance) stream.

int main(int argc, char** argv) {
    assert(argc >= 4);
    const char* idx_dir = argv[1];
    const char* hdf5    = argv[2];
    const char* golden  = argv[3];
    bool capture = (argc >= 5 && std::strcmp(argv[4], "--capture") == 0);
    const int k = 10, NQ = 200;

    NGTAQ::NGTAQIndex index = NGTAQ::NGTAQIndex::load(std::string(idx_dir) + "/aqindex");
    index.loadV2(idx_dir);
    const int D_eff = index.dEff();

    H5FloatDataset test_ds = h5_read_float(hdf5, "test");
    const int nq = std::min(NQ, test_ds.n_rows), D = test_ds.n_cols;
    std::vector<std::vector<float>> queries(nq, std::vector<float>(D_eff, 0.f));
    for (int i = 0; i < nq; ++i) {
        const float* src = test_ds.data.data() + (size_t)i * D;
        std::copy(src, src + D, queries[i].begin());
    }

    // Legacy searchV2 signature (delegates to (query,params,ctx) after Task 4).
    std::vector<std::pair<uint32_t, float>> stream;
    for (auto& q : queries) {
        auto res = index.searchV2(q, k);
        for (auto& r : res) stream.emplace_back(r.id, r.distance);
    }

    if (capture) {
        FILE* f = std::fopen(golden, "wb");
        uint64_t n = stream.size();
        std::fwrite(&n, sizeof(n), 1, f);
        std::fwrite(stream.data(), sizeof(stream[0]), n, f);
        std::fclose(f);
        std::printf("captured %llu tuples\n", (unsigned long long)n);
        return 0;
    }

    FILE* f = std::fopen(golden, "rb");
    assert(f && "golden file missing — run --capture first");
    uint64_t n = 0; (void)std::fread(&n, sizeof(n), 1, f);
    std::vector<std::pair<uint32_t, float>> gold(n);
    (void)std::fread(gold.data(), sizeof(gold[0]), n, f);
    std::fclose(f);

    if (stream.size() != gold.size()) { std::printf("FAIL size %zu != %llu\n", stream.size(), (unsigned long long)n); return 1; }
    for (size_t i = 0; i < n; ++i) {
        if (stream[i].first != gold[i].first ||
            std::memcmp(&stream[i].second, &gold[i].second, sizeof(float)) != 0) {
            std::printf("FAIL at %zu: got (%u,%a) want (%u,%a)\n",
                        i, stream[i].first, stream[i].second, gold[i].first, gold[i].second);
            return 1;
        }
    }
    std::printf("PASS byte-identical (%llu tuples)\n", (unsigned long long)n);
    return 0;
}
```

(The `--concurrent` mode used in Task 9 is added to this same file later; for now only `--capture`/`--verify` are needed.)

- [ ] **Step 2: Add the build target**

Mirror the existing `ann_bench` target in `tests/ngtaq/CMakeLists.txt` (it already pulls in `hdf5_io.h` and links HDF5 — copy its exact link/include vars). Add:
```cmake
add_executable(test_p0_prism test_p0_prism.cpp)
target_link_libraries(test_p0_prism ngt ${HDF5_LIBRARIES})
target_include_directories(test_p0_prism PRIVATE ${HDF5_INCLUDE_DIRS})
```

- [ ] **Step 3: Build the specific target (NOT `all`)**

Run: `cmake --build build_ngtaq --target test_p0_prism -j`
Expected: links cleanly. (If the build dir is stale, reconfigure: `cmake -S . -B build_ngtaq` first. Build only this target — never `all`.)

- [ ] **Step 4: Capture the golden file against the prebuilt SIFT index**

Run: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./build_ngtaq/tests/ngtaq/test_p0_prism $IDX $Q $GOLDEN --capture`
Expected: `captured 2000 tuples` (200 queries × k=10; fewer if any query returns <k).

- [ ] **Step 5: Verify capture is self-consistent**

Run: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./build_ngtaq/tests/ngtaq/test_p0_prism $IDX $Q $GOLDEN --verify`
Expected: `PASS byte-identical (2000 tuples)`.

- [ ] **Step 6: Commit**

```bash
git add tests/ngtaq/test_p0_prism.cpp tests/ngtaq/CMakeLists.txt
git commit -m "test(ngtaq): P0 golden byte-identical recall capture for searchV2"
```

---

## Task 2: Define SearchParameters, SearchContext, SearchContextPool

**Files:**
- Create: `lib/NGT/NGTAQ/SearchContext.h`

Field lists are taken verbatim from the reconnaissance: 16 `AQConfig` fields + 6 `searchV2` args + 4 setter-backed members → split into per-query `SearchParameters` vs per-thread `SearchContext` scratch. Defaults reproduce today's env-defaults exactly.

- [ ] **Step 1: Write the header**

```cpp
// lib/NGT/NGTAQ/SearchContext.h
#pragma once
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <utility>
#include "ADCDistance.h"   // ADCQueryState, ADCSlotMeta, ADC_SLOTS
#include "GlobalPQ4.h"     // GlobalPQ4LUT
// NbrCand, SeedScore, AQLinearPool are hoisted from AQIndex.cpp into this header
// in Step 0 below, so SearchContext owns them and the header is self-contained.

namespace NGTAQ {   // top-level namespace, confirmed AQIndex.h:23

// ---- per-query tuning (immutable during a search). Defaults == current env defaults ----
struct SearchParameters {
    int   k                = 10;
    int   ef               = 0;      // AQ_EF       (0 = derive from max_visits)
    int   max_visits       = 0;      // searchV2 arg(0 = derive)
    float gamma_enq        = 0.2f;   // searchV2 arg
    float gamma_term       = 0.4f;   // searchV2 arg
    float k_prime_factor   = -1.0f;  // setSearchGammas (-1 = default)
    int   rerank_factor    = 0;      // searchV2 arg
    int   rerank_n         = 0;      // AQ_RERANK_N
    int   refine_mult      = 0;      // AQ_REFINE_MULT
    int   rerank_sq8       = 0;      // AQ_RERANK_SQ8
    int   cq_probe         = 4;      // AQ_CQ_PROBE
    int   n_probe          = 0;      // n_probe_override_ (0 = default)
    int   seeds            = 0;      // AQ_SEEDS
    int   n_cluster_seeds  = 0;      // setNClusterSeeds -> prop_.n_cluster_seeds
    int   seeds_per_cluster= 0;      // setSeedsPerCluster -> prop_.seeds_per_cluster
    bool  seed_cap_topk    = false;  // AQ_SEED_CAP_TOPK
    float term_eps         = -1.0f;  // AQ_TERM_EPS
    float term_eps_fp16    = -1.0f;  // AQ_TERM_EPS_FP16
};

// ---- per-thread reusable scratch (one alloc, reused forever). 1:1 with the 25 thread_locals ----
struct SearchContext {
    // query transforms
    std::vector<float>  q_normalized;   // 1526 q_normalized_tl
    std::vector<float>  q_padded;       // 1542 q_padded_tl
    std::vector<float>  q_rot;          // 1543 q_rot_tl
    std::vector<float>  q_res;          // 1544 q_res_tl
    std::vector<float>  q_res_init;     // 1545 q_res_init_tl
    ADCQueryState       adc;            // 1548 adc_tl
    // LUT scratch (searchV2 + buildGlobalLUT/16 helpers)
    std::vector<float>  q_norm_lut;     // 1241/1287 q_norm_tl (helper-local)
    std::vector<float>  q_padded_lut;   // 1252/1296 q_padded_tl(helper-local)
    std::vector<float>  q_rot_lut;      // 1252/1296 q_rot_tl   (helper-local)
    std::vector<float>  ip;             // 1296 ip_tl
    std::vector<float>  t2_lut;         // 1827 t2_lut_tl
    std::vector<float>  t2_lut_probe;   // 2000 t2_lut_probe
    std::vector<float>  global_lut;     // 1854 global_lut_tl
    GlobalPQ4LUT        batch_lut;      // 1865 batch_lut_tl
    std::vector<float>  batch_ip;       // 1866 batch_ip_tl
    // ADC cache
    std::vector<int8_t>                 adc_int8;  // 1759 adc_int8_tl (8 slots × D)
    std::array<ADCSlotMeta, ADC_SLOTS>  adc_meta;  // 1760 adc_meta_tl
    // SQ8 query
    std::vector<int8_t> q_sq8;          // 1882 q_sq8_tl
    // frontier / results
    AQLinearPool        lp;             // 1659 lp
    std::vector<std::pair<float,uint32_t>> results;  // 1661 results_tl
    std::vector<float>  block_ip;       // 2284 block_ip_lp / 2429 block_ip_tl (unified)
    std::vector<NbrCand> nbr_buf;       // 2492 nbr_buf_tl
    // seeding scratch
    std::vector<SeedScore> scored;          // 1959/1991 scored
    std::vector<uint32_t>  probe_clusters;  // 1981 probe_clusters
    std::vector<SeedScore> clus_buf;        // 2068 clus_buf
    // visited (dual backend; runtime config selects which is used)
    std::vector<uint64_t> vis;          // 1679 t_vis
    std::vector<uint16_t> vis_ver;      // 1680 t_vis_ver
    uint16_t              vis_cur = 0;  // 1681 t_vis_cur
    // P2 placeholder (unused in P0): EBR published epoch
    std::atomic<uint64_t> epoch{0};
};

// ---- context pool (mutex-guarded free list; provably TSan-clean for P0).
// acquire/release run once per search vs the long search body under shared_lock,
// so the lock is uncontended. (Lock-free Treiber is a P2 option if profiling warrants;
// avoided here to sidestep offsetof-on-non-standard-layout UB and ABA.) ----
class SearchContextPool {
public:
    SearchContext* acquire() {
        std::lock_guard<std::mutex> g(m_);
        if (free_.empty()) { auto* c = new SearchContext(); all_.push_back(c); return c; }
        auto* c = free_.back(); free_.pop_back(); return c;
    }
    void release(SearchContext* c) {
        std::lock_guard<std::mutex> g(m_);
        free_.push_back(c);
    }
    ~SearchContextPool() { for (auto* c : all_) delete c; }
    SearchContextPool() = default;
    SearchContextPool(const SearchContextPool&) = delete;
    SearchContextPool& operator=(const SearchContextPool&) = delete;
private:
    std::mutex m_;
    std::vector<SearchContext*> free_;   // available
    std::vector<SearchContext*> all_;    // owns every allocated context
};

// RAII acquire/release
struct SearchContextGuard {
    SearchContextPool& pool; SearchContext* ctx;
    explicit SearchContextGuard(SearchContextPool& p) : pool(p), ctx(p.acquire()) {}
    ~SearchContextGuard() { pool.release(ctx); }
    SearchContext& operator*() { return *ctx; }
    SearchContext* operator->() { return ctx; }
};

} // namespace NGTAQ
```

- [ ] **Step 0 (do FIRST): hoist `AQLinearPool`/`NbrCand`/`SeedScore` into `SearchContext.h`**

Move `AQLinearPool` (`AQIndex.cpp:134-177`) and the `NbrCand`/`SeedScore` struct definitions (grep them in AQIndex.cpp) out of AQIndex.cpp into `SearchContext.h` (above `struct SearchContext`). Add `#include "SearchContext.h"` near the top of `AQIndex.cpp` (after the existing NGTAQ includes, before first use) so AQIndex.cpp still sees them. This makes the header self-contained. (Task 5 Step 1 then becomes a no-op verification.)

- [ ] **Step 2: Compile-check via the existing golden test target**

Because AQIndex.cpp now `#include`s SearchContext.h, building `test_p0_prism` compiles the new header (the structs are defined but not yet used by searchV2, so recall is unchanged).
Run: `cmake --build build_ngtaq --target test_p0_prism -j`
Then golden verify (Task 2 changes no logic → must stay byte-identical):
`eval "$RUN ./build_ngtaq/tests/ngtaq/test_p0_prism $IDX $Q $GOLDEN --verify"`
Expected: clean build + `PASS byte-identical (2000 tuples)`.

- [ ] **Step 3: Commit**

```bash
git add lib/NGT/NGTAQ/SearchContext.h
git commit -m "feat(ngtaq): add SearchParameters/SearchContext/SearchContextPool (P0 PRISM scaffolding)"
```

---

## Task 3: Resolve env once at construction into an immutable IndexRuntimeConfig

**Files:**
- Modify: `lib/NGT/NGTAQ/AQIndex.h` (add member + struct)
- Modify: `lib/NGT/NGTAQ/AQIndex.cpp:77-118` (AQConfig), constructor, `:3283` (`AQ_PACKED`)

The 6 structural flags become an immutable member resolved at construction. Per-query env reads (which there are none of in the hot loop — all 16 are read via `aq_cfg()` whose singleton already reads env once) keep identical values; we simply move ownership from a file-scope Meyers singleton to a member so it is injectable and testable.

- [ ] **Step 1: Add `IndexRuntimeConfig` to AQIndex.h**

```cpp
// AQIndex.h — near the other config structs
struct IndexRuntimeConfig {
    bool use_global_routing = false; // AQ_USE_GLOBAL_ROUTING
    bool batch_routing      = true;  // AQ_BATCH_ROUTING
    bool dist_lut           = true;  // AQ_DIST_LUT
    bool use_sq8            = false; // AQ_SQ8
    bool graph_entry        = false; // AQ_GRAPH_ENTRY
    bool versioned_vis      = false; // AQ_VERSIONED_VIS
};
```
Add member (near `AQIndex.h:222` where `prop_` lives): `IndexRuntimeConfig rt_cfg_;`

- [ ] **Step 2: Populate `rt_cfg_` in the constructor from env (move the reads out of `aq_cfg()`)**

In the `NGTAQIndex` constructor, read the 6 env vars once and assign to `rt_cfg_` (same defaults/parse as the current `AQConfig` struct at `AQIndex.cpp:77-116`). Keep `aq_cfg()` temporarily for the per-query tunables not yet migrated (removed in Task 6).

- [ ] **Step 3: Re-run golden verify (no behavior change expected)**

Run: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./build_ngtaq/tests/ngtaq/test_p0_prism <idx> <q> /tmp/p0_golden.bin --verify`
Expected: `PASS byte-identical`.

- [ ] **Step 4: Commit**

```bash
git add lib/NGT/NGTAQ/AQIndex.h lib/NGT/NGTAQ/AQIndex.cpp
git commit -m "refactor(ngtaq): resolve structural env flags into immutable IndexRuntimeConfig member"
```

---

## Task 4: Add the new injected `searchV2` overload (old one delegates)

**Files:**
- Modify: `lib/NGT/NGTAQ/AQIndex.h:110-113`
- Modify: `lib/NGT/NGTAQ/AQIndex.cpp:1491`

Introduce the pure signature without yet moving the body; the legacy signature builds a `SearchParameters` from its args + defaults and acquires a context, then calls the new one. This keeps `test_p0_prism` (which calls the legacy signature) green throughout.

- [ ] **Step 1: Declare the new overload in AQIndex.h**

```cpp
// AQIndex.h — alongside existing searchV2 decl
std::vector<SearchResult> searchV2(const std::vector<float>& query,
                                   const SearchParameters& params,
                                   SearchContext& ctx) const;
// legacy (kept; delegates):
std::vector<SearchResult> searchV2(const std::vector<float>& query, int k,
                                   float gamma_enq = 0.2f, float gamma_term = 0.4f,
                                   int rerank_factor = 0, int max_visits = 0) const;
```
Add members near `AQIndex.h:234`: `mutable SearchContextPool ctx_pool_;`

- [ ] **Step 2: Implement the delegating legacy body**

```cpp
// AQIndex.cpp — replace the current searchV2(query,k,...) entry; body moves to the new overload in Task 5
std::vector<SearchResult> NGTAQIndex::searchV2(const std::vector<float>& query, int k,
        float gamma_enq, float gamma_term, int rerank_factor, int max_visits) const {
    SearchParameters p;
    p.k = k; p.gamma_enq = gamma_enq; p.gamma_term = gamma_term;
    p.rerank_factor = rerank_factor; p.max_visits = max_visits;
    // pull the not-yet-migrated tunables from aq_cfg() so behavior is identical
    const auto& c = aq_cfg();
    p.ef = c.ef; p.rerank_n = c.rerank_n; p.refine_mult = c.refine_mult;
    p.rerank_sq8 = c.rerank_sq8; p.cq_probe = c.cq_probe; p.seeds = c.seeds;
    p.seed_cap_topk = c.seed_cap_topk; p.term_eps = c.term_eps; p.term_eps_fp16 = c.term_eps_fp16;
    p.k_prime_factor = searcher_.k_prime_factor;          // from setSearchGammas
    p.n_cluster_seeds = prop_.n_cluster_seeds;            // from setNClusterSeeds
    p.seeds_per_cluster = prop_.seeds_per_cluster;        // from setSeedsPerCluster
    p.n_probe = n_probe_override_;                        // from setNProbe
    SearchContextGuard g(ctx_pool_);
    return searchV2(query, p, *g);
}
```

- [ ] **Step 3: Stub the new overload (temporarily contains the old body verbatim)**

For this task, the new overload literally contains the **current** body of `searchV2` unchanged (still using `thread_local`/`aq_cfg()` internally), so it compiles and passes. Task 5 migrates its internals to `ctx`/`params`.

- [ ] **Step 4: Build + golden verify**

Run: `cmake --build build_ngtaq --target test_p0_prism -j && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./build_ngtaq/tests/ngtaq/test_p0_prism <idx> <q> /tmp/p0_golden.bin --verify`
Expected: `PASS byte-identical`.

- [ ] **Step 5: Commit**

```bash
git add lib/NGT/NGTAQ/AQIndex.h lib/NGT/NGTAQ/AQIndex.cpp
git commit -m "refactor(ngtaq): add injected searchV2(query,params,ctx) overload; legacy delegates"
```

---

## Task 5: Migrate the 25 `thread_local` scratch buffers to `ctx` fields

**Files:**
- Modify: `lib/NGT/NGTAQ/AQIndex.cpp` (searchV2 body `1491-2724`, helpers `buildGlobalLUT`/`buildGlobalLUT16`)
- Modify: `lib/NGT/NGTAQ/SearchContext.h` (hoist `NbrCand`, `SeedScore`, `AQLinearPool` definitions here from AQIndex.cpp so the header is self-contained)

**Migration table (apply each replacement; delete the `static thread_local` declaration):**

| Old `thread_local` (file:line) | New `ctx` field |
|---|---|
| `q_normalized_tl` (1526) | `ctx.q_normalized` |
| `q_padded_tl`/`q_rot_tl`/`q_res_tl`/`q_res_init_tl` (1542-1545) | `ctx.q_padded`/`ctx.q_rot`/`ctx.q_res`/`ctx.q_res_init` |
| `adc_tl` (1548) | `ctx.adc` |
| `q_norm_tl`/`q_padded_tl`/`q_rot_tl` in `buildGlobalLUT` (1241,1252) | `ctx.q_norm_lut`/`ctx.q_padded_lut`/`ctx.q_rot_lut` |
| `q_norm_tl`/`q_padded_tl`/`q_rot_tl`/`ip_tl` in `buildGlobalLUT16` (1287,1296) | `ctx.q_norm_lut`/`ctx.q_padded_lut`/`ctx.q_rot_lut`/`ctx.ip` |
| `lp` (1659) | `ctx.lp` |
| `results_tl` (1661) | `ctx.results` |
| `t_vis`/`t_vis_ver`/`t_vis_cur` (1679-1681) | `ctx.vis`/`ctx.vis_ver`/`ctx.vis_cur` |
| `adc_int8_tl`/`adc_meta_tl` (1759-1760) | `ctx.adc_int8`/`ctx.adc_meta` |
| `t2_lut_tl` (1827) | `ctx.t2_lut` |
| `global_lut_tl` (1854) | `ctx.global_lut` |
| `batch_lut_tl`/`batch_ip_tl` (1865-1866) | `ctx.batch_lut`/`ctx.batch_ip` |
| `q_sq8_tl` (1882) | `ctx.q_sq8` |
| `scored` (1959,1991) | `ctx.scored` |
| `probe_clusters` (1981) | `ctx.probe_clusters` |
| `t2_lut_probe` (2000) | `ctx.t2_lut_probe` |
| `clus_buf` (2068) | `ctx.clus_buf` |
| `block_ip_lp` (2284) + `block_ip_tl` (2429) | `ctx.block_ip` (unified) |
| `nbr_buf_tl` (2492) | `ctx.nbr_buf` |

- [ ] **Step 1: Hoist `NbrCand`, `SeedScore`, `AQLinearPool` into `SearchContext.h`**

Move their definitions (`AQLinearPool` at `AQIndex.cpp:134-177`; locate `NbrCand`/`SeedScore` structs in AQIndex.cpp) into `SearchContext.h` above `struct SearchContext`. Leave a `using` alias or just rely on the namespace. Confirm AQIndex.cpp still sees them via the include.

- [ ] **Step 2: Add `SearchContext& ctx` param to the helpers**

Change `buildGlobalLUT(...)` and `buildGlobalLUT16(...)` to take `SearchContext& ctx`, and pass `ctx` from `searchV2`. Replace their helper-local `thread_local`s per the table.

- [ ] **Step 3: Apply the migration table inside the new `searchV2(query,params,ctx)` overload**

For each row: delete the `static thread_local <type> <name>;` line, replace every use of `<name>` with `ctx.<field>`. The reset/`resize`/`memset` logic stays identical (still per-query) — it now operates on the pooled buffer, which is exactly the prior `thread_local` reuse semantics.

- [ ] **Step 4: Build + golden verify (the critical byte-identical gate)**

Run: `cmake --build build_ngtaq --target test_p0_prism -j && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./build_ngtaq/tests/ngtaq/test_p0_prism <idx> <q> /tmp/p0_golden.bin --verify`
Expected: `PASS byte-identical (2000 tuples)`. If FAIL: a buffer's reset/resize semantics differ — diff the failing index against the original `thread_local` reset path.

- [ ] **Step 5: Commit**

```bash
git add lib/NGT/NGTAQ/AQIndex.cpp lib/NGT/NGTAQ/SearchContext.h
git commit -m "refactor(ngtaq): migrate 25 searchV2 thread_local scratch buffers to injected SearchContext"
```

---

## Task 6: Migrate `aq_cfg()` reads to `params` / `rt_cfg_`; delete the singleton

**Files:**
- Modify: `lib/NGT/NGTAQ/AQIndex.cpp` (every `aq_cfg()` use inside `searchV2`; `AQConfig`/`aq_cfg()` at `77-118`)

- [ ] **Step 1: Replace per-query tunable reads with `params`**

Inside `searchV2(query,params,ctx)`, replace `aq_cfg().<tunable>` with `params.<field>` for: `ef, cq_probe, seeds, seed_cap_topk, term_eps, term_eps_fp16, refine_mult, rerank_n, rerank_sq8`.

- [ ] **Step 2: Replace structural reads with `rt_cfg_`**

Replace `aq_cfg().<structural>` with `rt_cfg_.<field>` for: `use_global_routing, batch_routing, dist_lut, use_sq8, graph_entry, versioned_vis` (line 1678 selector `use_versioned_vis = rt_cfg_.versioned_vis`).

- [ ] **Step 3: Delete `AQConfig`/`aq_cfg()` if no remaining references**

Grep: `grep -n "aq_cfg()" lib/NGT/NGTAQ/AQIndex.cpp`. If only build-path uses remain (e.g. the `AQ_*` build flags at 898-1214 are separate locals — leave those), delete the struct + singleton at `77-118`. Otherwise leave and note remaining users.

- [ ] **Step 4: Build + golden verify**

Run: `cmake --build build_ngtaq --target test_p0_prism -j && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./build_ngtaq/tests/ngtaq/test_p0_prism <idx> <q> /tmp/p0_golden.bin --verify`
Expected: `PASS byte-identical`.

- [ ] **Step 5: Commit**

```bash
git add lib/NGT/NGTAQ/AQIndex.cpp
git commit -m "refactor(ngtaq): route searchV2 tunables through SearchParameters/IndexRuntimeConfig; drop aq_cfg singleton"
```

---

## Task 7: Remove the 4 non-thread-safe setters; update internal callers

**Files:**
- Modify: `lib/NGT/NGTAQ/AQIndex.h:61-76` (remove setters)
- Modify: `lib/NGT/NGTAQ/AQIndex.cpp:2894` (`rebuildGraphSelf`), `searchBatch`, any external test callers

`setSearchGammas`/`setNClusterSeeds`/`setSeedsPerCluster`/`setNProbe` mutate `searcher_`/`prop_`/`n_probe_override_` which `searchV2` now reads via `SearchParameters`. Remove them; callers build a `SearchParameters` instead.

- [ ] **Step 1: Grep all callers**

Run: `grep -rn "setSearchGammas\|setNClusterSeeds\|setSeedsPerCluster\|setNProbe" lib tests samples benchmarks`
Note every call site.

- [ ] **Step 2: Convert each caller to populate `SearchParameters`**

For each call site, instead of `index.setX(v); index.searchV2(q,k);` build `SearchParameters p; p.field = v; ...; SearchContextGuard g(pool); index.searchV2(q, p, *g);` — or, where the legacy signature suffices, pass via the (still-present) legacy delegating overload's args. For `rebuildGraphSelf` @2894 (internal), pass an explicit `SearchParameters` + a local `SearchContext`.

- [ ] **Step 3: Delete the 4 setters and the now-unused `n_probe_override_` member if no remaining writers**

Verify `n_probe_override_` has no remaining writer (grep), then remove. Keep `searcher_`/`prop_` (used elsewhere).

- [ ] **Step 4: Build + golden verify**

Run: `cmake --build build_ngtaq --target test_p0_prism -j && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./build_ngtaq/tests/ngtaq/test_p0_prism <idx> <q> /tmp/p0_golden.bin --verify`
Expected: `PASS byte-identical`.

- [ ] **Step 5: Commit**

```bash
git add lib/NGT/NGTAQ/AQIndex.h lib/NGT/NGTAQ/AQIndex.cpp
git commit -m "refactor(ngtaq): remove non-thread-safe search setters; callers inject SearchParameters"
```

---

## Task 8: Public search path acquires from the pool (concurrent-reuse-ready)

**Files:**
- Modify: `lib/NGT/NGTAQ/AQIndex.cpp` (`search()` @324, `searchBatch()` @400)

- [ ] **Step 1: Route `search()`/`searchBatch()` through the pool**

Where the public entry points call into `searchV2`, wrap with `SearchContextGuard g(ctx_pool_)` and pass a `SearchParameters` built from the call args. The `std::shared_lock` at 324/1601 stays exactly as-is (P0 keeps the locking model; lock-free EBR is P2). Confirm no `thread_local` remains in the search path: `grep -n "thread_local" lib/NGT/NGTAQ/AQIndex.cpp` → only build-path (if any) may remain; search path must be clean.

- [ ] **Step 2: Build + golden verify**

Run: `cmake --build build_ngtaq --target test_p0_prism -j && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./build_ngtaq/tests/ngtaq/test_p0_prism <idx> <q> /tmp/p0_golden.bin --verify`
Expected: `PASS byte-identical`.

- [ ] **Step 3: Commit**

```bash
git add lib/NGT/NGTAQ/AQIndex.cpp
git commit -m "refactor(ngtaq): public search() acquires SearchContext from pool (concurrent-reuse-ready)"
```

---

## Task 9: Concurrent-safety test (ThreadSanitizer)

**Files:**
- Modify: `tests/ngtaq/test_p0_prism.cpp` (add a `--concurrent` mode)

Proves the redesign goal: many threads sharing one `NGTAQIndex`, each with its own pooled `SearchContext`, race-free. (No writers in P0 — the shared_lock already permits concurrent readers; this verifies the state migration removed the per-thread aliasing hazards.)

- [ ] **Step 1: Add a concurrent mode**

```cpp
// in main(): if argv[4] == "--concurrent"
//   launch 16 std::jthread, each runs all 200 queries via index.searchV2(q,k),
//   each acquiring its own SearchContext from the index's pool; collect per-thread
//   result streams and assert every thread's stream == the golden file.
```

- [ ] **Step 2: Build with TSan and run**

Run: `cmake --build build_ngtaq --target test_p0_prism -j` then a TSan build:
`clang++ -std=c++2c -fsanitize=thread -O1 -g -I lib tests/ngtaq/test_p0_prism.cpp -o /tmp/test_p0_tsan <link ngt objects>`
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/test_p0_tsan <idx> <q> /tmp/p0_golden.bin --concurrent`
Expected: `PASS` on all 16 threads, **zero** TSan data-race reports.

- [ ] **Step 3: Commit**

```bash
git add tests/ngtaq/test_p0_prism.cpp
git commit -m "test(ngtaq): P0 concurrent 16-thread shared-index race-free verification (TSan)"
```

---

## Task 10: C++26 bump + final sweep

**Files:**
- Modify: root `CMakeLists.txt` (`cxx_std_23` → `cxx_std_26` / `-std=c++2c`)

- [ ] **Step 1: Bump the standard**

Change the project `CMAKE_CXX_STANDARD`/target `cxx_std_23` to `26` (the C++26 probe confirmed `-std=c++2c`, `std::jthread`, `std::atomic_ref` available on clang21/libstdc++).

- [ ] **Step 2: Full search-path cleanliness sweep**

Run:
```
grep -n "getenv\|thread_local\|aq_cfg()" lib/NGT/NGTAQ/AQIndex.cpp
```
Expected: no occurrences inside the search path (`searchV2`, `search`, `searchBatch`, `buildGlobalLUT*`). Build-path `AQ_*` reads (898-1214, 3283) may remain (out of P0 scope; noted for a later build-path pass).

- [ ] **Step 3: Build + golden verify + concurrent verify**

Run all three: target build, `--verify`, `--concurrent`.
Expected: `PASS byte-identical` + race-free.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "build(ngtaq): bump to C++26 (-std=c++2c) after PRISM search-path statelessness"
```

---

## Self-Review

**1. Spec coverage (Directive 4 — stateless, env-var-free, injected API):**
- Injected `SearchContext`/`SearchParameters` → Tasks 2,4,5. ✓
- `getenv` eradication from search path → Tasks 3,6,10. ✓
- `static thread_local` eradication → Task 5. ✓
- Zero-allocation steady state → pooled `SearchContext` reuse (Task 2 pool, Task 8 wiring). ✓
- Thread-safety for multi-tenant / CGO-goroutine reuse → Task 9 TSan proof. ✓
- Non-thread-safe setters removed → Task 7. ✓

**2. Placeholder scan:** Struct definitions are complete (Task 2). The migration is a fully-enumerated table (Task 5), not "migrate the buffers". Every step has an exact build/run command + expected output. The only deliberately deferred items (build-path `AQ_*` flags, lock-free EBR) are explicitly scoped OUT of P0 with a reason.

**3. Type consistency:** `SearchParameters`/`SearchContext`/`SearchContextPool`/`SearchContextGuard` names and fields are identical across Tasks 2,4,5,6,7,8. `IndexRuntimeConfig`/`rt_cfg_` consistent across Tasks 3,6. `ctx_pool_` consistent across Tasks 4,8. The new `searchV2(query,params,ctx)` signature is identical in Tasks 4 and 5.

**Out of P0 scope (subsequent plans):** P1 RaBitQ routing quantizer (reuse map: SRHT/VNNI-dpbusd/fp16-rerank/sign-plane + scalar correction); P2 fixed-R Vamana + IP-DiskANN + EBR lock-free (replaces shared_mutex; uses the `ctx.epoch` placeholder); P3 (gated) DRIFT relayout.

---

## Execution Handoff

Plan complete and saved. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec then quality) between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session via executing-plans, batch with checkpoints.
