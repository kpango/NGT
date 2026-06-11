# ArcFlare — Beat QBG *and* QG by 2–10× QPS Across All ANN-Benchmarks Recall Levels

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ArcFlare (AQ-DABS) deliver 2–10× the QPS of **both** QBG (Quantized Blob Graph) **and** QG (NGT-qg, the actual ANN-Benchmarks leaderboard entry) at every recall level {0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99} on all five datasets (SIFT-128, GIST-960, FashionMNIST-784 = L2; GloVe-100, NYTimes-256 = angular). 0.99 is targeted only where physically reachable.

**Architecture:** Five-phase campaign. (0) Migrate the exact-rerank store `raw_flat_` from fp32 to fp16. (1) Establish two credible competitor baselines: QG with **official ANN-Benchmarks params verbatim**, and QBG self-tuned via its documented CLI knobs. (2) Generate the full single- and multi-thread recall-QPS grid for ArcFlare, QBG, and QG; a cell PASSES only when ArcFlare ≥ 2× the *stronger* competitor. (3) Profile the angular search path. (4) Apply profile-/grid-driven optimizations. (5) Re-measure and emit an honest per-cell report scoped to passing cells.

**Tech Stack:** C++17 (`lib/NGT/ArcFlare/`), AVX2+F16C SIMD, HDF5 I/O, std::thread inter-query parallelism, NGT `ngt`/`qbg` CLI for QG, Python 3 for grid comparison.

---

## ⚠️ Worktree / base-ref note

This plan was authored in worktree `arcflare-beat-qbg` (branch `worktree-arcflare-beat-qbg`, based at commit `522a328`). **The in-progress uncommitted changes to `ArcFlareIndex.cpp/.h` (searchV2 `n_probe_override_`, `fixHoleTombstones`, `rebuild_max_seeds_`, fp16-prep in `half.hpp`) are NOT in this worktree** — they live in the main checkout's working tree on `feat/arcflare-speedup`. **Execute this plan against that WIP state** (commit the WIP first, then branch, or run in the main checkout). Line numbers below reference the `feat/arcflare-speedup` *working tree*; the executing engineer should confirm anchors by the surrounding code snippets (re-grep if a line moved).

## Decisions locked in (2026-05-31 grill-me session)

| # | Branch | Decision |
|---|--------|----------|
| Q1 | Memory fairness | **fp16 rerank** — store `raw_flat_` as fp16 (≈½ memory), rerank in fp16. Memory caveat documented; note QBG's own `-R h` refinement is the fp16 analog. |
| Q2 | Dataset scope | **Angular included now** — NYTimes/GloVe in scope from the start; fundamental rework accepted. |
| Q3 | Angular fix method | **Profile first** — instrument `searchV2`, locate the real bottleneck, then fix the dominant cost. |
| Q4 | QPS protocol | **Report single-thread AND multi-thread.** Single-thread (threads=1, sequential) is the win basis; multi-thread supplementary. |
| Q5 | Baseline rigor | **Official ANN-Benchmarks params verbatim** — applies to **QG** (only QG is on ann-benchmarks). |
| Q6 | Done criteria | **Per-cell honest reporting + iterate on misses.** Claim 2–10× only on passing cells; never block on the hardest cells. |
| Q7 | Which baseline | **Both QBG and QG are co-primary.** A cell passes only if ArcFlare ≥ 2× **max(QBG, QG)** at that recall. |

**Key discovery (Q7 trigger):** ann-benchmarks has **no `qbg`** algorithm. The "NGT-qg" leaderboard entry is **QG** (`qbg create-qg/build-qg/search-qg`), reproducible verbatim. **QBG** (`qbg create/append/build/search`) is CLI-only with no official grid → self-tuned. `qsg_ngt` is a patched/external competition variant → **not reproducible, excluded**.

## Target matrix and current standing

| Dataset | Metric | N | D | 0.50 | 0.70 | 0.80 | 0.85 | 0.90 | 0.95 | 0.99 |
|---------|--------|---|---|------|------|------|------|------|------|------|
| SIFT-128 | L2 | 1M | 128 | ? | ? | ? | ? | ? | ? | ArcFlare 6046 vs QBG 2406 (2.5×); QG 未測 |
| FashionMNIST-784 | L2 | 60K | 784 | ? | ? | ? | ? | ? | ? | ArcFlare 0.9999@3098; both 未測 |
| GIST-960 | L2 | 1M | 960 | — | — | — | — | — | — | — (all 未測) |
| GloVe-100 | angular | 1.18M | 100 | ? | ? | ? | ? | △ | △ | ArcFlare ~0.99@291 (weak) |
| NYTimes-256 | angular | 290K | 256 | ArcFlare ~45 QPS 全帯（壊滅） |||||||

`?` = unmeasured on current build. `—` = no data. Only SIFT-128 @ 0.99 vs QBG is demonstrated; **QG never measured against ArcFlare on any cell.**

## Known risks (each has a mitigation task)

- **R1 — L2 mid/low-recall ceiling:** QBG/QG are flat-fast (≈6–14k+ QPS) at recall 0.5–0.9; QG especially (pure quantized graph) raises the bar. ArcFlare's fixed overhead (SRHT, per-cluster seeding, `EXPAND_N=200`, `refine_n=k_beam*100`, exact rerank) may cap QPS below 2×. → **Phase 4B BQ-only fast path.**
- **R2 — Angular 0.99:** current recall caps ≈0.984; reaching 0.99 and 2× needs rework, may never hit 2×. → **Phase 3+4A; Q6 scoping.**
- **R3 — fp16 precision at 0.99** (esp. GIST D=960). → **Phase 0.3 gate.**
- **R4 — Baseline parity:** QG official params must be applied per dataset; QBG must be genuinely well-tuned (no strawman). → **Phase 1.**
- **R5 — QG is fast:** beating QG 2× at low recall is harder than beating QBG; expect QG to be the binding competitor on L2. → Acknowledged; Q6 lets us report per-cell honestly.

## Standard environment (every shell step)

```bash
export NGT=/home/kpango/go/src/github.com/kpango/NGT
export BUILD=$NGT/build_arcflare
export DATA=$NGT/data/ann-benchmarks
export IDX=$NGT/data/indices          # persistent (NOT /tmp — GIST index/TSV are large)
export LOGS=$NGT/benchmarks/results/grid
export NGTBIN=$BUILD/bin/ngt/ngt
export QBGBIN=$BUILD/bin/qbg/qbg
export LD_LIBRARY_PATH=$BUILD/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
mkdir -p "$IDX" "$LOGS"
```

Dataset map (used throughout):

| HDF5 | ArcFlare/QBG metric | QG CLI `-D` | ArcFlare edge_size |
|------|------------------|-------------|-----------------|
| `sift-128-euclidean` | `l2` | `2` | 100 |
| `gist-960-euclidean` | `l2` | `2` | 100 |
| `fashion-mnist-784-euclidean` | `l2` | `2` | 100 |
| `glove-100-angular` | `angular` | `E` | 100 |
| `nytimes-256-angular` | `angular` | `E` | 100 |

> All ArcFlare angular builds use `edge_size=100` (the prior `edge_size=10` is the README-blamed thin-graph cause).

---

## Phase 0 — fp16 exact-rerank store (Q1)

Do first: it changes the on-disk index format, so it must precede all index builds; the recall gate surfaces R3 early.

**Files:** `lib/NGT/ArcFlare/SIMDUtils.h`, `lib/NGT/ArcFlare/ArcFlareIndex.h:150`, `lib/NGT/ArcFlare/ArcFlareIndex.cpp` (all `raw_flat_` sites), `tests/arcflare/test_l2_fp16.cpp` (new).

### Task 0.1: Add F16C `l2_sq_f32_fp16` + unit test

- [ ] **Step 1: Failing test** — `tests/arcflare/test_l2_fp16.cpp`

```cpp
// tests/arcflare/test_l2_fp16.cpp — fp32-query vs fp16-stored squared-L2 distance.
#include "NGT/ArcFlare/SIMDUtils.h"
#include "NGT/ArcFlare/VectorRecord.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace NGT::ArcFlare;
int main() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (int D : {64, 128, 256, 784, 960, 1024}) {
        std::vector<float> a(D), b(D); std::vector<uint16_t> bh(D);
        for (int i = 0; i < D; ++i){ a[i]=U(rng); b[i]=U(rng); bh[i]=float_to_fp16(b[i]); }
        float ref = 0.f;
        for (int i = 0; i < D; ++i){ float d=a[i]-fp16_to_float(bh[i]); ref+=d*d; }
        float got = l2_sq_f32_fp16(a.data(), bh.data(), D);
        float rel = std::fabs(got-ref)/(ref+1e-9f);
        printf("D=%4d ref=%.6f got=%.6f rel=%.2e\n", D, ref, got, rel);
        assert(rel < 1e-4f && "l2_sq_f32_fp16 mismatch vs scalar reference");
    }
    printf("test_l2_fp16: PASS\n"); return 0;
}
```

- [ ] **Step 2: Add target + run; expect compile failure.** Add `test_l2_fp16` to the non-HDF5 `foreach` in `tests/arcflare/CMakeLists.txt` (same pattern as other `test_*`). Then:

```bash
cd "$BUILD" && cmake . >/dev/null && cmake --build . --target test_l2_fp16 2>&1 | tail -5
# Expected: error: 'l2_sq_f32_fp16' is not a member of 'NGT::ArcFlare'
```

- [ ] **Step 3: Implement.** Add `#include "NGT/ArcFlare/VectorRecord.h"` at the top of `SIMDUtils.h`, then append after `l2_sq` (ends line 64), inside `namespace NGT { namespace ArcFlare {`:

```cpp
// Squared L2 between fp32 query `a` and fp16-packed `x` (dimension D). F16C path.
inline float l2_sq_f32_fp16(const float* __restrict__ a,
                            const uint16_t* __restrict__ x, int D) {
#if defined(__AVX2__) && defined(__F16C__)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= D; i += 16) {
        __m256 xb0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + i)));
        __m256 xb1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + i + 8)));
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),     xb0);
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), xb1);
        s0 = _mm256_fmadd_ps(d0, d0, s0); s1 = _mm256_fmadd_ps(d1, d1, s1);
    }
    for (; i + 8 <= D; i += 8) {
        __m256 xb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + i)));
        __m256 d  = _mm256_sub_ps(_mm256_loadu_ps(a + i), xb);
        s0 = _mm256_fmadd_ps(d, d, s0);
    }
    __m256 acc = _mm256_add_ps(s0, s1);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    float r = _mm_cvtss_f32(s);
    for (; i < D; ++i){ float d=a[i]-fp16_to_float(x[i]); r+=d*d; }
    return r;
#else
    float r = 0.f;
    for (int i = 0; i < D; ++i){ float d=a[i]-fp16_to_float(x[i]); r+=d*d; }
    return r;
#endif
}
```

- [ ] **Step 4: Ensure `-mf16c`.** `grep -rn "mavx2\|march\|mf16c" "$NGT/lib/NGT/CMakeLists.txt" "$NGT/CMakeLists.txt"`. If `-mf16c` absent, add it beside `-mavx2`.

- [ ] **Step 5: Build + run; expect PASS.**

```bash
cd "$BUILD" && cmake --build . --target test_l2_fp16 2>&1 | tail -3 && ./tests/arcflare/test_l2_fp16
# Expected last line: test_l2_fp16: PASS
```

- [ ] **Step 6: Commit.**

```bash
cd "$NGT" && git add lib/NGT/ArcFlare/SIMDUtils.h tests/arcflare/test_l2_fp16.cpp tests/arcflare/CMakeLists.txt
git commit -m "feat(arcflare): add l2_sq_f32_fp16 (F16C) for fp16 exact rerank

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 0.2: Migrate `raw_flat_` to fp16

Sites (verify by snippet): decl `ArcFlareIndex.h:150`; ctor `:35`; v1 search `:173-194`; `insert` `:258-262`; `rebuild` reorder `:321-333`; `save`/`load` `:381-384`; **searchV2 rerank `:1293-1298`**; `fixHoleTombstones` `:1390-1400`; `loadV2`/`rebuildGraphSelf` `:1438-1473`.

- [ ] **Step 1:** `ArcFlareIndex.h:150` → `std::vector<uint16_t> raw_flat_;`. Change `rawFlat()` (`:108`) to `const uint16_t*`. Fix callers: `grep -rn "rawFlat()" "$NGT/tests" "$NGT/lib"` and convert each via `fp16_to_float`.
- [ ] **Step 2:** searchV2 rerank `:1297-1298`:

```cpp
        const uint16_t* vec = raw_flat_.data() + static_cast<size_t>(id) * D;
        float exact_sq = NGT::ArcFlare::l2_sq_f32_fp16(q_ptr, vec, D);
```

- [ ] **Step 3:** v1 `search()` rerank `:183-194` → `l2_sq_f32_fp16(query_ptr, raw_flat_.data()+id*D, D)` (cosine branch unchanged otherwise).
- [ ] **Step 4:** `insert()` `:258-262` — push fp16, normalize via temp fp32:

```cpp
    for (int d = 0; d < D; ++d) raw_flat_.push_back(NGT::ArcFlare::float_to_fp16(vec[d]));
    if (is_angular_) {
        uint16_t* h = raw_flat_.data() + static_cast<size_t>(new_id) * D;
        float n2=0.f; for(int d=0;d<D;++d){ float f=NGT::ArcFlare::fp16_to_float(h[d]); n2+=f*f; }
        if (n2>1e-12f){ float inv=1.f/std::sqrt(n2);
            for(int d=0;d<D;++d) h[d]=NGT::ArcFlare::float_to_fp16(NGT::ArcFlare::fp16_to_float(h[d])*inv); }
    }
```

- [ ] **Step 5:** `rebuild()` reorder `:321-333` — change `new_flat` to `std::vector<uint16_t>`; the element copy works unchanged.
- [ ] **Step 6:** `save()`/`load()` `:381-384` — write/read `uint16_t` count×size; bump the serialization magic/version (`:351`) so old fp32 indices error cleanly.
- [ ] **Step 7:** `fixHoleTombstones` + `loadV2`/`rebuildGraphSelf` reads — replace each `std::vector<float>` row view with a convert loop:

```cpp
    std::vector<float> v(D);
    const uint16_t* h = raw_flat_.data() + i * static_cast<size_t>(D);
    for (int d = 0; d < D; ++d) v[d] = NGT::ArcFlare::fp16_to_float(h[d]);
```

- [ ] **Step 8: Build.** `cd "$BUILD" && cmake --build . --target ngt build_arcflarev2_ann ann_bench -j"$(nproc)" 2>&1 | tail -8` (expect no errors).
- [ ] **Step 9: Commit.**

```bash
cd "$NGT" && git add lib/NGT/ArcFlare/ArcFlareIndex.h lib/NGT/ArcFlare/ArcFlareIndex.cpp
git commit -m "feat(arcflare): store raw_flat_ as fp16; rerank via l2_sq_f32_fp16 (Q1)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 0.3: fp16 recall-regression gate (R3)

- [ ] **Step 1: Build SIFT-128 (fp16) at the known-good config + bench single-thread.**

```bash
cd "$BUILD/tests/arcflare"
./build_arcflarev2_ann "$DATA/sift-128-euclidean.hdf5" "$IDX/arcflare_sift-128-euclidean" l2 2000 64 1.2 100 2>&1 | tail -4
./ann_bench "$IDX/arcflare_sift-128-euclidean" "$DATA/sift-128-euclidean.hdf5" 10 0.20 0.40 3 1 0
```

- [ ] **Step 2: Gate.** fp32 baseline was recall@10 = 0.9895. **Expected: recall ≥ 0.9880.** If < 0.985, STOP: keep only the final `out_n` rerank in fp32 (small side-buffer) or revert Q1. Record the value.

---

## Phase 1 — Competitor baselines: QG (official) + QBG (self-tuned) (Q5, Q7, R4)

### Task 1A: QG with official ANN-Benchmarks params (verbatim)

Official `qg_ngt` pipeline (from `ann_benchmarks/algorithms/qg_ngt/module.py`): ANNG `ngt create` → ONNG `ngt reconstruct-graph` → `qbg create-qg` (numOfSubvectors defaults to **1**, no `-Q`) → `qbg build-qg`. Query: `qbg search-qg -p{result_expansion} -e{epsilon}` where **epsilon = raw − 1.0**.

**Verbatim build groups** (keys: edge, epsilon, indegree, max_edge, outdegree, sample):
```
s2000-e0.02 : edge100 eps0.02 indeg120 max_edge96 outdeg64 sample2000
s4000-e0.02 : edge100 eps0.02 indeg120 max_edge96 outdeg64 sample4000
s4000-e0.04 : edge100 eps0.04 indeg120 max_edge96 outdeg64 sample4000
s4000-e0.06 : edge100 eps0.06 indeg120 max_edge96 outdeg64 sample4000
s4000-e0.08 : edge100 eps0.08 indeg120 max_edge96 outdeg64 sample4000
s20000-e0.1 : edge100 eps0.1  indeg120 max_edge96 outdeg64 sample20000
```
**Verbatim 31-point query grid** `[result_expansion(-p), epsilon_raw]` (pass `-e = epsilon_raw − 1.0`):
```
[0,0.9][0,0.95][0,0.98][0,1.0]
[1.2,0.9][1.5,0.9][2,0.9][3,0.9]  [1.2,0.95][1.5,0.95][2,0.95][3,0.95]
[1.2,0.98][1.5,0.98][2,0.98][3,0.98]
[1.2,1.0][1.5,1.0][2,1.0][3,1.0][5,1.0][10,1.0][20,1.0]
[1.2,1.02][1.5,1.02][2,1.02][3,1.02]  [2,1.04][3,1.04][5,1.04][8,1.04]
```

- [ ] **Step 1: Dump train/query TSV from HDF5** — `benchmarks/results/grid/h5_to_tsv.py`

```python
#!/usr/bin/env python3
import h5py, numpy as np, sys
ds, out_train, out_query = sys.argv[1], sys.argv[2], sys.argv[3]
f = h5py.File(ds, "r")
np.savetxt(out_train, f["train"][:], delimiter="\t", fmt="%.7g")
np.savetxt(out_query, f["test"][:],  delimiter="\t", fmt="%.7g")
print("wrote", out_train, out_query)
```

> GIST TSV is large (~7 GB). Ensure `$IDX` has space; reuse across build groups.

- [ ] **Step 2: Build the QG index per group** — `benchmarks/results/grid/build_qg.sh` (builds the two representative groups `s4000-e0.04` and `s20000-e0.1`; add the other four for the full official Pareto if time permits). For each dataset with dim `d` and `-D{2|E}`:

```bash
#!/usr/bin/env bash
set -euo pipefail
build_qg () { # $1=ds $2=dim $3=Dmetric $4=eps $5=sample $6=tag
  local ds="$1" d="$2" Dm="$3" eps="$4" sample="$5" tag="$6"
  local tsv="$IDX/${ds}_train.tsv" anng="$IDX/qg_${ds}_${tag}_anng" onng="$IDX/qg_${ds}_${tag}"
  "$NGTBIN" create -it -p8 -b500 -ga -of -D"$Dm" -d"$d" -E100 -S40 -e"$eps" -P0 -B30 -T4 "$anng" "$tsv"
  "$NGTBIN" reconstruct-graph -mS -E64 -o64 -i120 "$anng" "$onng"
  "$QBGBIN" create-qg "$onng"
  "$QBGBIN" build-qg -o"$sample" -M6 -ib -I400 -Gz -Pn -E96 "$onng"
  rm -rf "$anng"
}
# usage examples (run for all 5 datasets, both groups):
# build_qg sift-128-euclidean 128 2 0.04 4000 s4000e04
# build_qg sift-128-euclidean 128 2 0.10 20000 s20000e1
# build_qg glove-100-angular 100 E 0.04 4000 s4000e04   ... etc.
```

- [ ] **Step 3: Bench QG.** First verify the repo's `qg_bench` (NGTQG::Index) can load the CLI-built ONNG-QG index:

```bash
cd "$BUILD/tests/arcflare"
./qg_bench "$IDX/qg_sift-128-euclidean_s20000e1" "$DATA/sift-128-euclidean.hdf5" 10 1 l2 2>&1 | head -20
```

  - **If it loads** (prints recall/QPS blocks): extend `qg_bench` to sweep the official `(result_expansion, epsilon)` grid via `NGTQG::SearchQuery::setEpsilon` and result-expansion (add `sq.setResultExpansion(p)` if the API exposes it; else fall back to the CLI path below). Run threads=1 and threads=`nproc`, save `qg_${ds}_t{1,N}.log`.
  - **If it does not load:** time via the CLI with a wrapper that emits the same `recall@k =`/`agg_QPS =` block. `qbg search-qg -n20 -p{p} -e{raw-1.0} <onng> <query.tsv>` prints neighbor IDs + timing; compute recall against HDF5 `neighbors` with a small Python parser (`benchmarks/results/grid/qg_recall.py`) and aggregate QPS = nq / total_search_time. Single-thread is the faithful QG number (ann-benchmarks measures QG single-thread); multi-thread via GNU `parallel`-style query partitioning is optional/supplementary.

- [ ] **Step 4: Commit** the QG drivers + `qg_official_params.md` recording the verbatim params above.

### Task 1B: QBG self-tuned (documented CLI knobs)

QBG has no official ann-benchmarks grid. Tune it genuinely (R4) via its C++ API in the existing `qbg_build.cpp`/`qbg_bench.cpp`. Knob reference (`bin/qbg/README.md`): build `-N numOfSubvectors` (divisor of extended dims), `-C pq4|sq8`, `-P r|R|p|n` (rotation), `-B s:c,...` (recursive clustering); search `-e` graph epsilon, `-B` blob epsilon (default 0), `-N` explored nodes (default 256), `-p` result_expansion (refinement), `-R f|h` (h = fp16 refinement — the fair analog to ArcFlare's fp16 rerank).

- [ ] **Step 1: Set per-dataset `numOfSubvectors`** in `qbg_build.cpp` (`params.creation.numOfSubvectors`), each a divisor of the padded dim: SIFT-128→64, GIST-960→120, FashionMNIST-784→98 (784=8·98) or use extended-dim divisor, GloVe-100→50, NYTimes-256→64. Use `-C pq4`, `-P R`.
- [ ] **Step 2: Extend `qbg_bench.cpp`** so the sweep covers the recall→QPS frontier with `setResultExpansion` and `-R h` refinement in addition to the existing `numOfProbes`/`graphExplorationSize` ramp. Keep the existing `PROBE_GRID`; add an inner `result_expansion ∈ {0,1.2,2,3,5}` loop and run once with refinement float, once with half.
- [ ] **Step 3: Build + bench QBG on all 5 datasets, threads=1 and `nproc`** (driver `run_qbg.sh`, same shape as Phase 2's drivers). Verify each `qbg_${ds}_t1.log` reaches recall ≥ 0.95 (≥0.99 L2); if it caps low, `numOfSubvectors`/clustering was misset — retune (R4).
- [ ] **Step 4: Commit.**

---

## Phase 2 — Full grid: ArcFlare vs max(QBG, QG) (Q4, Q6, Q7)

### Task 2.1: Build ArcFlare (fp16) indices for all 5 datasets

- [ ] **Step 1:** `benchmarks/results/grid/build_arcflare.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$BUILD/tests/arcflare"
for spec in sift-128-euclidean:l2:100 gist-960-euclidean:l2:100 \
            fashion-mnist-784-euclidean:l2:100 \
            glove-100-angular:angular:100 nytimes-256-angular:angular:100; do
  ds="${spec%%:*}"; rest="${spec#*:}"; m="${rest%%:*}"; edge="${rest##*:}"
  echo "=== build ArcFlare $ds ($m, edge=$edge) $(date) ==="
  ./build_arcflarev2_ann "$DATA/$ds.hdf5" "$IDX/arcflare_$ds" "$m" 2000 64 1.2 "$edge" \
    2>&1 | tee "$LOGS/arcflare_build_$ds.log"
done
```

- [ ] **Step 2:** Run in background; verify each log ends `[Done] Saved to:`.

### Task 2.2: Sweep ArcFlare recall-QPS (single + multi thread)

`ann_bench <idx> <hdf5> k gamma_enq gamma_term rerank_factor n_threads n_probe`.

- [ ] **Step 1:** `benchmarks/results/grid/run_arcflare.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$BUILD/tests/arcflare"; NPROC="$(nproc)"
declare -A METRIC=( [sift-128-euclidean]=l2 [gist-960-euclidean]=l2 \
  [fashion-mnist-784-euclidean]=l2 [glove-100-angular]=angular [nytimes-256-angular]=angular )
declare -A GENQ=( [l2]=0.20 [angular]=0.50 )
GT_GRID=(0.02 0.05 0.10 0.15 0.20 0.30 0.40 0.55 0.70 0.90)
for ds in "${!METRIC[@]}"; do
  m="${METRIC[$ds]}"; ge="${GENQ[$m]}"; hdf5="$DATA/$ds.hdf5"; idx="$IDX/arcflare_$ds"
  for T in 1 "$NPROC"; do
    log="$LOGS/arcflare_${ds}_t${T}.log"; : > "$log"
    for gt in "${GT_GRID[@]}"; do
      ./ann_bench "$idx" "$hdf5" 10 "$ge" "$gt" 3 "$T" 0 2>/dev/null | tee -a "$log"; done
    for cfg in "0.70 10 20" "0.90 20 40" "0.90 30 80"; do
      read -r gt rf np <<< "$cfg"
      ./ann_bench "$idx" "$hdf5" 10 "$ge" "$gt" "$rf" "$T" "$np" 2>/dev/null | tee -a "$log"; done
  done
done
```

- [ ] **Step 2:** Run in background; verify ≥13 result blocks per `arcflare_${ds}_t1.log`.

### Task 2.3: Per-cell comparison grid (beat the stronger competitor)

- [ ] **Step 1:** `benchmarks/results/grid/compare_grid.py`:

```python
#!/usr/bin/env python3
# compare_grid.py — ArcFlare vs max(QBG, QG) QPS ratio at fixed recall targets.
# Usage: compare_grid.py <arcflare_log> <qbg_log> <qg_log> [label]
import sys, re
TARGETS = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]

def parse(path):
    pts=[]; rec=qps=None
    try: lines=open(path)
    except FileNotFoundError: return pts
    for line in lines:
        m=re.search(r'recall@\d+\s*=\s*([\d.]+)', line);  rec=float(m.group(1)) if m else rec
        m=re.search(r'agg_QPS\s*=\s*([\d.]+)', line);      qps=float(m.group(1)) if m else qps
        if rec is not None and qps is not None: pts.append((rec,qps)); rec=qps=None
    return pts

def qps_at(pts, t):
    q=[p[1] for p in pts if p[0]>=t]      # best QPS while still meeting recall t
    if q: return max(q)
    s=sorted(pts)                          # else interpolate on the frontier
    for (r0,q0),(r1,q1) in zip(s,s[1:]):
        if r0<=t<=r1 and r1>r0: return q0+(q1-q0)*(t-r0)/(r1-r0)
    return None

def main():
    if len(sys.argv)<4: print("usage: compare_grid.py arcflare qbg qg [label]"); sys.exit(1)
    ng,qb,qg = parse(sys.argv[1]),parse(sys.argv[2]),parse(sys.argv[3])
    label=sys.argv[4] if len(sys.argv)>4 else ""
    ng_max=max((r for r,_ in ng),default=0.0)
    print(f"# {label}  ArcFlare_maxrecall={ng_max:.4f}")
    print(f"{'recall':>7} | {'ArcFlare':>8} | {'QBG':>8} | {'QG':>8} | {'vs max':>7} | verdict")
    print("-"*64)
    for t in TARGETS:
        a=qps_at(ng,t); b=qps_at(qb,t); c=qps_at(qg,t)
        comp=[x for x in (b,c) if x]; mx=max(comp) if comp else None
        if a is None and t>ng_max+1e-9:
            print(f"{t:>7.2f} | {'unreach':>8} | {fmt(b):>8} | {fmt(c):>8} | {'—':>7} | ArcFlare cannot reach"); continue
        ratio=(a/mx) if (a and mx) else None
        v=("PASS>=2x" if ratio and ratio>=2 else "win<2x" if ratio and ratio>=1
           else "LOSS" if ratio else "N/A")
        print(f"{t:>7.2f} | {fmt(a):>8} | {fmt(b):>8} | {fmt(c):>8} | "
              f"{(f'{ratio:.2f}x' if ratio else 'N/A'):>7} | {v}")

def fmt(x): return f"{x:.0f}" if x else "N/A"
if __name__=="__main__": main()
```

- [ ] **Step 2: Single-thread grid (win basis):**

```bash
cd "$LOGS"
for ds in sift-128-euclidean gist-960-euclidean fashion-mnist-784-euclidean \
          glove-100-angular nytimes-256-angular; do
  python3 compare_grid.py "arcflare_${ds}_t1.log" "qbg_${ds}_t1.log" "qg_${ds}_t1.log" "$ds (1T)"
done | tee grid_single_thread.txt
```

- [ ] **Step 3: Commit** the baseline grid + scripts. Every non-`PASS≥2x` cell is a Phase 4 work item.

---

## Phase 3 — Profile the angular path (Q3)

**Files:** `lib/NGT/ArcFlare/ArcFlareIndex.cpp` (`-DAQ_PROFILE`-gated timers in `searchV2`).

### Task 3.1: Add `-DAQ_PROFILE` instrumentation

- [ ] **Step 1:** Add timers attributing per-query µs to the five regions: seeding (`:1046-1152`), DABS loop (`:1154-1253`), refine (`:1255-1263`), expand (`:1265-1285`), rerank (`:1287-1313`).

```cpp
#ifdef AQ_PROFILE
#include <chrono>
struct AqProf { double seed=0,dabs=0,refine=0,expand=0,rerank=0; long n=0;
  ~AqProf(){ if(n) fprintf(stderr,
    "[AQ_PROFILE] n=%ld seed=%.1f dabs=%.1f refine=%.1f expand=%.1f rerank=%.1f us/q\n",
    n,seed/n,dabs/n,refine/n,expand/n,rerank/n); } };
static thread_local AqProf g_aqprof;
#define AQ_T0() auto _t=std::chrono::steady_clock::now()
#define AQ_ADD(f) do{auto _e=std::chrono::steady_clock::now(); \
  g_aqprof.f+=std::chrono::duration<double,std::micro>(_e-_t).count(); _t=_e;}while(0)
#else
#define AQ_T0()
#define AQ_ADD(f)
#endif
```

Place `AQ_T0();` before seeding; `AQ_ADD(seed);…AQ_ADD(dabs);…AQ_ADD(refine);…AQ_ADD(expand);…AQ_ADD(rerank);` at region boundaries; `g_aqprof.n++;` once per query.

- [ ] **Step 2: Profiling binary:** add `target_compile_definitions(ann_bench PRIVATE AQ_PROFILE)` temporarily (or a dedicated `ann_bench_prof`), `cmake --build . --target ann_bench`.

### Task 3.2: Run on NYTimes (single-thread, clean attribution)

```bash
cd "$BUILD/tests/arcflare"
./ann_bench "$IDX/arcflare_nytimes-256-angular" "$DATA/nytimes-256-angular.hdf5" \
  10 0.50 0.90 20 1 40 2>&1 | grep -E "AQ_PROFILE|recall|QPS"
```

- [ ] Record the breakdown in `benchmarks/results/grid/angular_profile.md`.

### Task 3.3: Decision gate

- [ ] Map dominant region → Phase 4A branch: **seeding**→4A-1, **DABS**→4A-2, **refine/expand/rerank**→4A-3. Record the choice.

---

## Phase 4 — Targeted optimization (apply only selected sub-tasks)

### Task 4A-1: Cap angular seeding (if Phase 3.3 = seeding)

Root cause: angular scores **all** members of all `n_probe`(=20) clusters (`take = members.size()`, `ArcFlareIndex.cpp:1084`) with per-cluster LUT rebuilds.

- [ ] **Step 1:** `ArcFlareIndex.cpp:1084-1086`:

```cpp
        const size_t take = std::min(members.size(),
            static_cast<size_t>(is_angular_ ? std::max(N_CLUSTER_SEEDS, 64) : N_CLUSTER_SEEDS));
```

- [ ] **Step 2:** Build; re-bench NYTimes+GloVe (threads=1) over `gt ∈ {0.10..0.90}`, n_probe=20; run `compare_grid.py`. Expect QPS up sharply; if recall regresses, raise cap (96/128).
- [ ] **Step 3:** Commit if the grid improved without recall loss.

### Task 4A-2: Angular tier-2 termination gate (if Phase 3.3 = DABS)

Root cause: angular disables tier-2 termination (`:1199-1201` `break`), forcing large `gamma_term`.

- [ ] **Step 1:** `:1191-1202` — build the tier-2 LUT from the popped node's own cluster residual (ADC already rebuilt at `:1204`; reorder so the gate uses current `q_res_tl`):

```cpp
        if (dk_tracker.size() >= (size_t)k_beam && dist_x > (1.f+gamma_term)*d_k) {
            maybe_rebuild_adc(rec_x.centroid_id());
            NGT::ArcFlare::build_tier2_lut_fast_m(q_res_tl.data(), M_PQ,
                tier2_codebook_T_.data(), t2_lut_tl.data());
            float nx2  = NGT::ArcFlare::fp16_to_float(rec_x.norm_fp16());
            float t2ip = NGT::ArcFlare::tier2_adc_pq_m(t2_lut_tl.data(), rec_x.tier2(), M_PQ);
            float d_t2 = adc.q_norm_sq + nx2*nx2 - 2.0f*t2ip;
            if (d_t2 > (1.f+gamma_term)*d_k) break;
        }
```

- [ ] **Step 2:** Build; re-bench; compare. Expect fewer visits at fixed recall; smaller `gamma_term` usable. **Step 3:** Commit if improved.

### Task 4A-3: Shrink angular refine/expand (if Phase 3.3 = refine/expand/rerank)

- [ ] **Step 1:** `:1259`, `:1271`:

```cpp
    const size_t refine_n = static_cast<size_t>(k_beam * (is_angular_ ? 40 : 100));
    const size_t EXPAND_N = is_angular_ ? 64 : 200;
```

- [ ] **Step 2:** Build; re-bench; compare; watch recall ≥0.95. **Step 3:** Commit if improved.

### Task 4B: L2 low-recall BQ-only fast path (if Phase 2.3 shows L2 not-PASS at recall ≤ 0.80)

Root cause R1/R5: at low recall ArcFlare still pays full rerank+expand while QBG/QG run lean.

- [ ] **Step 1:** Convention `rerank_factor < 0` ⇒ skip exact rerank, return top-`k` by tier-1 ADC. Insert after the DABS loop, before `refine_n` (`:1255`):

```cpp
    if (rerank_factor < 0) {
        const size_t out_n = std::min((size_t)k_out, results.size());
        std::partial_sort(results.begin(), results.begin()+out_n, results.end(),
            [](const auto&a,const auto&b){return a.first<b.first;});
        std::vector<SearchResult> out; out.reserve(out_n);
        for (size_t i=0;i<out_n;++i)
            out.push_back({results[i].second, std::sqrt(std::max(0.f,results[i].first)), results[i].first});
        return out;
    }
```

- [ ] **Step 2:** Sweep SIFT low band with `rerank_factor=-1`, merge with reranked high band, compare:

```bash
cd "$BUILD" && cmake --build . --target ngt ann_bench -j"$(nproc)" 2>&1 | tail -3
cd tests/arcflare; : > "$LOGS/arcflare_sift_bqonly_t1.log"
for gt in 0.02 0.05 0.10 0.15 0.20 0.30; do
  ./ann_bench "$IDX/arcflare_sift-128-euclidean" "$DATA/sift-128-euclidean.hdf5" 10 0.20 "$gt" -1 1 0 \
    2>/dev/null | tee -a "$LOGS/arcflare_sift_bqonly_t1.log"; done
cat "$LOGS/arcflare_sift-128-euclidean_t1.log" "$LOGS/arcflare_sift_bqonly_t1.log" > "$LOGS/arcflare_sift_full_t1.log"
python3 "$LOGS/compare_grid.py" "$LOGS/arcflare_sift_full_t1.log" \
   "$LOGS/qbg_sift-128-euclidean_t1.log" "$LOGS/qg_sift-128-euclidean_t1.log" "SIFT after 4B"
```

- [ ] **Step 3:** If it wins low-recall cells, extend to GIST+FashionMNIST; commit.

---

## Phase 5 — Final grid + honest report (Q6)

### Task 5.1: Regenerate complete grid (single + multi)

- [ ] Re-run `run_arcflare.sh` (optimized binary + BQ-only low band where it wins), `compare_grid.py` for all 5 datasets at threads=1 and `nproc`; save `grid_single_thread.txt`, `grid_multi_thread.txt`.

### Task 5.2: `benchmarks/results/grid/REPORT.md`

- [ ] **Step 1:** Per-cell verdict table (datasets × {0.50…0.99}): ArcFlare, QBG, QG QPS, ratio vs max, verdict. Single-thread primary; multi-thread alongside.
- [ ] **Step 2:** Honest claim: "ArcFlare ≥2× max(QBG,QG) on cells {…}; cells {…} are win<2× / unreached." Memory caveat (Q1): ArcFlare stores fp16 vectors (2 B/dim) + 38 B/record vs QBG PQ-only / QG. Record sizes:

```bash
cd "$IDX" && for d in arcflare_* qbg_* qg_*; do printf "%-40s %s\n" "$d" "$(du -sh "$d"|cut -f1)"; done \
  | tee "$LOGS/index_sizes.txt"
```

- [ ] **Step 3:** Commit `REPORT.md`, `grid_*.txt`, `index_sizes.txt`.
- [ ] **Step 4:** Per Q6, each remaining non-PASS cell re-enters Phase 3 (re-profile at that operating point) → Phase 4 → Phase 5, without blocking passing cells.

---

## Self-Review

**1. Spec coverage:**

| Decision | Task(s) |
|----------|---------|
| Q1 fp16 rerank | Phase 0 |
| Q2 angular in scope | 2.1 (dense build), 3, 4A |
| Q3 profile-first | Phase 3 |
| Q4 single+multi thread | 1A.3, 1B.3, 2.2, 5.1 |
| Q5 official QG params verbatim | 1A (build groups + 31-pt grid inline) |
| Q6 per-cell report + iterate | 2.3, 5.2, 5.4 |
| Q7 beat both QBG & QG | 2.3 `compare_grid.py` ratio vs max(QBG,QG) |
| R1/R5 L2 low/mid recall | 4B |
| R2 angular 0.99 | 4A + Q6 scoping |
| R3 fp16 precision | 0.3 gate |
| R4 baseline rigor | 1A (verbatim), 1B (genuine tuning) |
| 2–10× @ {0.5…0.99}, 5 datasets | 2.3 / 5.2 grid covers exactly these |

**2. Placeholder scan:** QG params are inlined verbatim (build groups + 31-pt grid). QBG `numOfSubvectors` values are concrete per dataset. Phase 1A.3 has a genuine load-vs-CLI verification fork (not a TODO — both branches specified). Phase 4 sub-tasks are profile/grid-gated by design (Q3/Q6) with complete code per branch. No "TBD"/"handle errors"/"similar to" remain.

**3. Type consistency:** `l2_sq_f32_fp16(const float*, const uint16_t*, int)` defined 0.1, used 0.2. `raw_flat_` is `std::vector<uint16_t>` everywhere post-0.2; `rawFlat()`→`const uint16_t*`. `float_to_fp16`/`fp16_to_float` are existing `VectorRecord.h` utils. `rerank_factor<0` (4B) is disjoint from `>1`/`0/1`. `compare_grid.py` reads the `recall@k =`/`agg_QPS =` blocks emitted by `ann_bench` and `qbg_bench` (and the QG bench/wrapper, which must emit the same two lines).

---

Plan saved. Supersedes the tool-building plan `2026-05-31-qg-qbg-baseline-benchmark.md` (its QG/QSG/QBG/ann_bench tools are already built and committed).
