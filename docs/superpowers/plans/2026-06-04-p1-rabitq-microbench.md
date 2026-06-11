# P1 RaBitQ Microbench — Measure-First Decision Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a STANDALONE offline microbench (`tests/arcflare/rabitq_bench.cpp` + `RaBitQ.h`) that answers ONE question on the two *validated* levers from obs-4118 — **does any RaBitQ routing variant beat production gpq4 on (1) per-neighbor footprint ≤32B at preserved end-to-end recall, OR (2) hops-to-iso-recall — on BOTH SIFT-128 and GIST-960** — without modifying the production `searchV2`. It emits a single PROCEED/STOP verdict for P1 RaBitQ integration.

**Architecture:** A new tool reusing existing machinery (SRHT, the VNNI/AVX2 int-dot kernels, PCAProjector, the prebuilt graph via one read-only accessor). It compares 4 routing variants — `gpq4` (production baseline, distance via `idx.gpq4Dist`), `rabitq-1bit`, `rabitq-2bit`, `rabitq-pca` — in two parts: **Part A** (offline footprint table + encode/distance correctness + a clearly-labeled NON-GATING rank-correlation diagnostic) and **Part B** (a *production-faithful* minimal greedy beam over the real graph + exact fp16 rerank, measuring recall-vs-hdf5@10 + hops). The PROCEED/STOP gate reads Part B ONLY.

**Tech Stack:** C++26, HDF5 (via `tests/arcflare/hdf5_io.h`), reuses `lib/NGT/ArcFlare/{SRHT,ADCDistance,PCAProjector,SoAGraph,VectorRecord}.h`. Dev machine is **AVX2-only** (no AVX-512/VNNI); VNNI kernels are compiled behind `#if defined(__AVX512VNNI__)` and the gate numbers (footprint/recall/hops) are path-independent.

---

## ⚠️ Critical correctness invariants (read before any task — these are the fixes 6 adversarial reviews demanded)

1. **NEVER gate on rerank-saturated quantities.** `recall@iso-candidate-budget` and rank-correlation are **DIAGNOSTIC ONLY** (obs-4118: ordering changes give ~0 end-to-end delta). The ONLY decision metric is **recall-vs-hdf5@10** from Part B, read jointly with footprint and hops.
2. **Footprint baseline = 66B/neighbor, NOT 104B.** The production batch path reads only the gpq4 block (`M*8+32` over 16 neighbors = 66B/neighbor at M=128); the 38B/node `VectorRecord` is **dead work on the batch path** (read only in the seed scan, ArcFlareIndex.cpp:2017,2210-2287). Compare RaBitQ per-neighbor vs **66B**. ⇒ SIFT 1-bit (20B) wins; **GIST 1-bit (132B) and 2-bit (262B) are LOSSES**; only `rabitq-pca` (D'≤224 → ≤32B) is a GIST win candidate.
3. **The beam MUST be production-faithful** (else it repeats the saturation mistake): rerank exactly **top-64 by the beam's own routing distance** (`refine_floor=64`, ArcFlareIndex.cpp:2563 — NOT k*100, NOT the whole pool); **no hop cap**; `ef = max(k_beam*2, max_visits)` (ArcFlareIndex.cpp:1721, `k_beam=(rerank_factor>1)?k*rerank_factor:k`); terminate on `has_next()==(cur<size && cur<ef)`; `term_eps`/`term_eps_fp16` gates OFF (default). Sweep `max_visits` to drive `ef` and build a recall-vs-hops curve.
4. **One SHARED seed set across all variants** (score a fixed random 256-node sample ONCE with exact fp16 L2 vs the raw query; reuse the same seed ids for every variant) so only the per-hop routing distance differs. Prefer production IVF cluster-probe seeding if reachable; else scope lever-2 to RELATIVE cross-variant hop ratios.
5. **PCA: `whiten=false` is the DEFAULT.** PCA-project is not orthonormal; `whiten=true` divides by `sqrt(eigenvalue)` → breaks the RaBitQ L2 identity and amplifies noise on low-variance axes. `whiten=false` gives an orthogonal sub-projection that preserves L2 on the top-D' subspace (the RaBitQ identity still holds up to truncation). Offer `--pca-whiten` as a sweep flag, never the default. Label any PCA win as a **"RaBitQ-pipeline (PCA front-end + 1-bit) win,"** not a quantizer win. gpq4 stays on full-D.
6. **Gate on hops, not dist_evals** (gpq4 scores 16 neighbors/block in one pass; RaBitQ scores per-neighbor — dist_evals is incomparable, informational only).
7. **Same single variant must win BOTH datasets** to PROCEED. SIFT=1bit + GIST=pca (two different schemes) is NOT a universal win → STOP or per-dataset scoped note.

## Build & run notes
- Build ONLY specific targets — **NEVER `all`** (`test_vector_record` broken). Mirror the `ann_bench` HDF5 target in `tests/arcflare/CMakeLists.txt` (foreach at ~lines 62-80: `ngt ${HDF5_C_LIBRARIES}`, `${HDF5_INCLUDE_DIRS}`, `cxx_std_26`).
- **Runner recipe (REQUIRED):**
  ```bash
  W=/home/kpango/go/src/github.com/kpango/NGT/.claude/worktrees/arcflare-beat-qbg
  RUN="LD_LIBRARY_PATH=$W/build_arcflare/lib/NGT:/tmp/blas-local/usr/lib/x86_64-linux-gnu/openblas-pthread:/tmp/blas-local/usr/lib/x86_64-linux-gnu/lapack OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1"
  SIFT_IDX=/home/kpango/go/src/github.com/kpango/NGT/data/indices/arcflare_sift-128-euclidean
  GIST_IDX=/home/kpango/go/src/github.com/kpango/NGT/data/indices/arcflare_gist-960-euclidean
  SIFT_H5=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/sift-128-euclidean.hdf5
  GIST_H5=/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/gist-960-euclidean.hdf5
  ```
- clangd diagnostics are false positives (no project `-Ilib`/HDF5 flags); the CMake build is the source of truth.

---

## Verified design reference (the implementer follows this)

### Encode (offline, per DB vector x; tool builds its OWN `SRHT(D,seed)`)
`D = next_pow2(dim)` (SIFT 128; GIST 960→1024, required by SRHT.h:24 assert). `c = mean(train over D)` (zero-pad x to D). Per x: `r = x_pad - c`; `srht.apply(r, rr)` (SRHT.h:37, orthonormal so `‖rr‖=‖r‖`); `nr = ‖rr‖`.
- **1-bit:** `bits[d] = (rr[d] >= 0) ? 1 : 0` packed `ceil(D/8)` B; `factor_x = nr / ((1/√D)·Σ|rr|)`. Store `bits + fp16(nr) + fp16(factor_x)`. `Sb = 2·popcount(bits) − D` recomputed at distance time. **Footprint `ceil(D/8)+4` B** (SIFT 20B, GIST 132B).
- **2-bit:** `s_x = max|rr|/1.5`; `u[d] = clamp(round(rr[d]/s_x), −2, +1)` packed `ceil(D/4)` B; `factor_x = nr·‖u‖ / Σ(u·rr)`. Store `u + fp16(nr,factor_x,s_x)`. `Su = Σu` recomputed. **Footprint `ceil(D/4)+6` B** (SIFT 38B, GIST 262B).

### Distance (per query: `qr = q_pad − c → srht → qrr`; `nq2 = ‖qrr‖²`; int4 grid `lo=min(qrr)`, `delta=(max−lo)/15`, `q_int[d]=clamp(round((qrr[d]−lo)/delta),0,15)`, `S_q=Σq_int`)
- **1-bit:** `dot = Σ q_int[d]·bits[d]`; `t = 2·dot − S_q`; `g = delta·t + lo·Sb`; `IPr = (factor_x/√D)·g`; `L2est = nq2 + nr² − 2·IPr`.
- **2-bit (lo·Su FIX — this term was missing and biased the prior design):** `dot = Σ q_int[d]·u[d]`; `g_u = delta·dot + lo·Su`; `IPr = (factor_x/√D)·g_u`; `L2est = nq2 + nr² − 2·IPr`.

### Kernels (q_int in vpdpbusd **unsigned** slot, data sign-bit/level in **signed** slot)
- 1-bit D=128 reuses real kernels `tier1_adc_avx2` (ADCDistance.h:68-130) / `tier1_adc_scalar` (:24) / `tier1_adc_vnni` (:39-58, the `2·sum_pos − q_sum` identity) under `#if defined(__AVX512VNNI__)`. (`dot_s8_s8` at :263 is int8×int8 — do NOT use for 1-bit; the first design fabricated that dependency.)
- The existing fast kernels are **hardwired to D=128** (`tier1_adc_fast_d`:148 falls back to scalar for D≠128) → **write a new generic-D loop** for GIST D=1024 (D/64 ZMM iters behind the VNNI guard; a D/32-block AVX2 generalization; D mult of 64 ⇒ no tail).
- 2-bit signed-level dot: new scalar (default) + AVX2 (`cvtepi8_epi16`+`madd_epi16`) + guarded VNNI.
- **Path-independent:** the integer dot is exact in every path → footprint/recall/hops are valid on the AVX2 dev machine; only the timing column needs VNNI HW (deploy/GHA, informational).

### Footprint table (per-neighbor, the lever-1 input)
| variant | SIFT (D=128) | GIST (D=1024) | ≤32B? |
|---|---|---|---|
| gpq4 (baseline) | 66B/neighbor | 66B/neighbor | — |
| rabitq-1bit | **20B** ✓ | 132B ✗ (2× loss) | SIFT only |
| rabitq-2bit | 38B ✗ | 262B ✗ | no |
| rabitq-pca (D'=128/192/224/256/384) | (control) | 20/28/**32**/36/52B | GIST candidate at D'≤224 |

### Decision gate (Part B only; same variant, BOTH datasets)
**PROCEED** iff one variant V satisfies, on SIFT-128 AND GIST-960:
- **Lever 1:** per-neighbor footprint ≤ 32B AND `|recall_V@10 − recall_gpq4@10| ≤ 0.005` at iso-hops (same `max_visits`), measured on ≥5000 queries (or the full test set); OR
- **Lever 2:** V reaches gpq4's recall@10≥0.95 operating point using ≤ 0.80× the hops gpq4 needs (≥20% hop cut).

**STOP** if no single variant clears either lever on both datasets. Single end metric = **recall-vs-hdf5@10** (reranked top-10 ∩ hdf5 `neighbors`). Diagnostics (rank-corr, recall@budget, dist_evals) never gate.

### Honest expected outcome (state in the report, do not pre-judge the gate)
Per the footprint math + obs-4118: SIFT 1-bit is the likely footprint win; GIST 1-bit/2-bit lose, so GIST hinges on `rabitq-pca` (D'≤224) holding recall — and an fp32-on-D' control must separate PCA-truncation loss from quantization loss. A realistic result is a **per-dataset split** (SIFT win, GIST conditional) → which under invariant #7 is a STOP-or-scoped verdict, not a universal PROCEED. That is a valid, valuable measure-first answer.

---

## File Structure
- **Create:** `tests/arcflare/rabitq_bench.cpp` (the tool: modes `--selftest --footprint --encode --pca-encode --diag-rank --e2e --gate`), `tests/arcflare/RaBitQ.h` (encode/distance/kernels).
- **Modify:** `lib/NGT/ArcFlare/ArcFlareIndex.h` (ONE read-only accessor `graphForDiag()`), `tests/arcflare/CMakeLists.txt` (add `rabitq_bench` to the HDF5 foreach).
- **Reuse (read-only):** SRHT.h, ADCDistance.h, PCAProjector.h, SoAGraph.h, VectorRecord.h, hdf5_io.h.

---

## Task 0: Diagnostic graph accessor + confirm gpq4 baseline path

**Files:** Modify `lib/NGT/ArcFlare/ArcFlareIndex.h`.

- [ ] **Step 1: Add the read-only accessor** near `rotateForDiag()`/`kmeansForDiag()` (~ArcFlareIndex.h:176), inside the public section:
```cpp
// Diagnostic-only: exposes the loaded graph for the standalone rabitq_bench. Read-only; does NOT touch searchV2.
const ::ArcFlare::SoAGraph* graphForDiag() const { return graph_.get(); }
```
- [ ] **Step 2: Build the ngt lib (NOT `all`)**: `cmake --build build_arcflare --target ngt -j` → clean.
- [ ] **Step 3: Confirm the gpq4 baseline path.** The prebuilt indices are expected to report `hasGPQ4()==true` (ann_bench drives SIFT to 0.9950 via the batch path). The bench will assert `idx.hasGPQ4()` and use `idx.gpq4Dist(...)` (ArcFlareIndex.h:318) as the baseline distance; only self-train a codebook if `false`. (No runtime check needed in this task — Task 8 asserts it.)
- [ ] **Step 4: Commit**
```bash
git add lib/NGT/ArcFlare/ArcFlareIndex.h
git commit -m "feat(arcflare): add read-only graphForDiag() accessor for the standalone rabitq_bench (P1)"
```

## Task 1: `RaBitQ.h` encode/distance + `--selftest`

**Files:** Create `tests/arcflare/RaBitQ.h`; create `tests/arcflare/rabitq_bench.cpp` (with `--selftest` mode).

- [ ] **Step 1: Implement `RaBitQ.h`** — the encode + distance per the design reference (1-bit + 2-bit WITH the `lo·Su` term). Use `NGT::ArcFlare::SRHT` (SRHT.h:21,37) for rotation and `float_to_fp16`/`fp16_to_float` (VectorRecord.h:29,68) for stored scalars. Provide a scalar reference `dot` first (kernels come in Task 2).
- [ ] **Step 2: Write `--selftest`** (D=128 synthetic, no index): assert (a) SRHT round-trip preserves `‖·‖` within 1e-4; (b) 1-bit and 2-bit `L2est` reproduce true-L2 ordering on 1000 random pairs with Spearman > 0.9; (c) the 2-bit `lo·Su` term is present — removing it strictly worsens mean estimate error (assert).
- [ ] **Step 3: Build + run** (after Task 10 wires CMake, or compile standalone): `cmake --build build_arcflare --target rabitq_bench -j && eval "$RUN ./build_arcflare/tests/arcflare/rabitq_bench --selftest"` → all asserts pass.
- [ ] **Step 4: Commit** `test(arcflare): RaBitQ.h encode/distance (1-bit + 2-bit lo·Su) + selftest`.

## Task 2: Integer-dot kernels (scalar + AVX2 generic-D + guarded VNNI)

**Files:** Modify `RaBitQ.h`.

- [ ] **Step 1: Implement** the three paths (q_int unsigned slot / data signed slot): scalar (always), AVX2 generic-D (D/32 blocks, generalizing `tier1_adc_avx2`), VNNI behind `#if defined(__AVX512VNNI__)` (D/64). 1-bit uses the `2·dot − S_q` identity; 2-bit uses the signed-level dot.
- [ ] **Step 2: Path-equivalence unit test** in `--selftest`: the AVX2 (and VNNI if built) integer dot is **byte-identical** to the scalar reference on random inputs at D∈{128,1024}.
- [ ] **Step 3: Build + run `--selftest`** on dev (AVX2) → pass; VNNI path compiles only under the guard.
- [ ] **Step 4: Commit** `feat(arcflare): RaBitQ int-dot kernels (scalar/AVX2 generic-D/guarded VNNI), path-equivalent`.

## Task 3: `--footprint` table

- [ ] **Step 1:** Implement `--footprint` printing the per-neighbor table (gpq4 66B; rabitq-1bit 20/132B; 2bit 38/262B; pca D'∈{128,192,224,256,384}) per dataset, flagging ≤32B. Pure arithmetic, no index.
- [ ] **Step 2: Run** → matches the design table; GIST 1-bit/2-bit flagged as losses, pca D'≤224 as the only GIST ≤32B candidate.
- [ ] **Step 3: Commit** `feat(arcflare): rabitq_bench --footprint (per-neighbor, 66B gpq4 baseline)`.

## Task 4: `--encode` offline pipeline over real data

- [ ] **Step 1:** `--encode <hdf5>`: read `train` (hdf5_io.h:33), pad to D=pow2, build `SRHT(D,seed)` + centroid `c=mean`, encode ALL train vectors into the 1-bit + 2-bit code arrays (RAM), persist to a scratch `.bin`. Verify `nr`/`factor_x` finite, bit/level histograms non-degenerate.
- [ ] **Step 2: Run** on SIFT + GIST → no NaN, sane histograms.
- [ ] **Step 3: Commit** `feat(arcflare): rabitq_bench --encode offline pipeline`.

## Task 5: PCA variant (`whiten=false` default + fp32-D' control)

- [ ] **Step 1:** `--pca-encode <hdf5> --dprime <D'>`: `PCAProjector(D,D',seed,whiten=false)` (PCAProjector.h:16), `fit` on a **100k subsample** (cost note: GIST cov is 1024² doubles), `project` all train (PCAProjector.h:91), encode rabitq-1bit on D'-space (`c'=0`). Sweep D'∈{128,192,224,256,384}. Provide `--pca-whiten` flag (non-default). Report `variance_ratio` cumulative coverage (PCAProjector.h:103) at each D'.
- [ ] **Step 2: fp32-on-D' control:** also store the projected fp32 vectors so Part B can run an EXACT-fp32-on-D' distance (no quantization) — this upper-bounds recall at each D' and separates PCA-truncation loss from 1-bit-quant loss (DIAGNOSTIC).
- [ ] **Step 3: Run** GIST D' sweep → footprint per D' matches table; variance_ratio reported.
- [ ] **Step 4: Commit** `feat(arcflare): rabitq_bench PCA variant (whiten=false default, fp32-D' control)`.

## Task 6: `--diag-rank` (NON-GATING diagnostic)

- [ ] **Step 1:** `--diag-rank <index_dir> <hdf5>`: 200 queries, score gt-neighbors + random sample with `L2est` vs true L2; print Spearman + route-recall@k per variant. Print banner **"DIAGNOSTIC ONLY — NOT A DECISION GATE"** (mirror diag_route_recall.cpp:56-81).
- [ ] **Step 2: Run + Commit** `feat(arcflare): rabitq_bench --diag-rank (non-gating)`.

## Task 7: Production-faithful beam infrastructure

**Files:** Modify `rabitq_bench.cpp`.

- [ ] **Step 1:** Implement the templated greedy beam faithful to the production batch DABS loop (per invariant #3): sorted pool (binary-search insert + cursor rewind, mirror AQLinearPool, SearchContext.h:20-74), bitvector visited, `ef = max(k_beam*2, max_visits)` with `k_beam=k` (rerank_factor=1 default), **no hop cap**, terminate on `has_next()`. Graph via `idx.graphForDiag()` → `g->size()`, `g->getNeighbors(x)`, `g->isTombstone` (SoAGraph.h:159,185,202). Instrument `hops` (popped nodes) + `dist_evals` + the candidate pool.
- [ ] **Step 2: Shared seed set** (invariant #4): score a fixed random 256-node sample ONCE with exact fp16 L2 vs the raw query; reuse the SAME seed ids for every variant. (If production IVF cluster-probe seeding is cleanly reachable via the loaded `v2_kmeans.bin` + centroid_ids, prefer it and document; else document that lever-2 is a RELATIVE cross-variant hop ratio.)
- [ ] **Step 3: Unit-test** the beam on a toy 1000-node graph: terminates, returns sorted candidates, `ef` drives capacity.
- [ ] **Step 4: Commit** `feat(arcflare): rabitq_bench production-faithful beam + shared seeds`.

## Task 8: `--e2e` end-to-end smoke (the gating measurement)

- [ ] **Step 1:** `--e2e <index_dir> <hdf5> [variant] [max_visits...]`: load index, assert `idx.hasGPQ4()`, `g=idx.graphForDiag()`. Run the beam with the per-variant `dist_fn` (rabitq variants use `RaBitQ.h` L2est over the tool's code arrays; **gpq4 uses `idx.gpq4Dist`**). **Rerank exactly the top-64 candidates BY THE BEAM'S ROUTING DISTANCE** (invariant #3, `refine_floor=64`), with exact squared L2 from `idx.rawFlat()` (ArcFlareIndex.h:169, fp16 → `fp16_to_float`) against the **RAW (unrotated)** query; take top-10; `recall-vs-hdf5@10` via `compute_recall_k` (hdf5_io.h:76) against `neighbors` (h5_read_int).
- [ ] **Step 2:** Sweep `max_visits ∈ {50,100,200,400,800}` → recall-vs-hops curve, per variant, on ≥5000 queries for SIFT AND GIST. `term_eps` gates OFF.
- [ ] **Step 3: Run** both datasets; record the per-(dataset×variant×max_visits) table {footprint, recall@10, hops, dist_evals}.
- [ ] **Step 4: Commit** `feat(arcflare): rabitq_bench --e2e production-faithful end-to-end smoke`.

## Task 9: `--gate` evaluation + report

- [ ] **Step 1:** `--gate`: aggregate Part B, evaluate Lever-1 (footprint≤32B AND |Δrecall|≤0.005 at iso-hops, same variant BOTH datasets) and Lever-2 (hops≤0.80× at recall@10≥0.95, same variant BOTH datasets). Gate on **hops + recall@10 only** (dist_evals informational). Emit a single `PROCEED <variant>` or `STOP` line + the per-dataset split note. Print Part A diagnostics under the non-gating banner.
- [ ] **Step 2: Run** the full pipeline on both datasets → a single verdict + the evidence table.
- [ ] **Step 3: Commit** `feat(arcflare): rabitq_bench --gate (un-gameable PROCEED/STOP on validated levers)`.

## Task 10: CMake wiring + dev smoke

- [ ] **Step 1:** Add `rabitq_bench` to `tests/arcflare/CMakeLists.txt` HDF5 foreach (~62-80), mirroring `ann_bench`.
- [ ] **Step 2:** `cmake --build build_arcflare --target rabitq_bench -j` clean; `eval "$RUN ./build_arcflare/tests/arcflare/rabitq_bench --footprint"` + `--selftest` green on dev (AVX2).
- [ ] **Step 3: Commit** `build(arcflare): wire rabitq_bench into the HDF5 test targets`.

---

## Self-Review
- **Invariant coverage:** #1 (no saturated gate) → Tasks 6/9 label diagnostics non-gating; #2 (66B baseline) → Task 3; #3 (faithful beam) → Task 7/8; #4 (shared seeds) → Task 7; #5 (PCA whiten=false + fp32 control) → Task 5; #6 (hops not dist_evals) → Task 9; #7 (same variant both datasets) → Task 9. ✓
- **Placeholder scan:** formulas (encode/distance with lo·Su), kernel mapping, beam params (ef/refine_floor), gate thresholds (≤32B, 0.005, 0.80×) are all concrete with file:line citations.
- **The only production change** is the read-only `graphForDiag()` accessor (Task 0); `searchV2` is untouched.

## Execution Handoff
Subagent-Driven Development (as for P0): fresh subagent per task, controller review for mechanical tasks + `code-reviewer` subagent for the math/kernel/beam tasks (1, 2, 7, 8).
