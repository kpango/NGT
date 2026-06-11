// tests/arcflare/rabitq_bench.cpp
// Offline RaBitQ-style residual-quantizer microbench for ArcFlare.
//
// Standalone tool. Does NOT touch production searchV2. Modes are dispatched on
// argv[1].
//
//   --selftest   synthetic D=128 verification of SRHT round-trip, L2-estimate
//                ordering (Spearman > 0.9 for 1-bit + 2-bit), and that the 2-bit
//                lo*Su term reduces the mean |L2est - trueL2| error.
//   --ktest      integer-dot kernel path-equivalence (AVX2/VNNI vs scalar).
//   --encode <hdf5>
//                offline encode pipeline: pads/centroid/SRHT/encodes the train
//                set into the reusable EncodedDB (RaBitQ1+RaBitQ2 SoA), then
//                verifies finiteness, sign-bit balance, and 2-bit level coverage.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <thread>
#include <queue>
#include <chrono>
#include <cassert>

#include "RaBitQ.h"
#include "RaBitQFastScan.h"
#include "RaBitQGraph.h"
#include "hdf5_io.h"

#include "NGT/ArcFlare/ArcFlareIndex.h"
#include "NGT/ArcFlare/SoAGraph.h"
#include "NGT/ArcFlare/KMeansCentering.h"
#include "NGT/ArcFlare/GlobalPQ4.h"

using namespace ArcFlare;

using NGT::ArcFlare::KMeansCentering;

// ---------------------------------------------------------------------------
// Reusable offline encode pipeline (also used by the Part B end-to-end task).
//
// PRODUCTION-FAITHFUL per-cluster IVF-RaBitQ geometry (NOT a single global
// centroid). The index trains K-means on SRHT-rotated DB vectors and stores a
// per-node centroid_id (ArcFlareIndex.cpp:731-751). Each residual is then SMALL —
// rotated DB vector minus its OWN cluster centroid — which is exactly what makes
// the 1-bit sign code informative. (An IVF-1 single global centroid leaves the
// residual huge → sign uninformative → the prior SIFT 1-bit recall of 0.41.)
//
//   centroids live in SRHT-ROTATED space (kmeans->train(rotated) at build time).
//   Since SRHT is linear, SRHT(x) - centroid[cid] == SRHT(x - raw_centroid[cid]),
//   so the rotated residual is simply:
//       rr = rotate(x_pad) - centroid[cid]
//   where rotate() is the INDEX SRHT (idx.rotateForDiag) and centroid[cid] is the
//   index's rotated-space cluster centroid. No second rotation after subtraction.
//
//   per vector i:  cid = centroid_id(i)  (the index's stored per-node assignment)
//                  rr  = rotate(x_pad) - centroid[cid]
//                  RaBitQ1/RaBitQ2 encode(rr)   + store cid
// ---------------------------------------------------------------------------
struct EncodedDB {
    int N = 0;
    int D = 0;
    int raw_dim = 0;
    std::vector<uint32_t> cids;     // per-node cluster id (= index centroid_id)
    std::vector<RaBitQ1>  code1;    // SoA, indexed by node id
    std::vector<RaBitQ2>  code2;    // SoA, indexed by node id
};

static inline int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Rotate raw (unpadded) vector x[raw_dim] into the index SRHT-rotated space,
// subtract the per-cluster rotated centroid, write rr[D].  rotate(x_pad) is the
// index SRHT (same one used to train the centroids and to route queries).
template <typename RotateFn>
static inline void ivf_rotated_residual(RotateFn&& rotate, const float* x, int raw_dim,
                                        const float* centroid_cid, int D, float* rr) {
    rotate(x, raw_dim, rr);                       // rr = SRHT_index(zero-pad(x))
    for (int d = 0; d < D; ++d) rr[d] -= centroid_cid[d];
}

// Per-cluster IVF encode.  cids[i] is the index's stored centroid_id for node i;
// km holds the rotated-space centroids; rotate() is idx.rotateForDiag.
template <typename RotateFn>
EncodedDB encode_all_ivf(const float* train, int N, int raw_dim, int D,
                         const uint32_t* node_cids, const KMeansCentering* km,
                         RotateFn&& rotate) {
    EncodedDB db;
    db.N = N;
    db.raw_dim = raw_dim;
    db.D = D;
    db.cids.assign(node_cids, node_cids + N);
    db.code1.resize(static_cast<size_t>(N));
    db.code2.resize(static_cast<size_t>(N));

    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int n_threads = std::max(1, std::min(N, (hw > 0) ? hw : 4));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<float> rr(static_cast<size_t>(D), 0.f);
            for (int i = t; i < N; i += n_threads) {
                const float* x = train + static_cast<size_t>(i) * raw_dim;
                const float* cc = km->centroid(db.cids[static_cast<size_t>(i)]);
                ivf_rotated_residual(rotate, x, raw_dim, cc, D, rr.data());
                db.code1[static_cast<size_t>(i)].encode(rr.data(), D);
                db.code2[static_cast<size_t>(i)].encode(rr.data(), D);
            }
        });
    }
    for (auto& th : threads) th.join();
    return db;
}

namespace {

// Spearman rank correlation between two equal-length samples.
double spearman(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = a.size();
    auto ranks = [n](const std::vector<double>& v) {
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) { return v[i] < v[j]; });
        std::vector<double> r(n);
        size_t i = 0;
        while (i < n) {
            size_t j = i;
            while (j + 1 < n && v[idx[j + 1]] == v[idx[i]]) ++j;
            double avg = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
            for (size_t k = i; k <= j; ++k) r[idx[k]] = avg;
            i = j + 1;
        }
        return r;
    };
    std::vector<double> ra = ranks(a), rb = ranks(b);
    double ma = std::accumulate(ra.begin(), ra.end(), 0.0) / n;
    double mb = std::accumulate(rb.begin(), rb.end(), 0.0) / n;
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < n; ++i) {
        double xa = ra[i] - ma, xb = rb[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if (da == 0 || db == 0) return 0.0;
    return num / std::sqrt(da * db);
}

int runSelftest() {
    const int D = 128;
    const uint64_t seed = 0xC0FFEEull;
    SRHT srht(D, seed);
    std::mt19937_64 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);

    // ---- (a) SRHT round-trip: norm preservation ----
    {
        float maxErr = 0.f;
        std::vector<float> v(D), out(D);
        for (int trial = 0; trial < 200; ++trial) {
            float n2 = 0.f;
            for (int d = 0; d < D; ++d) { v[d] = nd(rng); n2 += v[d] * v[d]; }
            srht.apply(v.data(), out.data());
            float o2 = 0.f;
            for (int d = 0; d < D; ++d) o2 += out[d] * out[d];
            float err = std::fabs(std::sqrt(o2) - std::sqrt(n2));
            maxErr = std::max(maxErr, err);
        }
        if (!(maxErr < 1e-4f)) {
            std::printf("FAIL selftest: SRHT round-trip max norm error %.3e >= 1e-4\n", maxErr);
            return 1;
        }
    }

    // ---- shared centroid (random, fixed) ----
    std::vector<float> c(D);
    for (int d = 0; d < D; ++d) c[d] = 0.3f * nd(rng);

    std::vector<float> qraw(D), xraw(D), qrr(D), xrr(D);

    // ---- (b) ordering: Spearman(L2est, trueL2) > 0.9 for 1-bit and 2-bit ----
    // Use the realistic ANN candidate-list regime: a single fixed query, with DB
    // vectors at a spread of distances (query + variable-magnitude perturbation).
    // This is the regime the estimator is actually used in (ranking a graph
    // neighbourhood).  Fully-independent Gaussian pairs are degenerate: the IP
    // term is pure noise relative to the norms, so even an exact 1-bit code caps
    // L2-ranking near 0.88 — a property of 1-bit DB codes, not the formula.
    const int N = 1000;
    std::vector<double> est1, est2, truth;
    est1.reserve(N); est2.reserve(N); truth.reserve(N);

    for (int d = 0; d < D; ++d) qraw[d] = nd(rng);
    rotateResidual(srht, qraw.data(), c.data(), D, qrr.data());
    RaBitQQuery qfixed = RaBitQQuery::prepare(qrr.data(), D);
    std::uniform_real_distribution<float> pert_dist(0.1f, 3.0f);
    for (int i = 0; i < N; ++i) {
        float pert = pert_dist(rng);
        for (int d = 0; d < D; ++d) xraw[d] = qraw[d] + pert * nd(rng);
        double tL2 = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = static_cast<double>(qraw[d]) - static_cast<double>(xraw[d]);
            tL2 += diff * diff;
        }
        truth.push_back(tL2);
        rotateResidual(srht, xraw.data(), c.data(), D, xrr.data());
        RaBitQ1 r1; r1.encode(xrr.data(), D);
        est1.push_back(r1.distance(qfixed));
        RaBitQ2 r2; r2.encode(xrr.data(), D);
        est2.push_back(r2.distance(qfixed));
    }

    double sp1 = spearman(est1, truth);
    double sp2 = spearman(est2, truth);

    // ---- (c) lo*Su matters: 2-bit WITH-term error < WITHOUT-term error ----
    // The lo*Su term corrects the int4 query-quantization offset bias.  Its effect
    // is a *systematic* bias, isolated cleanly on independent Gaussian pairs where
    // it dominates the absolute error (in the high-distance perturbation regime the
    // stochastic IP variance swamps the small offset bias).
    double err2_with = 0.0, err2_without = 0.0;
    const int M = 1000;
    for (int i = 0; i < M; ++i) {
        for (int d = 0; d < D; ++d) { qraw[d] = nd(rng); xraw[d] = nd(rng); }
        double tL2 = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = static_cast<double>(qraw[d]) - static_cast<double>(xraw[d]);
            tL2 += diff * diff;
        }
        rotateResidual(srht, qraw.data(), c.data(), D, qrr.data());
        rotateResidual(srht, xraw.data(), c.data(), D, xrr.data());
        RaBitQQuery q = RaBitQQuery::prepare(qrr.data(), D);
        RaBitQ2 r2; r2.encode(xrr.data(), D);
        float e2with = r2.distanceImpl(q, true);
        float e2without = r2.distanceImpl(q, false);
        err2_with += std::fabs(static_cast<double>(e2with) - tL2);
        err2_without += std::fabs(static_cast<double>(e2without) - tL2);
    }
    err2_with /= M;
    err2_without /= M;

    bool ok = true;
    if (!(sp1 > 0.9)) {
        std::printf("FAIL selftest: 1-bit Spearman %.4f <= 0.9\n", sp1);
        ok = false;
    }
    if (!(sp2 > 0.9)) {
        std::printf("FAIL selftest: 2-bit Spearman %.4f <= 0.9\n", sp2);
        ok = false;
    }
    if (!(err2_with < err2_without)) {
        std::printf("FAIL selftest: lo*Su term did not help (with=%.6g >= without=%.6g)\n",
                    err2_with, err2_without);
        ok = false;
    }

    std::printf("selftest metrics: spearman_1bit=%.4f spearman_2bit=%.4f "
                "err2_with_loSu=%.6g err2_without_loSu=%.6g\n",
                sp1, sp2, err2_with, err2_without);

    if (!ok) return 1;
    std::printf("PASS selftest\n");
    return 0;
}

// ---------------------------------------------------------------------------
// --ktest: integer-dot kernel path-equivalence.
//
// The integer dots are EXACT integer arithmetic, so the fast kernels (AVX2 by
// default; VNNI when compiled) must be byte-identical (==) to the scalar
// reference for both 1-bit and 2-bit, at D=128 and D=1024. Random q_int / packed
// codes over many trials; any mismatch is a hard FAIL.
// ---------------------------------------------------------------------------
int runKtest() {
    std::mt19937_64 rng(0xBADC0DEull);
    const int trials = 5000;
    const int dims[2] = {128, 1024};

    for (int di = 0; di < 2; ++di) {
        const int D = dims[di];
        std::vector<int32_t> q_int(static_cast<size_t>(D));
        std::vector<uint8_t> bits(static_cast<size_t>((D + 7) / 8));
        std::vector<uint8_t> levels(static_cast<size_t>((D + 3) / 4));
        std::uniform_int_distribution<int> q_dist(0, 15);    // int4 query operand
        std::uniform_int_distribution<int> byte_dist(0, 255);

        for (int t = 0; t < trials; ++t) {
            for (int d = 0; d < D; ++d) q_int[d] = q_dist(rng);
            for (auto& b : bits)   b = static_cast<uint8_t>(byte_dist(rng));
            for (auto& l : levels) l = static_cast<uint8_t>(byte_dist(rng));

            // 1-bit
            int64_t s1 = ArcFlare::rabitq_dot1_scalar(q_int.data(), bits.data(), D);
            int64_t a1 = ArcFlare::rabitq_dot1_avx2_or_scalar(q_int.data(), bits.data(), D);
            if (a1 != s1) {
                std::printf("FAIL ktest at D=%d variant=1bit-avx2 (avx2=%lld scalar=%lld trial=%d)\n",
                            D, (long long)a1, (long long)s1, t);
                return 1;
            }
#if defined(__AVX512VNNI__)
            int64_t v1 = ArcFlare::rabitq_dot1_vnni(q_int.data(), bits.data(), D);
            if (v1 != s1) {
                std::printf("FAIL ktest at D=%d variant=1bit-vnni (vnni=%lld scalar=%lld trial=%d)\n",
                            D, (long long)v1, (long long)s1, t);
                return 1;
            }
#endif

            // 2-bit
            int64_t s2 = ArcFlare::rabitq_dot2_scalar(q_int.data(), levels.data(), D);
            int64_t a2 = ArcFlare::rabitq_dot2_avx2_or_scalar(q_int.data(), levels.data(), D);
            if (a2 != s2) {
                std::printf("FAIL ktest at D=%d variant=2bit-avx2 (avx2=%lld scalar=%lld trial=%d)\n",
                            D, (long long)a2, (long long)s2, t);
                return 1;
            }
#if defined(__AVX512VNNI__)
            int64_t v2 = ArcFlare::rabitq_dot2_vnni(q_int.data(), levels.data(), D);
            if (v2 != s2) {
                std::printf("FAIL ktest at D=%d variant=2bit-vnni (vnni=%lld scalar=%lld trial=%d)\n",
                            D, (long long)v2, (long long)s2, t);
                return 1;
            }
#endif
        }
    }

#if defined(__AVX512VNNI__)
    const char* fast = "VNNI+AVX2";
#elif defined(__AVX2__)
    const char* fast = "AVX2";
#else
    const char* fast = "scalar-only";
#endif
    std::printf("ktest: %d trials each at D=128 and D=1024, 1-bit + 2-bit; fast path = %s\n",
                trials, fast);
    std::printf("PASS ktest\n");
    return 0;
}

// ---------------------------------------------------------------------------
// --encode <hdf5>: run encode_all_ivf on the train set and verify the codes.
//
// Checks:
//  - every stored nr/factor_x (1-bit + 2-bit) is finite, nr >= 0, factor_x >= ~1.
//  - 1-bit sign-bit population fraction ≈ 0.5 (mean popcount/D in 0.5 ± 0.05).
//  - 2-bit level histogram covers all of {-2,-1,0,+1} non-trivially.
// ---------------------------------------------------------------------------
int runEncode(const char* hdf5_path) {
    std::printf("[encode] loading train set from %s\n", hdf5_path);
    H5FloatDataset train_ds = h5_read_float(hdf5_path, "train");
    const int N = train_ds.n_rows;
    const int raw_dim = train_ds.n_cols;
    if (N <= 0 || raw_dim <= 0) {
        std::printf("FAIL encode: empty train set (N=%d raw_dim=%d)\n", N, raw_dim);
        return 1;
    }
    std::printf("[encode] N=%d raw_dim=%d -> encoding...\n", N, raw_dim);

    // No on-disk index here, so build a production-faithful IVF ourselves: SRHT-
    // rotate the train set, K-means (select_k clusters) on the rotated vectors,
    // assign each node, then encode per-cluster residuals (rotated x - centroid).
    const int D = next_pow2(raw_dim);
    SRHT srht(D, /*seed=*/0xC0FFEEu);
    auto rotate = [&](const float* x, int rd, float* out) {
        std::vector<float> pad(static_cast<size_t>(D), 0.f);
        std::copy(x, x + std::min(rd, D), pad.begin());
        srht.apply(pad.data(), out);
    };
    std::vector<float> rotated(static_cast<size_t>(N) * D);
    for (int i = 0; i < N; ++i)
        rotate(train_ds.row(i), raw_dim, rotated.data() + static_cast<size_t>(i) * D);
    const uint32_t K = NGT::ArcFlare::select_k(static_cast<uint64_t>(N));
    KMeansCentering km(K, D, /*seed=*/0xC0FFEEu ^ 0xFFFF);
    std::printf("[encode] IVF K-means K=%u on rotated train...\n", K);
    km.train(rotated.data(), static_cast<size_t>(N));
    std::vector<uint32_t> cids(static_cast<size_t>(N));
    km.assign(rotated.data(), static_cast<size_t>(N), cids.data());

    EncodedDB db = encode_all_ivf(train_ds.data.data(), N, raw_dim, D,
                                  cids.data(), &km, rotate);

    // ---- finiteness + range checks; running means + histograms ----
    long long bad_finite = 0;
    long long bad_nr = 0;        // nr < 0 or non-finite
    long long bad_factor = 0;    // factor_x < ~1 (allow tiny fp16 slack) or non-finite
    double sum_nr1 = 0.0, sum_factor1 = 0.0;
    double sum_nr2 = 0.0, sum_factor2 = 0.0;

    // 1-bit sign-bit population: mean popcount / D.
    double sum_pop_frac = 0.0;
    // 2-bit level histogram over {-2,-1,0,+1}; index = u+2 ∈ [0,3].
    long long level_hist[4] = {0, 0, 0, 0};
    const long long total_levels = static_cast<long long>(N) * D;

    const float FACTOR_MIN = 0.99f;  // fp16 round-trip slack below the ideal >= 1

    // -Ofast (fast-math) lets the compiler assume no inf/nan, so std::isfinite on
    // a float is unreliable. Detect non-finite directly on the fp16 bit pattern:
    // exponent bits all 1 (0x7C00) ⇒ inf (mantissa 0) or nan (mantissa != 0).
    auto fp16_nonfinite = [](uint16_t h) { return (h & 0x7C00u) == 0x7C00u; };

    for (int i = 0; i < N; ++i) {
        const RaBitQ1& r1 = db.code1[static_cast<size_t>(i)];
        const RaBitQ2& r2 = db.code2[static_cast<size_t>(i)];

        float nr1 = fp16_to_float(r1.nr_fp16);
        float fx1 = fp16_to_float(r1.factor_fp16);
        float nr2 = fp16_to_float(r2.nr_fp16);
        float fx2 = fp16_to_float(r2.factor_fp16);

        if (fp16_nonfinite(r1.nr_fp16) || fp16_nonfinite(r1.factor_fp16) ||
            fp16_nonfinite(r2.nr_fp16) || fp16_nonfinite(r2.factor_fp16) ||
            fp16_nonfinite(r2.sx_fp16))
            ++bad_finite;
        if (!(nr1 >= 0.f) || !(nr2 >= 0.f)) ++bad_nr;
        if (!(fx1 >= FACTOR_MIN) || !(fx2 >= FACTOR_MIN)) ++bad_factor;

        sum_nr1 += nr1; sum_factor1 += fx1;
        sum_nr2 += nr2; sum_factor2 += fx2;

        // 1-bit: popcount of packed sign bits.
        long long pc = 0;
        const uint8_t* p = r1.bits.data();
        const int nbytes = static_cast<int>(r1.bits.size());
        for (int b = 0; b < nbytes; ++b) pc += __builtin_popcount(static_cast<unsigned>(p[b]));
        sum_pop_frac += static_cast<double>(pc) / static_cast<double>(D);

        // 2-bit: decode each level.
        const uint8_t* lp = r2.levels.data();
        for (int d = 0; d < D; ++d) {
            int u = RaBitQ2::levelAt(lp, d);  // in [-2,+1]
            ++level_hist[u + 2];
        }
    }

    const double mean_nr1 = sum_nr1 / N, mean_factor1 = sum_factor1 / N;
    const double mean_nr2 = sum_nr2 / N, mean_factor2 = sum_factor2 / N;
    const double mean_pop_frac = sum_pop_frac / N;
    double lf[4];
    for (int j = 0; j < 4; ++j) lf[j] = static_cast<double>(level_hist[j]) / static_cast<double>(total_levels);

    std::printf("encode metrics: N=%d D=%d raw_dim=%d\n", N, D, raw_dim);
    std::printf("  1bit: mean_nr=%.6g mean_factor_x=%.6g  sign_pop_frac=%.4f\n",
                mean_nr1, mean_factor1, mean_pop_frac);
    std::printf("  2bit: mean_nr=%.6g mean_factor_x=%.6g\n", mean_nr2, mean_factor2);
    std::printf("  2bit level frac [-2,-1,0,+1] = [%.4f, %.4f, %.4f, %.4f]\n",
                lf[0], lf[1], lf[2], lf[3]);

    bool ok = true;
    if (bad_finite > 0) {
        std::printf("FAIL encode: %lld vectors with non-finite scalars\n", bad_finite);
        ok = false;
    }
    if (bad_nr > 0) {
        std::printf("FAIL encode: %lld vectors with nr < 0\n", bad_nr);
        ok = false;
    }
    if (bad_factor > 0) {
        std::printf("FAIL encode: %lld vectors with factor_x < %.3f\n", bad_factor, FACTOR_MIN);
        ok = false;
    }
    if (!(mean_pop_frac >= 0.45 && mean_pop_frac <= 0.55)) {
        std::printf("FAIL encode: sign-bit population fraction %.4f outside 0.5 +/- 0.05\n",
                    mean_pop_frac);
        ok = false;
    }
    for (int j = 0; j < 4; ++j) {
        if (!(lf[j] > 1e-4)) {
            static const char* names[4] = {"-2", "-1", "0", "+1"};
            std::printf("FAIL encode: 2-bit level %s unused (frac=%.6g)\n", names[j], lf[j]);
            ok = false;
        }
    }

    if (!ok) return 1;
    std::printf("PASS encode\n");
    return 0;
}

// ===========================================================================
// TASK 7+8: production-faithful beam + --e2e end-to-end smoke
// ===========================================================================
//
// The beam mirrors the production AQLinearPool dynamics (SearchContext.h:20-74,
// driven by ArcFlareIndex.cpp:1720-1722) EXACTLY:
//   - ef = max(k_beam*2, max_visits) drives the pool CAPACITY (not a hop cap).
//   - termination is solely has_next() == (cur < size && cur < ef). Hops are the
//     number of nodes popped (emergent), never a separate budget.
//   - sorted pool ascending by routing distance, binary-search insert, cursor
//     rewind when a node is inserted before the cursor.
//   - external bitvector visited set; on pop skip tombstones; expand neighbors,
//     skip visited / tombstone / id>=N, route, insert.
//
// The pool here is a standalone copy of AQLinearPool's semantics. We add a
// "candidate pool snapshot" (every node that ever entered the pool with its
// routing distance) so the e2e rerank can take the top-64 BY ROUTING DISTANCE.

// Standalone re-implementation of the production AQLinearPool semantics
// (SearchContext.h:31-74). Identical insert / pop / has_next behaviour; the only
// addition is instrumentation (hops = #pops, dist_evals counted by the caller's
// DistFn) and a separate flat record of every (id,dist) that ever entered the
// pool (used as the rerank candidate set).
struct BeamPool {
    struct Node { uint32_t id; float dist; };
    std::vector<Node> data_;
    int size_ = 0, cur_ = 0, ef_ = 0, capacity_ = 0;

    static constexpr uint32_t kMask = 0x7fffffffu;
    static inline uint32_t rawid(uint32_t id)    { return id & kMask; }
    static inline bool      checked(uint32_t id) { return (id >> 31) & 1u; }

    void reset(int ef) {
        ef_ = ef;
        capacity_ = ef;
        size_ = cur_ = 0;
        if ((int)data_.size() < capacity_ + 1) data_.resize(capacity_ + 1);
    }
    inline int find_bsearch(float dist) const {
        int lo = 0, hi = size_;
        while (lo < hi) { int mid = (lo + hi) >> 1;
            if (data_[mid].dist > dist) hi = mid; else lo = mid + 1; }
        return lo;
    }
    inline bool insert(uint32_t u, float dist) {
        if (size_ == capacity_ && dist >= data_[size_ - 1].dist) return false;
        int lo = find_bsearch(dist);
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Node));
        data_[lo] = {u, dist};
        if (size_ < capacity_) ++size_;
        if (lo < cur_) cur_ = lo;
        return true;
    }
    inline uint32_t pop() {
        data_[cur_].id |= (1u << 31);
        int pre = cur_;
        while (cur_ < size_ && checked(data_[cur_].id)) ++cur_;
        return rawid(data_[pre].id);
    }
    inline bool has_next() const { return cur_ < size_ && cur_ < ef_; }
};

// Result of one beam run: candidate set {(id, routing_dist)} for rerank + counters.
struct BeamResult {
    std::vector<std::pair<uint32_t, float>> cands;  // every node that entered the pool
    long long hops = 0;        // #pops (emergent, NOT a budget)
    long long dist_evals = 0;  // #DistFn calls
};

// Production-faithful beam walk over the graph.
//   g          : graph (read-only diagnostic view)
//   N          : node count (ids >= N are skipped, matching searchV2)
//   seeds      : initial frontier node ids (shared across variants — see e2e)
//   max_visits : the visit-budget knob; ef = max(k_beam*2, max_visits)
//   k_beam     : k (=10 here, rerank_factor=1)
//   route      : per-neighbor routing distance (the only thing that varies
//                across variants). Each call is counted in dist_evals.
template <typename DistFn>
BeamResult beam_walk(const SoAGraph* g, uint32_t N,
                     const std::vector<uint32_t>& seeds,
                     int max_visits, int k_beam, DistFn&& route) {
    BeamResult res;
    const int ef = std::max(k_beam * 2, max_visits);   // ArcFlareIndex.cpp:1720-1722

    BeamPool pool;
    pool.reset(ef);

    // External visited bitvector (mirrors searchV2 t_vis).
    std::vector<uint64_t> vis((static_cast<size_t>(N) + 63) / 64, 0ull);
    auto is_vis  = [&](uint32_t id) { return (vis[id >> 6] >> (id & 63)) & 1ull; };
    auto set_vis = [&](uint32_t id) { vis[id >> 6] |= (1ull << (id & 63)); };

    auto consider = [&](uint32_t id) {
        if (id >= N || is_vis(id) || g->isTombstone(id)) return;
        set_vis(id);
        ++res.dist_evals;
        float d = route(id);
        if (pool.insert(id, d)) res.cands.emplace_back(id, d);
    };

    for (uint32_t s : seeds) consider(s);

    while (pool.has_next()) {
        uint32_t cur = pool.pop();
        ++res.hops;
        if (g->isTombstone(cur)) continue;
        auto nbrs = g->getNeighbors(cur);
        for (const uint32_t* it = nbrs.begin(); it != nbrs.end(); ++it)
            consider(*it);
    }
    return res;
}

// ---- exact fp16 L2 against the RAW (unrotated) query, via rawFlat ----------
// rawFlat() is fp16-packed [N * stride] (stride = idx.dim(), the property dim,
// which may EXCEED the raw data dim — GIST: data=960, stride=1024 zero-padded).
// decode with fp16_to_float (VectorRecord.h:68). We sum only over q_dim real dims:
// rawFlat's padded tail [q_dim,stride) is 0 and the query has no values there, so
// those terms contribute 0 — summing q_dim dims is exact (and matches the ann-
// benchmarks ground-truth L2 computed over the real data dimensions).
static inline float exact_l2_raw(const uint16_t* raw, int stride, int q_dim,
                                 uint32_t id, const float* q_raw) {
    const uint16_t* p = raw + static_cast<size_t>(id) * stride;
    float acc = 0.f;
    for (int d = 0; d < q_dim; ++d) {
        float xv = fp16_to_float(p[d]);
        float df = xv - q_raw[d];
        acc += df * df;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Self-trained gpq4 codebook (FALLBACK path).
//
// WHY self-trained, not production gpq4Dist: the on-disk SIFT/GIST indices were
// built with meta_version=0 (no v2_gpq4_*.bin sidecars), so loadV2 leaves
// has_gpq4_=false and hasGPQ4()==false. The production single-node gpq4Dist
// reads private gpq4_codes_/gpq4_norm_sq_ that are neither loaded nor externally
// accessible. Per the task's documented fallback, we self-train a K=16 codebook
// (M subspaces) on the SRHT-rotated train residuals — the SAME rotated geometry
// production routes in — and replicate gpq4Dist's exact arithmetic:
//   gpq4Dist(node) = ||q_rot||^2 + recon_norm_sq[node] - 2 * Σ_s ip[s][code[node,s]]
// (ArcFlareIndex.h:321-327), using gpq4_ip_table for the per-query IP table
// (GlobalPQ4.h:185) — i.e. production's IP-LUT form. Footprint stays the analytic
// 66 B/neighbor either way (M 4-bit codes + recon-norm).
struct GPQ4Self {
    int M = 0;
    int D_sub = 0;
    int D = 0;
    std::vector<float>   cb;        // [M][16][D_sub] sub-major, code-major
    std::vector<float>   cb_T;      // [M][D_sub][16] transposed (for gpq4_ip_table)
    std::vector<uint8_t> codes;     // [N*M] per-node 4-bit codes
    std::vector<float>   norm_sq;   // [N] per-node reconstructed-norm^2

    // Train on rotated vectors xr_all [N * D] (already SRHT-rotated residuals),
    // M subspaces of D_sub = D/M dims each, K=16 per subspace.
    void train(const float* xr_all, int N, int Dim, int M_sub) {
        D = Dim;
        M = M_sub;
        D_sub = D / M;
        cb.assign(static_cast<size_t>(M) * NGT::ArcFlare::GPQ4_K * D_sub, 0.f);

        // Per-subspace k-means over the subspace slices. To feed KMeansCentering a
        // contiguous [N][D_sub] view, gather each subspace into a scratch buffer.
        std::vector<float> sub(static_cast<size_t>(N) * D_sub);
        for (int s = 0; s < M; ++s) {
            for (int i = 0; i < N; ++i)
                std::memcpy(sub.data() + static_cast<size_t>(i) * D_sub,
                            xr_all + static_cast<size_t>(i) * D + static_cast<size_t>(s) * D_sub,
                            static_cast<size_t>(D_sub) * sizeof(float));
            NGT::ArcFlare::KMeansCentering km(NGT::ArcFlare::GPQ4_K, D_sub,
                                           0xA5A5u + static_cast<uint64_t>(s));
            km.train(sub.data(), static_cast<size_t>(N));
            for (int k = 0; k < NGT::ArcFlare::GPQ4_K; ++k)
                std::memcpy(cb.data() + (static_cast<size_t>(s) * NGT::ArcFlare::GPQ4_K + k) * D_sub,
                            km.centroid(static_cast<uint32_t>(k)),
                            static_cast<size_t>(D_sub) * sizeof(float));
        }
        cb_T.resize(cb.size());
        NGT::ArcFlare::build_tier2_codebook_T(cb.data(), M, NGT::ArcFlare::GPQ4_K, D_sub, cb_T.data());

        // Encode every node: per-subspace argmin code + accumulate recon-norm^2.
        codes.assign(static_cast<size_t>(N) * M, 0);
        norm_sq.assign(static_cast<size_t>(N), 0.f);
        const int nT = std::max(1, std::min(N, (int)std::thread::hardware_concurrency()));
        std::vector<std::thread> th;
        for (int t = 0; t < nT; ++t) {
            th.emplace_back([&, t]() {
                for (int i = t; i < N; i += nT) {
                    const float* xr = xr_all + static_cast<size_t>(i) * D;
                    float rns = 0.f;
                    uint8_t* cd = codes.data() + static_cast<size_t>(i) * M;
                    for (int s = 0; s < M; ++s) {
                        const float* cbs = cb.data() + (size_t)s * NGT::ArcFlare::GPQ4_K * D_sub;
                        cd[s] = NGT::ArcFlare::gpq4_encode_sub(xr + (size_t)s * D_sub, cbs, D_sub, rns);
                    }
                    norm_sq[static_cast<size_t>(i)] = rns;
                }
            });
        }
        for (auto& x : th) x.join();
    }

    // Per-query IP table (gpq4_ip_table, GlobalPQ4.h:185); ||q_rot||^2 returned.
    float buildQueryIP(const float* q_rot, std::vector<float>& ip_out) const {
        ip_out.resize(static_cast<size_t>(M) * NGT::ArcFlare::GPQ4_K);
        NGT::ArcFlare::gpq4_ip_table(q_rot, M, cb_T.data(), D_sub, ip_out.data());
        float q_ns = 0.f;
        for (int d = 0; d < D; ++d) q_ns += q_rot[d] * q_rot[d];
        return q_ns;
    }

    // Replicates ArcFlareIndex::gpq4Dist (ArcFlareIndex.h:321-327) exactly.
    inline float dist(uint32_t id, const float* ip, float q_ns) const {
        const uint8_t* cd = codes.data() + static_cast<size_t>(id) * M;
        float ipsum = 0.f;
        for (int s = 0; s < M; ++s) ipsum += ip[(size_t)s * NGT::ArcFlare::GPQ4_K + cd[s]];
        return q_ns + norm_sq[static_cast<size_t>(id)] - 2.0f * ipsum;
    }
};

// Take the top-64 candidates BY ROUTING DISTANCE, rerank with exact fp16 L2 vs the
// RAW query, return the top-k node ids.  refine_floor = 64 (NOT the whole pool).
static std::vector<int> rerank_topk(const BeamResult& br,
                                    const uint16_t* raw, int stride, int q_dim,
                                    const float* q_raw, int k, int refine_floor) {
    std::vector<std::pair<uint32_t, float>> cands = br.cands;
    // Partial-sort the candidate pool by routing distance, keep the best refine_floor.
    const int rf = std::min<int>(refine_floor, (int)cands.size());
    std::partial_sort(cands.begin(), cands.begin() + rf, cands.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
    std::vector<std::pair<float, uint32_t>> exact;
    exact.reserve(rf);
    for (int i = 0; i < rf; ++i) {
        uint32_t id = cands[i].first;
        exact.emplace_back(exact_l2_raw(raw, stride, q_dim, id, q_raw), id);
    }
    const int kk = std::min<int>(k, (int)exact.size());
    std::partial_sort(exact.begin(), exact.begin() + kk, exact.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<int> out;
    out.reserve(kk);
    for (int i = 0; i < kk; ++i) out.push_back((int)exact[i].second);
    return out;
}

// ---------------------------------------------------------------------------
// --beamtest: unit test of the beam on a toy 1000-node in-RAM graph.
//
// Graph: a ring lattice (each node connected to +-1..+-4 neighbours) so the walk
// can reach any node. Trivial exact distance = |id - target| (a 1-D embedding).
// Asserts: beam terminates; candidate pool sorted (ascending) when reranked;
// increasing max_visits (=> larger ef) non-decreases recall vs brute-force top-k.
// ---------------------------------------------------------------------------
int runBeamtest() {
    const uint32_t N = 1000;
    // Build a toy CSR ring lattice via SoAGraph public API.
    SoAGraph g(/*words=*/1);
    std::vector<uint64_t> bq(2, 0);  // 2*words_ dummy bq words per node
    for (uint32_t i = 0; i < N; ++i) g.addNode(bq.data());
    g.finalizeCSR();
    // Dense ring lattice (+-1..+-16) so even the smallest ef reaches the target
    // region; a couple of long-range "express" edges keep the diameter small.
    for (uint32_t i = 0; i < N; ++i) {
        std::vector<uint32_t> nbrs;
        for (int off = -16; off <= 16; ++off) {
            if (off == 0) continue;
            long j = (long)i + off;
            if (j >= 0 && j < (long)N) nbrs.push_back((uint32_t)j);
        }
        // long-range express edges (skip-list style) shrink the walk diameter.
        for (int hop : {64, 128, 256}) {
            long a = (long)i + hop, b = (long)i - hop;
            if (a < (long)N) nbrs.push_back((uint32_t)a);
            if (b >= 0)      nbrs.push_back((uint32_t)b);
        }
        g.setNeighbors(i, nbrs);
    }

    const uint32_t target = 723;  // the query "point" in the 1-D embedding
    auto route = [&](uint32_t id) -> float {
        return std::fabs((float)id - (float)target);
    };
    const int k = 10;
    const std::vector<uint32_t> seeds = {0u};  // start far from target

    // Brute-force ground truth: the k ids closest to target.
    std::vector<uint32_t> bf(N);
    std::iota(bf.begin(), bf.end(), 0u);
    std::partial_sort(bf.begin(), bf.begin() + k, bf.end(),
                      [&](uint32_t a, uint32_t b) { return route(a) < route(b); });
    std::vector<int> gt(bf.begin(), bf.begin() + k);

    bool ok = true;
    int prev_pool = -1;
    double prev_recall = -1.0;
    const int sweep[5] = {10, 20, 40, 80, 160};
    for (int mv : sweep) {
        BeamResult br = beam_walk(&g, N, seeds, mv, k, route);

        // (1) terminates: beam_walk returned (no infinite loop) and made progress.
        if (br.hops <= 0) { std::printf("FAIL beamtest: no hops at mv=%d\n", mv); ok = false; }

        // (2) candidate pool sortable ascending by routing distance; verify the
        // sorted order is monotonic and the best candidate is the true nearest.
        std::vector<std::pair<uint32_t,float>> c = br.cands;
        std::sort(c.begin(), c.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
        for (size_t i = 1; i < c.size(); ++i)
            if (c[i].second < c[i-1].second) { std::printf("FAIL beamtest: pool not sortable\n"); ok = false; break; }

        // recall@k vs brute force, from the routing-distance top-k of the pool.
        std::vector<int> topk;
        for (int i = 0; i < k && i < (int)c.size(); ++i) topk.push_back((int)c[i].first);
        int found = 0;
        for (int id : topk) for (int gtid : gt) if (id == gtid) { ++found; break; }
        double recall = (double)found / k;

        // (3) increasing ef non-decreases pool size and recall.
        int pool_sz = (int)br.cands.size();
        if (prev_pool >= 0 && pool_sz < prev_pool) {
            std::printf("FAIL beamtest: pool shrank with larger ef (%d -> %d at mv=%d)\n",
                        prev_pool, pool_sz, mv);
            ok = false;
        }
        if (prev_recall >= 0 && recall < prev_recall - 1e-9) {
            std::printf("FAIL beamtest: recall dropped with larger ef (%.3f -> %.3f at mv=%d)\n",
                        prev_recall, recall, mv);
            ok = false;
        }
        std::printf("  beamtest mv=%-4d ef=%-4d hops=%-5lld pool=%-4d recall@%d=%.3f\n",
                    mv, std::max(k*2, mv), br.hops, pool_sz, k, recall);
        prev_pool = pool_sz;
        prev_recall = recall;
    }

    // Final largest ef must reach full recall (the ring is fully connected to target).
    if (prev_recall < 1.0 - 1e-9) {
        std::printf("FAIL beamtest: largest ef did not reach recall 1.0 (got %.3f)\n", prev_recall);
        ok = false;
    }

    if (!ok) { std::printf("FAIL beamtest\n"); return 1; }
    std::printf("PASS beamtest\n");
    return 0;
}

// ---------------------------------------------------------------------------
// --e2e <index_dir> <hdf5> [max_visits_list]
// ---------------------------------------------------------------------------
int runE2E(const std::string& idx_dir, const std::string& hdf5_path,
           const std::vector<int>& mv_list) {
    std::printf("[e2e] loading index from %s\n", idx_dir.c_str());
    ArcFlare::ArcFlareIndex idx = ArcFlare::ArcFlareIndex::load(idx_dir + "/aqindex");
    idx.loadV2(idx_dir);

    // stride = idx.dim() is the rawFlat per-node width (= the index property dim,
    // possibly > the raw data dim — GIST: data=960, stride=1024 zero-padded).
    const int stride  = idx.dim();
    const int D       = idx.dEff();
    const SoAGraph* g = idx.graphForDiag();
    const uint16_t* raw = idx.rawFlat();
    const KMeansCentering* km = idx.kmeansForDiag();
    if (!g)   { std::printf("FAIL e2e: graphForDiag() null\n"); return 1; }
    if (!raw) { std::printf("FAIL e2e: rawFlat() null\n"); return 1; }
    if (!km)  { std::printf("BLOCKED e2e: kmeansForDiag() null — no IVF centroids\n"); return 1; }
    const uint32_t N = (uint32_t)g->size();
    const uint32_t K = km->num_clusters();
    if (km->dim() != D) {
        std::printf("BLOCKED e2e: kmeans dim %d != D_eff %d\n", km->dim(), D);
        return 1;
    }
    std::printf("[e2e] N=%u rawFlat_stride=%d D_eff=%d hasGPQ4=%d IVF_K=%u\n",
                N, stride, D, (int)idx.hasGPQ4(), K);

    // ---- load train (encode) + test + neighbors ----
    std::printf("[e2e] loading hdf5 %s\n", hdf5_path.c_str());
    H5FloatDataset train_ds = h5_read_float(hdf5_path, "train");
    H5FloatDataset test_ds  = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt        = h5_read_int(hdf5_path, "neighbors");
    const int Ntrain = train_ds.n_rows;
    const int nq     = test_ds.n_rows;
    // q_dim = the real data dimension from the hdf5 (960 for GIST, 128 for SIFT).
    // encode_all_ivf / rotateForDiag / exact L2 against rawFlat all key off this; rawFlat
    // is indexed with `stride` but only the first q_dim dims are non-zero (and match
    // the query), so exact L2 over q_dim dims is exact.
    const int q_dim = test_ds.n_cols;
    if (train_ds.n_cols != q_dim)
        std::printf("[e2e] WARN: train cols (%d) != test cols (%d)\n", train_ds.n_cols, q_dim);
    if (q_dim > stride)
        std::printf("[e2e] WARN: data dim (%d) > rawFlat stride (%d)\n", q_dim, stride);

    // ---- per-node IVF centroid ids (the index's stored assignment) -----------
    // The bench assumes train row i == index node i (the existing db.code1[id]
    // contract, id == graph node id). Read each node's centroid_id from the index
    // record so DB residuals route on the SAME clusters production routes on.
    if (Ntrain != (int)N)
        std::printf("[e2e] WARN: train rows (%d) != index nodes (%u)\n", Ntrain, N);
    std::vector<uint32_t> node_cids(static_cast<size_t>(Ntrain), 0u);
    if (!g->hasV2Records()) {
        std::printf("BLOCKED e2e: graph has no v2 records — cannot read per-node centroid_id\n");
        return 1;
    }
    {
        long long bad_cid = 0;
        for (int i = 0; i < Ntrain && (uint32_t)i < N; ++i) {
            uint32_t cid = g->getRecordConstView((uint32_t)i).centroid_id();
            if (cid >= K) { cid = 0; ++bad_cid; }
            node_cids[static_cast<size_t>(i)] = cid;
        }
        if (bad_cid > 0)
            std::printf("[e2e] WARN: %lld nodes had out-of-range centroid_id (clamped to 0)\n", bad_cid);
    }
    // rotate() = index SRHT (idx.rotateForDiag), the SAME rotation the centroids
    // were trained in and that queries are routed through.
    auto rotate = [&](const float* x, int rd, float* out) { idx.rotateForDiag(x, rd, out); };

    // ---- offline RaBitQ encode of the train set (per-cluster IVF residuals) ----
    std::printf("[e2e] RaBitQ encode_all_ivf (N=%d q_dim=%d, per-cluster K=%u)...\n",
                Ntrain, q_dim, K);
    EncodedDB db = encode_all_ivf(train_ds.data.data(), Ntrain, q_dim, D,
                                  node_cids.data(), km, rotate);

    // ---- gpq4 BASELINE: production gpq4Dist if available, else self-trained ----
    // The production single-node path needs hasGPQ4(); these on-disk indices have
    // meta_version=0 (no gpq4 sidecar) so hasGPQ4()==false → self-train fallback.
    const bool prod_gpq4 = idx.hasGPQ4();
    GPQ4Self gpq4;
    if (!prod_gpq4) {
        // Train the codebook on the index-SRHT-rotated PER-CLUSTER RESIDUALS — the
        // SAME per-cluster geometry production routes in (gpq4 routes on residuals
        // to the node's cluster centroid), keeping the comparison apples-to-apples
        // with RaBitQ. M=128 (task spec), K=16. D_sub = D/M.
        const int M = std::min(128, D);
        std::printf("[e2e] gpq4 self-train (M=%d K=16 D_sub=%d) on per-cluster rotated residuals...\n",
                    M, D / M);
        std::vector<float> xr_all(static_cast<size_t>(Ntrain) * D);
        const int nT = std::max(1, std::min(Ntrain, (int)std::thread::hardware_concurrency()));
        std::vector<std::thread> th;
        for (int t = 0; t < nT; ++t) {
            th.emplace_back([&, t]() {
                std::vector<float> rr(static_cast<size_t>(D));
                for (int i = t; i < Ntrain; i += nT) {
                    const float* cc = km->centroid(node_cids[static_cast<size_t>(i)]);
                    ivf_rotated_residual(rotate, train_ds.row(i), q_dim, cc, D, rr.data());
                    std::copy(rr.begin(), rr.end(), xr_all.begin() + (size_t)i * D);
                }
            });
        }
        for (auto& x : th) x.join();
        gpq4.train(xr_all.data(), Ntrain, D, M);
    } else {
        std::printf("[e2e] gpq4 production path (hasGPQ4=true) — NOT expected for these indices\n");
    }

    const int k = 10;          // k_beam = k, rerank_factor = 1
    const int k_beam = k;
    const int refine_floor = 64;

    // ---- shared seed set: identical across ALL variants -----------------------
    // For each query, score a FIXED random sample of 256 graph nodes with EXACT
    // fp16 L2 vs the RAW query (rawFlat), take the best S=8 as seeds. The same 256
    // sample ids are used for every query AND every variant, so seeds are entirely
    // distance-scheme-independent → hops are comparable across variants (the gate's
    // lever-2 is the CROSS-VARIANT ratio under identical seeds). Absolute hops will
    // NOT match production's IVF/cluster seeding — that's expected and fine.
    const int SAMPLE = 256, S = 8;
    std::vector<uint32_t> sample;
    sample.reserve(SAMPLE);
    {
        std::mt19937_64 rng(0x5EED5EEDull);
        std::vector<uint32_t> all;
        all.reserve(N);
        for (uint32_t i = 0; i < N; ++i) if (!g->isTombstone(i)) all.push_back(i);
        std::shuffle(all.begin(), all.end(), rng);
        for (int i = 0; i < SAMPLE && i < (int)all.size(); ++i) sample.push_back(all[i]);
    }
    auto seeds_for_query = [&](const float* q_raw) {
        std::vector<std::pair<float,uint32_t>> sc;
        sc.reserve(sample.size());
        for (uint32_t id : sample) sc.emplace_back(exact_l2_raw(raw, stride, q_dim, id, q_raw), id);
        const int ss = std::min<int>(S, (int)sc.size());
        std::partial_sort(sc.begin(), sc.begin() + ss, sc.end(),
                          [](const auto& a, const auto& b){ return a.first < b.first; });
        std::vector<uint32_t> seeds;
        for (int i = 0; i < ss; ++i) seeds.push_back(sc[i].second);
        return seeds;
    };

    // ---- ASYMMETRIC IVF-RaBitQ query geometry (per-cluster, lazily cached) -----
    // Every variant rotates the raw query ONCE through the INDEX SRHT into q_rot
    // (idx.rotateForDiag), the same rotation the centroids were trained in. The
    // per-candidate query residual is then q_rot - centroid[cid] where cid is the
    // CANDIDATE's centroid_id (asymmetric IVF: the query is re-residualized against
    // each visited cluster's centroid). A query touches a bounded set of clusters
    // during its walk, so the per-cluster query representation (RaBitQ int4 grid /
    // gpq4 IP table) is built lazily on first touch and cached keyed by cid.
    enum Variant { V_GPQ4 = 0, V_RB1 = 1, V_RB2 = 2, NVAR = 3 };
    const char* vname[NVAR] = {"gpq4", "rabitq-1bit", "rabitq-2bit"};

    // analytic per-neighbor footprint (bytes). gpq4: M codes (4-bit => M/2) + recon
    // norm (~2B) ≈ 66 B at M=128. RaBitQ 1-bit: D/8 + 4 (fp16 nr + fp16 factor).
    // RaBitQ 2-bit: D/4 + 6 (fp16 nr + factor + s_x).
    auto footprint = [&](int v) -> int {
        switch (v) {
            case V_GPQ4: return 66;
            case V_RB1:  return D / 8 + 4;   // SIFT D=128 -> 20 ; GIST D=1024 -> 132
            case V_RB2:  return D / 4 + 6;   // SIFT -> 38 ; GIST -> 262
        }
        return 0;
    };

    std::printf("\n=== e2e results: dataset=%s  N=%u  D_eff=%d  k=%d  refine_floor=%d ===\n",
                hdf5_path.c_str(), N, D, k, refine_floor);
    std::printf("%-12s %-10s %-9s %-12s %-14s %-10s\n",
                "variant", "max_visit", "recall@10", "mean_hops", "mean_dist_ev", "footprint");

    // Per-cluster query caches (one slot per cid, reset between queries via an epoch
    // stamp so we never re-zero K entries). built[cid]==epoch ⇒ slot is valid.
    std::vector<float>        q_rot(static_cast<size_t>(D));     // SRHT(q), shared
    std::vector<float>        q_res(static_cast<size_t>(D));     // scratch q_rot - c[cid]
    std::vector<uint32_t>     cache_epoch(K, 0u);
    uint32_t                  epoch = 0;
    std::vector<RaBitQQuery>  rb_cache(K);                       // RaBitQ int4 grid per cid
    std::vector<std::vector<float>> gpq4_ip_cache(K);            // IP table per cid
    std::vector<float>        gpq4_qns_cache(K, 0.f);            // ||q_res||^2 per cid

    // Build (once per query) q_rot = SRHT(q); bump epoch to invalidate all slots.
    auto begin_query = [&](const float* q_raw) {
        idx.rotateForDiag(q_raw, q_dim, q_rot.data());
        ++epoch;
    };
    // Lazily build + cache the per-cluster query residual against centroid[cid].
    auto ensure_rb = [&](uint32_t cid) -> const RaBitQQuery& {
        if (cache_epoch[cid] != epoch) {
            const float* cc = km->centroid(cid);
            for (int d = 0; d < D; ++d) q_res[d] = q_rot[d] - cc[d];
            rb_cache[cid] = RaBitQQuery::prepare(q_res.data(), D);
            cache_epoch[cid] = epoch;
        }
        return rb_cache[cid];
    };
    auto ensure_gpq4 = [&](uint32_t cid) -> std::pair<const float*, float> {
        if (cache_epoch[cid] != epoch) {
            const float* cc = km->centroid(cid);
            for (int d = 0; d < D; ++d) q_res[d] = q_rot[d] - cc[d];
            gpq4_qns_cache[cid] = gpq4.buildQueryIP(q_res.data(), gpq4_ip_cache[cid]);
            cache_epoch[cid] = epoch;
        }
        return {gpq4_ip_cache[cid].data(), gpq4_qns_cache[cid]};
    };
    auto cid_of = [&](uint32_t id) -> uint32_t {
        uint32_t c = g->getRecordConstView(id).centroid_id();
        return c < K ? c : 0u;
    };

    for (int v = 0; v < NVAR; ++v) {
        for (int mv : mv_list) {
            double sum_hops = 0, sum_ev = 0;
            std::vector<std::vector<int>> results(nq);

            for (int qi = 0; qi < nq; ++qi) {
                const float* q_raw = test_ds.row(qi);
                std::vector<uint32_t> seeds = seeds_for_query(q_raw);
                begin_query(q_raw);

                BeamResult br;
                if (v == V_GPQ4) {
                    br = beam_walk(g, N, seeds, mv, k_beam, [&](uint32_t id) {
                        auto [ip, q_ns] = ensure_gpq4(cid_of(id));
                        return gpq4.dist(id, ip, q_ns);
                    });
                } else if (v == V_RB1) {
                    br = beam_walk(g, N, seeds, mv, k_beam, [&](uint32_t id) {
                        return db.code1[id].distance(ensure_rb(cid_of(id)));
                    });
                } else {
                    br = beam_walk(g, N, seeds, mv, k_beam, [&](uint32_t id) {
                        return db.code2[id].distance(ensure_rb(cid_of(id)));
                    });
                }

                sum_hops += br.hops;
                sum_ev   += br.dist_evals;
                // rerank top-64 BY ROUTING DISTANCE with exact fp16 L2 vs RAW query.
                results[qi] = rerank_topk(br, raw, stride, q_dim, q_raw, k, refine_floor);
            }

            double recall = compute_recall_k(gt, results, k);
            std::printf("%-12s %-10d %-9.4f %-12.1f %-14.1f %d B\n",
                        vname[v], mv, recall, sum_hops / nq, sum_ev / nq, footprint(v));
        }
    }

    std::printf("=== end e2e (gpq4 baseline path: %s) ===\n",
                prod_gpq4 ? "PRODUCTION gpq4Dist"
                          : "SELF-TRAINED gpq4 (M=128,K=16 on per-cluster rotated residuals)");
    return 0;
}

// --fstest: FastScan-over-RaBitQ-bit block kernel + 3-factor estimator validation.
//   (a) AVX2 FastScan block result == scalar logical dot  (HARD kernel gate)
//   (b) Spearman(appro_dist, exact L2)   (c) routing recall@10
// Single global centroid (standalone) -> absolute ordering is centroid-limited (cf. the
// per-cluster centroid-fix finding); kernel correctness is gate (a).
static int runFstest() {
    const char* H5 = "/home/kpango/go/src/github.com/kpango/NGT/data/ann-benchmarks/sift-128-euclidean.hdf5";
    H5FloatDataset tr, te;
    try { tr = h5_read_float(H5, "train"); te = h5_read_float(H5, "test"); }
    catch (const std::exception& e) { std::printf("FAIL fstest: cannot read %s (%s)\n", H5, e.what()); return 1; }
    const int raw_dim = tr.n_cols;
    const int n = std::min(10000, tr.n_rows);
    const int q = std::min(200, te.n_rows);
    std::printf("[fstest] N=%d Q=%d raw_dim=%d\n", n, q, raw_dim);

    rbfs::FSIndex idx(raw_dim, 0xF5C0DEULL);
    idx.encode(tr.data.data(), n);
    std::printf("[fstest] encoded: D=%d num_codebook=%d Npad=%d\n", idx.D, idx.num_codebook, idx.Npad);

    // (a) AVX2 FastScan block result == scalar logical dot, all n vectors, query 0
    long fs_mismatch = 0;
    {
        rbfs::FSIndex::Query q0; idx.query_prepare(te.data.data(), q0);
        std::vector<uint16_t> res(32);
        const size_t bb = (size_t)idx.D * 4;
        for (int blk = 0; blk < idx.Npad; blk += 32) {
            rbfs::accumulate_block(idx.D, idx.blocks.data() + (size_t)(blk / 32) * bb,
                                   q0.lut.data(), res.data());
            for (int j = 0; j < 32; ++j) {
                int iv = blk + j;
                if (iv >= n) break;
                if ((int)res[j] != idx.scalar_fs_result(iv, q0.byte_q.data())) ++fs_mismatch;
            }
        }
    }

    // (b)/(c) over q queries
    auto l2 = [&](const float* a, const float* b) {
        double s = 0; for (int d = 0; d < raw_dim; ++d) { double e = a[d] - b[d]; s += e * e; } return s;
    };
    auto topk = [&](const std::vector<float>& v, int k) {
        std::vector<int> ix(n); std::iota(ix.begin(), ix.end(), 0);
        std::partial_sort(ix.begin(), ix.begin() + k, ix.end(),
                          [&](int x, int y) { return v[x] < v[y]; });
        ix.resize(k); return ix;
    };
    double sp_sum = 0, rec_sum = 0;
    std::vector<float> appro, exact(n);
    for (int qi = 0; qi < q; ++qi) {
        const float* qr = te.data.data() + (size_t)qi * raw_dim;
        rbfs::FSIndex::Query qq; idx.query_prepare(qr, qq);
        idx.estimate(qq, appro);
        for (int i = 0; i < n; ++i) exact[i] = (float)l2(qr, tr.data.data() + (size_t)i * raw_dim);
        int S = std::min(3000, n);
        std::vector<double> ea(S), tx(S);
        for (int i = 0; i < S; ++i) { ea[i] = appro[i]; tx[i] = exact[i]; }
        sp_sum += spearman(ea, tx);
        auto ta = topk(appro, 10), tx10 = topk(exact, 10);
        int hit = 0;
        for (int x : ta) for (int y : tx10) if (x == y) { ++hit; break; }
        rec_sum += hit / 10.0;
    }
    double sp = sp_sum / q, rec = rec_sum / q;
    std::printf("fstest metrics: fs_block_vs_scalar_mismatch=%ld spearman=%.4f routing_recall@10=%.4f\n",
                fs_mismatch, sp, rec);
    if (fs_mismatch != 0) {
        std::printf("FAIL fstest: AVX2 FastScan block != scalar logical dot (%ld mismatches)\n", fs_mismatch);
        return 1;
    }
    std::printf("PASS fstest (FastScan kernel exact vs scalar; spearman=%.4f recall@10=%.4f)\n", sp, rec);
    return 0;
}

// ===========================================================================
// --qgbench: SymphonyQG-class STATIC on-graph index (vertex-relative RaBitQ-
// FastScan routing + implicit-rerank dual-pool beam). The recall@10-vs-QPS
// milestone. Standalone; does NOT touch production searchV2/lib.
//
// Design (ported-adapted from SymphonyQG SIGMOD'25, Apache-2.0):
//   - graph: reuse the prebuilt index adjacency, degree-padded to EXACTLY R=32
//     (FastScan batch alignment): nearest-32 by L2 (+2-hop fill), self-pad short.
//   - vertex-relative encode: for vertex v with neighbors {u}, code(u|v) =
//     sign(SRHT(raw(u)) - SRHT(raw(v))) packed into one 32-vec FastScan block,
//     plus the 3 RaBitQ factors (c = raw(v)). SRHT is linear so the residual is
//     just (u_rot - v_rot) — rotate each raw vector ONCE.
//   - search: dual-pool beam. beam ordered by FastScan ESTIMATE; result pool
//     ordered by EXACT ||q - raw(v)||^2 computed FREE per popped vertex (it is
//     needed by the estimator and the row is in cache). Multi-estimate: a vertex
//     may be pushed many times (different parents); dedup only on pop. The query
//     LUT is built once from q_rot and SHARED across all vertices (the c-dependent
//     term is folded into the factors). top-k = smallest EXACT among visited
//     (implicit rerank) + a 1-hop top-up.
// ===========================================================================
struct QGIndex {
    static constexpr int R = 32;
    int N = 0, D = 0, stride = 0, q_dim = 0, num_codebook = 0;
    size_t block_bytes = 0;
    NGT::ArcFlare::SRHT srht;
    const uint16_t* raw = nullptr;          // rawFlat (fp16), stride per node
    std::vector<uint8_t>  blocks;           // N * block_bytes  (FastScan codes)
    std::vector<float>    trx, fdq, fvq;    // N*R  (3-factor estimator)
    std::vector<uint32_t> nbr;              // N*R  (neighbor ids; self-pad)
    std::vector<float>    vrot;             // N*D  (transient; freed after build)

    QGIndex(int D_, int stride_, int q_dim_, uint64_t seed)
        : D(D_), stride(stride_), q_dim(q_dim_), num_codebook(D_ / 4),
          block_bytes((size_t)D_ * 4), srht(D_, seed) {}

    inline void decode_raw(uint32_t id, float* out) const {
        const uint16_t* p = raw + (size_t)id * stride;
        for (int d = 0; d < D; ++d) out[d] = (d < q_dim) ? fp16_to_float(p[d]) : 0.f;
    }
    inline float l2_rot(uint32_t a, uint32_t b) const {
        const float* x = &vrot[(size_t)a * D];
        const float* y = &vrot[(size_t)b * D];
        float s = 0.f;
        for (int d = 0; d < D; ++d) { float e = x[d] - y[d]; s += e * e; }
        return s;
    }

    size_t bytes_per_vertex() const { return block_bytes + (size_t)R * (3 * 4 + 4); }

    void build(const SoAGraph* g, const uint16_t* raw_, int N_) {
        raw = raw_; N = N_;
        const int hw = std::max(1, (int)std::thread::hardware_concurrency());
        const int nT = std::max(1, std::min(N, hw));

        // 1) rotate every raw vector ONCE (SRHT linearity gives residuals for free).
        vrot.assign((size_t)N * D, 0.f);
        {
            std::vector<std::thread> th;
            for (int t = 0; t < nT; ++t) th.emplace_back([&, t]() {
                std::vector<float> tmp(D);
                for (int i = t; i < N; i += nT) {
                    decode_raw((uint32_t)i, tmp.data());
                    srht.apply(tmp.data(), &vrot[(size_t)i * D]);
                }
            });
            for (auto& x : th) x.join();
        }

        // 2) per-vertex: gather R neighbors, encode vertex-relative, pack block.
        blocks.assign((size_t)N * block_bytes, 0);
        trx.assign((size_t)N * R, 0.f);
        fdq.assign((size_t)N * R, 0.f);
        fvq.assign((size_t)N * R, 0.f);
        nbr.assign((size_t)N * R, 0u);
        const int words = D / 64;
        const float fac_norm = 1.0f / std::sqrt((float)D);
        {
            std::vector<std::thread> th;
            for (int t = 0; t < nT; ++t) th.emplace_back([&, t]() {
                std::vector<uint64_t> bin32((size_t)R * words);
                std::vector<int> bin(D);
                std::vector<std::pair<float, uint32_t>> cand;
                for (int v = t; v < N; v += nT) {
                    // ---- gather neighbors -> exactly R ----
                    cand.clear();
                    auto add = [&](uint32_t u) {
                        if (u >= (uint32_t)N || u == (uint32_t)v || g->isTombstone(u)) return;
                        for (auto& c : cand) if (c.second == u) return;  // dedup (cand small)
                        cand.emplace_back(l2_rot((uint32_t)v, u), u);
                    };
                    auto nb = g->getNeighbors((uint32_t)v);
                    for (const uint32_t* it = nb.begin(); it != nb.end(); ++it) add(*it);
                    // 2-hop fill if short (cap candidate pool ~4R)
                    if ((int)cand.size() < R) {
                        size_t base = cand.size();
                        for (size_t ci = 0; ci < base && (int)cand.size() < 4 * R; ++ci) {
                            auto nb2 = g->getNeighbors(cand[ci].second);
                            for (const uint32_t* it = nb2.begin();
                                 it != nb2.end() && (int)cand.size() < 4 * R; ++it)
                                add(*it);
                        }
                    }
                    int keep = std::min<int>(R, (int)cand.size());
                    std::partial_sort(cand.begin(), cand.begin() + keep, cand.end(),
                                      [](const auto& a, const auto& b) { return a.first < b.first; });
                    uint32_t* nbv = &nbr[(size_t)v * R];
                    for (int j = 0; j < R; ++j)
                        nbv[j] = (j < keep) ? cand[(size_t)j].second : (uint32_t)v;  // self-pad

                    // ---- vertex-relative encode of the R neighbors ----
                    const float* vr = &vrot[(size_t)v * D];
                    std::fill(bin32.begin(), bin32.end(), 0ull);
                    for (int j = 0; j < R; ++j) {
                        const float* ur = &vrot[(size_t)nbv[j] * D];
                        double nr2 = 0, ipabs = 0, ipc = 0;
                        for (int d = 0; d < D; ++d) {
                            float r = ur[d] - vr[d];
                            int b = r > 0.f ? 1 : 0;
                            bin[d] = b;
                            float sgn = 2.f * b - 1.f;
                            nr2 += (double)r * r;
                            ipabs += (double)r * sgn;
                            ipc += (double)vr[d] * sgn;
                        }
                        rbfs::pack_binary(bin.data(), &bin32[(size_t)j * words], D);
                        int popc = 0;
                        for (int w = 0; w < words; ++w)
                            popc += __builtin_popcountll(bin32[(size_t)j * words + w]);
                        float nr = (float)std::sqrt(nr2);
                        float fac_x0 = (float)(ipabs * fac_norm) / (nr > 1e-12f ? nr : 1e-12f);
                        float fac_x1 = (float)(ipc * fac_norm);
                        float x_x0 = nr / (fac_x0 != 0.f ? fac_x0 : 1e-12f);
                        trx[(size_t)v * R + j] = nr * nr + 2.f * x_x0 * fac_x1;
                        fdq[(size_t)v * R + j] = -2.f * x_x0 * fac_norm;
                        fvq[(size_t)v * R + j] = fdq[(size_t)v * R + j] * (float)(2 * popc - D);
                    }
                    rbfs::pack_codes(D, bin32.data(), R, &blocks[(size_t)v * block_bytes]);
                }
            });
            for (auto& x : th) x.join();
        }
        vrot.clear(); vrot.shrink_to_fit();   // not needed at query time
    }

    // per-query shared scratch
    struct Q {
        std::vector<uint8_t> lut, byte_q;
        float width = 0, vl = 0;
        int32_t sumq = 0;
        std::vector<float> q_rot;
    };
    void query_prepare(const float* q_raw, Q& o) const {
        o.q_rot.assign(D, 0.f);
        std::vector<float> pad(D, 0.f);
        std::memcpy(pad.data(), q_raw, (size_t)q_dim * sizeof(float));
        srht.apply(pad.data(), o.q_rot.data());
        float lo, hi; rbfs::data_range(o.q_rot.data(), D, lo, hi);
        o.width = (hi - lo) / (float)((1 << rbfs::QG_BQUERY) - 1);
        if (o.width <= 0.f) o.width = 1e-6f;
        o.vl = lo;
        o.byte_q.assign(D, 0);
        rbfs::quantize_q(o.byte_q.data(), o.q_rot.data(), D, lo, o.width, o.sumq);
        o.lut.assign((size_t)num_codebook * 16, 0);
        rbfs::pack_lut(D, o.byte_q.data(), o.lut.data());
    }

    // implicit-rerank dual-pool beam. ef = result-pool size (quality knob).
    // visited via epoch stamp (O(1) reset). returns top-k node ids (smallest exact).
    void search(const float* q_raw, int ef, int k, const std::vector<uint32_t>& seeds,
                std::vector<uint32_t>& stamp, uint32_t epoch, std::vector<int>& out) const {
        Q q; query_prepare(q_raw, q);
        // beam: min-heap by estimate; result: max-heap by exact, capped at ef.
        using P = std::pair<float, uint32_t>;
        std::priority_queue<P, std::vector<P>, std::greater<P>> beam;
        std::priority_queue<P> result;  // max on top
        uint16_t res32[32];
        auto seen = [&](uint32_t id) { return stamp[id] == epoch; };
        auto mark = [&](uint32_t id) { stamp[id] = epoch; };

        for (uint32_t s : seeds) {
            float ev = exact_l2_raw(raw, stride, q_dim, s, q_raw);
            beam.emplace(ev, s);
        }
        while (!beam.empty()) {
            auto [est_v, v] = beam.top(); beam.pop();
            if ((int)result.size() >= ef && est_v > result.top().first) break;  // converged
            if (seen(v)) continue;
            mark(v);
            float ev = exact_l2_raw(raw, stride, q_dim, v, q_raw);
            result.emplace(ev, v);
            if ((int)result.size() > ef) result.pop();
            rbfs::accumulate_block(D, &blocks[(size_t)v * block_bytes], q.lut.data(), res32);
            const uint32_t* nbv = &nbr[(size_t)v * R];
            const float* tx = &trx[(size_t)v * R];
            const float* dq = &fdq[(size_t)v * R];
            const float* vq = &fvq[(size_t)v * R];
            for (int j = 0; j < R; ++j) {
                uint32_t u = nbv[j];
                if (u == v || seen(u)) continue;            // self-pad / dedup-on-pop
                float fsr = (float)((int)res32[j] * 2 - q.sumq);
                float est = tx[j] + ev + dq[j] * q.width * fsr + vq[j] * q.vl;
                beam.emplace(est, u);
            }
        }
        // 1-hop top-up of the best result vertex (cheap completeness guard).
        // collect result, find best, scan its neighbors exactly.
        std::vector<P> got;
        got.reserve(result.size());
        while (!result.empty()) { got.push_back(result.top()); result.pop(); }
        // got is in descending-exact order; best = back. top-up from best's neighbors.
        if (!got.empty()) {
            uint32_t best = got.back().second;
            const uint32_t* nbv = &nbr[(size_t)best * R];
            for (int j = 0; j < R; ++j) {
                uint32_t u = nbv[j];
                if (u == best) continue;
                float du = exact_l2_raw(raw, stride, q_dim, u, q_raw);
                got.emplace_back(du, u);
            }
        }
        std::sort(got.begin(), got.end(),
                  [](const P& a, const P& b) { return a.first < b.first; });
        out.clear();
        uint32_t last = 0xffffffffu;
        for (auto& p : got) {
            if (p.second == last) continue;  // dedup (top-up may repeat)
            out.push_back((int)p.second);
            last = p.second;
            if ((int)out.size() >= k) break;
        }
    }
};

int runQGBench(const std::string& idx_dir, const std::string& hdf5_path) {
    std::printf("[qgbench] loading index from %s\n", idx_dir.c_str());
    ArcFlare::ArcFlareIndex idx = ArcFlare::ArcFlareIndex::load(idx_dir + "/aqindex");
    idx.loadV2(idx_dir);
    const int stride = idx.dim();
    const int D = idx.dEff();
    const SoAGraph* g = idx.graphForDiag();
    const uint16_t* raw = idx.rawFlat();
    if (!g)   { std::printf("FAIL qgbench: graphForDiag() null\n"); return 1; }
    if (!raw) { std::printf("FAIL qgbench: rawFlat() null\n"); return 1; }
    const int N = (int)g->size();

    std::printf("[qgbench] loading hdf5 %s\n", hdf5_path.c_str());
    H5FloatDataset test_ds = h5_read_float(hdf5_path, "test");
    H5IntDataset   gt       = h5_read_int(hdf5_path, "neighbors");
    const int nq = test_ds.n_rows;
    const int q_dim = test_ds.n_cols;
    std::printf("[qgbench] N=%d rawFlat_stride=%d D_eff=%d q_dim=%d nq=%d\n",
                N, stride, D, q_dim, nq);

    // ---- build the SymphonyQG-class index from the prebuilt graph ----
    auto t0 = std::chrono::steady_clock::now();
    QGIndex qg(D, stride, q_dim, /*seed=*/0x5117A11ull);
    qg.build(g, raw, N);
    double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[qgbench] build done in %.1fs  bytes/vertex=%zu (block=%zu + R*(3f+id))\n",
                build_s, qg.bytes_per_vertex(), qg.block_bytes);

    // ---- shared seed set: nearest-8 of a fixed random 256-node sample (exact L2) --
    const int SAMPLE = 256, S = 8, k = 10;
    std::vector<uint32_t> sample;
    {
        std::mt19937_64 rng(0x5EED5EEDull);
        std::vector<uint32_t> all;
        all.reserve(N);
        for (uint32_t i = 0; i < (uint32_t)N; ++i) if (!g->isTombstone(i)) all.push_back(i);
        std::shuffle(all.begin(), all.end(), rng);
        for (int i = 0; i < SAMPLE && i < (int)all.size(); ++i) sample.push_back(all[i]);
    }
    auto seeds_for_query = [&](const float* q_raw) {
        std::vector<std::pair<float, uint32_t>> sc;
        sc.reserve(sample.size());
        for (uint32_t id : sample) sc.emplace_back(exact_l2_raw(raw, stride, q_dim, id, q_raw), id);
        int ss = std::min<int>(S, (int)sc.size());
        std::partial_sort(sc.begin(), sc.begin() + ss, sc.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<uint32_t> seeds;
        for (int i = 0; i < ss; ++i) seeds.push_back(sc[i].second);
        return seeds;
    };

    std::printf("\n=== qgbench: dataset=%s  N=%d  D_eff=%d  k=%d  R=%d (SymphonyQG-class) ===\n",
                hdf5_path.c_str(), N, D, k, QGIndex::R);
    std::printf("%-6s %-9s %-12s\n", "ef", "recall@10", "QPS(1T)");

    const int efs[] = {16, 32, 48, 64, 96, 128, 192, 256};
    std::vector<uint32_t> stamp((size_t)N, 0u);
    uint32_t epoch = 0;
    for (int ef : efs) {
        std::vector<std::vector<int>> results(nq);
        std::vector<int> out;
        auto ts = std::chrono::steady_clock::now();
        for (int qi = 0; qi < nq; ++qi) {
            const float* q = test_ds.row(qi);
            std::vector<uint32_t> seeds = seeds_for_query(q);
            ++epoch;
            qg.search(q, ef, k, seeds, stamp, epoch, out);
            results[qi] = out;
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        double recall = compute_recall_k(gt, results, k);
        std::printf("%-6d %-9.4f %-12.1f\n", ef, recall, nq / el);
    }
    std::printf("=== end qgbench (build %.1fs, %zu B/vertex) ===\n", build_s, qg.bytes_per_vertex());
    return 0;
}

} // namespace

// --churntest: dynamic insert + IP-DiskANN delete correctness under churn.
// Build n0, +add1, -del, +add2; at each checkpoint recall@10 vs brute-force GT of
// the CURRENT active set; compare final dynamic recall to a from-scratch rebuild.
static double churn_recall(ArcFlare::RaBitQGraph& g, const float* train, int q_dim,
                           const std::vector<uint32_t>& active,
                           const float* Q, int nq, int k, int ef) {
    // GT (parallel over queries): top-k active ids by exact L2.
    std::vector<std::vector<uint32_t>> gt(nq);
    int nT = std::max(1, std::min(nq, (int)std::thread::hardware_concurrency()));
    {
        std::vector<std::thread> th;
        for (int t = 0; t < nT; ++t) th.emplace_back([&, t]() {
            for (int qi = t; qi < nq; qi += nT) {
                const float* q = Q + (size_t)qi * q_dim;
                std::priority_queue<std::pair<float, uint32_t>> heap;  // max on top, size k
                for (uint32_t id : active) {
                    const float* x = train + (size_t)id * q_dim;
                    float s = 0.f;
                    for (int d = 0; d < q_dim; ++d) { float e = x[d] - q[d]; s += e * e; }
                    if ((int)heap.size() < k) heap.emplace(s, id);
                    else if (s < heap.top().first) { heap.pop(); heap.emplace(s, id); }
                }
                auto& v = gt[qi]; v.reserve(k);
                while (!heap.empty()) { v.push_back(heap.top().second); heap.pop(); }
            }
        });
        for (auto& x : th) x.join();
    }
    // dynamic search (serial — epoch/stamp is single-thread state).
    double sum = 0.0;
    std::vector<uint32_t> dyn;
    for (int qi = 0; qi < nq; ++qi) {
        g.search(Q + (size_t)qi * q_dim, k, ef, dyn);
        int hit = 0;
        for (uint32_t a : dyn) for (uint32_t b : gt[qi]) if (a == b) { ++hit; break; }
        sum += (double)hit / (double)k;
    }
    return sum / (double)nq;
}

int runChurnTest(const std::string& hdf5_path) {
    auto tr = h5_read_float(hdf5_path, "train");
    auto te = h5_read_float(hdf5_path, "test");
    const int q_dim = tr.n_cols, Ntr = tr.n_rows;
    const float* train = tr.data.data();
    const float* Q = te.data.data();
    const int nq = std::min(500, te.n_rows), k = 10, ef = 128;
    int n0 = std::min(50000, Ntr / 2), add1 = 30000, del = 20000, add2 = 20000;
    if (n0 + add1 + add2 > Ntr) { add1 = (Ntr - n0) / 2; add2 = add1; del = add1 * 2 / 3; }
    std::printf("[churn] q_dim=%d Ntr=%d nq=%d k=%d ef=%d | build=%d +%d -%d +%d\n",
                q_dim, Ntr, nq, k, ef, n0, add1, del, add2);

    ArcFlare::RaBitQGraph g(q_dim, 12345);
    std::vector<uint8_t> act(Ntr, 0);
    auto activelist = [&]() { std::vector<uint32_t> a; for (uint32_t i = 0; i < (uint32_t)Ntr; ++i) if (act[i]) a.push_back(i); return a; };
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

    // build n0
    auto t0 = now();
    for (int i = 0; i < n0; ++i) { g.insert((uint32_t)i, train + (size_t)i * q_dim); act[i] = 1; }
    double t_build = ms(t0, now());
    double r0 = churn_recall(g, train, q_dim, activelist(), Q, nq, k, ef);
    std::printf("[churn] built %d in %.1fs (%.3f ms/insert)  recall@10=%.4f\n",
                n0, t_build / 1000, t_build / n0, r0);

    // +add1
    t0 = now();
    for (int i = n0; i < n0 + add1; ++i) { g.insert((uint32_t)i, train + (size_t)i * q_dim); act[i] = 1; }
    double t_ins1 = ms(t0, now());
    double r1 = churn_recall(g, train, q_dim, activelist(), Q, nq, k, ef);
    std::printf("[churn] +%d inserts (%.3f ms/op)  active=%d  recall@10=%.4f\n",
                add1, t_ins1 / add1, (int)activelist().size(), r1);

    // -del random
    std::vector<uint32_t> cur = activelist();
    std::mt19937 rng(7);
    std::shuffle(cur.begin(), cur.end(), rng);
    t0 = now();
    for (int i = 0; i < del; ++i) { g.remove(cur[i]); act[cur[i]] = 0; }
    double t_del = ms(t0, now());
    double r2 = churn_recall(g, train, q_dim, activelist(), Q, nq, k, ef);
    std::printf("[churn] -%d deletes (%.3f ms/op)  active=%d  recall@10=%.4f\n",
                del, t_del / del, (int)activelist().size(), r2);

    // +add2 (fresh ids)
    int base = n0 + add1;
    t0 = now();
    for (int i = base; i < base + add2; ++i) { g.insert((uint32_t)i, train + (size_t)i * q_dim); act[i] = 1; }
    double t_ins2 = ms(t0, now());
    double r3 = churn_recall(g, train, q_dim, activelist(), Q, nq, k, ef);
    std::printf("[churn] +%d inserts (%.3f ms/op)  active=%d  recall@10=%.4f\n",
                add2, t_ins2 / add2, (int)activelist().size(), r3);

    // static rebuild reference on the SAME final active set
    std::vector<uint32_t> fin = activelist();
    ArcFlare::RaBitQGraph gs(q_dim, 12345);
    std::vector<float> sdata((size_t)fin.size() * q_dim);
    for (size_t i = 0; i < fin.size(); ++i)
        std::memcpy(&sdata[i * q_dim], train + (size_t)fin[i] * q_dim, (size_t)q_dim * sizeof(float));
    t0 = now();
    gs.build(sdata.data(), (int)fin.size(), fin.data());
    double t_static = ms(t0, now());
    double rs = churn_recall(gs, train, q_dim, fin, Q, nq, k, ef);
    std::printf("[churn] static-rebuild (%d active, %.1fs)  recall@10=%.4f\n",
                (int)fin.size(), t_static / 1000, rs);

    std::printf("\n=== churntest trajectory recall@10: build=%.4f +ins=%.4f -del=%.4f +ins=%.4f | static-ref=%.4f ===\n",
                r0, r1, r2, r3, rs);
    bool pass = (r3 >= rs - 0.05) && (r3 >= 0.80);
    std::printf("%s churntest (dynamic final %.4f vs static %.4f; gate: dynamic >= static-0.05 AND >=0.80)\n",
                pass ? "PASS" : "FAIL", r3, rs);
    return pass ? 0 : 1;
}

static void dyn_l2norm(float* v, int d);  // forward decl (defined later)

// --build-save: build RaBitQGraph and save to disk (full-core build, decoupled from bench).
static int runBuildSave(const std::string& hdf5_path, int Ncap, bool angular,
                        int Rdeg, const std::string& save_path) {
    auto tr = h5_read_float(hdf5_path, "train");
    const int q_dim = tr.n_cols;
    const int N  = (Ncap > 0) ? std::min(Ncap, tr.n_rows) : tr.n_rows;
    std::vector<float> train((size_t)N * q_dim);
    std::memcpy(train.data(), tr.data.data(), (size_t)N * q_dim * sizeof(float));
    if (angular) {
        for (int i = 0; i < N; ++i) dyn_l2norm(&train[(size_t)i * q_dim], q_dim);
    }
    ArcFlare::RaBitQGraph g(q_dim, 12345, Rdeg, angular);
    std::printf("[build-save] %s q_dim=%d N=%d angular=%d R=%d\n",
                hdf5_path.c_str(), q_dim, N, (int)angular, g.R);
    std::vector<uint32_t> ids(N); for (int i = 0; i < N; ++i) ids[i] = (uint32_t)i;
    auto t0 = std::chrono::steady_clock::now();
    g.build_parallel(train.data(), N, ids.data());
    double t_build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[build-save] built %d in %.1fs (%.3f ms/insert)\n", N, t_build, t_build * 1000 / N);
    g.save(save_path);
    return 0;
}

// --bench-pure: load saved index, sweep ef/rerank_factor for pure-routing + final-rerank.
static int runBenchPureLoad(const std::string& idx_path, const std::string& hdf5_path) {
    auto te = h5_read_float(hdf5_path, "test");
    ArcFlare::RaBitQGraph g = ArcFlare::RaBitQGraph::load(idx_path);
    const int q_dim = g.q_dim, N = g.nalive, k = 10;
    const int nq = std::min(1000, te.n_rows);
    // Check if angular (query norms all 1.0 → angular mode was used at build)
    bool angular = false;  // user should pass --angular flag if needed
    std::vector<float> Q((size_t)nq * q_dim);
    std::memcpy(Q.data(), te.data.data(), (size_t)nq * q_dim * sizeof(float));
    // Load GT from HDF5
    std::vector<std::vector<uint32_t>> gt(nq);
    H5IntDataset gtds = h5_read_int(hdf5_path, "neighbors");
    for (int qi = 0; qi < nq; ++qi) {
        auto& v = gt[qi]; v.reserve(k);
        for (int j = 0; j < k && j < gtds.n_cols; ++j)
            v.push_back((uint32_t)gtds.data[(size_t)qi * gtds.n_cols + j]);
    }
    std::printf("[bench-pure] idx=%s N=%d R=%d q_dim=%d nq=%d\n",
                idx_path.c_str(), N, g.R, q_dim, nq);
    int efs[] = {16, 32, 64, 96, 128, 192, 256, 384, 512};
    int rfs[] = {1, 2, 3, 5, 10};
    std::vector<uint32_t> res;
    for (int rf : rfs) {
        std::printf("--- rerank_factor=%d ---\nef     recall@10 QPS(1T)\n", rf);
        for (int ef : efs) {
            if (ef < k) continue;
            auto s0 = std::chrono::steady_clock::now();
            double sum = 0;
            for (int qi = 0; qi < nq; ++qi) {
                g.search_pure(&Q[(size_t)qi * q_dim], k, ef, rf, res);
                int hit = 0; for (uint32_t a : res) for (uint32_t b : gt[qi]) if (a == b) { ++hit; break; }
                sum += (double)hit / k;
            }
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
            std::printf("%-6d %-9.4f %-11.1f\n", ef, sum / nq, nq / el);
        }
    }
    return 0;
}

// --bench-beam: load saved index, sweep ef for regular beam (implicit-rerank).
static int runBenchBeamLoad(const std::string& idx_path, const std::string& hdf5_path) {
    auto te = h5_read_float(hdf5_path, "test");
    ArcFlare::RaBitQGraph g = ArcFlare::RaBitQGraph::load(idx_path);
    const int q_dim = g.q_dim, N = g.nalive, k = 10;
    const int nq = std::min(1000, te.n_rows);
    std::vector<float> Q((size_t)nq * q_dim);
    std::memcpy(Q.data(), te.data.data(), (size_t)nq * q_dim * sizeof(float));
    std::vector<std::vector<uint32_t>> gt(nq);
    H5IntDataset gtds = h5_read_int(hdf5_path, "neighbors");
    for (int qi = 0; qi < nq; ++qi) {
        auto& v = gt[qi]; v.reserve(k);
        for (int j = 0; j < k && j < gtds.n_cols; ++j)
            v.push_back((uint32_t)gtds.data[(size_t)qi * gtds.n_cols + j]);
    }
    std::printf("[bench-beam] idx=%s N=%d R=%d q_dim=%d nq=%d\n",
                idx_path.c_str(), N, g.R, q_dim, nq);
    int efs[] = {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
    std::vector<uint32_t> res;
    std::printf("ef     recall@10 QPS(1T)\n");
    for (int ef : efs) {
        if (ef < k) continue;
        auto s0 = std::chrono::steady_clock::now();
        double sum = 0;
        for (int qi = 0; qi < nq; ++qi) {
            g.search(&Q[(size_t)qi * q_dim], k, ef, res);
            int hit = 0; for (uint32_t a : res) for (uint32_t b : gt[qi]) if (a == b) { ++hit; break; }
            sum += (double)hit / k;
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        std::printf("%-6d %-9.4f %-11.1f\n", ef, sum / nq, nq / el);
    }
    return 0;
}

// --dynbench: build the proper incremental-Vamana RaBitQGraph at scale + sweep ef.
// Answers the high-recall question for GIST/glove where the reused-graph --qgbench
// plateaued (~0.90). recall@10 is vs brute-force GT of the N built vectors (subset
// if N<full). Angular datasets: L2-normalize train+test so L2 ranks as cosine.
static void dyn_l2norm(float* v, int d) {
    double s = 0; for (int i = 0; i < d; ++i) s += (double)v[i] * v[i];
    if (s > 1e-30) { float inv = (float)(1.0 / std::sqrt(s)); for (int i = 0; i < d; ++i) v[i] *= inv; }
}
static int runDynBench(const std::string& hdf5_path, int Ncap, bool angular, int Rdeg = 32) {
    auto tr = h5_read_float(hdf5_path, "train");
    auto te = h5_read_float(hdf5_path, "test");
    const int q_dim = tr.n_cols;
    const int N  = (Ncap > 0) ? std::min(Ncap, tr.n_rows) : tr.n_rows;
    const int nq = std::min(1000, te.n_rows), k = 10;
    std::vector<float> train((size_t)N * q_dim), Q((size_t)nq * q_dim);
    std::memcpy(train.data(), tr.data.data(), (size_t)N * q_dim * sizeof(float));
    std::memcpy(Q.data(), te.data.data(), (size_t)nq * q_dim * sizeof(float));
    if (angular) {
        for (int i = 0; i < N;  ++i) dyn_l2norm(&train[(size_t)i * q_dim], q_dim);
        for (int i = 0; i < nq; ++i) dyn_l2norm(&Q[(size_t)i * q_dim], q_dim);
    }
    ArcFlare::RaBitQGraph g(q_dim, 12345, Rdeg, angular);
    std::printf("[dynbench] %s q_dim=%d N=%d nq=%d k=%d angular=%d R=%d\n",
                hdf5_path.c_str(), q_dim, N, nq, k, (int)angular, g.R);
    std::vector<uint32_t> ids(N); for (int i = 0; i < N; ++i) ids[i] = (uint32_t)i;
    auto t0 = std::chrono::steady_clock::now();
    {
        int l_build = angular ? 256 : 128;
        int n_iters = angular ? 5  : 3;
        g.build_parallel(train.data(), N, ids.data(), n_iters, l_build);
    }
    if (N >= 100000) g.build_seeds();  // greedy K-center seeds for both Euclidean and angular
    double t_build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Ground truth top-k. At full-N use the hdf5 "neighbors" (exact GT for the full
    // train set — avoids the O(N*nq*D) brute-force that is slow + memory-heavy at 1M).
    // Subset N: brute-force over the built set (the hdf5 GT is only valid for full-N).
    std::vector<std::vector<uint32_t>> gt(nq);
    if (N >= tr.n_rows) {
        H5IntDataset gtds = h5_read_int(hdf5_path, "neighbors");
        for (int qi = 0; qi < nq; ++qi) {
            auto& v = gt[qi]; v.reserve(k);
            for (int j = 0; j < k && j < gtds.n_cols; ++j)
                v.push_back((uint32_t)gtds.data[(size_t)qi * gtds.n_cols + j]);
        }
    } else {
        int nT = std::max(1, (int)std::thread::hardware_concurrency());
        std::vector<std::thread> th;
        for (int t = 0; t < nT; ++t) th.emplace_back([&, t]() {
            for (int qi = t; qi < nq; qi += nT) {
                const float* q = &Q[(size_t)qi * q_dim];
                std::priority_queue<std::pair<float, uint32_t>> heap;
                for (int id = 0; id < N; ++id) {
                    const float* x = &train[(size_t)id * q_dim];
                    float s = 0.f; for (int d = 0; d < q_dim; ++d) { float e = x[d] - q[d]; s += e * e; }
                    if ((int)heap.size() < k) heap.emplace(s, (uint32_t)id);
                    else if (s < heap.top().first) { heap.pop(); heap.emplace(s, (uint32_t)id); }
                }
                auto& v = gt[qi]; while (!heap.empty()) { v.push_back(heap.top().second); heap.pop(); }
            }
        });
        for (auto& x : th) x.join();
    }
    std::printf("[dynbench] build %d in %.1fs (%.3f ms/insert)\n", N, t_build, t_build * 1000 / N);
    std::printf("ef     recall@10 QPS(1T)\n");
    int efs[] = {16, 32, 64, 96, 128, 192, 256, 384, 512};
    std::vector<uint32_t> res;
    for (int ef : efs) {
        if (ef < k) continue;
        auto s0 = std::chrono::steady_clock::now();
        double sum = 0;
        for (int qi = 0; qi < nq; ++qi) {
            g.search(&Q[(size_t)qi * q_dim], k, ef, res);
            int hit = 0; for (uint32_t a : res) for (uint32_t b : gt[qi]) if (a == b) { ++hit; break; }
            sum += (double)hit / k;
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        std::printf("%-6d %-9.4f %-11.1f\n", ef, sum / nq, nq / el);
    }
    std::printf("=== dynbench done (build %.1fs, %zu B/vertex) ===\n", t_build, g.bytes_per_vertex());
    return 0;
}

// --dynbench-pure: same build as --dynbench, but search via the PURE-ROUTING + final-rerank
// path (search_pure). Sweeps rerank_factor to trace recall@10/QPS. Tests whether dropping the
// per-hop exact L2 (Glass-style) beats Glass low-recall QPS without losing high recall.
static int runDynBenchPure(const std::string& hdf5_path, int Ncap, bool angular, int Rdeg) {
    auto tr = h5_read_float(hdf5_path, "train");
    auto te = h5_read_float(hdf5_path, "test");
    const int q_dim = tr.n_cols;
    const int N  = (Ncap > 0) ? std::min(Ncap, tr.n_rows) : tr.n_rows;
    const int nq = std::min(1000, te.n_rows), k = 10;
    std::vector<float> train((size_t)N * q_dim), Q((size_t)nq * q_dim);
    std::memcpy(train.data(), tr.data.data(), (size_t)N * q_dim * sizeof(float));
    std::memcpy(Q.data(), te.data.data(), (size_t)nq * q_dim * sizeof(float));
    if (angular) {
        for (int i = 0; i < N;  ++i) dyn_l2norm(&train[(size_t)i * q_dim], q_dim);
        for (int i = 0; i < nq; ++i) dyn_l2norm(&Q[(size_t)i * q_dim], q_dim);
    }
    // angular_=true: exact-L2 routing (1-bit SNR ~0.24 insufficient for unit-sphere)
    // angular_=false: 1-bit vertex-relative routing + ev anchor (correct for Euclidean)
    ArcFlare::RaBitQGraph g(q_dim, 12345, Rdeg, angular);
    std::printf("[dynbench-pure] %s q_dim=%d N=%d nq=%d k=%d angular=%d R=%d\n",
                hdf5_path.c_str(), q_dim, N, nq, k, (int)angular, g.R);
    std::fflush(stdout);
    std::vector<uint32_t> ids(N); for (int i = 0; i < N; ++i) ids[i] = (uint32_t)i;
    auto t0 = std::chrono::steady_clock::now();
    // Angular unit-sphere graphs need higher L_build + more iters for navigability:
    // random build entry (per-vertex) helps, but L=128/iters=3 still produces under-connected
    // graphs on high-D unit spheres. L=256/iters=5 gives better long-range coverage.
    int l_build = angular ? 256 : 128;
    int n_iters = angular ? 5  : 3;
    g.build_parallel(train.data(), N, ids.data(), n_iters, l_build);
    if (N >= 100000) g.build_seeds();  // greedy K-center seeds for both Euclidean and angular
    double t_build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<std::vector<uint32_t>> gt(nq);
    if (N >= tr.n_rows) {
        H5IntDataset gtds = h5_read_int(hdf5_path, "neighbors");
        for (int qi = 0; qi < nq; ++qi) {
            auto& v = gt[qi]; v.reserve(k);
            for (int j = 0; j < k && j < gtds.n_cols; ++j)
                v.push_back((uint32_t)gtds.data[(size_t)qi * gtds.n_cols + j]);
        }
    } else {
        int nT = std::max(1, (int)std::thread::hardware_concurrency());
        std::vector<std::thread> th;
        for (int t = 0; t < nT; ++t) th.emplace_back([&, t]() {
            for (int qi = t; qi < nq; qi += nT) {
                const float* q = &Q[(size_t)qi * q_dim];
                std::priority_queue<std::pair<float, uint32_t>> heap;
                for (int id = 0; id < N; ++id) {
                    const float* x = &train[(size_t)id * q_dim];
                    float s = 0.f; for (int d = 0; d < q_dim; ++d) { float e = x[d] - q[d]; s += e * e; }
                    if ((int)heap.size() < k) heap.emplace(s, (uint32_t)id);
                    else if (s < heap.top().first) { heap.pop(); heap.emplace(s, (uint32_t)id); }
                }
                auto& v = gt[qi]; while (!heap.empty()) { v.push_back(heap.top().second); heap.pop(); }
            }
        });
        for (auto& x : th) x.join();
    }
    std::printf("[dynbench-pure] build %d in %.1fs\n", N, t_build);
    std::fflush(stdout);
    int efs[] = {16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048};
    int rfs[] = {2, 3, 5, 10};
    std::vector<uint32_t> res;
    for (int rf : rfs) {
        std::printf("--- rerank_factor=%d ---\nef     recall@10 QPS(1T)\n", rf);
        std::fflush(stdout);
        for (int ef : efs) {
            if (ef < k) continue;
            auto s0 = std::chrono::steady_clock::now();
            double sum = 0;
            for (int qi = 0; qi < nq; ++qi) {
                g.search_pure(&Q[(size_t)qi * q_dim], k, ef, rf, res);
                int hit = 0; for (uint32_t a : res) for (uint32_t b : gt[qi]) if (a == b) { ++hit; break; }
                sum += (double)hit / k;
            }
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
            std::printf("%-6d %-9.4f %-11.1f\n", ef, sum / nq, nq / el);
            std::fflush(stdout);
        }
    }
    std::printf("=== dynbench-pure done (build %.1fs) ===\n", t_build);
    std::fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s --selftest | --ktest | --fstest | --encode <hdf5> | "
                    "--beamtest | --e2e <index_dir> <hdf5> [mv_list] | --qgbench <index_dir> <hdf5> | "
                    "--churntest <hdf5> | --dynbench <hdf5> [N] [--angular]\n", argv[0]);
        return 2;
    }
    std::string mode = argv[1];
    if (mode == "--selftest") {
        return runSelftest();
    }
    if (mode == "--fstest") {
        return runFstest();
    }
    if (mode == "--ktest") {
        return runKtest();
    }
    if (mode == "--encode") {
        if (argc < 3) {
            std::printf("usage: %s --encode <hdf5>\n", argv[0]);
            return 2;
        }
        return runEncode(argv[2]);
    }
    if (mode == "--beamtest") {
        return runBeamtest();
    }
    if (mode == "--e2e") {
        if (argc < 4) {
            std::printf("usage: %s --e2e <index_dir> <hdf5> [mv_list e.g. 50,100,200,400,800]\n",
                        argv[0]);
            return 2;
        }
        std::vector<int> mv = {50, 100, 200, 400, 800};
        if (argc >= 5) {
            mv.clear();
            std::string s = argv[4];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!tok.empty()) mv.push_back(std::stoi(tok));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            if (mv.empty()) mv = {50, 100, 200, 400, 800};
        }
        return runE2E(argv[2], argv[3], mv);
    }
    if (mode == "--qgbench") {
        if (argc < 4) {
            std::printf("usage: %s --qgbench <index_dir> <hdf5>\n", argv[0]);
            return 2;
        }
        return runQGBench(argv[2], argv[3]);
    }
    if (mode == "--churntest") {
        if (argc < 3) { std::printf("usage: %s --churntest <hdf5>\n", argv[0]); return 2; }
        return runChurnTest(argv[2]);
    }
    if (mode == "--build-save") {
        if (argc < 4) { std::printf("usage: %s --build-save <hdf5> <save_file> [N] [R] [--angular]\n", argv[0]); return 2; }
        std::string save_path = argv[3];
        int Ncap = 0, Rreq = 0, nnum = 0; bool ang = false;
        for (int a = 4; a < argc; ++a) {
            std::string s = argv[a];
            if (s == "--angular") ang = true;
            else { int v = std::atoi(argv[a]); if (nnum++ == 0) Ncap = v; else Rreq = v; }
        }
        return runBuildSave(argv[2], Ncap, ang, Rreq > 0 ? Rreq : 32, save_path);
    }
    if (mode == "--bench-pure") {
        if (argc < 4) { std::printf("usage: %s --bench-pure <idx_file> <hdf5>\n", argv[0]); return 2; }
        return runBenchPureLoad(argv[2], argv[3]);
    }
    if (mode == "--bench-beam") {
        if (argc < 4) { std::printf("usage: %s --bench-beam <idx_file> <hdf5>\n", argv[0]); return 2; }
        return runBenchBeamLoad(argv[2], argv[3]);
    }
    if (mode == "--dynbench") {
        if (argc < 3) { std::printf("usage: %s --dynbench <hdf5> [N] [--angular]\n", argv[0]); return 2; }
        int Ncap = 0, Rreq = 0, nnum = 0; bool ang = false;
        for (int a = 3; a < argc; ++a) {
            std::string s = argv[a];
            if (s == "--angular") ang = true;
            else { int v = std::atoi(argv[a]); if (nnum++ == 0) Ncap = v; else Rreq = v; }
        }
        return runDynBench(argv[2], Ncap, ang, Rreq > 0 ? Rreq : 32);
    }
    if (mode == "--dynbench-pure") {
        if (argc < 3) { std::printf("usage: %s --dynbench-pure <hdf5> [N] [R] [--angular]\n", argv[0]); return 2; }
        int Ncap = 0, Rreq = 0, nnum = 0; bool ang = false;
        for (int a = 3; a < argc; ++a) {
            std::string s = argv[a];
            if (s == "--angular") ang = true;
            else { int v = std::atoi(argv[a]); if (nnum++ == 0) Ncap = v; else Rreq = v; }
        }
        return runDynBenchPure(argv[2], Ncap, ang, Rreq > 0 ? Rreq : 32);
    }
    std::printf("unknown mode: %s (only --selftest, --ktest, --encode, --beamtest, --e2e, --qgbench implemented)\n",
                mode.c_str());
    return 2;
}
