// tests/ngtaq/benchmark_comparison.cpp
// Compares QBG (Quantized Blob Graph) vs NGTAQ (AQ-DABS) recall-QPS on SIFT-1M.
//
// Usage:
//   ./benchmark_comparison <ngt_path> <base.fvecs> <query.fvecs> <gt.ivecs>
//                          [k=10] [ngtaq_cache] [qbg_index_dir=/tmp/qbg_bench]
//
// First run builds both indexes (QBG takes 15-60 min for 1M vecs).
// Subsequent runs reuse cached indexes.
// Delete ngtaq_cache or qbg_index_dir to force rebuild.
//
// SIFT-1M ground truth IDs are 0-indexed.
// NGTAQ returns 0-indexed IDs. QBG returns 1-indexed IDs (subtract 1 before GT lookup).
#include "NGT/NGTAQ/AQIndex.h"
#ifndef NGT_QBG_DISABLED
#include "NGT/NGTQ/QuantizedBlobGraph.h"
#endif
#include "fvecs_io.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// --------------------------------------------------------------------------
// Recall@k computation
// result_ids: 0-indexed result IDs (up to k items)
// gt:         0-indexed ground truth (sorted nearest first)
// --------------------------------------------------------------------------
static double computeRecall(const std::vector<uint32_t>& result_ids,
                            const std::vector<int32_t>&  gt,
                            int k)
{
    const int gt_k = std::min(static_cast<int>(gt.size()), k);
    if (gt_k == 0) return 0.0;
    std::unordered_set<int32_t> gt_set(gt.begin(), gt.begin() + gt_k);
    int hits = 0;
    const int res_k = std::min(static_cast<int>(result_ids.size()), k);
    for (int j = 0; j < res_k; ++j)
        if (gt_set.count(static_cast<int32_t>(result_ids[j]))) ++hits;
    return static_cast<double>(hits) / static_cast<double>(gt_k);
}

// --------------------------------------------------------------------------
// Percentile helper (input: sorted vector)
// --------------------------------------------------------------------------
static double pct(const std::vector<double>& sv, double p) {
    if (sv.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p / 100.0 * static_cast<double>(sv.size()));
    if (idx >= sv.size()) idx = sv.size() - 1;
    return sv[idx];
}

// --------------------------------------------------------------------------
// BenchRow: one sweep point result
// --------------------------------------------------------------------------
struct BenchRow {
    std::string param_label;
    double      param_val;
    double      recall;
    double      qps;
    double      p50_us;
    double      p99_us;
};

static void printTable(const std::string& title,
                       const std::string& param_name,
                       const std::vector<BenchRow>& rows,
                       int k)
{
    std::cout << "\n" << title << "\n";
    std::cout << std::left  << std::setw(14) << param_name
              << std::right << std::setw(12) << ("recall@" + std::to_string(k))
              << std::setw(12) << "QPS"
              << std::setw(12) << "P50(us)"
              << std::setw(12) << "P99(us)" << "\n";
    std::cout << std::string(62, '-') << "\n";
    for (const auto& r : rows) {
        std::cout << std::left  << std::setw(14) << r.param_label
                  << std::right << std::fixed
                  << std::setw(12) << std::setprecision(4) << r.recall
                  << std::setw(12) << std::setprecision(0) << r.qps
                  << std::setw(12) << std::setprecision(1) << r.p50_us
                  << std::setw(12) << std::setprecision(1) << r.p99_us << "\n";
    }
}

// --------------------------------------------------------------------------
// NGTAQ recall-QPS sweep over gamma_term values
// --------------------------------------------------------------------------
static std::vector<BenchRow> sweepNGTAQ(
    NGTAQ::NGTAQIndex& aq,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int32_t>>& gt,
    int k)
{
    static const float GAMMA_TERMS[] = {0.05f, 0.10f, 0.20f, 0.35f, 0.50f, 0.75f, 1.00f};
    const size_t nq = queries.size();
    std::vector<BenchRow> rows;

    // Warmup (200 queries, default params)
    aq.setSearchGammas(0.15f, 0.35f);
    for (size_t i = 0; i < std::min(nq, size_t(200)); ++i)
        aq.search(queries[i], k);

    for (float gt_val : GAMMA_TERMS) {
        const float enq_val = gt_val * 0.43f;
        aq.setSearchGammas(enq_val, gt_val);

        std::vector<double> lats(nq);
        double total_recall = 0.0;

        for (size_t i = 0; i < nq; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            auto res = aq.search(queries[i], k);
            const auto t1 = std::chrono::high_resolution_clock::now();
            lats[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();

            std::vector<uint32_t> ids;
            ids.reserve(res.size());
            for (auto& r : res) ids.push_back(r.id);
            total_recall += computeRecall(ids, gt[i], k);
        }

        double total_us = std::accumulate(lats.begin(), lats.end(), 0.0);
        std::sort(lats.begin(), lats.end());

        BenchRow row;
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << gt_val;
            row.param_label = oss.str();
        }
        row.param_val = gt_val;
        row.recall    = total_recall / static_cast<double>(nq);
        row.qps       = static_cast<double>(nq) / (total_us / 1e6);
        row.p50_us    = pct(lats, 50.0);
        row.p99_us    = pct(lats, 99.0);
        rows.push_back(row);

        std::cout << "  NGTAQ gamma_term=" << std::fixed << std::setprecision(3) << gt_val
                  << "  recall=" << std::setprecision(4) << row.recall
                  << "  QPS=" << std::setprecision(0) << row.qps << "\n";
        std::cout.flush();
    }

    // Restore default
    aq.setSearchGammas(0.15f, 0.35f);
    return rows;
}

// --------------------------------------------------------------------------
// NGTAQ k_prime_factor sweep (fixed gamma) — diagnoses BQ vs graph quality
// A rapidly saturating recall curve means the graph is the bottleneck.
// A slowly increasing recall means BQ is too lossy.
// --------------------------------------------------------------------------
static void sweepNGTAQKprime(
    NGTAQ::NGTAQIndex& aq,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int32_t>>& gt,
    int k)
{
    static const float KPF_VALS[] = {2.f, 5.f, 10.f, 20.f, 50.f, 100.f};
    const size_t nq = queries.size();

    std::cout << "\n=== NGTAQ k_prime_factor sweep (gamma_term=0.50, k=" << k << ") ===\n";
    std::cout << std::left  << std::setw(12) << "k_prime"
              << std::right << std::setw(12) << ("recall@" + std::to_string(k))
              << std::setw(12) << "candidates"
              << std::setw(12) << "QPS" << "\n";
    std::cout << std::string(48, '-') << "\n";

    for (float kpf : KPF_VALS) {
        aq.setSearchGammas(0.50f * 0.43f, 0.50f, kpf);  // gamma_term=0.50

        double total_recall = 0.0;
        double total_us = 0.0;

        for (size_t i = 0; i < nq; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            auto res = aq.search(queries[i], k);
            const auto t1 = std::chrono::high_resolution_clock::now();
            total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();

            std::vector<uint32_t> ids;
            ids.reserve(res.size());
            for (auto& r : res) ids.push_back(r.id);
            total_recall += computeRecall(ids, gt[i], k);
        }

        double recall = total_recall / static_cast<double>(nq);
        double qps    = static_cast<double>(nq) / (total_us / 1e6);
        int    k_prime = static_cast<int>(k * kpf);

        std::cout << std::left  << std::fixed
                  << std::setw(12) << kpf
                  << std::right
                  << std::setw(12) << std::setprecision(4) << recall
                  << std::setw(12) << k_prime
                  << std::setw(12) << std::setprecision(0) << qps << "\n";
        std::cout.flush();
    }
    // Restore defaults
    aq.setSearchGammas(0.15f, 0.35f, 2.0f);
}

// --------------------------------------------------------------------------
// QBG recall-QPS sweep over graphExplorationSize values
// --------------------------------------------------------------------------
#ifndef NGT_QBG_DISABLED
static std::vector<BenchRow> sweepQBG(
    QBG::Index& qbg,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int32_t>>& gt,
    int k)
{
    // Sweep (numOfProbes, graphExplorationSize) pairs to trace the full recall-QPS curve.
    // numOfProbes: number of QBG blobs/clusters searched (more probes = higher recall + lower QPS)
    // graphExplorationSize: size of graph expansion inside each probe (minor effect)
    struct QBGParam { size_t probes; size_t ges; };
    static const QBGParam PARAMS[] = {
        {1,  10}, {2,  10}, {3,  10}, {5,  10}, {8,  10},
        {10, 10}, {15, 20}, {20, 20}, {30, 50}, {50, 100},
        {64, 200}
    };
    const size_t nq = queries.size();
    std::vector<BenchRow> rows;

    auto doSearch = [&](size_t probes, size_t ges) {
        std::vector<double> lats(nq);
        double total_recall = 0.0;

        for (size_t i = 0; i < nq; ++i) {
            std::vector<float> qv = queries[i];  // mutable copy required by setObjectVector
            QBG::SearchContainer sc;
            sc.setObjectVector(qv);
            sc.setSize(k);
            sc.setEpsilon(0.1f);
            sc.setBlobEpsilon(0.0f);
            sc.setNumOfProbes(probes);
            sc.setGraphExplorationSize(ges);
            sc.setRefinementExpansion(3.0f);
            NGT::ObjectDistances results;
            sc.setResults(&results);

            const auto t0 = std::chrono::high_resolution_clock::now();
            try { qbg.searchInTwoSteps(sc); }
            catch (const std::exception& e) {
                std::cerr << "QBG search error at query " << i << ": " << e.what() << "\n";
            }
            const auto t1 = std::chrono::high_resolution_clock::now();
            lats[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();

            std::vector<uint32_t> ids;
            ids.reserve(results.size());
            for (const auto& obj : results)
                ids.push_back(static_cast<uint32_t>(obj.id - 1));  // 1-based → 0-based
            total_recall += computeRecall(ids, gt[i], k);
        }
        return std::make_pair(total_recall, lats);
    };

    // Warmup
    doSearch(10, 50);

    constexpr int N_PARAMS = (int)(sizeof(PARAMS) / sizeof(PARAMS[0]));
    // Measure heaviest first (most probes) to avoid thermal throttle skewing fast results
    struct QBGRaw { float recall; float qps; float p50; float p99; };
    std::vector<QBGRaw> raw(N_PARAMS);
    for (int pi = N_PARAMS - 1; pi >= 0; --pi) {
        auto [total_recall, lats] = doSearch(PARAMS[pi].probes, PARAMS[pi].ges);
        double total_us = std::accumulate(lats.begin(), lats.end(), 0.0);
        std::vector<double> slats = lats;
        std::sort(slats.begin(), slats.end());
        raw[pi] = {
            (float)(total_recall / (double)nq),
            (float)((double)nq / (total_us / 1e6)),
            (float)pct(slats, 50.0),
            (float)pct(slats, 99.0)
        };
    }

    for (int pi = 0; pi < N_PARAMS; ++pi) {
        {
            std::ostringstream oss;
            oss << "p" << PARAMS[pi].probes << "g" << PARAMS[pi].ges;
            BenchRow row;
            row.param_label = oss.str();
            row.param_val   = static_cast<double>(PARAMS[pi].probes);
            row.recall      = raw[pi].recall;
            row.qps         = raw[pi].qps;
            row.p50_us      = raw[pi].p50;
            row.p99_us      = raw[pi].p99;
            rows.push_back(row);
        }
        std::cout << "  QBG probes=" << PARAMS[pi].probes << " ges=" << PARAMS[pi].ges
                  << "  recall=" << std::fixed << std::setprecision(4) << raw[pi].recall
                  << "  QPS=" << std::setprecision(0) << raw[pi].qps
                  << "  P50=" << std::setprecision(1) << raw[pi].p50 << "us"
                  << "  P99=" << raw[pi].p99 << "us\n";
        std::cout.flush();
    }
    return rows;
}
#endif // NGT_QBG_DISABLED

// --------------------------------------------------------------------------
// sweepNGTAQv2: gamma_term sweep for NGTAQv2 (ADC search)
// --------------------------------------------------------------------------
static std::vector<BenchRow> sweepNGTAQv2(
    NGTAQ::NGTAQIndex& idx,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int32_t>>& gt,
    int k)
{
    static const float GAMMA_TERMS[] = {0.10f, 0.12f, 0.14f, 0.16f, 0.18f, 0.20f, 0.30f, 0.50f, 0.70f, 1.00f};
    constexpr int N_GAMMAS = (int)(sizeof(GAMMA_TERMS) / sizeof(GAMMA_TERMS[0]));
    const size_t nq = queries.size();

    // Warmup: run heaviest gamma to prime cluster membership cache and visited bitvector
    for (size_t i = 0; i < std::min(nq, size_t(20)); ++i)
        idx.searchV2(queries[i], k, 0.2f, 1.0f);

    // Measure from heaviest to lightest gamma to avoid thermal throttling on fast gammas
    struct RawResult { float recall; float qps; float p50; float p99; };
    std::vector<RawResult> raw(N_GAMMAS);

    for (int gi = N_GAMMAS - 1; gi >= 0; --gi) {
        float gt_val = GAMMA_TERMS[gi];
        std::vector<double> lats(nq);
        double total_recall = 0.0;

        for (size_t i = 0; i < nq; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            auto res = idx.searchV2(queries[i], k, 0.2f, gt_val);
            const auto t1 = std::chrono::high_resolution_clock::now();
            lats[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();

            std::vector<uint32_t> ids;
            ids.reserve(res.size());
            for (auto& r : res) ids.push_back(r.id);
            total_recall += computeRecall(ids, gt[i], k);
        }

        double total_us = std::accumulate(lats.begin(), lats.end(), 0.0);
        std::vector<double> slats = lats;
        std::sort(slats.begin(), slats.end());
        raw[gi] = {
            (float)(total_recall / (double)nq),
            (float)((double)nq / (total_us / 1e6)),
            (float)pct(slats, 50.0),
            (float)pct(slats, 99.0)
        };
    }

    // Collect results in ascending gamma order and print
    std::vector<BenchRow> rows;
    for (int gi = 0; gi < N_GAMMAS; ++gi) {
        float gt_val = GAMMA_TERMS[gi];
        BenchRow row;
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << gt_val;
            row.param_label = oss.str();
        }
        row.param_val = gt_val;
        row.recall    = raw[gi].recall;
        row.qps       = raw[gi].qps;
        row.p50_us    = raw[gi].p50;
        row.p99_us    = raw[gi].p99;
        rows.push_back(row);

        std::cout << "  NGTAQv2 gamma_term=" << std::fixed << std::setprecision(3) << gt_val
                  << "  recall=" << std::setprecision(4) << row.recall
                  << "  QPS=" << std::setprecision(0) << row.qps
                  << "  P50=" << std::setprecision(1) << row.p50_us << "us"
                  << "  P99=" << row.p99_us << "us\n";
        std::cout.flush();
    }

    return rows;
}

// --------------------------------------------------------------------------
// Find the BenchRow closest to a target recall
// --------------------------------------------------------------------------
static const BenchRow* nearest(const std::vector<BenchRow>& rows, double target) {
    const BenchRow* best = nullptr;
    double best_diff = 1e9;
    for (const auto& r : rows) {
        double d = std::abs(r.recall - target);
        if (d < best_diff) { best_diff = d; best = &r; }
    }
    return best;
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr
            << "Usage: " << argv[0]
            << " <ngt_path> <base.fvecs> <query.fvecs> <gt.ivecs>"
            << " [k=10] [ngtaq_cache] [qbg_index_dir=/tmp/qbg_bench]\n"
            << "\n"
            << "  ngt_path:      NGT index directory (source for NGTAQ fromNGT())\n"
            << "  base.fvecs:    Base vectors in fvecs format (source for QBG append)\n"
            << "  query.fvecs:   Query vectors in fvecs format\n"
            << "  gt.ivecs:      Ground-truth top-k IDs in ivecs format (0-indexed)\n"
            << "  k:             Number of nearest neighbors (default: 10)\n"
            << "  ngtaq_cache:   Path to save/load NGTAQ index cache\n"
            << "  qbg_index_dir: Directory for QBG index (default: /tmp/qbg_bench)\n";
        return 1;
    }

    const std::string ngt_path   = argv[1];
    const std::string base_path  = argv[2];
    const std::string query_path = argv[3];
    const std::string gt_path    = argv[4];
    const int k          = (argc > 5) ? std::stoi(argv[5]) : 10;
    const std::string ngtaq_cache = (argc > 6) ? argv[6] : "";
#ifndef NGT_QBG_DISABLED
    const std::string qbg_dir     = (argc > 7) ? argv[7] : "/tmp/qbg_bench";
#endif

    // ---- Load shared data ----
    std::cout << "[Load] base vectors: " << base_path << "\n"; std::cout.flush();
    auto base_vecs = NGTAQ::loadFvecs(base_path);
    const size_t N = base_vecs.size();
    const int D = N > 0 ? static_cast<int>(base_vecs[0].size()) : 128;
    std::cout << "  N=" << N << "  D=" << D << "\n";

    std::cout << "[Load] queries: " << query_path << "\n"; std::cout.flush();
    auto queries = NGTAQ::loadFvecs(query_path);
    const size_t nq = queries.size();

    std::cout << "[Load] ground truth: " << gt_path << "\n"; std::cout.flush();
    auto ground_truth = NGTAQ::loadIvecs(gt_path);

    if (queries.size() != ground_truth.size()) {
        std::cerr << "Size mismatch: " << queries.size() << " queries vs "
                  << ground_truth.size() << " GT entries\n";
        return 1;
    }
    std::cout << "  " << nq << " queries  k=" << k << "\n\n";

    // ==================== QBG ====================
#ifndef NGT_QBG_DISABLED
    std::cout << "========================================\n";
    std::cout << " QBG Index: " << qbg_dir << "\n";
    std::cout << "========================================\n";
    std::cout.flush();

    // grp = QBG graph file, created only after Phase 3 (full build) completes.
    // Old-style QBG: grp at top level; new-style: global/grp.
    const bool qbg_prebuilt = std::filesystem::exists(qbg_dir + "/grp") ||
                              std::filesystem::exists(qbg_dir + "/global/grp");

    if (!qbg_prebuilt) {
        // Create index skeleton
        QBG::BuildParameters bp;
        bp.creation.genuineDimension     = static_cast<size_t>(D);
        bp.creation.dimensionOfSubvector = 2;  // 2 dims/subvector → D/2 subvectors
        bp.creation.distanceType         = NGTQ::DistanceType::DistanceTypeL2;
        bp.creation.dataType             = NGTQ::DataTypeFloat;

        std::cout << "[QBG] Creating index skeleton (D=" << D
                  << ", dimensionOfSubvector=2)...\n";
        std::cout.flush();
        // QBG::Index::create() creates the directory itself — do NOT mkdir first.
        try {
            QBG::Index::create(qbg_dir, bp);
        } catch (const std::exception& e) {
            std::cerr << "QBG create failed: " << e.what() << "\n";
            return 1;
        }

        // Append base vectors
        std::cout << "[QBG] Appending " << N << " vectors...\n"; std::cout.flush();
        {
            QBG::Index qbg(qbg_dir, false);
            for (size_t i = 0; i < N; ++i) {
                qbg.append(base_vecs[i]);
                if ((i + 1) % 100000 == 0 || i == N - 1) {
                    std::cout << "  " << (i + 1) << "/" << N << "\r"; std::cout.flush();
                }
            }
            std::cout << "\n[QBG] Saving...\n"; std::cout.flush();
            qbg.save();
        }

        // Build phase 1/3: hierarchical k-means clustering → creates ws/hkc_* files
        std::cout << "[QBG] Phase 1/3: Hierarchical K-means clustering (may take ~10 min)...\n";
        std::cout.flush();
        const auto t0 = std::chrono::steady_clock::now();
        try {
            QBG::HierarchicalKmeans hkm(bp);
            hkm.clustering(qbg_dir);
        } catch (const std::exception& e) {
            std::cerr << "QBG Phase 1 (clustering) failed: " << e.what() << "\n";
            return 1;
        }
        std::cout << "[QBG] Phase 1 done in "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - t0).count()
                  << "s\n"; std::cout.flush();

        // Build phase 2/3: PQ codebook rotation optimization → creates rotation/codebook files
        std::cout << "[QBG] Phase 2/3: Optimizer...\n"; std::cout.flush();
        try {
            QBG::Optimizer opt(bp);
            opt.optimize(qbg_dir, 0 /*all available threads*/);
        } catch (const std::exception& e) {
            std::cerr << "QBG Phase 2 (optimizer) failed: " << e.what() << "\n";
            return 1;
        }
        std::cout << "[QBG] Phase 2 done (total "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - t0).count()
                  << "s)\n"; std::cout.flush();

        // Build phase 3/3: load workspace files + build NGTQ + build QBG blob graph
        std::cout << "[QBG] Phase 3/3: Build blob graph...\n"; std::cout.flush();
        try {
            QBG::Index::build(qbg_dir, /*verbose=*/false);
        } catch (const std::exception& e) {
            std::cerr << "QBG Phase 3 (build) failed: " << e.what() << "\n";
            return 1;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "[QBG] Build complete in " << elapsed << "s\n\n";
        std::cout.flush();
    } else {
        std::cout << "[QBG] Reusing prebuilt index.\n\n";
    }

    std::cout << "[QBG] Opening prebuilt index...\n"; std::cout.flush();
    QBG::Index qbg(qbg_dir, true);
    std::cout << "[QBG] Ready.\n\n";
#endif // NGT_QBG_DISABLED

    // ==================== NGTAQ ====================
    std::cout << "========================================\n";
    std::cout << " NGTAQ Index\n";
    std::cout << "========================================\n";
    std::cout.flush();

    NGTAQ::NGTAQIndex::Property prop;
    prop.dimension      = D;
    prop.n_tau_samples  = 10000;
    prop.metric         = NGT::ObjectSpace::DistanceTypeL2;
    prop.gamma_term     = 0.35f;
    prop.gamma_enq      = 0.15f;
    prop.k_prime_factor = 2.0f;

    NGTAQ::NGTAQIndex aq = [&]() -> NGTAQ::NGTAQIndex {
        if (!ngtaq_cache.empty()) {
            std::ifstream probe(ngtaq_cache);
            if (probe.good()) {
                probe.close();
                std::cout << "[NGTAQ] Loading cache: " << ngtaq_cache << "\n"; std::cout.flush();
                try {
                    auto idx = NGTAQ::NGTAQIndex::load(ngtaq_cache + "/aqindex");
                    idx.loadV2(ngtaq_cache);
                    std::cout << "[NGTAQ] Loaded (v2). size=" << idx.size() << "\n\n"; std::cout.flush();
                    return idx;
                } catch (const std::exception& e) {
                    std::cerr << "[NGTAQ] Cache load failed (" << e.what() << "), rebuilding...\n";
                }
            }
        }
        std::cout << "[NGTAQ] Building from NGT: " << ngt_path << " ...\n"; std::cout.flush();
        const auto t0 = std::chrono::steady_clock::now();
        auto idx = NGTAQ::NGTAQIndex::fromNGT(ngt_path, prop);
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "[NGTAQ] Built in " << elapsed << "s. size=" << idx.size() << "\n";
        std::cout.flush();
        if (!ngtaq_cache.empty()) {
            try {
                idx.save(ngtaq_cache);
                std::cout << "[NGTAQ] Saved to cache: " << ngtaq_cache << "\n";
                std::cout.flush();
            } catch (const std::exception& e) {
                std::cerr << "[NGTAQ] Save failed: " << e.what() << "\n";
            }
        }
        return idx;
    }();
    std::cout << "\n";

    // ==================== NGTAQv2 ====================
    const std::string v2_cache = ngtaq_cache.empty() ? std::string("") : ngtaq_cache + "_v2";
    NGTAQ::NGTAQIndex aq_v2 = [&]() -> NGTAQ::NGTAQIndex {
        if (!v2_cache.empty()) {
            std::ifstream probe(v2_cache + "/v2_records.bin");
            if (probe.good()) {
                probe.close();
                std::cout << "[NGTAQv2] Loading cache: " << v2_cache << "\n"; std::cout.flush();
                try {
                    auto idx = NGTAQ::NGTAQIndex::load(v2_cache + "/aqindex");
                    idx.loadV2(v2_cache);
                    std::cout << "[NGTAQv2] Loaded.\n"; std::cout.flush();
                    return idx;
                } catch (const std::exception& e) {
                    std::cerr << "[NGTAQv2] Cache load failed (" << e.what() << "), rebuilding...\n";
                }
            }
        }
        std::cout << "[NGTAQv2] Building from NGT: " << ngt_path << " ...\n"; std::cout.flush();
        NGTAQ::NGTAQIndex::Property v2_prop;
        v2_prop.dimension = D;
        const auto t0 = std::chrono::steady_clock::now();
        auto idx = NGTAQ::NGTAQIndex::fromNGTv2(ngt_path, v2_prop);
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "[NGTAQv2] Built in " << elapsed << "s\n"; std::cout.flush();
        if (!v2_cache.empty()) {
            try {
                std::filesystem::create_directories(v2_cache);
                idx.save(v2_cache + "/aqindex");
                idx.saveV2(v2_cache);
                std::cout << "[NGTAQv2] Saved to cache: " << v2_cache << "\n"; std::cout.flush();
            } catch (const std::exception& e) {
                std::cerr << "[NGTAQv2] Save failed: " << e.what() << "\n";
            }
        }
        return idx;
    }();
    std::cout << "\n";

    // ==================== Sweeps ====================
    // Diagnostic: k_prime_factor sweep to distinguish BQ quality vs graph bottleneck
    sweepNGTAQKprime(aq, queries, ground_truth, k);

    std::cout << "\n=== NGTAQ sweep (gamma_term) ===\n"; std::cout.flush();
    auto ngtaq_rows = sweepNGTAQ(aq, queries, ground_truth, k);

    std::cout << "\n=== NGTAQv2 sweep (gamma_term) ===\n"; std::cout.flush();
    auto v2_rows = sweepNGTAQv2(aq_v2, queries, ground_truth, k);

#ifndef NGT_QBG_DISABLED
    std::cout << "\n=== QBG sweep (graphExplorationSize) ===\n"; std::cout.flush();
    auto qbg_rows = sweepQBG(qbg, queries, ground_truth, k);
#endif

    // ==================== Summary ====================
    std::cout << "\n\n";
    std::cout << "================================================================\n";
    std::cout << "  Comparison: NGTAQ (AQ-DABS BQ) vs QBG (Quantized Blob Graph)\n";
    std::cout << "  N=" << N << "  D=" << D << "  queries=" << nq << "  k=" << k << "\n";
    std::cout << "================================================================\n";

    printTable("--- NGTAQ (AQ-DABS Binary Quantization) ---", "gamma_term", ngtaq_rows, k);
    printTable("--- NGTAQv2 (ADC Quantization) ---",          "gamma_term", v2_rows,    k);
#ifndef NGT_QBG_DISABLED
    printTable("--- QBG (Quantized Blob Graph, PQ4) ---",     "graphExp",   qbg_rows,   k);
#endif

    // Find crossover near various recall targets
    for (double target_recall : {0.80, 0.85, 0.90, 0.95}) {
        std::cout << "\n--- Comparison at recall ~" << std::fixed << std::setprecision(2) << target_recall << " ---\n";
        const auto* nv   = nearest(ngtaq_rows, target_recall);
        const auto* v2v  = nearest(v2_rows,    target_recall);
#ifndef NGT_QBG_DISABLED
        const auto* qv   = nearest(qbg_rows,   target_recall);
#endif
        if (nv)  std::cout << "  NGTAQ:   recall=" << std::setprecision(4) << nv->recall  << "  QPS=" << std::setprecision(0) << nv->qps  << "  P50=" << std::setprecision(1) << nv->p50_us  << "us  P99=" << nv->p99_us  << "us\n";
        if (v2v) std::cout << "  NGTAQv2: recall=" << std::setprecision(4) << v2v->recall << "  QPS=" << std::setprecision(0) << v2v->qps << "  P50=" << std::setprecision(1) << v2v->p50_us << "us  P99=" << v2v->p99_us << "us\n";
#ifndef NGT_QBG_DISABLED
        if (qv)  std::cout << "  QBG:     recall=" << std::setprecision(4) << qv->recall  << "  QPS=" << std::setprecision(0) << qv->qps  << "  P50=" << std::setprecision(1) << qv->p50_us  << "us  P99=" << qv->p99_us  << "us\n";
        if (v2v && qv && qv->qps > 0) {
            double ratio = v2v->qps / qv->qps;
            std::cout << "  NGTAQv2 vs QBG: ";
            if (ratio > 1.05) std::cout << "NGTAQv2 " << std::setprecision(2) << ratio << "x faster\n";
            else if (ratio < 0.95) std::cout << "QBG " << std::setprecision(2) << (1.0/ratio) << "x faster\n";
            else std::cout << "~equivalent\n";
        }
#endif
    }

    // Legacy comparison summary at ~0.90 for backward compat
    std::cout << "\n--- Legacy comparison at recall ~0.90 ---\n" << std::fixed;
    const auto* n90  = nearest(ngtaq_rows, 0.90);
    const auto* v290 = nearest(v2_rows,    0.90);
    if (n90) {
        std::cout << "  NGTAQ (gamma_term=" << std::setprecision(3) << n90->param_val << "):"
                  << "  recall=" << std::setprecision(4) << n90->recall
                  << "  QPS="    << std::setprecision(0) << n90->qps
                  << "  P50="    << std::setprecision(1) << n90->p50_us << "us"
                  << "  P99="    << n90->p99_us << "us\n";
    }
    if (v290) {
        std::cout << "  NGTAQv2 (gamma_term=" << std::setprecision(3) << v290->param_val << "):"
                  << "  recall=" << std::setprecision(4) << v290->recall
                  << "  QPS="    << std::setprecision(0) << v290->qps
                  << "  P50="    << std::setprecision(1) << v290->p50_us << "us"
                  << "  P99="    << v290->p99_us << "us\n";
    }
    if (n90 && v290 && v290->qps > 0 && n90->qps > 0) {
        const double ratio = v290->qps / n90->qps;
        std::cout << "  NGTAQv2 vs NGTAQ at recall~0.90: ";
        if (ratio > 1.05)
            std::cout << "NGTAQv2 is " << std::setprecision(2) << ratio << "x faster\n";
        else if (ratio < 0.95)
            std::cout << "NGTAQ is " << std::setprecision(2) << (1.0/ratio) << "x faster\n";
        else
            std::cout << "Roughly equivalent (" << std::setprecision(2) << ratio << "x)\n";
    }
#ifndef NGT_QBG_DISABLED
    const auto* q90  = nearest(qbg_rows,   0.90);
    if (q90) {
        std::cout << "  QBG (graphExp=" << static_cast<size_t>(q90->param_val) << "):"
                  << "  recall=" << std::setprecision(4) << q90->recall
                  << "  QPS="    << std::setprecision(0) << q90->qps
                  << "  P50="    << std::setprecision(1) << q90->p50_us << "us"
                  << "  P99="    << q90->p99_us << "us\n";
    }
    if (n90 && q90 && q90->qps > 0) {
        const double ratio = n90->qps / q90->qps;
        std::cout << "  NGTAQ vs QBG at recall~0.90: ";
        if (ratio > 1.05)
            std::cout << "NGTAQ is " << std::setprecision(2) << ratio << "x faster\n";
        else if (ratio < 0.95)
            std::cout << "QBG is " << std::setprecision(2) << (1.0/ratio) << "x faster\n";
        else
            std::cout << "Roughly equivalent (" << std::setprecision(2) << ratio << "x)\n";
    }
    if (v290 && q90 && q90->qps > 0) {
        const double ratio = v290->qps / q90->qps;
        std::cout << "  NGTAQv2 vs QBG at recall~0.90: ";
        if (ratio > 1.05)
            std::cout << "NGTAQv2 is " << std::setprecision(2) << ratio << "x faster\n";
        else if (ratio < 0.95)
            std::cout << "QBG is " << std::setprecision(2) << (1.0/ratio) << "x faster than NGTAQv2\n";
        else
            std::cout << "Roughly equivalent (" << std::setprecision(2) << ratio << "x)\n";
    }
#endif // NGT_QBG_DISABLED

    return 0;
}
