// lib/NGT/NGTAQ/AQIndex.cpp
#include "NGT/NGTAQ/AQIndex.h"

#include "NGT/Graph.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <omp.h>
#include <queue>
#include <random>
#include <shared_mutex>
#include <stdexcept>
// unordered_set removed: visited tracking uses flat bitvector (see searchV2)

namespace NGTAQ {

// ---------------------------------------------------------------------------
// Private constructor
// ---------------------------------------------------------------------------
NGTAQIndex::NGTAQIndex(Property prop, BinaryQuantizer bq,
                       std::unique_ptr<SoAGraph> graph,
                       std::vector<uint32_t> eps,
                       std::vector<float> raw_flat)
    : prop_(prop)
    , bq_(std::move(bq))
    , graph_(std::move(graph))
    , pruner_(prop.alpha, prop.kappa)
    , entry_points_(std::move(eps))
    , raw_flat_(std::move(raw_flat))
{
    searcher_.gamma_enq      = prop_.gamma_enq;
    searcher_.gamma_term     = prop_.gamma_term;
    searcher_.k_prime_factor = prop_.k_prime_factor;
}

// ---------------------------------------------------------------------------
// fromNGT
// ---------------------------------------------------------------------------
NGTAQIndex NGTAQIndex::fromNGT(const std::string& ngt_path, const Property& prop) {
    if (prop.dimension % 64 != 0)
        throw std::invalid_argument("NGTAQIndex: dimension must be a multiple of 64");

    NGT::Index ngt(ngt_path);
    NGT::ObjectSpace& objspace = ngt.getObjectSpace();
    const size_t repo_size = objspace.getRepository().size();
    const size_t N = repo_size - 1;
    const int D = prop.dimension;
    const int words = D / 64;

    // Load all float vectors into flat array: raw_flat[i*D .. i*D+D-1] = vec i
    std::vector<float> raw_flat(N * static_cast<size_t>(D), 0.0f);
    std::vector<bool> is_hole(N, false);
    std::vector<float> tmp(D);
    for (size_t i = 1; i <= N; i++) {
        try {
            objspace.getObject(static_cast<NGT::ObjectID>(i), tmp);
            std::copy(tmp.begin(), tmp.end(), raw_flat.begin() + static_cast<ptrdiff_t>((i - 1) * D));
        } catch (...) {
            is_hole[i - 1] = true;
        }
    }

    // Pre-normalize for cosine metric
    if (prop.metric == NGT::ObjectSpace::DistanceTypeAngle ||
        prop.metric == NGT::ObjectSpace::DistanceTypeCosine) {
        for (size_t i = 0; i < N; ++i) {
            float* v = raw_flat.data() + i * D;
            float norm_sq = 0.0f;
            for (int j = 0; j < D; ++j) norm_sq += v[j] * v[j];
            if (norm_sq > 0.0f) {
                float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (int j = 0; j < D; ++j) v[j] *= inv_norm;
            }
        }
    }

    BinaryQuantizer bq;
    bq.init(D);
    bq.setRandomRotation();

    std::vector<const float*> ptrs(N);
    for (size_t i = 0; i < N; i++) ptrs[i] = raw_flat.data() + i * D;
    bq.calibrateTau(ptrs, prop.n_tau_samples, prop.metric);

    // Encode all vectors and build SoAGraph
    auto graph = std::make_unique<SoAGraph>(words);
    std::vector<uint64_t> bq_buf(static_cast<size_t>(words) * 2);
    for (size_t i = 0; i < N; i++) {
        bq.encode(raw_flat.data() + i * D, bq_buf.data());
        graph->addNode(bq_buf.data());
    }
    graph->finalizeCSR();

    // Tombstone ghost nodes
    for (size_t i = 0; i < N; ++i) {
        if (is_hole[i]) graph->removeNode(static_cast<uint32_t>(i));
    }

    // Build alpha-CG graph from NGT edges (O(N·k) via resetEdges)
    AlphaCGPruner pruner(prop.alpha, prop.kappa);
    const float tau = bq.tau();
    NGT::GraphIndex& gi = static_cast<NGT::GraphIndex&>(ngt.getIndex());

    std::vector<std::vector<uint32_t>> adj(N);
    for (size_t i = 1; i <= N; i++) {
        uint32_t aq_id = static_cast<uint32_t>(i - 1);
        NGT::GraphNode* node = nullptr;
        try {
            node = gi.getNode(static_cast<NGT::ObjectID>(i));
        } catch (...) {
            continue;
        }
        if (!node || node->empty()) continue;

        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(node->size());
        for (auto& edge : *node) {
            if (edge.id == 0 || edge.id > static_cast<unsigned int>(N)) continue;
            uint32_t nbr = static_cast<uint32_t>(edge.id - 1);
            float d = bqDistance(graph->getNodeBQ(aq_id), graph->getNodeBQ(nbr), words, D);
            candidates.push_back({nbr, d});
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (static_cast<int>(candidates.size()) > prop.max_edges)
            candidates.resize(static_cast<size_t>(prop.max_edges));

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(graph->getNodeBQ(v), graph->getNodeBQ(u), words, D);
        };
        adj[aq_id] = pruner.prune(candidates, tau, dist_fn);
    }
    graph->resetEdges(adj);

    int n_ep = std::min(prop.n_entry_points, static_cast<int>(N));
    auto entry_points = selectEntryPoints(*graph, n_ep);

    return NGTAQIndex(prop, std::move(bq), std::move(graph),
                      std::move(entry_points), std::move(raw_flat));
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------
std::vector<SearchResult> NGTAQIndex::search(
    const std::vector<float>& query, int k) const
{
    if (static_cast<int>(query.size()) < prop_.dimension)
        throw std::invalid_argument("NGTAQIndex::search: query dimension mismatch");
    const int D = prop_.dimension;
    const int words = D / 64;

    // Encode query to interleaved BQ
    std::vector<uint64_t> q_bq(static_cast<size_t>(words) * 2);
    bq_.encode(query.data(), q_bq.data());

    std::shared_lock<std::shared_mutex> lock(graph_->mutex());

    auto cand_ids = searcher_.route(q_bq.data(), k, *graph_, entry_points_);

    // Prefetch raw float vectors for refinement (PREFETCH_AHEAD=8 candidates)
    constexpr int PREFETCH_AHEAD = 8;
    const int n_cands = static_cast<int>(cand_ids.size());
    for (int ci = 0; ci < n_cands; ++ci) {
        if (ci + PREFETCH_AHEAD < n_cands) {
            uint32_t nxt = cand_ids[static_cast<size_t>(ci + PREFETCH_AHEAD)];
            if (nxt * static_cast<size_t>(D) < raw_flat_.size()) {
                __builtin_prefetch(raw_flat_.data() + nxt * static_cast<size_t>(D), 0, 1);
            }
        }
    }

    // Exact-distance refinement
    std::vector<SearchResult> results;
    results.reserve(cand_ids.size());
    for (uint32_t id : cand_ids) {
        if (static_cast<size_t>(id) * D + D > raw_flat_.size()) continue;
        const float* vec = raw_flat_.data() + static_cast<size_t>(id) * D;
        float exact_dist = 0.0f;
        if (prop_.metric == NGT::ObjectSpace::DistanceTypeL2) {
            float sq = 0.0f;
            for (int j = 0; j < D; ++j) {
                float d = query[j] - vec[j];
                sq += d * d;
            }
            exact_dist = std::sqrt(sq);
        } else {
            // Cosine: raw_flat_ stores pre-normalized vectors; normalize query too.
            std::vector<float> qn(query.data(), query.data() + D);
            float norm_sq = 0.0f;
            for (float x : qn) norm_sq += x * x;
            if (norm_sq > 0.0f) {
                float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (float& x : qn) x *= inv_norm;
            }
            float dot = 0.0f;
            for (int j = 0; j < D; ++j) dot += qn[j] * vec[j];
            exact_dist = 1.0f - dot;
        }
        float bq_dist = bqDistance(q_bq.data(), graph_->getNodeBQ(id), words, D);
        results.push_back({id, exact_dist, bq_dist});
    }

    std::sort(results.begin(), results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        });
    if (static_cast<int>(results.size()) > k)
        results.resize(static_cast<size_t>(k));
    return results;
}

// ---------------------------------------------------------------------------
// searchBatch
// ---------------------------------------------------------------------------
std::vector<std::vector<SearchResult>> NGTAQIndex::searchBatch(
    const std::vector<std::vector<float>>& queries, int k) const
{
    const int nq = static_cast<int>(queries.size());
    std::vector<std::vector<SearchResult>> out(static_cast<size_t>(nq));

    const int nt = (prop_.n_search_threads <= 0)
                   ? omp_get_max_threads()
                   : prop_.n_search_threads;

    // Note: each search() acquires a shared_lock; multiple shared locks coexist safely.
    // If the caller is already in an OMP parallel region, set n_search_threads=1
    // (or omp_set_nested(false)) to avoid nested parallelism overhead.
#pragma omp parallel for schedule(dynamic, 8) num_threads(nt)
    for (int qi = 0; qi < nq; ++qi) {
        out[static_cast<size_t>(qi)] = search(queries[static_cast<size_t>(qi)], k);
    }
    return out;
}

// ---------------------------------------------------------------------------
// insert
// ---------------------------------------------------------------------------
uint32_t NGTAQIndex::insert(const std::vector<float>& vec) {
    if (static_cast<int>(vec.size()) < prop_.dimension)
        throw std::invalid_argument("NGTAQIndex::insert: vector dimension mismatch");
    const int D = prop_.dimension;
    const int words = D / 64;

    std::vector<uint64_t> bq_buf(static_cast<size_t>(words) * 2);
    bq_.encode(vec.data(), bq_buf.data());

    std::unique_lock<std::shared_mutex> lock(graph_->mutex());
    uint32_t new_id = graph_->addNode(bq_buf.data());

    // Append raw float vector to flat array
    raw_flat_.insert(raw_flat_.end(), vec.begin(), vec.begin() + D);
    // Normalize in-place for cosine metric
    if (prop_.metric == NGT::ObjectSpace::DistanceTypeAngle ||
        prop_.metric == NGT::ObjectSpace::DistanceTypeCosine) {
        float* v = raw_flat_.data() + static_cast<size_t>(new_id) * D;
        float norm_sq = 0.0f;
        for (int j = 0; j < D; ++j) norm_sq += v[j] * v[j];
        if (norm_sq > 0.0f) {
            float inv_norm = 1.0f / std::sqrt(norm_sq);
            for (int j = 0; j < D; ++j) v[j] *= inv_norm;
        }
    }

    if (graph_->size() > 1) {
        graph_->finalizeCSR();

        auto cand_ids = searcher_.route(bq_buf.data(),
            std::min(prop_.max_edges, static_cast<int>(graph_->size()) - 1),
            *graph_, entry_points_);

        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(cand_ids.size());
        for (uint32_t cid : cand_ids) {
            if (cid == new_id) continue;
            float d = bqDistance(graph_->getNodeBQ(new_id), graph_->getNodeBQ(cid), words, D);
            candidates.push_back({cid, d});
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(graph_->getNodeBQ(v), graph_->getNodeBQ(u), words, D);
        };
        auto pruned = pruner_.prune(candidates, bq_.tau(), dist_fn);
        graph_->setNeighbors(new_id, pruned);
    } else {
        graph_->finalizeCSR();
    }
    return new_id;
}

// ---------------------------------------------------------------------------
// remove
// ---------------------------------------------------------------------------
void NGTAQIndex::remove(uint32_t id) {
    std::unique_lock<std::shared_mutex> lock(graph_->mutex());
    graph_->removeNode(id);
}

// ---------------------------------------------------------------------------
// rebuild
// ---------------------------------------------------------------------------
void NGTAQIndex::rebuild() {
    std::unique_lock<std::shared_mutex> lock(graph_->mutex());

    const size_t N = graph_->size();
    const int D = prop_.dimension;
    std::vector<uint32_t> old_to_new(N, static_cast<uint32_t>(-1));
    uint32_t next_id = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
        if (!graph_->isTombstone(i)) old_to_new[i] = next_id++;
    }

    // Reorder raw_flat_ to match post-rebuild node ordering
    std::vector<float> new_flat(static_cast<size_t>(next_id) * D);
    for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
        if (old_to_new[i] == static_cast<uint32_t>(-1)) continue;
        size_t src_off = static_cast<size_t>(i) * D;
        size_t dst_off = static_cast<size_t>(old_to_new[i]) * D;
        if (src_off + D <= raw_flat_.size()) {
            std::copy(raw_flat_.begin() + static_cast<ptrdiff_t>(src_off),
                      raw_flat_.begin() + static_cast<ptrdiff_t>(src_off + D),
                      new_flat.begin() + static_cast<ptrdiff_t>(dst_off));
        }
    }
    raw_flat_ = std::move(new_flat);

    graph_->rebuild();

    int n_ep = std::min(prop_.n_entry_points, static_cast<int>(graph_->size()));
    entry_points_ = selectEntryPoints(*graph_, n_ep);
}

// ---------------------------------------------------------------------------
// size
// ---------------------------------------------------------------------------
size_t NGTAQIndex::size() const {
    return graph_->activeCount();
}

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------
void NGTAQIndex::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("NGTAQIndex::save: cannot open " + path);

    // 1. Property (raw bytes)
    os.write(reinterpret_cast<const char*>(&prop_), sizeof(prop_));

    // 2. BinaryQuantizer
    bq_.serialize(os);

    // 3. SoAGraph
    graph_->serialize(os);

    // 4. Entry points
    uint32_t n_ep = static_cast<uint32_t>(entry_points_.size());
    os.write(reinterpret_cast<const char*>(&n_ep), sizeof(n_ep));
    if (n_ep > 0)
        os.write(reinterpret_cast<const char*>(entry_points_.data()),
                 n_ep * sizeof(uint32_t));

    // 5. Raw flat vectors: uint64_t n_floats, then float array
    uint64_t n_floats = static_cast<uint64_t>(raw_flat_.size());
    os.write(reinterpret_cast<const char*>(&n_floats), sizeof(n_floats));
    if (n_floats > 0)
        os.write(reinterpret_cast<const char*>(raw_flat_.data()),
                 static_cast<std::streamsize>(n_floats * sizeof(float)));

    os.flush();
    if (!os) throw std::runtime_error("NGTAQIndex::save: write error on " + path);
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
NGTAQIndex NGTAQIndex::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("NGTAQIndex::load: cannot open " + path);

    // 1. Property
    Property prop;
    is.read(reinterpret_cast<char*>(&prop), sizeof(prop));
    if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read property from " + path);
    if (prop.dimension <= 0 || prop.dimension > 65536)
        throw std::runtime_error("NGTAQIndex::load: invalid dimension in file");

    // 2. BinaryQuantizer
    BinaryQuantizer bq;
    bq.deserialize(is);

    // 3. SoAGraph
    auto graph = std::make_unique<SoAGraph>(prop.dimension / 64);
    graph->deserialize(is);

    // 4. Entry points
    uint32_t n_ep = 0;
    is.read(reinterpret_cast<char*>(&n_ep), sizeof(n_ep));
    if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read entry point count");
    // Upper bound: graph node count (already loaded above). Corrupt files may set n_ep
    // to an absurdly large value — cap it at graph size as a sanity check.
    if (n_ep > static_cast<uint64_t>(graph->size()))
        throw std::runtime_error("NGTAQIndex::load: n_ep exceeds graph size (file corrupt?)");
    std::vector<uint32_t> entry_points(n_ep);
    if (n_ep > 0)
        is.read(reinterpret_cast<char*>(entry_points.data()), n_ep * sizeof(uint32_t));

    // 5. Raw flat vectors
    uint64_t n_floats = 0;
    is.read(reinterpret_cast<char*>(&n_floats), sizeof(n_floats));
    if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read vec count");
    // Upper bound = dimension × maximum reasonable vector count (20M vectors).
    // Full consistency check (n_floats == prop_.dimension * graph->size()) is done
    // implicitly: the read below will fail or produce wrong results if corrupt.
    const uint64_t max_reasonable = static_cast<uint64_t>(prop.dimension) * 20000000ULL;
    if (n_floats > max_reasonable)
        throw std::runtime_error("NGTAQIndex::load: n_floats exceeds limit (file corrupt?)");
    std::vector<float> raw_flat(n_floats);
    if (n_floats > 0)
        is.read(reinterpret_cast<char*>(raw_flat.data()),
                static_cast<std::streamsize>(n_floats * sizeof(float)));

    if (!is)
        throw std::runtime_error("NGTAQIndex::load: stream error reading " + path);

    return NGTAQIndex(prop, std::move(bq), std::move(graph),
                      std::move(entry_points), std::move(raw_flat));
}

// ---------------------------------------------------------------------------
// selectEntryPoints
// ---------------------------------------------------------------------------
std::vector<uint32_t> NGTAQIndex::selectEntryPoints(
    const SoAGraph& graph, int n, uint32_t seed)
{
    if (n <= 0 || graph.size() == 0) return {};

    const int words = graph.words();
    const int D = words * 64;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> pick(
        0, static_cast<uint32_t>(graph.size()) - 1);

    std::vector<uint32_t> selected;
    selected.reserve(static_cast<size_t>(n));

    for (int attempt = 0; attempt < 1000 && selected.empty(); ++attempt) {
        uint32_t c = pick(rng);
        if (!graph.isTombstone(c)) selected.push_back(c);
    }
    if (selected.empty()) return {};

    while (static_cast<int>(selected.size()) < n) {
        const int cand_size = std::min(200, static_cast<int>(graph.activeCount()));
        if (cand_size == 0) break;
        float best_min_dist = -1.0f;
        uint32_t best_id = selected[0];

        for (int t = 0; t < cand_size; ++t) {
            uint32_t c = pick(rng);
            if (graph.isTombstone(c)) continue;

            bool already = false;
            for (uint32_t s : selected) {
                if (s == c) { already = true; break; }
            }
            if (already) continue;

            float min_d = std::numeric_limits<float>::infinity();
            for (uint32_t s : selected) {
                float d = bqDistance(graph.getNodeBQ(s), graph.getNodeBQ(c), words, D);
                if (d < min_d) min_d = d;
            }
            if (min_d > best_min_dist) {
                best_min_dist = min_d;
                best_id = c;
            }
        }
        if (best_min_dist < 0.0f) break;
        selected.push_back(best_id);
    }
    return selected;
}

// ---------------------------------------------------------------------------
// fromNGTv2: SRHT + K-means + PCA + VectorRecord + cluster-aware graph
// ---------------------------------------------------------------------------
NGTAQIndex NGTAQIndex::fromNGTv2(const std::string& ngt_path, const Property& prop) {
    const int D = prop.dimension;
    if ((D & (D - 1)) != 0)
        throw std::invalid_argument("fromNGTv2: dimension must be a power of 2");
    if (D % 64 != 0)
        throw std::invalid_argument("fromNGTv2: dimension must be divisible by 64");

    // ---- 1. Load all float vectors from NGT ----
    NGT::Index ngt(ngt_path);
    NGT::ObjectSpace& objspace = ngt.getObjectSpace();
    const size_t repo_size = objspace.getRepository().size();
    const size_t N = repo_size - 1;
    const int words = D / 64;

    std::vector<float> raw_flat(N * static_cast<size_t>(D), 0.f);
    std::vector<bool> is_hole(N, false);
    std::vector<float> tmp(D);
    for (size_t i = 1; i <= N; i++) {
        try {
            objspace.getObject(static_cast<NGT::ObjectID>(i), tmp);
            std::copy(tmp.begin(), tmp.end(),
                      raw_flat.begin() + static_cast<ptrdiff_t>((i - 1) * D));
        } catch (...) {
            is_hole[i - 1] = true;
        }
    }
    fprintf(stderr, "[NGTAQv2] Loaded %zu vectors D=%d\n", N, D);

    // ---- 2. SRHT: rotate all vectors ----
    const uint64_t seed = 0xCAFEBABE12345678ULL;
    auto srht = std::make_unique<NGT::NGTAQ::SRHT>(D, seed);
    std::vector<float> rotated(N * static_cast<size_t>(D));
    for (size_t i = 0; i < N; ++i)
        srht->apply(raw_flat.data() + i*D, rotated.data() + i*D);

    // ---- 3. K-means on rotated vectors ----
    uint32_t K = NGT::NGTAQ::select_k(N);
    auto kmeans = std::make_unique<NGT::NGTAQ::KMeansCentering>(K, D, seed ^ 0xFFFF);
    fprintf(stderr, "[NGTAQv2] K-means K=%u...\n", K);
    kmeans->train(rotated.data(), N);

    // ---- 4. Assign and compute residuals ----
    std::vector<uint32_t> centroid_ids(N);
    kmeans->assign(rotated.data(), N, centroid_ids.data());
    std::vector<float> residuals(N * static_cast<size_t>(D));
    for (size_t i = 0; i < N; ++i)
        kmeans->get_residual(rotated.data() + i*D, centroid_ids[i], residuals.data() + i*D);

    // ---- 5. PCA top-32 on residuals ----
    auto pca = std::make_unique<NGT::NGTAQ::PCAProjector>(D, 32, seed ^ 0x1234);
    size_t fit_n = std::min(N, (size_t)262144);
    fprintf(stderr, "[NGTAQv2] PCA fit on %zu residuals...\n", fit_n);
    pca->fit(residuals.data(), fit_n);

    // ---- 6. PCA-project all residuals ----
    std::vector<float> pca_residuals(N * 32);
    for (size_t i = 0; i < N; ++i)
        pca->project(residuals.data() + i*D, pca_residuals.data() + i*32);

    // ---- 7. Tier-2 PQ: 16 sub-codebooks on SRHT residuals (all D dims) ----
    // M=16 sub-spaces × D/16 dims each (D_sub=8 for D=128). K=256 centroids (8 bit/sub).
    // Layout: tier2_cb[(sub*256 + code)*D_sub + dim]
    // 16 sub-spaces × 8 bits = 128 bits → uses all 16 bytes of tier2[16].
    // SRHT isotropizes data → equal per-sub-space variance → balanced PQ.
    // 8× more centroids than previous M=32 K=16 → much higher quantization precision.
    const int M_PQ  = 16;
    const int K_PQ  = 256;
    const int D_sub = D / M_PQ;  // = 8 for D=128
    std::vector<float> tier2_cb((size_t)M_PQ * K_PQ * D_sub, 0.f);
    fprintf(stderr, "[NGTAQv2] Training %d PQ sub-codebooks (K=%d, D_sub=%d) on SRHT residuals...\n",
            M_PQ, K_PQ, D_sub);
    for (int sub = 0; sub < M_PQ; ++sub) {
        std::vector<float> sub_data(N * (size_t)D_sub);
        for (size_t i = 0; i < N; ++i)
            memcpy(sub_data.data() + i*D_sub,
                   residuals.data() + i*D + sub*D_sub,
                   (size_t)D_sub * sizeof(float));
        NGT::NGTAQ::KMeansCentering sub_km(K_PQ, D_sub, seed ^ (0xABCD1234ULL + (uint64_t)sub));
        sub_km.train(sub_data.data(), N, 262144, 50);
        for (int code = 0; code < K_PQ; ++code)
            memcpy(tier2_cb.data() + (sub*K_PQ + code)*D_sub,
                   sub_km.centroid(code),
                   (size_t)D_sub * sizeof(float));
    }
    fprintf(stderr, "[NGTAQv2] PQ sub-codebooks done.\n");

    // ---- 8. Encode all vectors into VectorRecord ----
    // Build the BQ-compatible SoAGraph (needed for existing graph infra + v1 compat)
    BinaryQuantizer bq;
    bq.init(D);
    bq.setRandomRotation();
    {
        std::vector<const float*> ptrs(N);
        for (size_t i = 0; i < N; i++) ptrs[i] = raw_flat.data() + i * D;
        bq.calibrateTau(ptrs, prop.n_tau_samples, prop.metric);
    }
    auto graph = std::make_unique<SoAGraph>(words);
    {
        std::vector<uint64_t> bq_buf(static_cast<size_t>(words) * 2);
        for (size_t i = 0; i < N; i++) {
            bq.encode(raw_flat.data() + i * D, bq_buf.data());
            graph->addNode(bq_buf.data());
        }
    }
    graph->finalizeCSR();
    for (size_t i = 0; i < N; ++i)
        if (is_hole[i]) graph->removeNode(static_cast<uint32_t>(i));

    // Fill v2 VectorRecords
    graph->reserveV2(N);
    for (size_t i = 0; i < N; ++i) {
        if (is_hole[i]) continue;
        NGT::NGTAQ::VectorRecord rec = {};
        rec.centroid_id = centroid_ids[i];

        // tier-1: sign bits of SRHT residual (128 bits → 16 bytes)
        const float* res = residuals.data() + i*D;
        for (int b = 0; b < D; ++b)
            NGT::NGTAQ::set_tier1_bit(rec, b, res[b] >= 0.f);

        // norm_fp16: L2 norm of residual
        float norm2 = 0.f;
        for (int d = 0; d < D; ++d) norm2 += res[d] * res[d];
        rec.norm_fp16 = NGT::NGTAQ::float_to_fp16(std::sqrt(norm2));

        // tier-2: 16 independent PQ codes (bytes 0..15 of tier2[16])
        // Each byte encodes nearest centroid (0-255) for D_sub-dim sub-vector of SRHT residual
        const float* sv_base = residuals.data() + i * D;
        for (int sub = 0; sub < M_PQ; ++sub) {
            const float* sv = sv_base + sub * D_sub;
            float best_d = std::numeric_limits<float>::max();
            uint8_t best_code = 0;
            for (int code = 0; code < K_PQ; ++code) {
                const float* c = tier2_cb.data() + (sub*K_PQ + code) * D_sub;
                float dist = 0.f;
                for (int dd = 0; dd < D_sub; ++dd) {
                    float diff = sv[dd] - c[dd]; dist += diff * diff;
                }
                if (dist < best_d) { best_d = dist; best_code = (uint8_t)code; }
            }
            NGT::NGTAQ::set_tier2_byte(rec, sub, best_code);
        }

        graph->setRecord(static_cast<uint32_t>(i), rec);
    }

    // ---- 9. Build cluster-aware graph from NGT edges ----
    NGT::GraphIndex& gi = static_cast<NGT::GraphIndex&>(ngt.getIndex());
    AlphaCGPruner pruner(prop.alpha, prop.kappa);
    const float tau = bq.tau();

    std::vector<std::vector<uint32_t>> adj(N);
    for (size_t i = 1; i <= N; i++) {
        uint32_t aq_id = static_cast<uint32_t>(i - 1);
        if (is_hole[aq_id]) continue;
        NGT::GraphNode* node = nullptr;
        try { node = gi.getNode(static_cast<NGT::ObjectID>(i)); }
        catch (...) { continue; }
        if (!node || node->empty()) continue;

        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(node->size());
        for (auto& edge : *node) {
            if (edge.id == 0 || edge.id > static_cast<unsigned int>(N)) continue;
            uint32_t nbr = static_cast<uint32_t>(edge.id - 1);
            float d_bq = bqDistance(graph->getNodeBQ(aq_id), graph->getNodeBQ(nbr), words, D);
            candidates.push_back({nbr, d_bq});
        }
        // Cluster-aware sort: same centroid neighbors first, then by BQ distance
        std::stable_sort(candidates.begin(), candidates.end(),
            [&](const auto& a, const auto& b) {
                bool a_same = (centroid_ids[a.first] == centroid_ids[aq_id]);
                bool b_same = (centroid_ids[b.first] == centroid_ids[aq_id]);
                if (a_same != b_same) return a_same > b_same;
                return a.second < b.second;
            });
        if (static_cast<int>(candidates.size()) > prop.max_edges)
            candidates.resize(static_cast<size_t>(prop.max_edges));

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(graph->getNodeBQ(v), graph->getNodeBQ(u), words, D);
        };
        adj[aq_id] = pruner.prune(candidates, tau, dist_fn);
    }
    graph->resetEdges(adj);

    int n_ep = std::min(prop.n_entry_points, static_cast<int>(N));
    auto entry_points = selectEntryPoints(*graph, n_ep);

    // Construct index
    NGTAQIndex idx(prop, std::move(bq), std::move(graph),
                   std::move(entry_points), std::move(raw_flat));
    idx.is_v2_ = true;
    idx.srht_v2_        = std::move(srht);
    idx.kmeans_v2_      = std::move(kmeans);
    idx.pca_v2_         = std::move(pca);
    idx.tier2_codebook_ = std::move(tier2_cb);
    // Build transposed codebook [M][D_sub][K] for AVX2 FMA LUT build
    {
        const int Dv2 = D;
        const int Mv2 = M_PQ, Kv2 = K_PQ, DSv2 = D_sub;
        idx.tier2_codebook_T_.resize((size_t)Mv2 * DSv2 * Kv2);
        NGT::NGTAQ::build_tier2_codebook_T(
            idx.tier2_codebook_.data(), Mv2, Kv2, DSv2,
            idx.tier2_codebook_T_.data());
        (void)Dv2;
    }
    idx.v2_entry_points_ = idx.entry_points_;  // reuse existing entry points for v2

    fprintf(stderr, "[NGTAQv2] Build complete. N=%zu K=%u\n", N, K);
    return idx;
}

// ---------------------------------------------------------------------------
// searchV2: ADC search with lazy centroid switch
// ---------------------------------------------------------------------------
std::vector<SearchResult> NGTAQIndex::searchV2(
    const std::vector<float>& query, int k,
    float gamma_enq, float gamma_term) const
{
    if (!is_v2_)
        throw std::runtime_error("searchV2: call fromNGTv2() first");
    if (static_cast<int>(query.size()) < prop_.dimension)
        throw std::invalid_argument("searchV2: query dimension mismatch");

    const int D = prop_.dimension;

    // 1. Rotate query
    std::vector<float> q_rot(D);
    srht_v2_->apply(query.data(), q_rot.data());

    // 2. Find query's nearest centroid
    uint32_t active_cid = kmeans_v2_->nearest_public(q_rot.data());
    std::vector<float> q_res(D);

    // 3. Build initial ADC state (tier-1 + tier-2 PQ on SRHT residuals)
    NGT::NGTAQ::ADCQueryState adc = {};
    float q_norm_sq = 0.f;
    NGT::NGTAQ::compute_residual_and_tier1(
        q_rot.data(), kmeans_v2_->centroid(active_cid), D,
        q_res.data(), adc.q_norm_sq, adc.q_int8, adc.q_sum);
    adc.q_norm = std::sqrt(adc.q_norm_sq);
    q_norm_sq = adc.q_norm_sq;
    const float inv_sqrt_D = 1.f / std::sqrt((float)D);

    // Save initial cluster residual for tier-2 LUT build post-routing.
    // maybe_rebuild_adc overwrites q_res/q_norm_sq on cluster transitions.
    const uint32_t initial_cid = active_cid; (void)initial_cid;
    std::vector<float> q_res_initial(q_res);
    const float q_norm_sq_initial = q_norm_sq;

    std::shared_lock<std::shared_mutex> lock(graph_->mutex());
    const size_t N = graph_->size();

    // Lazy-build cluster inverted list + precomputed cluster neighbors (once per index lifetime).
    // Done under shared_lock (N stable). call_once provides thread-safe one-shot semantics.
    std::call_once(*cluster_members_once_, [this, N]() {
        const uint32_t K = kmeans_v2_->num_clusters();
        const int Dim    = prop_.dimension;

        // 1. Build inverted list: cluster_id → [node_ids]
        cluster_members_v2_.resize(K);
        for (size_t i = 0; i < N; ++i) {
            if (graph_->isTombstone(static_cast<uint32_t>(i))) continue;
            uint32_t cid = graph_->getRecord(static_cast<uint32_t>(i)).centroid_id;
            if (cid < K)
                cluster_members_v2_[cid].push_back(static_cast<uint32_t>(i));
        }

        // 2. Precompute top-N_EXTRA_CLUSTERS nearest clusters for each cluster.
        // Cost: K² × D scalar ops (K=1000, D=128 → 128M ops, ~15ms) — one-time amortized.
        constexpr int CLUSTER_NBRS = 2;
        cluster_neighbors_v2_.resize(K);
        using CD = std::pair<float, uint32_t>;
        std::vector<CD> dists;
        dists.reserve(K);
        for (uint32_t c = 0; c < K; ++c) {
            const float* cc = kmeans_v2_->centroid(c);
            dists.clear();
            for (uint32_t c2 = 0; c2 < K; ++c2) {
                if (c2 == c) continue;
                const float* cc2 = kmeans_v2_->centroid(c2);
                float d2 = NGT::NGTAQ::KMeansCentering::l2sq(cc, cc2, Dim);
                dists.push_back({d2, c2});
            }
            int take = std::min((int)dists.size(), CLUSTER_NBRS);
            std::partial_sort(dists.begin(), dists.begin() + take, dists.end());
            cluster_neighbors_v2_[c].resize(static_cast<size_t>(take));
            for (int i = 0; i < take; ++i)
                cluster_neighbors_v2_[c][i] = dists[i].second;
        }
    });

    // 4. DABS search with asymmetric PQ ADC + lazy centroid rebuild
    using Entry = std::pair<float, uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> cand_q;
    std::priority_queue<float> dk_tracker;
    std::vector<std::pair<float, uint32_t>> results;
    // Reserve 30× k upfront to avoid repeated reallocations during DABS routing.
    // At γ=0.18 ~200-300 candidates are popped; 30×10=300 covers the common case.
    results.reserve(static_cast<size_t>(k * 30));
    // Flat bitvector for visited tracking: N=1M → 15,625 uint64_t = 125KB (fits in L2)
    // thread_local avoids heap allocation after first query on each thread
    static thread_local std::vector<uint64_t> t_vis;
    t_vis.assign((N + 63) / 64, 0ULL);
    auto is_visited = [&](uint32_t id) -> bool {
        return (t_vis[id >> 6] >> (id & 63)) & 1ULL;
    };
    auto mark_visited = [&](uint32_t id) {
        t_vis[id >> 6] |= 1ULL << (id & 63);
    };
    float d_k = std::numeric_limits<float>::infinity();

    // ADC state cache: skip get_residual + q_norm_sq loop + build_tier1_query on
    // repeated visits to the same cluster.
    // 8 slots × 144 bytes = 1152 bytes (fits in L1). DABS visits 3-15 unique clusters
    // per query; 8 slots eliminate evictions for typical cases, reducing tail latency.
    constexpr int ADC_SLOTS = 8;
    struct ADCSlot {
        uint32_t cid;
        float    q_norm_sq;
        float    q_norm;
        int32_t  q_sum;
        int8_t   q_int8[128];
    };
    ADCSlot adc_cache[ADC_SLOTS];
    // Slot 0 = initial cluster (ADC state already computed above)
    adc_cache[0].cid       = active_cid;
    adc_cache[0].q_norm_sq = adc.q_norm_sq;
    adc_cache[0].q_norm    = adc.q_norm;
    adc_cache[0].q_sum     = adc.q_sum;
    std::memcpy(adc_cache[0].q_int8, adc.q_int8, sizeof(adc.q_int8));
    for (int s = 1; s < ADC_SLOTS; ++s) adc_cache[s].cid = UINT32_MAX;
    int adc_cache_hand = 1;

    // Tier-1 routing: fast RaBitQ ADC for DABS beam search.
    // Tier-2 contributes only via seeding (pre-sorts cluster candidates before routing).
    // Note: tier-2 routing was tested and regressed QPS 8x — accurate distances delay
    // DABS stopping criterion, causing many more node visits than tier-1 noise allows.
    auto adc_dist = [&](const NGT::NGTAQ::VectorRecord& rec) -> float {
        float norm_x = NGT::NGTAQ::fp16_to_float(rec.norm_fp16);
        float t1 = NGT::NGTAQ::tier1_adc_fast(adc.q_int8, rec.tier1, adc.q_sum);
        float t1_ip = adc.q_norm * norm_x * NGT::NGTAQ::RABITQ_SCALE * t1 * inv_sqrt_D / 127.f;
        return adc.q_norm_sq + norm_x * norm_x - 2.0f * t1_ip;
    };

    // Rebuild tier-1 ADC tables on cluster boundary crossing, with cache lookup.
    // Cache hit (~10ns): restore from 576-byte L1-resident table, skip all recomputation.
    // Cache miss: full rebuild (get_residual + q_norm_sq + sqrt + build_tier1_query ~150ns).
    auto maybe_rebuild_adc = [&](uint32_t cid) {
        if (cid == active_cid) return;
        for (int s = 0; s < ADC_SLOTS; ++s) {
            if (adc_cache[s].cid == cid) {
                active_cid    = cid;
                q_norm_sq     = adc_cache[s].q_norm_sq;
                adc.q_norm_sq = adc_cache[s].q_norm_sq;
                adc.q_norm    = adc_cache[s].q_norm;
                adc.q_sum     = adc_cache[s].q_sum;
                std::memcpy(adc.q_int8, adc_cache[s].q_int8, sizeof(adc.q_int8));
                return;
            }
        }
        // Cache miss: full rebuild
        active_cid = cid;
        NGT::NGTAQ::compute_residual_and_tier1(
            q_rot.data(), kmeans_v2_->centroid(active_cid), D,
            q_res.data(), adc.q_norm_sq, adc.q_int8, adc.q_sum);
        adc.q_norm = std::sqrt(adc.q_norm_sq);
        q_norm_sq = adc.q_norm_sq;
        // Store in cache (round-robin eviction)
        const int slot = adc_cache_hand;
        adc_cache_hand = (adc_cache_hand + 1) % ADC_SLOTS;
        adc_cache[slot].cid       = cid;
        adc_cache[slot].q_norm_sq = adc.q_norm_sq;
        adc_cache[slot].q_norm    = adc.q_norm;
        adc_cache[slot].q_sum     = adc.q_sum;
        std::memcpy(adc_cache[slot].q_int8, adc.q_int8, sizeof(adc.q_int8));
    };

    // Tier-2 seed scoring: build LUT from initial cluster, pre-sort all cluster seeds
    // by tier-2 distance. Better seeds → routing converges faster → higher recall at
    // same gamma_term. Cost: 1 LUT build (~0.5μs AVX2) + N_seeds × 10ns ADC.
    // LUT built from q_res_initial (saved before routing modifies q_res).
    float t2_seed_lut[16][256];
    NGT::NGTAQ::build_tier2_lut_fast(q_res_initial.data(), D,
                                      tier2_codebook_T_.data(),
                                      t2_seed_lut);

    // Cluster-aware seeding: use members of the query's nearest cluster as entry points.
    // This places the search start close to where the true neighbors are, typically
    // requiring fewer hops to converge → better recall at same gamma_term → higher QPS.
    // N_CLUSTER_SEEDS controls the breadth; 32 is a good default for K≈1000, N=1M.
    constexpr int N_CLUSTER_SEEDS = 32;
    // Neighboring clusters seeded via precomputed cluster_neighbors_v2_ (built in call_once)
    {
        // Gather seed IDs from the nearest cluster(s)
        std::vector<uint32_t> seeds;
        seeds.reserve(static_cast<size_t>(N_CLUSTER_SEEDS * 4)); // primary + up to 2 neighbor clusters

        // Primary cluster
        if (active_cid < cluster_members_v2_.size()) {
            const auto& primary = cluster_members_v2_[active_cid];
            const size_t take = std::min(primary.size(),
                                         static_cast<size_t>(N_CLUSTER_SEEDS));
            for (size_t i = 0; i < take; ++i) seeds.push_back(primary[i]);
        }

        // Expand to neighboring clusters using precomputed cluster neighbor table (O(1)).
        // cluster_neighbors_v2_[active_cid] gives the nearest CLUSTER_NBRS clusters,
        // precomputed offline during call_once — zero per-query centroid scan overhead.
        if (active_cid < cluster_neighbors_v2_.size()) {
            for (uint32_t cid2 : cluster_neighbors_v2_[active_cid]) {
                if (cid2 >= cluster_members_v2_.size()) continue;
                const auto& nbr_members = cluster_members_v2_[cid2];
                const size_t take = std::min(nbr_members.size(),
                                              static_cast<size_t>(N_CLUSTER_SEEDS));
                for (size_t i = 0; i < take; ++i) seeds.push_back(nbr_members[i]);
            }
        }

        // Fall back to static entry points if cluster membership is empty
        if (seeds.empty()) {
            for (uint32_t ep : entry_points_) seeds.push_back(ep);
        }

        // Tier-2 sort seeds before routing: score by PQ ADC using initial-cluster LUT.
        // Seeds in the same cluster as the query get accurate scores; cross-cluster seeds
        // get an approximation (different residual basis), but still better than random order.
        // Sorting ~96 floats costs ~0.5μs; benefit: routing starts from the closest seed.
        {
            // Bulk prefetch all seed records before the scoring loop.
            // Issuing all ~96 prefetches at once lets the hardware overlap multiple
            // DRAM requests (max ~20 outstanding) instead of serialising them.
            // Expected gain: ~5μs → ~0.5μs for the cold-miss phase.
            for (uint32_t ep : seeds) {
                if (ep < N) graph_->prefetchRecord(ep);
            }

            struct SeedScore { float score; uint32_t id; };
            std::vector<SeedScore> scored;
            scored.reserve(seeds.size());
            for (uint32_t ep : seeds) {
                if (ep >= N || graph_->isTombstone(ep)) continue;
                const auto& rec = graph_->getRecord(ep);
                float norm_x = NGT::NGTAQ::fp16_to_float(rec.norm_fp16);
                float t2_ip = NGT::NGTAQ::tier2_adc_pq(t2_seed_lut, rec.tier2);
                // Use initial-cluster residual norm for L2 estimate
                float d_approx = q_norm_sq_initial + norm_x * norm_x - 2.0f * t2_ip;
                scored.push_back({d_approx, ep});
            }
            std::sort(scored.begin(), scored.end(),
                      [](const SeedScore& a, const SeedScore& b){ return a.score < b.score; });
            // Push sorted seeds into priority queue (best first for warm d_k)
            for (const auto& s : scored) {
                if (is_visited(s.id)) continue;
                mark_visited(s.id);
                const auto& rec = graph_->getRecord(s.id);
                maybe_rebuild_adc(rec.centroid_id);
                float d = adc_dist(rec);
                cand_q.push({d, s.id});
                graph_->prefetchOffset(s.id);
            }
        }
    }

    while (!cand_q.empty()) {
        auto [dist_x, x] = cand_q.top(); cand_q.pop();

        if (dk_tracker.size() >= static_cast<size_t>(k) &&
            dist_x > (1.f + gamma_term) * d_k) break;

        // Prefetch next-popped node's data while we process current node
        if (!cand_q.empty()) {
            uint32_t nxt = cand_q.top().second;
            graph_->prefetchRecord(nxt);
            graph_->prefetchNeighbors(nxt);
        }
        // Prefetch current node's neighbor list (hides CSR access latency)
        graph_->prefetchNeighbors(x);

        const auto& rec_x = graph_->getRecord(x);
        maybe_rebuild_adc(rec_x.centroid_id);
        float d_approx = adc_dist(rec_x);

        // Add ALL popped candidates to results for exact reranking.
        // We only use d_k for ROUTING termination, not for result filtering.
        results.push_back({d_approx, x});
        dk_tracker.push(d_approx);
        if (static_cast<int>(dk_tracker.size()) > k) {
            dk_tracker.pop();
            d_k = dk_tracker.top();
        } else if (static_cast<int>(dk_tracker.size()) == k) {
            d_k = dk_tracker.top();
        }

        auto neighbors = graph_->getNeighbors(x);
        const size_t n_nbrs = neighbors.size();
        // Sliding-window prefetch (PFDIST=8): issue record prefetch 8 iterations ahead.
        // With 8-slot ADC cache, hit-path cost ~10ns → 8×10=80ns look-ahead covers DRAM
        // (~100ns). Bulk (n_nbrs=20-40 simultaneous) overloads the CPU LSQ/MSHR, competing
        // with centroid and offset prefetches already in-flight, causing P99 spikes.
        constexpr int PFDIST = 8;
        for (size_t pf = 0; pf < std::min((size_t)PFDIST, n_nbrs); ++pf)
            graph_->prefetchRecord(neighbors[pf]);

        for (size_t ni = 0; ni < n_nbrs; ++ni) {
            if (ni + PFDIST < n_nbrs)
                graph_->prefetchRecord(neighbors[ni + PFDIST]);

            uint32_t u = neighbors[ni];
            if (u >= N || graph_->isTombstone(u)) continue;
            if (is_visited(u)) continue;
            mark_visited(u);
            const auto& rec_u = graph_->getRecord(u);
            maybe_rebuild_adc(rec_u.centroid_id);
            float d_u = adc_dist(rec_u);
            // Skip hopeless candidates: when d_k is initialized, a node with
            // d_u > (1+gamma)*d_k would trigger the outer-loop termination as
            // soon as it's popped — it can never contribute to the top-k result.
            // Skipping the push avoids a wasted heap insertion+extraction.
            if (static_cast<int>(dk_tracker.size()) >= k &&
                d_u > (1.f + gamma_term) * d_k)
                continue;
            cand_q.push({d_u, u});
            // Prefetch offset for u: when u is eventually popped and
            // prefetchNeighbors(u) is called, offsets_[u] will already be
            // in L1/L2 cache, eliminating the blocking DRAM read that
            // gates the neighbor-list prefetch.
            graph_->prefetchOffset(u);
        }
    }

    // Select top k*15 candidates by approximate score for exact L2 reranking.
    // nth_element (O(n)) is faster than sort (O(n log n)) when n >> refine_n.
    // tier-1 has ~0.5bit noise but good rank correlation for initial filtering.
    const size_t refine_n = static_cast<size_t>(k * 15);
    if (results.size() > refine_n) {
        std::nth_element(results.begin(), results.begin() + refine_n, results.end());
        results.resize(refine_n);
    }

    // Exact L2 refinement: l2_sq_avx2 for D=128 ≈ 5ns/vector × 150 = 0.75μs.
    // Store squared distances to avoid redundant sqrts during sort; take sqrt only
    // for the final top-k (10 sqrts instead of 150).
    std::vector<SearchResult> final_results;
    final_results.reserve(results.size());
    for (auto& [approx_d, id] : results) {
        if (static_cast<size_t>(id) * D + D > raw_flat_.size()) continue;
        const float* vec = raw_flat_.data() + static_cast<size_t>(id) * D;
#if defined(__AVX2__)
        float exact_sq = NGT::NGTAQ::l2_sq_avx2(query.data(), vec, D);
#else
        float exact_sq = 0.f;
        for (int j = 0; j < D; ++j) { float d = query[j] - vec[j]; exact_sq += d*d; }
#endif
        // Store exact_sq in .distance temporarily (sqrt deferred until after sort).
        final_results.push_back({id, exact_sq, approx_d});
    }
    // partial_sort: O(n log k) ≈ 500 comparisons vs std::sort O(n log n) ≈ 1080.
    // We only need the top-k; the rest are discarded.
    const size_t out_n = std::min(static_cast<size_t>(k), final_results.size());
    std::partial_sort(final_results.begin(), final_results.begin() + out_n,
        final_results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        });
    final_results.resize(out_n);
    // Now apply sqrt to the top-k distances (deferred from above).
    for (auto& r : final_results) r.distance = std::sqrt(r.distance);
    return final_results;
}

// ---------------------------------------------------------------------------
// saveV2 / loadV2
// ---------------------------------------------------------------------------
void NGTAQIndex::saveV2(const std::string& dir) const {
    if (!is_v2_) return;
    // SoAGraph v2 records
    graph_->saveV2Records(dir + "/v2_records.bin");
    // SRHT diagonal
    {
        std::vector<float> diag;
        srht_v2_->serialize(diag);
        std::ofstream f(dir + "/v2_srht.bin", std::ios::binary);
        uint32_t Dsz = (uint32_t)diag.size();
        f.write(reinterpret_cast<const char*>(&Dsz), sizeof(Dsz));
        f.write(reinterpret_cast<const char*>(diag.data()), Dsz * sizeof(float));
    }
    // K-means centroids
    {
        std::ofstream f(dir + "/v2_kmeans.bin", std::ios::binary);
        uint32_t K = kmeans_v2_->num_clusters();
        uint32_t Dim = (uint32_t)kmeans_v2_->dim();
        f.write(reinterpret_cast<const char*>(&K), sizeof(K));
        f.write(reinterpret_cast<const char*>(&Dim), sizeof(Dim));
        const auto& c = kmeans_v2_->centroids_data();
        f.write(reinterpret_cast<const char*>(c.data()), c.size() * sizeof(float));
    }
    // PCA components + mean + eigenvalues
    {
        std::ofstream f(dir + "/v2_pca.bin", std::ios::binary);
        uint32_t Dim = (uint32_t)pca_v2_->in_dim();
        uint32_t Top = (uint32_t)pca_v2_->out_dim();
        f.write(reinterpret_cast<const char*>(&Dim), sizeof(Dim));
        f.write(reinterpret_cast<const char*>(&Top), sizeof(Top));
        const auto& comp = pca_v2_->components();
        const auto& mean = pca_v2_->mean();
        const auto& eig  = pca_v2_->eigenvalues();
        f.write(reinterpret_cast<const char*>(comp.data()), comp.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(mean.data()), mean.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(eig.data()),  eig.size()  * sizeof(float));
    }
    // Tier-2 codebook
    {
        std::ofstream f(dir + "/v2_codebook.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(tier2_codebook_.data()),
                tier2_codebook_.size() * sizeof(float));
    }
}

void NGTAQIndex::loadV2(const std::string& dir) {
    graph_->loadV2Records(dir + "/v2_records.bin");
    // SRHT
    {
        std::ifstream f(dir + "/v2_srht.bin", std::ios::binary);
        uint32_t Dsz; f.read(reinterpret_cast<char*>(&Dsz), sizeof(Dsz));
        std::vector<float> diag(Dsz);
        f.read(reinterpret_cast<char*>(diag.data()), Dsz * sizeof(float));
        srht_v2_ = std::make_unique<NGT::NGTAQ::SRHT>((int)Dsz, 0);
        srht_v2_->deserialize(diag);
    }
    // K-means
    {
        std::ifstream f(dir + "/v2_kmeans.bin", std::ios::binary);
        uint32_t K, Dim;
        f.read(reinterpret_cast<char*>(&K), sizeof(K));
        f.read(reinterpret_cast<char*>(&Dim), sizeof(Dim));
        kmeans_v2_ = std::make_unique<NGT::NGTAQ::KMeansCentering>(K, (int)Dim, 0);
        std::vector<float> c((size_t)K * Dim);
        f.read(reinterpret_cast<char*>(c.data()), c.size() * sizeof(float));
        kmeans_v2_->set_centroids(std::move(c));
    }
    // PCA
    {
        std::ifstream f(dir + "/v2_pca.bin", std::ios::binary);
        uint32_t Dim, Top;
        f.read(reinterpret_cast<char*>(&Dim), sizeof(Dim));
        f.read(reinterpret_cast<char*>(&Top), sizeof(Top));
        pca_v2_ = std::make_unique<NGT::NGTAQ::PCAProjector>((int)Dim, (int)Top, 0);
        std::vector<float> comp((size_t)Top*Dim), mean(Dim), eig(Top);
        f.read(reinterpret_cast<char*>(comp.data()), comp.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(mean.data()), mean.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(eig.data()),  eig.size()  * sizeof(float));
        pca_v2_->set_state(std::move(comp), std::move(mean), std::move(eig));
    }
    // Tier-2 codebook: M=16 sub-spaces × K=256 codes × D_sub dims
    {
        std::ifstream f(dir + "/v2_codebook.bin", std::ios::binary);
        const int D_sub = prop_.dimension / 16;
        tier2_codebook_.resize((size_t)16 * 256 * D_sub);
        f.read(reinterpret_cast<char*>(tier2_codebook_.data()),
               tier2_codebook_.size() * sizeof(float));
        // Build transposed codebook [M][D_sub][K] for AVX2 FMA LUT build
        tier2_codebook_T_.resize(tier2_codebook_.size());
        NGT::NGTAQ::build_tier2_codebook_T(
            tier2_codebook_.data(), 16, 256, D_sub,
            tier2_codebook_T_.data());
    }
    is_v2_ = true;
}

} // namespace NGTAQ
