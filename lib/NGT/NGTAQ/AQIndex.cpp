// lib/NGT/NGTAQ/AQIndex.cpp
#include "NGT/NGTAQ/AQIndex.h"

#include "NGT/Graph.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <stdexcept>

namespace NGTAQ {

// ---------------------------------------------------------------------------
// Private constructor
// ---------------------------------------------------------------------------
NGTAQIndex::NGTAQIndex(Property prop, BinaryQuantizer bq,
                       std::unique_ptr<SoAGraph> graph,
                       std::vector<uint32_t> eps,
                       std::vector<std::vector<float>> raw_vecs)
    : prop_(prop)
    , bq_(std::move(bq))
    , graph_(std::move(graph))
    , pruner_(prop.alpha, prop.kappa)
    , entry_points_(std::move(eps))
    , raw_vecs_(std::move(raw_vecs))
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
    // NGT object IDs are 1-based; slot 0 is a null/dummy entry.
    const size_t repo_size = objspace.getRepository().size();
    // Count valid (non-null) objects: IDs 1 .. repo_size-1
    const size_t N = repo_size - 1;
    const int D = prop.dimension;
    const int words = D / 64;

    // Load all float vectors (NGT uses 1-based IDs)
    std::vector<std::vector<float>> raw_vecs(N, std::vector<float>(D));
    for (size_t i = 1; i <= N; i++) {
        objspace.getObject(static_cast<NGT::ObjectID>(i), raw_vecs[i - 1]);
    }

    // Init BQ with identity rotation
    BinaryQuantizer bq;
    bq.init(D);
    bq.setIdentityRotation();

    // Calibrate tau (NO D param)
    std::vector<const float*> ptrs(N);
    for (size_t i = 0; i < N; i++) ptrs[i] = raw_vecs[i].data();
    bq.calibrateTau(ptrs, prop.n_tau_samples, prop.metric);

    // Encode all vectors to BQ
    auto graph = std::make_unique<SoAGraph>(words);
    std::vector<uint64_t> sign_buf(words), mag_buf(words);
    for (size_t i = 0; i < N; i++) {
        bq.encode(raw_vecs[i].data(), sign_buf.data(), mag_buf.data());
        graph->addNode(sign_buf.data(), mag_buf.data());
    }
    graph->finalizeCSR();

    // Build alpha-CG graph from NGT edges
    AlphaCGPruner pruner(prop.alpha, prop.kappa);
    const float tau = bq.tau();
    NGT::GraphIndex& gi = static_cast<NGT::GraphIndex&>(ngt.getIndex());

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
            float d = bqDistance(
                graph->getSignPlane(aq_id), graph->getMagPlane(aq_id),
                graph->getSignPlane(nbr),   graph->getMagPlane(nbr),
                words, D);
            candidates.push_back({nbr, d});
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (static_cast<int>(candidates.size()) > prop.max_edges)
            candidates.resize(static_cast<size_t>(prop.max_edges));

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(
                graph->getSignPlane(v), graph->getMagPlane(v),
                graph->getSignPlane(u), graph->getMagPlane(u),
                words, D);
        };
        auto pruned = pruner.prune(candidates, tau, dist_fn);
        graph->setNeighbors(aq_id, pruned);
    }

    // Select entry points
    int n_ep = std::min(prop.n_entry_points, static_cast<int>(N));
    auto entry_points = selectEntryPoints(*graph, n_ep);

    return NGTAQIndex(prop, std::move(bq), std::move(graph),
                      std::move(entry_points), std::move(raw_vecs));
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------
std::vector<SearchResult> NGTAQIndex::search(
    const std::vector<float>& query, int k) const
{
    assert(static_cast<int>(query.size()) >= prop_.dimension);
    const int D = prop_.dimension;
    const int words = D / 64;

    // Encode query
    std::vector<uint64_t> q_sign(words), q_mag(words);
    bq_.encode(query.data(), q_sign.data(), q_mag.data());

    // Route in BQ space
    std::vector<uint32_t> cand_ids;
    {
        std::shared_lock<std::shared_mutex> lock(graph_->mutex());
        cand_ids = searcher_.route(q_sign.data(), q_mag.data(),
                                   k, *graph_, entry_points_);
    }

    // Refine with exact distances
    std::vector<SearchResult> results;
    results.reserve(cand_ids.size());
    for (uint32_t id : cand_ids) {
        if (id >= static_cast<uint32_t>(raw_vecs_.size())) continue;
        const auto& vec = raw_vecs_[id];
        float exact_dist = 0.0f;
        if (prop_.metric == NGT::ObjectSpace::DistanceTypeL2) {
            float sq = 0.0f;
            for (int j = 0; j < D; ++j) {
                float d = query[j] - vec[j];
                sq += d * d;
            }
            exact_dist = std::sqrt(sq);
        } else {
            // Cosine: assume vectors are normalized, dist = 1 - dot
            float dot = 0.0f;
            for (int j = 0; j < D; ++j) dot += query[j] * vec[j];
            exact_dist = 1.0f - dot;
        }
        float bq_dist = bqDistance(q_sign.data(), q_mag.data(),
                                    graph_->getSignPlane(id), graph_->getMagPlane(id),
                                    words, D);
        results.push_back({id, exact_dist, bq_dist});
    }

    // Sort by exact distance, keep top-k
    std::sort(results.begin(), results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        });
    if (static_cast<int>(results.size()) > k)
        results.resize(static_cast<size_t>(k));
    return results;
}

// ---------------------------------------------------------------------------
// insert
// ---------------------------------------------------------------------------
uint32_t NGTAQIndex::insert(const std::vector<float>& vec) {
    assert(static_cast<int>(vec.size()) >= prop_.dimension);
    const int D = prop_.dimension;
    const int words = D / 64;

    std::vector<uint64_t> sign(words), mag(words);
    bq_.encode(vec.data(), sign.data(), mag.data());

    std::unique_lock<std::shared_mutex> lock(graph_->mutex());
    uint32_t new_id = graph_->addNode(sign.data(), mag.data());
    raw_vecs_.push_back(std::vector<float>(vec.begin(), vec.begin() + D));

    if (graph_->size() > 1) {
        // finalizeCSR() to create the slot for new_id before routing/setNeighbors
        graph_->finalizeCSR();

        auto cand_ids = searcher_.route(sign.data(), mag.data(),
            std::min(prop_.max_edges, static_cast<int>(graph_->size()) - 1),
            *graph_, entry_points_);

        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(cand_ids.size());
        for (uint32_t cid : cand_ids) {
            if (cid == new_id) continue;
            float d = bqDistance(
                graph_->getSignPlane(new_id), graph_->getMagPlane(new_id),
                graph_->getSignPlane(cid),    graph_->getMagPlane(cid),
                words, D);
            candidates.push_back({cid, d});
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(
                graph_->getSignPlane(v), graph_->getMagPlane(v),
                graph_->getSignPlane(u), graph_->getMagPlane(u),
                words, D);
        };
        AlphaCGPruner pruner(prop_.alpha, prop_.kappa);
        auto pruned = pruner.prune(candidates, bq_.tau(), dist_fn);
        graph_->setNeighbors(new_id, pruned);
    } else {
        // Only one node: just finalize
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

    // Build old_to_new mapping before rebuild() shuffles IDs
    const size_t N = graph_->size();
    std::vector<uint32_t> old_to_new(N, UINT32_MAX);
    uint32_t new_id = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
        if (!graph_->isTombstone(i)) old_to_new[i] = new_id++;
    }

    graph_->rebuild();

    // Compact raw_vecs_ to match new IDs
    std::vector<std::vector<float>> new_raw_vecs(new_id);
    for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
        if (old_to_new[i] != UINT32_MAX && i < static_cast<uint32_t>(raw_vecs_.size())) {
            new_raw_vecs[old_to_new[i]] = std::move(raw_vecs_[i]);
        }
    }
    raw_vecs_ = std::move(new_raw_vecs);

    // Re-select entry points
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

    // 5. Raw vecs: uint64_t n_vecs, then for each: uint64_t dim + float array
    uint64_t n_vecs = static_cast<uint64_t>(raw_vecs_.size());
    os.write(reinterpret_cast<const char*>(&n_vecs), sizeof(n_vecs));
    for (const auto& v : raw_vecs_) {
        uint64_t dim = static_cast<uint64_t>(v.size());
        os.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        if (dim > 0)
            os.write(reinterpret_cast<const char*>(v.data()), dim * sizeof(float));
    }
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

    // 2. BinaryQuantizer
    BinaryQuantizer bq;
    bq.deserialize(is);

    // 3. SoAGraph
    auto graph = std::make_unique<SoAGraph>(prop.dimension / 64);
    graph->deserialize(is);

    // 4. Entry points
    uint32_t n_ep = 0;
    is.read(reinterpret_cast<char*>(&n_ep), sizeof(n_ep));
    std::vector<uint32_t> entry_points(n_ep);
    if (n_ep > 0)
        is.read(reinterpret_cast<char*>(entry_points.data()), n_ep * sizeof(uint32_t));

    // 5. Raw vecs
    uint64_t n_vecs = 0;
    is.read(reinterpret_cast<char*>(&n_vecs), sizeof(n_vecs));
    std::vector<std::vector<float>> raw_vecs(n_vecs);
    for (uint64_t i = 0; i < n_vecs; ++i) {
        uint64_t dim = 0;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        raw_vecs[i].resize(dim);
        if (dim > 0)
            is.read(reinterpret_cast<char*>(raw_vecs[i].data()), dim * sizeof(float));
    }

    if (!is)
        throw std::runtime_error("NGTAQIndex::load: stream error reading " + path);

    return NGTAQIndex(prop, std::move(bq), std::move(graph),
                      std::move(entry_points), std::move(raw_vecs));
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

    // Find first active node
    for (int attempt = 0; attempt < 1000 && selected.empty(); ++attempt) {
        uint32_t c = pick(rng);
        if (!graph.isTombstone(c)) selected.push_back(c);
    }
    if (selected.empty()) return {};

    while (static_cast<int>(selected.size()) < n) {
        const int cand_size = std::min(200, static_cast<int>(graph.size()));
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
                float d = bqDistance(
                    graph.getSignPlane(s), graph.getMagPlane(s),
                    graph.getSignPlane(c), graph.getMagPlane(c),
                    words, D);
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

} // namespace NGTAQ
