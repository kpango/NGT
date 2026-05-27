// lib/NGT/NGTAQ/AQIndex.cpp
#include "NGT/NGTAQ/AQIndex.h"

#include "NGT/Graph.h"

#include <algorithm>
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
    std::vector<bool> is_hole(N, false);
    for (size_t i = 1; i <= N; i++) {
        try {
            objspace.getObject(static_cast<NGT::ObjectID>(i), raw_vecs[i - 1]);
        } catch (...) {
            // NGT may have holes (deleted objects); leave slot as zero-initialized.
            raw_vecs[i - 1].assign(D, 0.0f);
            is_hole[i - 1] = true;
        }
    }

    // Pre-normalize for cosine metric so that dot(q_norm, v_norm) == cosine sim.
    if (prop.metric == NGT::ObjectSpace::DistanceTypeAngle ||
        prop.metric == NGT::ObjectSpace::DistanceTypeCosine) {
        for (auto& v : raw_vecs) {
            float norm_sq = 0.0f;
            for (float x : v) norm_sq += x * x;
            if (norm_sq > 0.0f) {
                float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (float& x : v) x *= inv_norm;
            }
        }
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

    // Tombstone ghost nodes (holes in the NGT object repository).
    for (size_t i = 0; i < N; ++i) {
        if (is_hole[i]) graph->removeNode(static_cast<uint32_t>(i));
    }

    // Build alpha-CG graph from NGT edges.
    // Collect all pruned adjacency lists first, then call resetEdges() in one
    // O(N·k) pass. Sequential setNeighbors() would be O(N²) due to CSR shifts.
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
        adj[aq_id] = pruner.prune(candidates, tau, dist_fn);
    }
    graph->resetEdges(adj);

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
    if (static_cast<int>(query.size()) < prop_.dimension)
        throw std::invalid_argument("NGTAQIndex::search: query dimension mismatch");
    const int D = prop_.dimension;
    const int words = D / 64;

    // Encode query
    std::vector<uint64_t> q_sign(words), q_mag(words);
    bq_.encode(query.data(), q_sign.data(), q_mag.data());

    // Acquire shared lock once, covering both routing and refinement.
    // insert() holds unique_lock when pushing to raw_vecs_, so shared_lock
    // here prevents concurrent raw_vecs_ modification during refinement.
    std::shared_lock<std::shared_mutex> lock(graph_->mutex());

    auto cand_ids = searcher_.route(q_sign.data(), q_mag.data(),
                                    k, *graph_, entry_points_);

    // Refine with exact distances (raw_vecs_ is protected by the shared_lock above)
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
            // Cosine: raw_vecs_ stores pre-normalized vectors; normalize query too.
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
    if (static_cast<int>(vec.size()) < prop_.dimension)
        throw std::invalid_argument("NGTAQIndex::insert: vector dimension mismatch");
    const int D = prop_.dimension;
    const int words = D / 64;

    std::vector<uint64_t> sign(words), mag(words);
    bq_.encode(vec.data(), sign.data(), mag.data());

    std::unique_lock<std::shared_mutex> lock(graph_->mutex());
    uint32_t new_id = graph_->addNode(sign.data(), mag.data());
    raw_vecs_.push_back(std::vector<float>(vec.begin(), vec.begin() + D));
    if (prop_.metric == NGT::ObjectSpace::DistanceTypeAngle ||
        prop_.metric == NGT::ObjectSpace::DistanceTypeCosine) {
        auto& v = raw_vecs_.back();
        float norm_sq = 0.0f;
        for (float x : v) norm_sq += x * x;
        if (norm_sq > 0.0f) {
            float inv_norm = 1.0f / std::sqrt(norm_sq);
            for (float& x : v) x *= inv_norm;
        }
    }

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
        auto pruned = pruner_.prune(candidates, bq_.tau(), dist_fn);
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

    // Compute old_to_new mapping from current tombstone state.
    // SoAGraph::rebuild() applies the same logic internally.
    // By holding the unique_lock throughout, both scans see identical state.
    const size_t N = graph_->size();
    std::vector<uint32_t> old_to_new(N, static_cast<uint32_t>(-1));
    uint32_t next_id = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
        if (!graph_->isTombstone(i)) old_to_new[i] = next_id++;
    }

    // Reorder raw_vecs_ to match the post-rebuild node ordering.
    std::vector<std::vector<float>> new_vecs(next_id);
    for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
        if (old_to_new[i] != static_cast<uint32_t>(-1)) {
            if (i < static_cast<uint32_t>(raw_vecs_.size())) {
                new_vecs[old_to_new[i]] = std::move(raw_vecs_[i]);
            }
        }
    }
    raw_vecs_ = std::move(new_vecs);

    // SoAGraph::rebuild() compacts active nodes using the same isTombstone
    // scan; since we hold the unique_lock, no concurrent modifications can
    // cause the two mappings to diverge.
    graph_->rebuild();

    // Re-select entry points after compaction.
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
    if (n_ep > 65536) throw std::runtime_error("NGTAQIndex::load: n_ep too large, file may be corrupt");
    std::vector<uint32_t> entry_points(n_ep);
    if (n_ep > 0)
        is.read(reinterpret_cast<char*>(entry_points.data()), n_ep * sizeof(uint32_t));

    // 5. Raw vecs
    uint64_t n_vecs = 0;
    is.read(reinterpret_cast<char*>(&n_vecs), sizeof(n_vecs));
    if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read vec count");
    if (n_vecs > 100000000ULL) throw std::runtime_error("NGTAQIndex::load: vec count too large");
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
        // Use activeCount so we don't waste all samples on tombstones.
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
