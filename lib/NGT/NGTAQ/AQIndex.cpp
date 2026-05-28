// lib/NGT/NGTAQ/AQIndex.cpp
#include "NGT/NGTAQ/AQIndex.h"

#include "NGT/Graph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <omp.h>
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

} // namespace NGTAQ
