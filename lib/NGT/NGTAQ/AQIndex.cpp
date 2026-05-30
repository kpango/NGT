// lib/NGT/NGTAQ/AQIndex.cpp
#include "NGT/NGTAQ/AQIndex.h"
#include "NGT/NGTAQ/DimUtils.h"

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
// Magic number identifying the versioned binary format.
// Old format: first 4 bytes = prop_.dimension (small positive int, typically 128).
// New format: first 4 bytes = kPropMagic, followed by uint32_t prop_size, then prop bytes.
static constexpr uint32_t kPropMagic = 0xAE17AE17u;

void NGTAQIndex::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("NGTAQIndex::save: cannot open " + path);

    // 1. Property — versioned format: [magic][prop_size][prop_bytes]
    //    Backward-compat: old files start with prop_.dimension; new files with kPropMagic.
    const uint32_t prop_size = sizeof(prop_);
    os.write(reinterpret_cast<const char*>(&kPropMagic), 4);
    os.write(reinterpret_cast<const char*>(&prop_size),  4);
    os.write(reinterpret_cast<const char*>(&prop_), prop_size);

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

    // 1. Property — detect format by magic number.
    //    Old format (pre-k_clusters/n_cluster_seeds): first 4 bytes = dimension (e.g. 128).
    //    New format: first 4 bytes = kPropMagic (0xAE17AE17), then uint32_t prop_size, then prop bytes.
    Property prop;
    memset(&prop, 0, sizeof(prop));
    prop.k_clusters      = 0;   // default: use select_k(N)
    prop.n_cluster_seeds = 32;  // default

    uint32_t maybe_magic;
    is.read(reinterpret_cast<char*>(&maybe_magic), 4);
    if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read header from " + path);

    if (maybe_magic == kPropMagic) {
        // New versioned format: [magic(4)][prop_size(4)][prop_bytes(prop_size)]
        uint32_t stored_size;
        is.read(reinterpret_cast<char*>(&stored_size), 4);
        if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read prop_size from " + path);
        const uint32_t read_size = std::min(stored_size, static_cast<uint32_t>(sizeof(prop)));
        is.read(reinterpret_cast<char*>(&prop), read_size);
        if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read property from " + path);
        // Skip unknown future fields (forward compatibility)
        if (stored_size > static_cast<uint32_t>(sizeof(prop)))
            is.seekg(static_cast<std::streamoff>(stored_size - sizeof(prop)), std::ios::cur);
    } else {
        // Old format: first 4 bytes = dimension. Property had 11 fields (44 bytes total),
        // without k_clusters or n_cluster_seeds. Use a local layout-compatible struct.
        struct OldProperty {
            int     dimension;
            float   alpha;
            float   kappa;
            float   gamma_enq;
            float   gamma_term;
            float   k_prime_factor;
            int     n_tau_samples;
            int     n_entry_points;
            int     max_edges;
            int     n_search_threads;
            int32_t metric;  // NGT::ObjectSpace::DistanceType, int-sized enum
        };
        static_assert(sizeof(OldProperty) == 44, "OldProperty layout mismatch");
        OldProperty old{};
        old.dimension = static_cast<int>(maybe_magic);  // first 4 bytes already consumed
        is.read(reinterpret_cast<char*>(&old.alpha),
                static_cast<std::streamsize>(sizeof(OldProperty) - sizeof(int)));
        if (!is) throw std::runtime_error("NGTAQIndex::load: failed to read old-format property from " + path);
        prop.dimension        = old.dimension;
        prop.alpha            = old.alpha;
        prop.kappa            = old.kappa;
        prop.gamma_enq        = old.gamma_enq;
        prop.gamma_term       = old.gamma_term;
        prop.k_prime_factor   = old.k_prime_factor;
        prop.n_tau_samples    = old.n_tau_samples;
        prop.n_entry_points   = old.n_entry_points;
        prop.max_edges        = old.max_edges;
        prop.n_search_threads = old.n_search_threads;
        // k_clusters / n_cluster_seeds retain defaults set above
        prop.metric = static_cast<NGT::ObjectSpace::DistanceType>(old.metric);
    }

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
    const int D_orig = prop.dimension;
    const int D = NGT::NGTAQ::pad_dim_for_v2(D_orig);  // pad to next power-of-2 divisible by 64
    // D_orig may differ from D (e.g., D_orig=100 → D=128, D_orig=960 → D=1024);
    // raw vectors are zero-padded when loaded.

    // ---- 1. Load all float vectors from NGT ----
    NGT::Index ngt(ngt_path);
    NGT::ObjectSpace& objspace = ngt.getObjectSpace();
    const size_t repo_size = objspace.getRepository().size();
    const size_t N = repo_size - 1;
    const int words = D / 64;

    std::vector<float> raw_flat(N * static_cast<size_t>(D), 0.f);  // zero-padded to D
    std::vector<bool> is_hole(N, false);
    std::vector<float> tmp(D_orig);  // NGT stores D_orig dims; extras stay zero
    for (size_t i = 1; i <= N; i++) {
        try {
            objspace.getObject(static_cast<NGT::ObjectID>(i), tmp);
            // Copy D_orig dims into the D-padded slot; padding dims remain 0
            std::copy(tmp.begin(), tmp.end(),
                      raw_flat.begin() + static_cast<ptrdiff_t>((i - 1) * D));
        } catch (...) {
            is_hole[i - 1] = true;
        }
    }
    fprintf(stderr, "[NGTAQv2] Loaded %zu vectors D_orig=%d D_eff=%d\n", N, D_orig, D);

    // ---- 1b. Angular/Cosine: L2-normalize all raw vectors ----
    // Zero/degenerate vectors (norm <= 1e-6) are marked as holes: they cannot be
    // meaningfully normalized and would pollute cluster centroids and graph edges,
    // causing probe queries to become trapped in zero-vector clusters with distance=q_norm_sq.
    int n_zero = 0;
    if (prop.metric == NGT::ObjectSpace::DistanceTypeAngle ||
        prop.metric == NGT::ObjectSpace::DistanceTypeCosine) {
        for (size_t i = 0; i < N; ++i) {
            float* v = raw_flat.data() + static_cast<ptrdiff_t>(i * static_cast<size_t>(D));
            float norm2 = 0.f;
            for (int d = 0; d < D; ++d) norm2 += v[d] * v[d];
            if (norm2 > 1e-12f) {
                float inv = 1.f / std::sqrt(norm2);
                for (int d = 0; d < D; ++d) v[d] *= inv;
            } else {
                // Degenerate vector: mark as hole so it is excluded from the graph
                is_hole[i] = true;
                ++n_zero;
            }
        }
        fprintf(stderr, "[NGTAQv2] Angular: L2-normalized %zu vectors (%d degenerate holes)\n",
                N, n_zero);
    }

    // ---- 2. SRHT: rotate all vectors ----
    const uint64_t seed = 0xCAFEBABE12345678ULL;
    auto srht = std::make_unique<NGT::NGTAQ::SRHT>(D, seed);
    std::vector<float> rotated(N * static_cast<size_t>(D));
    for (size_t i = 0; i < N; ++i)
        srht->apply(raw_flat.data() + i*D, rotated.data() + i*D);

    // ---- 3. K-means on rotated vectors ----
    uint32_t K = (prop.k_clusters > 0)
                 ? static_cast<uint32_t>(prop.k_clusters)
                 : NGT::NGTAQ::select_k(N);
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

    // ---- 7. Tier-2 PQ: M_PQ sub-codebooks on SRHT residuals (all D dims) ----
    // M_PQ = D/8 sub-spaces × D_sub=8 dims each. K=256 centroids (8 bit/sub).
    // Layout: tier2_cb[(sub*256 + code)*D_sub + dim]
    // M_PQ sub-spaces × 8 bits → uses all M_PQ bytes of tier2 storage.
    // SRHT isotropizes data → equal per-sub-space variance → balanced PQ.
    // D=128: M_PQ=16, D=256: M_PQ=32, D=1024: M_PQ=128.
    const int M_PQ  = D / 8;   // D_sub = 8 fixed; M_PQ scales with D
    const int K_PQ  = 256;
    const int D_sub = 8;       // always 8 dims per sub-space
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

    // Fill v2 VectorRecords using VectorRecordView (variable-D safe)
    graph->reserveV2(N, D/8, D/8);  // tier1_n = D/8, tier2_n = D/8
    for (size_t i = 0; i < N; ++i) {
        if (is_hole[i]) continue;
        auto view = graph->getRecordView(static_cast<uint32_t>(i));
        view.set_centroid_id(centroid_ids[i]);

        // tier-1: sign bits of SRHT residual (D bits → D/8 bytes)
        const float* res = residuals.data() + i*D;
        for (int b = 0; b < D; ++b)
            view.set_tier1_bit(b, res[b] >= 0.f);

        // norm_fp16: L2 norm of residual
        float norm2 = 0.f;
        for (int d = 0; d < D; ++d) norm2 += res[d] * res[d];
        view.set_norm_fp16(NGT::NGTAQ::float_to_fp16(std::sqrt(norm2)));

        // tier-2: M_PQ independent PQ codes (M_PQ bytes, 8-bit each)
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
            view.set_tier2_byte(sub, best_code);
        }
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
        // Pure BQ distance sort (no cluster priority — hurts navigability).
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

    idx.is_angular_ = (prop.metric == NGT::ObjectSpace::DistanceTypeAngle ||
                       prop.metric == NGT::ObjectSpace::DistanceTypeCosine);
    idx.d_eff_ = D;
    idx.m_pq_  = M_PQ;

    fprintf(stderr, "[NGTAQv2] Build complete. N=%zu K=%u\n", N, K);
    return idx;
}

// ---------------------------------------------------------------------------
// searchV2: ADC search with lazy centroid switch
// ---------------------------------------------------------------------------
std::vector<SearchResult> NGTAQIndex::searchV2(
    const std::vector<float>& query, int k,
    float gamma_enq, float gamma_term,
    int rerank_factor) const
{
    if (!is_v2_)
        throw std::runtime_error("searchV2: call fromNGTv2() first");
    if (static_cast<int>(query.size()) < prop_.dimension)
        throw std::invalid_argument("searchV2: query dimension mismatch");

    // rerank_factor: widen beam by searching for k_beam candidates, return top k_out.
    // rerank_factor <= 1: standard behavior (k_beam == k).
    // rerank_factor >  1: search k*rerank_factor internally, trim to k at output.
    const int k_beam = (rerank_factor > 1) ? k * rerank_factor : k;
    const int k_out  = k;

    const int D     = (d_eff_ > 0) ? d_eff_ : prop_.dimension;
    const int M_PQ  = (m_pq_ > 0) ? m_pq_ : 16;
    const int D_orig = (int)query.size();

    // 0. Angular/Cosine: L2-normalize query (over D_orig dims)
    static thread_local std::vector<float> q_normalized_tl;
    const float* q_src = query.data();
    if (is_angular_) {
        q_normalized_tl.assign(query.begin(), query.begin() + std::min(D_orig, D));
        float norm2 = 0.f;
        for (float x : q_normalized_tl) norm2 += x * x;
        if (norm2 > 1e-12f) {
            float inv = 1.f / std::sqrt(norm2);
            for (float& x : q_normalized_tl) x *= inv;
        }
        q_src = q_normalized_tl.data();
    }

    // Query padding: zero-pad to D (for D_orig < D, e.g., GloVe-100 → D=128)
    // Thread-local scratch buffers: reused across queries on same thread to avoid
    // per-query heap allocation overhead (~50KB for D=1024, significant for QPS).
    static thread_local std::vector<float> q_padded_tl;
    static thread_local std::vector<float> q_rot_tl;
    static thread_local std::vector<float> q_res_tl;
    static thread_local std::vector<float> q_res_init_tl;
    // ADC state: q_int8 buffer lives in a thread_local struct (one alloc per thread).
    // Using a reference alias so all existing `adc.*` accesses compile unchanged.
    static thread_local NGT::NGTAQ::ADCQueryState adc_tl(0);

    q_padded_tl.assign(static_cast<size_t>(D), 0.f);
    std::copy(q_src, q_src + std::min(D_orig, D), q_padded_tl.begin());
    const float* q_ptr = q_padded_tl.data();

    // 1. Rotate query
    q_rot_tl.resize(static_cast<size_t>(D));
    srht_v2_->apply(q_ptr, q_rot_tl.data());

    // 2. Find query's nearest centroid
    uint32_t active_cid = kmeans_v2_->nearest_public(q_rot_tl.data());
    q_res_tl.resize(static_cast<size_t>(D));

    // 3. Build initial ADC state (tier-1 + tier-2 PQ on SRHT residuals)
    // Grow q_int8 once on first use (or if D changed); no allocation on steady-state.
    if (static_cast<int>(adc_tl.q_int8.size()) < D)
        adc_tl.q_int8.assign(static_cast<size_t>(D), int8_t(0));
    NGT::NGTAQ::ADCQueryState& adc = adc_tl;
    adc.q_norm_sq = 0.f; adc.q_norm = 0.f; adc.q_sum = 0;
    float q_norm_sq = 0.f;
    NGT::NGTAQ::compute_residual_and_tier1(
        q_rot_tl.data(), kmeans_v2_->centroid(active_cid), D,
        q_res_tl.data(), adc.q_norm_sq, adc.q_int8.data(), adc.q_sum);
    adc.q_norm = std::sqrt(adc.q_norm_sq);
    q_norm_sq = adc.q_norm_sq;
    const float inv_sqrt_D = 1.f / std::sqrt((float)D);

    // Save initial cluster residual for tier-2 LUT build post-routing.
    // maybe_rebuild_adc overwrites q_res/q_norm_sq on cluster transitions.
    const uint32_t initial_cid = active_cid;
    q_res_init_tl.assign(q_res_tl.begin(), q_res_tl.end());
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
            uint32_t cid = graph_->getRecordConstView(static_cast<uint32_t>(i)).centroid_id();
            if (cid < K)
                cluster_members_v2_[cid].push_back(static_cast<uint32_t>(i));
        }

        // 2. Precompute top-N_EXTRA_CLUSTERS nearest clusters for each cluster.
        // Cost: K² × D scalar ops (K=2000, D=256 → 1B ops, ~50ms) — one-time amortized.
        // Angular data: use 20 neighbor clusters for accurate multi-cluster seeding.
        // DABS graph traversal bridges inter-cluster gaps via ANNG edges.
        // Precompute enough cluster neighbors to support any n_probe value.
        // n_probe_override_ may be set before the first search triggers this call_once.
        const int default_nbrs = is_angular_ ? 20 : 4;
        const int CLUSTER_NBRS = std::max(default_nbrs, n_probe_override_);
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
    // Thread-local results buffer: capacity persists across queries (no malloc on steady-state).
    static thread_local std::vector<std::pair<float, uint32_t>> results_tl;
    results_tl.clear();
    if (results_tl.capacity() < static_cast<size_t>(k_beam * 30))
        results_tl.reserve(static_cast<size_t>(k_beam * 30));
    auto& results = results_tl;
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
    // Thread_local flat int8 buffer (ADC_SLOTS × D) + scalar metadata avoids
    // per-query heap allocation (~8KB for D=1024).
    constexpr int ADC_SLOTS = 8;
    struct ADCSlotMeta {
        uint32_t cid       = UINT32_MAX;
        float    q_norm_sq = 0.f;
        float    q_norm    = 0.f;
        int32_t  q_sum     = 0;
    };
    static thread_local std::vector<int8_t> adc_int8_tl;  // ADC_SLOTS * D int8 entries
    static thread_local std::array<ADCSlotMeta, ADC_SLOTS> adc_meta_tl;
    adc_int8_tl.resize(static_cast<size_t>(ADC_SLOTS) * static_cast<size_t>(D));
    for (auto& m : adc_meta_tl) m.cid = UINT32_MAX;
    // Slot 0 = initial cluster (ADC state already computed above)
    adc_meta_tl[0] = {active_cid, adc.q_norm_sq, adc.q_norm, adc.q_sum};
    std::memcpy(adc_int8_tl.data(), adc.q_int8.data(), static_cast<size_t>(D));
    int adc_cache_hand = 1;

    // Tier-1 routing: fast RaBitQ ADC for DABS beam search.
    // Tier-2 contributes via seeding (pre-sorts cluster candidates) AND d_k tracking:
    // using the fixed initial-cluster LUT for dk_tracker gives ~10% noise vs tier-1's
    // ~44%, allowing smaller gamma_term at same recall → higher QPS. No per-cluster
    // LUT rebuild needed — costs only ~5μs (1000 tier2_adc_pq calls × ~5ns) per query.
    // Note: full tier-2 heap routing was tested and regressed QPS 8x — accurate distances
    // delay DABS stopping criterion, causing many more node visits than tier-1 allows.
    auto adc_dist = [&](const NGT::NGTAQ::VectorRecordConstView& rec) -> float {
        float norm_x = NGT::NGTAQ::fp16_to_float(rec.norm_fp16());
        float t1 = NGT::NGTAQ::tier1_adc_fast_d(adc.q_int8.data(), rec.tier1(), adc.q_sum, D);
        float t1_ip = adc.q_norm * norm_x * NGT::NGTAQ::RABITQ_SCALE * t1 * inv_sqrt_D / 127.f;
        return adc.q_norm_sq + norm_x * norm_x - 2.0f * t1_ip;
    };

    // Rebuild tier-1 ADC tables on cluster boundary crossing, with cache lookup.
    // Cache hit (~10ns): restore from thread_local L1-resident flat buffer, skip recompute.
    // Cache miss: full rebuild (get_residual + q_norm_sq + sqrt + build_tier1_query ~150ns).
    auto maybe_rebuild_adc = [&](uint32_t cid) {
        if (cid == active_cid) return;
        for (int s = 0; s < ADC_SLOTS; ++s) {
            if (adc_meta_tl[s].cid == cid) {
                active_cid    = cid;
                q_norm_sq     = adc_meta_tl[s].q_norm_sq;
                adc.q_norm_sq = adc_meta_tl[s].q_norm_sq;
                adc.q_norm    = adc_meta_tl[s].q_norm;
                adc.q_sum     = adc_meta_tl[s].q_sum;
                std::memcpy(adc.q_int8.data(),
                            adc_int8_tl.data() + static_cast<size_t>(s) * static_cast<size_t>(D),
                            static_cast<size_t>(D));
                return;
            }
        }
        // Cache miss: full rebuild
        active_cid = cid;
        NGT::NGTAQ::compute_residual_and_tier1(
            q_rot_tl.data(), kmeans_v2_->centroid(active_cid), D,
            q_res_tl.data(), adc.q_norm_sq, adc.q_int8.data(), adc.q_sum);
        adc.q_norm = std::sqrt(adc.q_norm_sq);
        q_norm_sq = adc.q_norm_sq;
        // Store in cache (round-robin eviction)
        const int slot = adc_cache_hand;
        adc_cache_hand = (adc_cache_hand + 1) % ADC_SLOTS;
        adc_meta_tl[slot] = {cid, adc.q_norm_sq, adc.q_norm, adc.q_sum};
        std::memcpy(adc_int8_tl.data() + static_cast<size_t>(slot) * static_cast<size_t>(D),
                    adc.q_int8.data(), static_cast<size_t>(D));
    };

    // Tier-2 seed scoring: build LUT from initial cluster, pre-sort all cluster seeds
    // by tier-2 distance. Better seeds → routing converges faster → higher recall at
    // same gamma_term. Cost: 1 LUT build (~0.5μs AVX2) + N_seeds × 10ns ADC.
    // LUT built from q_res_initial (saved before routing modifies q_res).
    static thread_local std::vector<float> t2_lut_tl;
    t2_lut_tl.resize(static_cast<size_t>(M_PQ) * 256);
    NGT::NGTAQ::build_tier2_lut_fast_m(q_res_init_tl.data(), M_PQ,
                                        tier2_codebook_T_.data(),
                                        t2_lut_tl.data());

    // Cluster-aware seeding: probe top-n_probe nearest clusters with PER-CLUSTER ADC.
    //
    // Root cause of angular data failure (NYTimes-256, GloVe-100):
    //   The original code scored ALL seeds with the initial cluster's LUT. Cross-cluster
    //   seeds are computed with the wrong residual basis → inaccurate ADC estimates →
    //   d_k initialized too small from primary-cluster-only hits → gamma gate prunes
    //   all cross-cluster graph edges → recall collapses to ~cluster_size/N * k.
    //
    // Fix: per-cluster probing. For each probed cluster c_i:
    //   1. rebuild ADC state to centroid(c_i) → correct q_res
    //   2. build tier-2 LUT from q_res(c_i)   → accurate per-cluster ADC
    //   3. score ALL cluster members with accurate LUT
    // This mirrors IVF nprobe: angular data requires more probing because unit vectors
    // spread true neighbors across many clusters (no magnitude diversity).
    //
    // After seeding, restore t2_lut_tl to initial cluster (used by DABS termination gate).
    const int N_CLUSTER_SEEDS = prop_.n_cluster_seeds;
    // n_probe: angular data seeds from more clusters than L2 for accurate d_k initialization.
    // Per-cluster tier-2 scoring gives cross-cluster seeds with correct ADC residuals,
    // so d_k is properly initialized before DABS beam search begins.
    // L2 data: DABS with 3 neighbor clusters is highly effective.
    const int n_probe = (n_probe_override_ > 0) ? n_probe_override_
                                                 : (is_angular_ ? 20 : 3);
    {
        // Build list of clusters to probe: primary cluster + top-(n_probe-1) neighbors.
        // cluster_neighbors_v2_[active_cid] is sorted by cluster-centroid L2 distance.
        std::vector<uint32_t> probe_clusters;
        probe_clusters.reserve(static_cast<size_t>(n_probe));
        probe_clusters.push_back(initial_cid);  // primary cluster first
        if (initial_cid < cluster_neighbors_v2_.size()) {
            for (uint32_t c2 : cluster_neighbors_v2_[initial_cid]) {
                if (static_cast<int>(probe_clusters.size()) >= n_probe) break;
                probe_clusters.push_back(c2);
            }
        }

        struct SeedScore { float score; uint32_t id; };
        std::vector<SeedScore> scored;
        scored.reserve(static_cast<size_t>(n_probe) * 200);

        // Score seeds from each probed cluster with the CORRECT centroid's LUT.
        // For each cluster c_i: rebuild ADC → build LUT(c_i) → score all members.
        static thread_local std::vector<float> t2_lut_probe;
        t2_lut_probe.resize(static_cast<size_t>(M_PQ) * 256);
        for (uint32_t cid_p : probe_clusters) {
            if (cid_p >= cluster_members_v2_.size()) continue;
            const auto& members = cluster_members_v2_[cid_p];
            if (members.empty()) continue;

            // Angular: rebuild ADC + tier-2 LUT per cluster for accurate cross-cluster scoring.
            // L2: initial-cluster LUT is accurate (magnitude diversity makes cross-cluster
            //     residual error negligible for seeding); skip expensive per-cluster rebuild.
            if (is_angular_) {
                maybe_rebuild_adc(cid_p);
                NGT::NGTAQ::build_tier2_lut_fast_m(q_res_tl.data(), M_PQ,
                                                    tier2_codebook_T_.data(),
                                                    t2_lut_probe.data());
            }
            const float q_ns  = is_angular_ ? adc.q_norm_sq : q_norm_sq_initial;
            const float* lut_p = is_angular_ ? t2_lut_probe.data() : t2_lut_tl.data();
            // Score all members of this cluster with the correct LUT
            const size_t take = is_angular_
                ? members.size()                                         // scan full cluster for angular
                : std::min(members.size(), static_cast<size_t>(N_CLUSTER_SEEDS)); // limit for L2
            // Prefetch only what we'll score: avoids cache pollution for L2 (full-cluster
            // prefetch evicts DABS hot data from L1/L2 when only 32/N_CLUSTER_SEEDS are used).
            for (size_t mi = 0; mi < take; ++mi) {
                if (members[mi] < N) graph_->prefetchRecord(members[mi]);
            }
            for (size_t mi = 0; mi < take; ++mi) {
                uint32_t ep = members[mi];
                if (ep >= N || graph_->isTombstone(ep)) continue;
                auto rec = graph_->getRecordConstView(ep);
                float norm_x = NGT::NGTAQ::fp16_to_float(rec.norm_fp16());
                float t2_ip  = NGT::NGTAQ::tier2_adc_pq_m(lut_p, rec.tier2(), M_PQ);
                float d_approx = q_ns + norm_x * norm_x - 2.0f * t2_ip;
                scored.push_back({d_approx, ep});
            }
        }

        // Fall back to static entry points if cluster membership is empty
        if (scored.empty()) {
            for (uint32_t ep : entry_points_) {
                if (ep < N && !graph_->isTombstone(ep)) {
                    auto rec = graph_->getRecordConstView(ep);
                    scored.push_back({std::numeric_limits<float>::infinity(), ep});
                }
            }
        }

        // Sort seeds by accurate per-cluster ADC estimate; best-first ensures d_k
        // is initialized from the true near-neighbors rather than random seeds.
        std::sort(scored.begin(), scored.end(),
                  [](const SeedScore& a, const SeedScore& b){ return a.score < b.score; });

        // During rebuildGraphSelf on angular data, limit total seeds to avoid queue
        // flooding (n_probe=20 × avg_cluster_members can be 20k+ seeds per query).
        // rebuild_max_seeds_ = 0 (default) means no limit (normal search path).
        if (rebuild_max_seeds_ > 0 &&
            static_cast<int>(scored.size()) > rebuild_max_seeds_)
            scored.resize(static_cast<size_t>(rebuild_max_seeds_));

        {
            // ── DABS path for all metrics ─────────────────────────────────────────
            // Per-cluster tier-2 seeding (above) gives accurate cross-cluster d_k
            // initialization. DABS graph traversal then bridges inter-cluster gaps
            // via ANNG edges (built on true angular proximity for normalized vectors).
            // gamma_enq ≥ 0.50 (caller should use this for angular) ensures tier-1
            // noise (~44% max deflation) doesn't prune true near-neighbors.
            for (const auto& s : scored) {
                if (is_visited(s.id)) continue;
                mark_visited(s.id);
                auto rec = graph_->getRecordConstView(s.id);
                maybe_rebuild_adc(rec.centroid_id());
                float d = adc_dist(rec);
                cand_q.push({d, s.id});
                graph_->prefetchOffset(s.id);
            }

            // Angular only: restore t2_lut_tl and ADC to initial cluster.
            // The DABS termination gate (two-gate tier-2 check) uses t2_lut_tl with the
            // initial-cluster residual. L2 never rebuilt LUT/ADC during seeding → no restore.
            if (is_angular_) {
                NGT::NGTAQ::build_tier2_lut_fast_m(q_res_init_tl.data(), M_PQ,
                                                    tier2_codebook_T_.data(),
                                                    t2_lut_tl.data());
                maybe_rebuild_adc(initial_cid);
            }
        }
    }

    while (!cand_q.empty()) {
        auto [dist_x, x] = cand_q.top(); cand_q.pop();

        // Skip tombstoned (hole) nodes: zero-norm train vectors excluded during
        // normalization. Their raw_flat_ entry is all-zeros, so exact L2 reranking
        // gives dist = ||q_norm|| = 1.0, which beats true NNs with dist > 1.0
        // (cos_sim < 0.5) and causes recall collapse on angular datasets (NYTimes).
        if (graph_->isTombstone(x)) continue;

        // Fast-path termination: node is so far that even maximum tier-1 deflation
        // (3σ ≈ 2.32×) of a true near-neighbor can't explain this distance.
        // Avoids rec_x cache miss for clearly-far nodes.
        if (dk_tracker.size() >= static_cast<size_t>(k_beam) &&
            dist_x > (1.f + gamma_term) * d_k * 2.0f) break;

        // Prefetch next-popped node's data while we process current node
        if (!cand_q.empty()) {
            uint32_t nxt = cand_q.top().second;
            graph_->prefetchRecord(nxt);
            graph_->prefetchNeighbors(nxt);
        }
        // Prefetch current node's neighbor list (hides CSR access latency)
        graph_->prefetchNeighbors(x);

        auto rec_x = graph_->getRecordConstView(x);

        // Two-gate gamma_term: tier-1 noise (~44% std) can inflate a true near-neighbor's
        // heap estimate above threshold. Verify with tier-2 PQ (lower noise, ~10%) before
        // terminating. Only fires in the "uncertain zone" (1× to 2× threshold): nodes that
        // tier-1 calls too far but tier-2 may recognize as close. Cost: 1 fp16_to_float +
        // 1 tier2_adc_pq (~5ns) per uncertain termination candidate.
        //
        // Angular EXCEPTION: t2_lut_tl uses the initial cluster's residual, but DABS pops
        // nodes from any cluster (after maybe_rebuild_adc). For cross-cluster nodes, this
        // gives a grossly wrong distance (wrong residual basis) → overestimates → false
        // termination → recall collapse for angular. Skip tier-2 for angular; rely solely
        // on tier-1 with a suitably large gamma_term (≥0.50 recommended for angular).
        if (dk_tracker.size() >= static_cast<size_t>(k_beam) &&
            dist_x > (1.f + gamma_term) * d_k) {
            if (!is_angular_) {
                float nx2  = NGT::NGTAQ::fp16_to_float(rec_x.norm_fp16());
                float t2ip = NGT::NGTAQ::tier2_adc_pq_m(t2_lut_tl.data(), rec_x.tier2(), M_PQ);
                float d_t2 = q_norm_sq_initial + nx2 * nx2 - 2.0f * t2ip;
                if (d_t2 > (1.f + gamma_term) * d_k) break;
                // tier-2 override: tier-1 overestimated this node — continue processing
            } else {
                break;  // angular: trust tier-1 termination (no 2-gate with wrong LUT)
            }
        }

        maybe_rebuild_adc(rec_x.centroid_id());
        float d_approx = adc_dist(rec_x);

        // Add ALL popped candidates to results for exact reranking.
        // We only use d_k for ROUTING termination, not for result filtering.
        results.push_back({d_approx, x});
        dk_tracker.push(d_approx);
        if (static_cast<int>(dk_tracker.size()) > k_beam) {
            dk_tracker.pop();
            d_k = dk_tracker.top();
        } else if (static_cast<int>(dk_tracker.size()) == k_beam) {
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
            auto rec_u = graph_->getRecordConstView(u);
            maybe_rebuild_adc(rec_u.centroid_id());
            float d_u = adc_dist(rec_u);
            // Skip hopeless candidates: when d_k is initialized, a node with
            // d_u > (1+gamma)*d_k would trigger the outer-loop termination as
            // soon as it's popped — it can never contribute to the top-k result.
            // Skipping the push avoids a wasted heap insertion+extraction.
            if (static_cast<int>(dk_tracker.size()) >= k_beam &&
                d_u > (1.f + gamma_enq) * d_k)
                continue;
            cand_q.push({d_u, u});
            // Prefetch offset for u: when u is eventually popped and
            // prefetchNeighbors(u) is called, offsets_[u] will already be
            // in L1/L2 cache, eliminating the blocking DRAM read that
            // gates the neighbor-list prefetch.
            graph_->prefetchOffset(u);
        }
    }

    // Select top candidates by approximate score for exact L2 reranking.
    // BQ ADC noise ~44% causes true NNs to appear at high ADC ranks; k_beam*15 was
    // too aggressive and threw away genuine neighbors.  k_beam*100 keeps a much
    // wider ADC window so true NNs survive to the exact-distance stage.
    const size_t refine_n = static_cast<size_t>(k_beam * 100);
    if (results.size() > refine_n) {
        std::nth_element(results.begin(), results.begin() + refine_n, results.end());
        results.resize(refine_n);
    }

    // 1-hop expansion from top-EXPAND_N candidates (all metrics).
    // For angular DABS: ANNG edges bridge inter-cluster gaps; expanding top seeds
    // finds true NNs in neighboring clusters not reached by beam search.
    // EXPAND_N raised from 20→200: covers secondary cluster seeds whose true NNs
    // are reachable 2 hops away from any of the top-200 ADC candidates.
    {
        constexpr size_t EXPAND_N = 200;
        const size_t expand_from = std::min(EXPAND_N, results.size());
        for (size_t ei = 0; ei < expand_from; ++ei) {
            uint32_t node = results[ei].second;
            for (uint32_t u : graph_->getNeighbors(node)) {
                if (u >= static_cast<uint32_t>(N) || graph_->isTombstone(u)) continue;
                if (is_visited(u)) continue;
                mark_visited(u);
                auto rec_u = graph_->getRecordConstView(u);
                maybe_rebuild_adc(rec_u.centroid_id());
                float d_u = adc_dist(rec_u);
                results.push_back({d_u, u});
            }
        }
    }

    // Exact L2 refinement: l2_sq_avx2 for D=128 ≈ 5ns/vector × 150 = 0.75μs.
    // Store squared distances to avoid redundant sqrts during sort; take sqrt only
    // for the final top-k (10 sqrts instead of 150).
    std::vector<SearchResult> final_results;
    final_results.reserve(results.size());
    for (auto& [approx_d, id] : results) {
        if (static_cast<size_t>(id) * D + D > raw_flat_.size()) continue;
        // Safety net: tombstoned nodes have raw_flat_=zeros → exact dist=1.0
        // which beats angular NNs with dist>1.0. Filter here as belt-and-suspenders.
        if (graph_->isTombstone(static_cast<uint32_t>(id))) continue;
        const float* vec = raw_flat_.data() + static_cast<size_t>(id) * D;
        float exact_sq = NGT::NGTAQ::l2_sq(q_ptr, vec, D);
        // Store exact_sq in .distance temporarily (sqrt deferred until after sort).
        final_results.push_back({id, exact_sq, approx_d});
    }
    // partial_sort: O(n log k) ≈ 500 comparisons vs std::sort O(n log n) ≈ 1080.
    // We only need the top-k; the rest are discarded.
    const size_t out_n = std::min(static_cast<size_t>(k_out), final_results.size());
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
// rebuildGraphFromNGT: hot-swap graph edges from a denser NGT source index.
// Reuses existing SRHT/K-means/PCA/PQ/BQ encoding — only rebuilds edges.
// ~50s vs ~400s for a full fromNGTv2 rebuild.
// ---------------------------------------------------------------------------
void NGTAQIndex::rebuildGraphFromNGT(const std::string& ngt_path,
                                      float new_alpha,
                                      int   new_max_edges)
{
    if (new_alpha    > 0.0f) prop_.alpha     = new_alpha;
    if (new_max_edges > 0)   prop_.max_edges = new_max_edges;

    // Use graph_->size() (= state_.size(), includes tombstones) not size()
    // (= activeCount()).  resetEdges() requires adj.size() == state_.size().
    const size_t N    = graph_->size();
    const int    D    = prop_.dimension;
    const int    words = D / 64;

    AlphaCGPruner pruner(prop_.alpha, prop_.kappa);
    const float   tau  = bq_.tau();

    NGT::Index ngt(ngt_path);
    NGT::GraphIndex& gi = static_cast<NGT::GraphIndex&>(ngt.getIndex());

    std::vector<std::vector<uint32_t>> adj(N);
    for (size_t i = 1; i <= N; ++i) {
        uint32_t aq_id = static_cast<uint32_t>(i - 1);
        if (graph_->isTombstone(aq_id)) continue;

        NGT::GraphNode* node = nullptr;
        try { node = gi.getNode(static_cast<NGT::ObjectID>(i)); }
        catch (...) { continue; }
        if (!node || node->empty()) continue;

        // Use getRecordConstView (correct variable stride) for cluster-aware sort.
        uint32_t own_cid = graph_->getRecordConstView(aq_id).centroid_id();

        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(node->size());
        for (auto& edge : *node) {
            if (edge.id == 0 || edge.id > static_cast<unsigned int>(N)) continue;
            uint32_t nbr = static_cast<uint32_t>(edge.id - 1);
            float d_bq = bqDistance(graph_->getNodeBQ(aq_id),
                                     graph_->getNodeBQ(nbr), words, D);
            candidates.push_back({nbr, d_bq});
        }

        // Pure BQ distance sort (no cluster priority — see rebuildGraphSelf comment).
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (static_cast<int>(candidates.size()) > prop_.max_edges)
            candidates.resize(static_cast<size_t>(prop_.max_edges));

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(graph_->getNodeBQ(v), graph_->getNodeBQ(u), words, D);
        };
        adj[aq_id] = pruner.prune(candidates, tau, dist_fn);
    }

    graph_->resetEdges(adj);

    // Re-select entry points from the new graph
    int n_ep = std::min(prop_.n_entry_points, static_cast<int>(N));
    entry_points_   = selectEntryPoints(*graph_, n_ep);
    v2_entry_points_ = entry_points_;

    // Invalidate lazy cluster tables — rebuilt on next searchV2 call
    cluster_members_once_ = std::make_unique<std::once_flag>();
    cluster_members_v2_.clear();
    cluster_neighbors_v2_.clear();
}

// ---------------------------------------------------------------------------
// fixHoleTombstones: post-hoc tombstone repair.
// Scan raw_flat_ for zero-norm vectors (hole nodes created when zero-norm
// train vectors were excluded during normalization but not properly tombstoned
// in the serialized graph). Tombstones them so they cannot appear in search results.
// ---------------------------------------------------------------------------
int NGTAQIndex::fixHoleTombstones() {
    const int D = (d_eff_ > 0) ? d_eff_ : prop_.dimension;
    const size_t N = graph_->size();
    int n_fixed = 0;
    for (size_t i = 0; i < N; ++i) {
        if (graph_->isTombstone(static_cast<uint32_t>(i))) continue;
        const float* v = raw_flat_.data() + i * static_cast<size_t>(D);
        float norm2 = 0.f;
        for (int d = 0; d < D; ++d) norm2 += v[d] * v[d];
        if (norm2 < 1e-12f) {
            graph_->removeNode(static_cast<uint32_t>(i));
            ++n_fixed;
        }
    }
    fprintf(stderr, "[fixHoleTombstones] tombstoned %d zero-norm (hole) nodes\n", n_fixed);
    return n_fixed;
}

// ---------------------------------------------------------------------------
// rebuildGraphSelf: self-referential graph refinement.
// Runs searchV2 on every node to find high-quality candidate neighbors, then
// re-prunes with AlphaCGPruner. One pass raises recall ceiling by ~5-10%.
// ---------------------------------------------------------------------------
void NGTAQIndex::rebuildGraphSelf(int   k_search,
                                   float gamma,
                                   int   n_threads,
                                   float new_alpha,
                                   int   new_max_edges)
{
    if (new_alpha    > 0.0f) prop_.alpha     = new_alpha;
    if (new_max_edges > 0)   prop_.max_edges = new_max_edges;

    // BUGFIX: use graph_->size() (= state_.size(), includes tombstones) not size()
    // (= activeCount(), excludes tombstones).  resetEdges() requires adj.size() ==
    // state_.size(); if adj is undersized it reads past the end → SIGSEGV.
    const size_t N    = graph_->size();
    const int    D    = prop_.dimension;
    const int    words = D / 64;

    AlphaCGPruner pruner(prop_.alpha, prop_.kappa);
    const float   tau  = bq_.tau();

    // Warm up lazy cluster tables with a dummy query (triggers call_once).
    // This ensures subsequent parallel queries don't race on cluster_members_once_.
    if (!raw_flat_.empty()) {
        std::vector<float> dummy(raw_flat_.data(), raw_flat_.data() + D);
        searchV2(dummy, 1, gamma, gamma);
    }

    // During rebuild we want maximum neighbor diversity, not maximum QPS.
    // Default: probe ALL K clusters so every node's true k-NN can be discovered
    // across cluster boundaries.  Caller can override via setNProbe() before calling
    // rebuildGraphSelf() (e.g. to cap rebuild time on large datasets).
    // rebuild_max_seeds_ is set to match so all probed centroids are used as seeds.
    int saved_n_probe_override = n_probe_override_;
    if (is_angular_) {
        const int K_clusters = static_cast<int>(kmeans_v2_ ? kmeans_v2_->num_clusters() : 0);
        if (K_clusters > 0) {
            // If caller pre-set via setNProbe(), honour that value (capped at K).
            // Otherwise default to probing all K clusters.
            const int caller_probe = saved_n_probe_override;  // 0 = "not set by caller"
            const int effective_probe = (caller_probe > 0)
                ? std::min(caller_probe, K_clusters)
                : K_clusters;
            n_probe_override_  = effective_probe;
            rebuild_max_seeds_ = effective_probe;
        }
    }

    // Parallel search: for each node i, find its k_search approximate NN.
    // searchV2 uses shared_lock — safe for concurrent reads.
    std::vector<std::vector<uint32_t>> adj(N);

#ifdef _OPENMP
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 512)
#endif
    for (size_t i = 0; i < N; ++i) {
        if (graph_->isTombstone(static_cast<uint32_t>(i))) continue;
        std::vector<float> q(raw_flat_.data() + i * static_cast<size_t>(D),
                             raw_flat_.data() + (i + 1) * static_cast<size_t>(D));
        // Search for k_search+1: self may appear in results; we'll skip it.
        auto results = searchV2(q, k_search + 1, gamma, gamma);

        // Build candidate list sorted by EXACT distance (two-phase approach).
        //
        // Problem: original code sorted/trimmed by BQ distance, which is
        // cluster-relative (residual BQ).  Cross-cluster true NNs have noisy
        // inter-cluster BQ distances and were ranked at positions 50-200+,
        // well outside the max_edges=128 cutoff → never reached alpha-CG.
        //
        // Fix phase 1: sort and trim by EXACT distance so cross-cluster true
        // NNs (small exact_dist) survive the max_edges cutoff.
        // Fix phase 2: replace with BQ distances for alpha-CG input so the
        // pruner retains its original navigability behaviour (same tau).
        //
        // Why BQ for alpha-CG: cross-cluster candidates have high BQ dist →
        // high pruning threshold → hard to be "dominated" → they are KEPT.
        // In-cluster candidates behave as before (accurate BQ distances).
        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(results.size());
        for (auto& r : results) {
            if (r.id == static_cast<uint32_t>(i)) continue;  // skip self
            candidates.push_back({r.id, r.distance});  // exact dist for ranking
        }

        // Phase 1: trim by exact distance (cross-cluster true NNs survive).
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (static_cast<int>(candidates.size()) > prop_.max_edges)
            candidates.resize(static_cast<size_t>(prop_.max_edges));

        // Phase 2: cluster-aware selective alpha-CG.
        //
        // Cross-cluster candidates (confirmed true NNs by Phase 1 exact trim) are
        // added unconditionally to the graph.  Applying BQ alpha-CG to them is
        // counterproductive: inter-cluster BQ distances are meaningless (different
        // residual frames), so the pruning decision is random noise and removes
        // valid cross-cluster edges that are critical for navigability.
        //
        // In-cluster candidates still use BQ alpha-CG which IS meaningful within a
        // cluster (same residual frame) and produces a navigable in-cluster sub-graph.
        const uint32_t cid_i = graph_->getRecordConstView(static_cast<uint32_t>(i)).centroid_id();

        std::vector<std::pair<uint32_t, float>> in_cluster_cands;
        std::vector<uint32_t> cross_cluster_ids;
        in_cluster_cands.reserve(candidates.size());
        // candidates is currently sorted by exact distance (Phase 1 order).
        // We iterate in that order so cross_cluster_ids preserves exact-dist rank.
        for (auto& c : candidates) {
            const uint32_t cid_c = graph_->getRecordConstView(c.first).centroid_id();
            const float bq_d = bqDistance(graph_->getNodeBQ(static_cast<uint32_t>(i)),
                                           graph_->getNodeBQ(c.first), words, D);
            if (cid_c == cid_i) {
                in_cluster_cands.push_back({c.first, bq_d});
            } else {
                cross_cluster_ids.push_back(c.first);  // exact-dist order preserved
            }
        }

        std::sort(in_cluster_cands.begin(), in_cluster_cands.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        auto dist_fn = [&](uint32_t v, uint32_t u) -> float {
            return bqDistance(graph_->getNodeBQ(v), graph_->getNodeBQ(u), words, D);
        };

        // Alpha-CG on in-cluster candidates → navigable in-cluster sub-graph.
        adj[i] = pruner.prune(in_cluster_cands, tau, dist_fn);

        // Append cross-cluster candidates unconditionally (closest first, exact-dist order).
        const int max_adj = prop_.max_edges;
        for (uint32_t ccid : cross_cluster_ids) {
            if (static_cast<int>(adj[i].size()) >= max_adj) break;
            adj[i].push_back(ccid);
        }
    }

    // Reset overrides before any post-rebuild searches.
    rebuild_max_seeds_ = 0;
    n_probe_override_  = saved_n_probe_override;

    graph_->resetEdges(adj);

    // Re-select entry points from the refined graph
    int n_ep = std::min(prop_.n_entry_points, static_cast<int>(N));
    entry_points_    = selectEntryPoints(*graph_, n_ep);
    v2_entry_points_ = entry_points_;

    // Invalidate lazy cluster tables — rebuilt on next searchV2 call
    cluster_members_once_ = std::make_unique<std::once_flag>();
    cluster_members_v2_.clear();
    cluster_neighbors_v2_.clear();
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
    // New metadata: is_angular_, d_eff_, m_pq_
    {
        std::string meta_path = dir + "/v2_meta.bin";
        FILE* f = fopen(meta_path.c_str(), "wb");
        if (f) {
            int32_t meta[4] = {(int32_t)is_angular_, d_eff_, m_pq_, 0};
            fwrite(meta, sizeof(meta), 1, f);
            fclose(f);
        }
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
    // New metadata: is_angular_, d_eff_, m_pq_ (optional — may not exist in old indices)
    // Must be read BEFORE codebook so M_cb and D_sub are known.
    {
        std::string meta_path = dir + "/v2_meta.bin";
        FILE* f = fopen(meta_path.c_str(), "rb");
        if (f) {
            int32_t meta[4] = {0, 0, 16, 0};
            if (fread(meta, sizeof(meta), 1, f) == 1) {
                is_angular_ = (bool)meta[0];
                d_eff_      = meta[1];
                m_pq_       = meta[2];
            }
            fclose(f);
        }
        if (m_pq_ <= 0) m_pq_ = 16;
        if (d_eff_ <= 0) d_eff_ = prop_.dimension;
    }
    // Tier-2 codebook: M_cb sub-spaces × K=256 codes × D_sub=8 dims
    {
        std::ifstream f(dir + "/v2_codebook.bin", std::ios::binary);
        const int M_cb  = m_pq_;   // derived from metadata
        const int D_sub = 8;       // always 8 dims per sub-space
        tier2_codebook_.resize((size_t)M_cb * 256 * D_sub);
        f.read(reinterpret_cast<char*>(tier2_codebook_.data()),
               tier2_codebook_.size() * sizeof(float));
        // Build transposed codebook [M][D_sub][K] for AVX2 FMA LUT build
        tier2_codebook_T_.resize(tier2_codebook_.size());
        NGT::NGTAQ::build_tier2_codebook_T(
            tier2_codebook_.data(), M_cb, 256, D_sub,
            tier2_codebook_T_.data());
    }
    is_v2_ = true;
}

} // namespace NGTAQ
