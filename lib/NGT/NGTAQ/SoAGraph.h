// lib/NGT/NGTAQ/SoAGraph.h
#pragma once

#include "NGT/NGTAQ/VectorRecord.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace NGTAQ {

// SoA graph with CSR edge lists, tombstone-based deletion, and epoch rebuild.
//
// Layout:
//   bq_data_[node_id * 2 * words_ + i*2]   — sign-plane word i for node node_id
//   bq_data_[node_id * 2 * words_ + i*2+1] — magnitude-plane word i for node node_id
//   offsets_[node_id]..offsets_[node_id+1] — range in edge_ids_ for node node_id's neighbors
//   state_[node_id]                         — ACTIVE or TOMBSTONE
//
// Usage:
//   1. addNode() for each node (accumulates offsets_ sentinel per node)
//   2. finalizeCSR() to append the final sentinel to offsets_
//   3. setNeighbors() to set neighbors per node (may be called multiple times)
//   4. getNeighbors() to read CSR neighbor range (requires finalizeCSR() called)
//   5. removeNode() to tombstone a node
//   6. rebuild() to compact out tombstones (caller must hold exclusive lock)
class SoAGraph {
public:
    enum State : uint8_t { ACTIVE = 0, TOMBSTONE = 1 };

    // Lightweight non-owning view over a contiguous neighbor array (C++17 compatible).
    struct NeighborView {
        const uint32_t* data;
        size_t          sz;
        size_t          size()  const { return sz; }
        const uint32_t* begin() const { return data; }
        const uint32_t* end()   const { return data + sz; }
        const uint32_t& operator[](size_t i) const { return data[i]; }
    };

    explicit SoAGraph(int words) : words_(words) {}

    // Add a new node. Returns assigned node ID.
    // bq must point to 2*words_ uint64_t in interleaved layout:
    //   bq[i*2] = sign word i, bq[i*2+1] = mag word i
    // Thread-unsafe: caller must hold exclusive lock.
    uint32_t addNode(const uint64_t* bq) {
        uint32_t id = static_cast<uint32_t>(state_.size());
        state_.push_back(ACTIVE);
        bq_data_.insert(bq_data_.end(), bq, bq + words_ * 2);
        offsets_.push_back(static_cast<uint32_t>(edge_ids_.size()));
        return id;
    }

    // Finalize CSR by appending the sentinel (end offset).
    // Must be called after all addNode() calls and before any getNeighbors() calls.
    void finalizeCSR() {
        if (offsets_.size() == state_.size() + 1) return;  // already finalized
        offsets_.push_back(static_cast<uint32_t>(edge_ids_.size()));
    }

    // Rebuild all edge data from a full adjacency list in O(N·k).
    // Replaces edge_ids_ and offsets_ entirely; state_/bq_data_ unchanged.
    // Thread-unsafe: caller must hold exclusive lock.
    void resetEdges(const std::vector<std::vector<uint32_t>>& adj) {
        assert(adj.size() == state_.size());
        const size_t N = state_.size();
        size_t total = 0;
        for (const auto& nbrs : adj) total += nbrs.size();
        edge_ids_.clear();
        edge_ids_.reserve(total);
        offsets_.resize(N + 1);
        offsets_[0] = 0;
        for (size_t i = 0; i < N; ++i) {
            for (uint32_t nbr : adj[i]) edge_ids_.push_back(nbr);
            offsets_[i + 1] = static_cast<uint32_t>(edge_ids_.size());
        }
    }

    // Replace neighbor list for node_id. Adjusts all subsequent offsets.
    // Thread-unsafe: caller must hold exclusive lock.
    //
    // NOTE: This is O(N) per call (shifts all offsets_). Use resetEdges() + finalizeCSR()
    // for bulk construction. Incremental inserts calling setNeighbors() per node are O(N²)
    // in total — acceptable for small graphs, avoid for N > 10k.
    void setNeighbors(uint32_t node_id, std::vector<uint32_t> neighbors) {
        assert(static_cast<size_t>(node_id) + 1 < offsets_.size() && "call finalizeCSR() before setNeighbors()");
        uint32_t begin = offsets_[node_id];
        uint32_t end   = offsets_[node_id + 1];
        int old_count  = static_cast<int>(end - begin);
        int new_count  = static_cast<int>(neighbors.size());
        int delta      = new_count - old_count;

        if (delta == 0) {
            // In-place update
            std::copy(neighbors.begin(), neighbors.end(), edge_ids_.begin() + begin);
        } else {
            edge_ids_.erase(edge_ids_.begin() + begin, edge_ids_.begin() + end);
            edge_ids_.insert(edge_ids_.begin() + begin, neighbors.begin(), neighbors.end());
            for (uint32_t i = node_id + 1; i < offsets_.size(); ++i)
                offsets_[i] = static_cast<uint32_t>(static_cast<int>(offsets_[i]) + delta);
        }
    }

    // Returns view into edge_ids_ for node_id's neighbors.
    // Requires finalizeCSR() to have been called.
    NeighborView getNeighbors(uint32_t node_id) const {
        assert(static_cast<size_t>(node_id) + 1 < offsets_.size() && "call finalizeCSR() before getNeighbors()");
        uint32_t begin = offsets_[node_id];
        uint32_t end   = offsets_[node_id + 1];
        return {edge_ids_.data() + begin, end - begin};
    }

    // Returns pointer to the interleaved BQ data for node_id.
    // Layout: [s0, m0, s1, m1, ..., s_{words-1}, m_{words-1}]
    const uint64_t* getNodeBQ(uint32_t node_id) const {
        assert(node_id < state_.size());
        return bq_data_.data() + static_cast<size_t>(node_id) * words_ * 2;
    }

    void removeNode(uint32_t node_id) {
        assert(node_id < state_.size());
        state_[node_id] = TOMBSTONE;
    }

    bool isTombstone(uint32_t node_id) const {
        assert(node_id < state_.size());
        return state_[node_id] == TOMBSTONE;
    }

    size_t size() const { return state_.size(); }

    size_t activeCount() const {
        return static_cast<size_t>(
            std::count(state_.begin(), state_.end(), ACTIVE));
    }

    int words() const { return words_; }

    // Compact out all tombstone nodes: re-maps IDs, purges tombstone-ID neighbors.
    // Caller must hold exclusive lock on mutex().
    void rebuild() {
        finalizeCSR();  // ensure CSR sentinel present before getNeighbors()
        size_t N = state_.size();
        std::vector<uint32_t> old_to_new(N, UINT32_MAX);
        uint32_t new_id = 0;
        for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i)
            if (state_[i] == ACTIVE) old_to_new[i] = new_id++;

        SoAGraph fresh(words_);
        fresh.bq_data_.reserve(static_cast<size_t>(new_id) * words_ * 2);
        fresh.state_.reserve(new_id);
        fresh.offsets_.reserve(static_cast<size_t>(new_id) + 1);

        for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
            if (state_[i] != ACTIVE) continue;
            const uint64_t* bq = getNodeBQ(i);
            fresh.bq_data_.insert(fresh.bq_data_.end(), bq, bq + words_ * 2);
            fresh.state_.push_back(ACTIVE);
            fresh.offsets_.push_back(static_cast<uint32_t>(fresh.edge_ids_.size()));

            auto nbrs = getNeighbors(i);
            for (uint32_t nbr : nbrs) {
                if (nbr >= static_cast<uint32_t>(N)) continue;  // guard corrupted IDs
                if (state_[nbr] == ACTIVE)
                    fresh.edge_ids_.push_back(old_to_new[nbr]);
            }
        }
        fresh.offsets_.push_back(static_cast<uint32_t>(fresh.edge_ids_.size()));

        // Move all data members individually — mutex_ cannot be moved/copied.
        words_    = fresh.words_;
        state_    = std::move(fresh.state_);
        offsets_  = std::move(fresh.offsets_);
        edge_ids_ = std::move(fresh.edge_ids_);
        bq_data_  = std::move(fresh.bq_data_);
    }

    // RW lock for external thread safety
    std::shared_mutex& mutex() { return mutex_; }

    void serialize(std::ostream& os) {
        finalizeCSR();  // ensure sentinel present (idempotent)
        auto n     = static_cast<uint32_t>(state_.size());
        auto w     = static_cast<uint32_t>(words_);
        auto edges = static_cast<uint32_t>(edge_ids_.size());
        os.write(reinterpret_cast<const char*>(&n),     sizeof(n));
        os.write(reinterpret_cast<const char*>(&w),     sizeof(w));
        os.write(reinterpret_cast<const char*>(&edges), sizeof(edges));
        os.write(reinterpret_cast<const char*>(state_.data()),    n * sizeof(State));
        os.write(reinterpret_cast<const char*>(offsets_.data()),  (n + 1) * sizeof(uint32_t));
        os.write(reinterpret_cast<const char*>(edge_ids_.data()), edges * sizeof(uint32_t));
        // Interleaved bq_data_: n * 2 * words_ uint64_t
        os.write(reinterpret_cast<const char*>(bq_data_.data()),
                 static_cast<std::streamsize>(static_cast<size_t>(n) * words_ * 2 * sizeof(uint64_t)));
    }

    void deserialize(std::istream& is) {
        uint32_t n = 0, w = 0, edges = 0;
        is.read(reinterpret_cast<char*>(&n),     sizeof(n));
        is.read(reinterpret_cast<char*>(&w),     sizeof(w));
        is.read(reinterpret_cast<char*>(&edges), sizeof(edges));
        if (!is) return;
        words_ = static_cast<int>(w);
        state_.resize(n);
        offsets_.resize(static_cast<size_t>(n) + 1);
        edge_ids_.resize(edges);
        bq_data_.resize(static_cast<size_t>(n) * words_ * 2);
        is.read(reinterpret_cast<char*>(state_.data()),    n * sizeof(State));
        is.read(reinterpret_cast<char*>(offsets_.data()),  (n + 1) * sizeof(uint32_t));
        is.read(reinterpret_cast<char*>(edge_ids_.data()), edges * sizeof(uint32_t));
        is.read(reinterpret_cast<char*>(bq_data_.data()),
                static_cast<std::streamsize>(static_cast<size_t>(n) * words_ * 2 * sizeof(uint64_t)));
        // Clear on stream failure to avoid partially-initialized state
        if (!is) {
            state_.clear(); offsets_.clear(); edge_ids_.clear();
            bq_data_.clear();
            words_ = 0;
        }
    }

    // ---- PackedV2Node: DiskANN-style cache-line co-location ----
    // Record (38B) + n_nbrs (1B) + pad (1B) + nbrs[64] (256B) + pad2 (24B)
    // = 320 bytes = 5 × 64B cache lines.  Single prefetch covers record AND
    // all neighbor IDs — breaks the CSR load-use dependency chain.

    static constexpr int PACKED_V2_MAX_NBRS = 64;

    struct alignas(64) PackedV2Node {
        NGT::NGTAQ::VectorRecord rec;            // 38 bytes [0..37]
        uint8_t  n_nbrs;                          //  1 byte  [38]
        uint8_t  _pad0;                           //  1 byte  [39]
        uint32_t nbrs[PACKED_V2_MAX_NBRS];        // 256 bytes [40..295]
        uint8_t  _pad1[24];                       //  24 bytes [296..319]
    };
    static_assert(sizeof(PackedV2Node) == 320, "PackedV2Node must be 320 bytes");

    // Build packed layout from v2_records_ + CSR.  Call after all setRecord()
    // and finalizeCSR() are done (fromNGTv2 / loadV2 tail).
    void buildPackedV2() {
        const size_t N = state_.size();
        packed_v2_.resize(N);
        for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
            auto& pn = packed_v2_[i];
            if (i < v2_records_.size()) pn.rec = v2_records_[i];
            // Read neighbors from CSR
            uint32_t off_b = offsets_[i];
            uint32_t off_e = (i + 1 < offsets_.size()) ? offsets_[i + 1] : off_b;
            uint32_t cnt = static_cast<uint32_t>(
                std::min<size_t>(off_e - off_b, PACKED_V2_MAX_NBRS));
            pn.n_nbrs = static_cast<uint8_t>(cnt);
            pn._pad0 = 0;
            if (cnt > 0)
                std::memcpy(pn.nbrs, edge_ids_.data() + off_b, cnt * sizeof(uint32_t));
        }
    }

    bool hasPackedV2() const { return !packed_v2_.empty(); }

    const PackedV2Node* getPackedNode(uint32_t id) const {
        return packed_v2_.data() + id;
    }

    // Issue 5 consecutive __builtin_prefetch for 5 × 64B cache lines of one node.
    void prefetchPackedNode(uint32_t id) const {
        const char* p = reinterpret_cast<const char*>(packed_v2_.data() + id);
        __builtin_prefetch(p,       0, 1);
        __builtin_prefetch(p +  64, 0, 1);
        __builtin_prefetch(p + 128, 0, 1);
        __builtin_prefetch(p + 192, 0, 1);
        __builtin_prefetch(p + 256, 0, 1);
    }

    // ---- v2 VectorRecord storage (optional, empty if not built) ----

    void reserveV2(size_t n) {
        v2_records_.resize(n);
    }

    void setRecord(uint32_t node_id, const NGT::NGTAQ::VectorRecord& rec) {
        if (static_cast<size_t>(node_id) >= v2_records_.size())
            v2_records_.resize(node_id + 1);
        v2_records_[node_id] = rec;
    }

    const NGT::NGTAQ::VectorRecord& getRecord(uint32_t node_id) const {
        return v2_records_[node_id];
    }

    // Prefetch helpers: issue non-blocking cache hints to hide DRAM latency.
    // Call PREFETCH_DIST iterations ahead of the actual access.
    // locality 3=L1, 2=L2, 1=L3, 0=nontemporal
    void prefetchRecord(uint32_t node_id) const {
        if (node_id < v2_records_.size())
            __builtin_prefetch(&v2_records_[node_id], 0, 3);
    }
    void prefetchNeighbors(uint32_t node_id) const {
        if (static_cast<size_t>(node_id) + 1 < offsets_.size())
            __builtin_prefetch(edge_ids_.data() + offsets_[node_id], 0, 2);
    }
    // Prefetch the CSR offset entry for node_id into L1/L2.
    // Call when pushing a node to the candidate queue: by the time the node is
    // popped and prefetchNeighbors() is called, offsets_[node_id] will be cached,
    // eliminating the synchronous DRAM read that gates the neighbor-list prefetch.
    void prefetchOffset(uint32_t node_id) const {
        if (static_cast<size_t>(node_id) + 1 < offsets_.size())
            __builtin_prefetch(&offsets_[node_id], 0, 3);
    }

    bool hasV2Records() const { return !v2_records_.empty(); }

    void saveV2Records(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("SoAGraph::saveV2Records: cannot open " + path);
        uint64_t n = v2_records_.size();
        fwrite(&n, sizeof(n), 1, f);
        if (n > 0)
            fwrite(v2_records_.data(), sizeof(NGT::NGTAQ::VectorRecord), n, f);
        fclose(f);
    }

    void loadV2Records(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("SoAGraph::loadV2Records: cannot open " + path);
        uint64_t n = 0;
        fread(&n, sizeof(n), 1, f);
        v2_records_.resize(n);
        if (n > 0)
            fread(v2_records_.data(), sizeof(NGT::NGTAQ::VectorRecord), n, f);
        fclose(f);
    }

private:
    int                   words_;
    std::vector<State>    state_;
    std::vector<uint32_t> offsets_;    // CSR offsets [N+1], sentinel appended by finalizeCSR()
    std::vector<uint32_t> edge_ids_;   // CSR edge data
    std::vector<uint64_t> bq_data_;    // interleaved BQ [N * 2 * words_]: [s0,m0,s1,m1,...]
    mutable std::shared_mutex mutex_;
    std::vector<NGT::NGTAQ::VectorRecord> v2_records_;
    std::vector<PackedV2Node>             packed_v2_;  // DiskANN-style packed layout
};

} // namespace NGTAQ
