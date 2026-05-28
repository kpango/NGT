// lib/NGT/NGTAQ/SoAGraph.h
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <shared_mutex>
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

private:
    int                   words_;
    std::vector<State>    state_;
    std::vector<uint32_t> offsets_;    // CSR offsets [N+1], sentinel appended by finalizeCSR()
    std::vector<uint32_t> edge_ids_;   // CSR edge data
    std::vector<uint64_t> bq_data_;    // interleaved BQ [N * 2 * words_]: [s0,m0,s1,m1,...]
    mutable std::shared_mutex mutex_;
};

} // namespace NGTAQ
