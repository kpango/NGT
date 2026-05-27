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

// Three-array SoA graph with CSR edge lists, tombstone-based deletion, and epoch rebuild.
//
// Layout:
//   bq_sign_[node_id * words_ + w]  — sign plane word w for node node_id
//   bq_mag_[node_id * words_ + w]   — magnitude plane word w for node node_id
//   offsets_[node_id]..offsets_[node_id+1] — range in edge_ids_ for node node_id's neighbors
//   state_[node_id]                 — ACTIVE or TOMBSTONE
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
    // Thread-unsafe: caller must hold exclusive lock.
    uint32_t addNode(const uint64_t* sign, const uint64_t* mag) {
        uint32_t id = static_cast<uint32_t>(state_.size());
        state_.push_back(ACTIVE);
        bq_sign_.insert(bq_sign_.end(), sign, sign + words_);
        bq_mag_.insert(bq_mag_.end(),   mag,  mag  + words_);
        // Push start offset for this node (edge_ids_ is empty initially)
        offsets_.push_back(static_cast<uint32_t>(edge_ids_.size()));
        return id;
    }

    // Finalize CSR by appending the sentinel (end offset).
    // Must be called after all addNode() calls and before any getNeighbors() calls.
    void finalizeCSR() {
        if (offsets_.size() == state_.size() + 1) return;  // already finalized
        offsets_.push_back(static_cast<uint32_t>(edge_ids_.size()));
    }

    // Replace neighbor list for node_id. Adjusts all subsequent offsets.
    // Thread-unsafe: caller must hold exclusive lock.
    void setNeighbors(uint32_t node_id, std::vector<uint32_t> neighbors) {
        assert(node_id + 1 < offsets_.size() && "call finalizeCSR() before setNeighbors()");
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
        assert(node_id + 1 < offsets_.size() && "call finalizeCSR() before getNeighbors()");
        uint32_t begin = offsets_[node_id];
        uint32_t end   = offsets_[node_id + 1];
        return {edge_ids_.data() + begin, end - begin};
    }

    const uint64_t* getSignPlane(uint32_t node_id) const {
        assert(node_id < state_.size());
        return bq_sign_.data() + static_cast<size_t>(node_id) * words_;
    }

    const uint64_t* getMagPlane(uint32_t node_id) const {
        assert(node_id < state_.size());
        return bq_mag_.data() + static_cast<size_t>(node_id) * words_;
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
        size_t N = state_.size();
        std::vector<uint32_t> old_to_new(N, UINT32_MAX);
        uint32_t new_id = 0;
        for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i)
            if (state_[i] == ACTIVE) old_to_new[i] = new_id++;

        SoAGraph fresh(words_);
        fresh.bq_sign_.reserve(static_cast<size_t>(new_id) * words_);
        fresh.bq_mag_.reserve(static_cast<size_t>(new_id) * words_);
        fresh.state_.reserve(new_id);
        fresh.offsets_.reserve(static_cast<size_t>(new_id) + 1);

        for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
            if (state_[i] != ACTIVE) continue;
            const uint64_t* s = getSignPlane(i);
            const uint64_t* m = getMagPlane(i);
            fresh.bq_sign_.insert(fresh.bq_sign_.end(), s, s + words_);
            fresh.bq_mag_.insert(fresh.bq_mag_.end(),   m, m + words_);
            fresh.state_.push_back(ACTIVE);
            fresh.offsets_.push_back(static_cast<uint32_t>(fresh.edge_ids_.size()));

            auto nbrs = getNeighbors(i);
            for (uint32_t nbr : nbrs) {
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
        bq_sign_  = std::move(fresh.bq_sign_);
        bq_mag_   = std::move(fresh.bq_mag_);
    }

    // RW lock for external thread safety
    std::shared_mutex& mutex() { return mutex_; }

    void serialize(std::ostream& os) const {
        auto n     = static_cast<uint32_t>(state_.size());
        auto w     = static_cast<uint32_t>(words_);
        auto edges = static_cast<uint32_t>(edge_ids_.size());
        os.write(reinterpret_cast<const char*>(&n),     sizeof(n));
        os.write(reinterpret_cast<const char*>(&w),     sizeof(w));
        os.write(reinterpret_cast<const char*>(&edges), sizeof(edges));
        os.write(reinterpret_cast<const char*>(state_.data()),    n * sizeof(State));
        os.write(reinterpret_cast<const char*>(offsets_.data()),  (n + 1) * sizeof(uint32_t));
        os.write(reinterpret_cast<const char*>(edge_ids_.data()), edges * sizeof(uint32_t));
        os.write(reinterpret_cast<const char*>(bq_sign_.data()),  static_cast<size_t>(n) * words_ * sizeof(uint64_t));
        os.write(reinterpret_cast<const char*>(bq_mag_.data()),   static_cast<size_t>(n) * words_ * sizeof(uint64_t));
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
        bq_sign_.resize(static_cast<size_t>(n) * words_);
        bq_mag_.resize(static_cast<size_t>(n) * words_);
        is.read(reinterpret_cast<char*>(state_.data()),    n * sizeof(State));
        is.read(reinterpret_cast<char*>(offsets_.data()),  (n + 1) * sizeof(uint32_t));
        is.read(reinterpret_cast<char*>(edge_ids_.data()), edges * sizeof(uint32_t));
        is.read(reinterpret_cast<char*>(bq_sign_.data()),  static_cast<size_t>(n) * words_ * sizeof(uint64_t));
        is.read(reinterpret_cast<char*>(bq_mag_.data()),   static_cast<size_t>(n) * words_ * sizeof(uint64_t));
    }

private:
    int                   words_;
    std::vector<State>    state_;
    std::vector<uint32_t> offsets_;    // CSR offsets [N+1], sentinel appended by finalizeCSR()
    std::vector<uint32_t> edge_ids_;   // CSR edge data
    std::vector<uint64_t> bq_sign_;    // sign planes [N * words_]
    std::vector<uint64_t> bq_mag_;     // magnitude planes [N * words_]
    mutable std::shared_mutex mutex_;
};

} // namespace NGTAQ
