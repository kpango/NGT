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

    // CSR out-degree of node_id (used to validate PackedV2Node alignment with GPQ4 blocks).
    uint32_t neighborCount(uint32_t node_id) const {
        if (static_cast<size_t>(node_id) + 1 >= offsets_.size()) return 0;
        return offsets_[node_id + 1] - offsets_[node_id];
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

        // Migrate v2 flat storage: compact active nodes only, preserving stride config.
        if (!v2_records_flat_.empty()) {
            fresh.v2_tier1_n_    = v2_tier1_n_;
            fresh.v2_tier2_n_    = v2_tier2_n_;
            fresh.v2_rec_stride_ = v2_rec_stride_;
            fresh.v2_records_flat_.resize((size_t)new_id * v2_rec_stride_, 0);
            uint32_t dst = 0;
            for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
                if (state_[i] != ACTIVE) continue;
                if ((size_t)i * v2_rec_stride_ + v2_rec_stride_ <= v2_records_flat_.size())
                    memcpy(fresh.v2_records_flat_.data() + (size_t)dst * v2_rec_stride_,
                           v2_records_flat_.data()       + (size_t)i   * v2_rec_stride_,
                           v2_rec_stride_);
                ++dst;
            }
        }

        // Move all data members individually — mutex_ cannot be moved/copied.
        words_           = fresh.words_;
        state_           = std::move(fresh.state_);
        offsets_         = std::move(fresh.offsets_);
        edge_ids_        = std::move(fresh.edge_ids_);
        bq_data_         = std::move(fresh.bq_data_);
        v2_records_flat_ = std::move(fresh.v2_records_flat_);
        v2_tier1_n_      = fresh.v2_tier1_n_;
        v2_tier2_n_      = fresh.v2_tier2_n_;
        v2_rec_stride_   = fresh.v2_rec_stride_;
        // GPQ4 store is keyed on pre-compaction node IDs + edge layout; invalidate it.
        // Caller must rebuildGPQ4() after rebuild() if batch routing is in use.
        gpq4_m_ = 0;
        gpq4_offsets_.clear();
        gpq4_blocks_.clear();
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

    // Build packed layout from v2_records_flat_ + CSR.  Call after all setRecord()
    // and finalizeCSR() are done (fromNGTv2 / loadV2 tail).
    void buildPackedV2() {
        const size_t N = state_.size();
        packed_v2_.resize(N);
        for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
            auto& pn = packed_v2_[i];
            if ((size_t)i * 38 + 38 <= v2_records_flat_.size())
                memcpy(&pn.rec, v2_records_flat_.data() + (size_t)i * 38, 38);
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

    // ---- v2 VectorRecord flat byte storage (optional, empty if not built) ----
    // stride = v2_rec_stride_ = v2_tier1_n_ + v2_tier2_n_ + 6
    // Default: D_eff=128 → tier1=16B, tier2=16B, stride=38B

    void reserveV2(size_t n, int tier1_n = 16, int tier2_n = 16) {
        v2_tier1_n_    = tier1_n;
        v2_tier2_n_    = tier2_n;
        v2_rec_stride_ = tier1_n + tier2_n + 6;
        v2_records_flat_.assign((size_t)n * v2_rec_stride_, 0);
    }

    // D=128 backward compat: assumes stride=38
    void setRecord(uint32_t node_id, const NGT::NGTAQ::VectorRecord& rec) {
        if ((size_t)node_id * 38 + 38 > v2_records_flat_.size())
            v2_records_flat_.resize(((size_t)node_id + 1) * 38, 0);
        memcpy(v2_records_flat_.data() + (size_t)node_id * 38, &rec, 38);
    }

    void setRecordView(uint32_t node_id, NGT::NGTAQ::VectorRecordView view) {
        memcpy(v2_records_flat_.data() + (size_t)node_id * v2_rec_stride_, view.ptr, v2_rec_stride_);
    }

    NGT::NGTAQ::VectorRecordView getRecordView(uint32_t node_id) {
        return NGT::NGTAQ::vrec_view(v2_records_flat_.data(), node_id, v2_tier1_n_, v2_tier2_n_);
    }

    // D=128 only: reinterpret flat bytes as VectorRecord
    const NGT::NGTAQ::VectorRecord& getRecord(uint32_t node_id) const {
        return *reinterpret_cast<const NGT::NGTAQ::VectorRecord*>(
            v2_records_flat_.data() + (size_t)node_id * 38);
    }

    NGT::NGTAQ::VectorRecordConstView getRecordConstView(uint32_t node_id) const {
        return NGT::NGTAQ::vrec_const_view(v2_records_flat_.data(), node_id, v2_tier1_n_, v2_tier2_n_);
    }

    // Prefetch helpers: issue non-blocking cache hints to hide DRAM latency.
    // Call PREFETCH_DIST iterations ahead of the actual access.
    // locality 3=L1, 2=L2, 1=L3, 0=nontemporal
    void prefetchRecord(uint32_t node_id) const {
        if ((size_t)node_id * v2_rec_stride_ < v2_records_flat_.size())
            __builtin_prefetch(v2_records_flat_.data() + (size_t)node_id * v2_rec_stride_, 0, 3);
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

    bool hasV2Records() const { return !v2_records_flat_.empty(); }
    int  v2Tier1N()    const { return v2_tier1_n_; }
    int  v2Tier2N()    const { return v2_tier2_n_; }
    int  v2RecStride() const { return v2_rec_stride_; }

    // ---- Stage B: GPQ4 contiguous per-node neighbor-code store ----
    // For each node, its neighbors' 16-centroid (4-bit) global PQ codes are arranged
    // block-16 transposed + uint4-packed (mirrors NGTQ arrangeQuantizedObject), with
    // each neighbor's reconstructed-norm^2 stored inline as fp16. One node's blocks are
    // contiguous → a single prefetch covers codes AND norms (no gather in the kernel).
    //
    // Per-block layout (bytes): [ (M/2)*16 code-plane bytes ][ 16 × fp16 norm = 32 bytes ].
    //   block_bytes = M*8 + 32.   A node with n_nbrs neighbors uses ceil(n_nbrs/16) blocks.
    // gpq4_offsets_[id] = starting BLOCK index for node id's store (units of blocks).

    int gpq4M() const { return gpq4_m_; }
    bool hasGPQ4() const { return !gpq4_blocks_.empty() && gpq4_m_ > 0; }
    size_t gpq4BlockBytes() const { return (size_t)gpq4_m_ * 8 + 32; }
    // True if the inline per-neighbor norm is the usable bf16 form (built/loaded from a
    // fused-norm index). When false (legacy fp16-norm index), callers must gather instead.
    bool gpq4FusedNorm() const { return gpq4_fused_norm_; }
    // Read neighbor `n`'s (0..15) fused bf16 norm^2 from a block pointer (codes+norms).
    // Norm region starts at offset ((M+1)/2)*16 bytes into the block (16 × bf16).
    float gpq4BlockNorm(const uint8_t* block, int n) const {
        const uint16_t* normp = reinterpret_cast<const uint16_t*>(
            block + (size_t)((gpq4_m_ + 1) / 2) * 16);
        return NGT::NGTAQ::bf16_to_float(normp[n]);
    }

    // Build the neighbor-code store from a per-node global 4-bit code table and per-node
    // reconstructed-norm^2. Call after finalizeCSR() (edges final). M = #subspaces.
    //   node_codes[id*M + s] = 4-bit global PQ code of node id, subspace s.
    //   node_norm_sq[id]     = ||reconstructed rotated PQ vector||^2 of node id.
    void buildGPQ4(int M, const uint8_t* node_codes, const float* node_norm_sq);

    // Pointer to node id's contiguous neighbor blocks (codes+norms). nullptr if absent.
    const uint8_t* gpq4Blocks(uint32_t id) const {
        if (!hasGPQ4() || (size_t)id + 1 >= gpq4_offsets_.size()) return nullptr;
        return gpq4_blocks_.data() + (size_t)gpq4_offsets_[id] * gpq4BlockBytes();
    }
    // Number of 16-blocks stored for node id (== ceil(n_nbrs/16)).
    uint32_t gpq4NumBlocks(uint32_t id) const {
        if ((size_t)id + 1 >= gpq4_offsets_.size()) return 0;
        return gpq4_offsets_[id + 1] - gpq4_offsets_[id];
    }
    void prefetchGPQ4(uint32_t id) const {
        if (!hasGPQ4() || (size_t)id + 1 >= gpq4_offsets_.size()) return;
        const uint8_t* p = gpq4_blocks_.data() + (size_t)gpq4_offsets_[id] * gpq4BlockBytes();
        const uint8_t* e = gpq4_blocks_.data() + (size_t)gpq4_offsets_[id + 1] * gpq4BlockBytes();
        for (const uint8_t* c = p; c < e; c += 64) __builtin_prefetch(c, 0, 2);
    }

    void saveGPQ4(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("SoAGraph::saveGPQ4: cannot open " + path);
        uint64_t n_off = gpq4_offsets_.size();
        uint64_t n_blk_bytes = gpq4_blocks_.size();
        // hdr[1] = norm format: 0 = legacy fp16 inline norm (overflows → gather fallback),
        //                       1 = bf16 inline norm (fused, read in-loop, no gather).
        uint32_t hdr[2] = {(uint32_t)gpq4_m_, (uint32_t)1};
        fwrite(hdr, sizeof(hdr), 1, f);
        fwrite(&n_off, sizeof(n_off), 1, f);
        fwrite(gpq4_offsets_.data(), sizeof(uint32_t), n_off, f);
        fwrite(&n_blk_bytes, sizeof(n_blk_bytes), 1, f);
        fwrite(gpq4_blocks_.data(), 1, n_blk_bytes, f);
        fclose(f);
    }
    // Returns true if the file was present and loaded.
    bool loadGPQ4(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        uint32_t hdr[2] = {0, 0};
        if (fread(hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return false; }
        gpq4_m_ = (int)hdr[0];
        gpq4_fused_norm_ = (hdr[1] == 1);  // bf16 inline norm present & usable in-loop
        uint64_t n_off = 0;
        if (fread(&n_off, sizeof(n_off), 1, f) != 1) { fclose(f); return false; }
        gpq4_offsets_.resize(n_off);
        if (n_off && fread(gpq4_offsets_.data(), sizeof(uint32_t), n_off, f) != n_off) {
            fclose(f); gpq4_offsets_.clear(); gpq4_m_ = 0; return false;
        }
        uint64_t n_blk_bytes = 0;
        if (fread(&n_blk_bytes, sizeof(n_blk_bytes), 1, f) != 1) { fclose(f); return false; }
        gpq4_blocks_.resize(n_blk_bytes);
        if (n_blk_bytes && fread(gpq4_blocks_.data(), 1, n_blk_bytes, f) != n_blk_bytes) {
            fclose(f); gpq4_blocks_.clear(); gpq4_offsets_.clear(); gpq4_m_ = 0; return false;
        }
        fclose(f);
        return true;
    }

    void saveV2Records(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("SoAGraph::saveV2Records: cannot open " + path);
        int stride = std::max(v2_rec_stride_, 1);
        uint32_t n_recs = (uint32_t)(v2_records_flat_.size() / stride);
        uint32_t hdr[4] = {n_recs, (uint32_t)v2_tier1_n_, (uint32_t)v2_tier2_n_, (uint32_t)v2_rec_stride_};
        fwrite(hdr, sizeof(hdr), 1, f);
        fwrite(v2_records_flat_.data(), 1, v2_records_flat_.size(), f);
        fclose(f);
    }

    void loadV2Records(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("loadV2Records: cannot open " + path);
        // Get total file size
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        // Try reading a 4×uint32 header
        uint32_t hdr[4] = {0, 16, 16, 38};
        bool got_hdr = (fread(hdr, sizeof(hdr), 1, f) == 1);
        // Validate new format: tier1_n == tier2_n, both powers-of-2 in [8,128],
        // stride == tier1_n + tier2_n + 6, file size == n_recs*stride + 16(header)
        bool new_fmt = got_hdr
            && hdr[1] == hdr[2]
            && hdr[3] == (uint32_t)(hdr[1] + hdr[2] + 6)
            && hdr[1] >= 8 && hdr[1] <= 128
            && (hdr[1] & (hdr[1] - 1)) == 0  // power of 2
            && sz == (long)(16 + (uint64_t)hdr[0] * hdr[3]);
        if (new_fmt) {
            v2_tier1_n_    = (int)hdr[1];
            v2_tier2_n_    = (int)hdr[2];
            v2_rec_stride_ = (int)hdr[3];
            v2_records_flat_.resize((size_t)hdr[0] * v2_rec_stride_);
            fread(v2_records_flat_.data(), 1, v2_records_flat_.size(), f);
        } else {
            // Old format: raw VectorRecord[N] without header (stride=38, D=128 only)
            fseek(f, 0, SEEK_SET);
            v2_tier1_n_ = 16; v2_tier2_n_ = 16; v2_rec_stride_ = 38;
            v2_records_flat_.resize(sz);
            fread(v2_records_flat_.data(), 1, sz, f);
        }
        fclose(f);
    }

private:
    int                   words_;
    std::vector<State>    state_;
    std::vector<uint32_t> offsets_;    // CSR offsets [N+1], sentinel appended by finalizeCSR()
    std::vector<uint32_t> edge_ids_;   // CSR edge data
    std::vector<uint64_t> bq_data_;    // interleaved BQ [N * 2 * words_]: [s0,m0,s1,m1,...]
    mutable std::shared_mutex mutex_;
    std::vector<uint8_t>  v2_records_flat_;   // flat byte array, stride = v2_rec_stride_
    int v2_tier1_n_    = 16;   // D_eff / 8  (16 for D_eff=128)
    int v2_tier2_n_    = 16;   // D_eff / 8  (16 for D_eff=128)
    int v2_rec_stride_ = 38;   // tier1+tier2+norm+centroid = D_eff/4+6
    std::vector<PackedV2Node>             packed_v2_;  // DiskANN-style packed layout

    // Stage B: GPQ4 contiguous neighbor-code store (see buildGPQ4 / gpq4Blocks).
    int                   gpq4_m_ = 0;          // #subspaces (0 = absent)
    bool                  gpq4_fused_norm_ = false;  // block holds usable bf16 inline norm
    std::vector<uint32_t> gpq4_offsets_;        // [N+1] starting block index per node
    std::vector<uint8_t>  gpq4_blocks_;         // flat: blocks of (M*8 + 32) bytes
};

inline void SoAGraph::buildGPQ4(int M, const uint8_t* node_codes, const float* node_norm_sq) {
    finalizeCSR();
    gpq4_m_ = M;
    gpq4_fused_norm_ = true;  // norms stored inline as bf16 (see normp write below)
    constexpr uint32_t BLK = 16;  // neighbors per SIMD block
    const size_t N = state_.size();
    const size_t blk_bytes = (size_t)M * 8 + 32;
    const int planes = (M + 1) / 2;
    // 1. Count blocks per node → offsets.
    gpq4_offsets_.assign(N + 1, 0);
    for (size_t i = 0; i < N; ++i) {
        uint32_t nn = offsets_[i + 1] - offsets_[i];
        gpq4_offsets_[i + 1] = gpq4_offsets_[i] + (nn + BLK - 1) / BLK;
    }
    const size_t total_blocks = gpq4_offsets_[N];
    gpq4_blocks_.assign(total_blocks * blk_bytes, 0);
    // 2. Fill each node's blocks.
    for (size_t i = 0; i < N; ++i) {
        uint32_t b = offsets_[i];
        uint32_t nn = offsets_[i + 1] - b;
        uint8_t* base = gpq4_blocks_.data() + (size_t)gpq4_offsets_[i] * blk_bytes;
        for (uint32_t blk = 0; blk * BLK < nn; ++blk) {
            uint8_t* blkp = base + (size_t)blk * blk_bytes;
            uint8_t* codep = blkp;                 // code planes
            uint16_t* normp = reinterpret_cast<uint16_t*>(blkp + (size_t)planes * 16);
            int n_real = (int)std::min<uint32_t>(BLK, nn - blk * BLK);
            for (int n = 0; n < n_real; ++n) {
                uint32_t nbr = edge_ids_[b + blk * BLK + n];
                const uint8_t* nc = node_codes + (size_t)nbr * M;
                for (int s = 0; s < M; ++s) {
                    const int plane = s / 2;
                    const int half  = (s & 1) * 8;
                    const uint8_t code = nc[s] & 0x0f;
                    uint8_t* dst = codep + (size_t)plane * 16 + half + (n / 2);
                    if (n & 1) *dst |= (code << 4);
                    else       *dst |= code;
                }
                // Fused per-neighbor norm: bf16 (NOT fp16) so SIFT recon norms ~2e5 don't
                // overflow (fp16 max 65504). The batch loop reads this inline instead of
                // gathering gpq4_norm_sq_[u] from a 4MB array (kills the hot-path cache miss).
                normp[n] = NGT::NGTAQ::float_to_bf16(node_norm_sq[nbr]);
            }
        }
    }
}

} // namespace NGTAQ
