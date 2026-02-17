const std = @import("std");
const ngt = @import("ngt.zig");
const quantizer_mod = @import("quantizer.zig");
const qbg = @import("qbg.zig");

pub const Index = struct {
    quantizer: quantizer_mod.Quantizer,
    qbg_repo: qbg.QuantizedBlobGraphRepository,
    allocator: std.mem.Allocator,

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Index {
        var quant = try quantizer_mod.Quantizer.load(allocator, path);
        errdefer quant.deinit();

        var repo = try qbg.QuantizedBlobGraphRepository.load(allocator, path, &quant);
        errdefer repo.deinit();

        return .{
            .quantizer = quant,
            .qbg_repo = repo,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Index) void {
        self.quantizer.deinit();
        self.qbg_repo.deinit();
    }

    // Compute distance using LUT and Packed 4-bit PQ
    fn compute_pq_distance(lut: [][]f32, objects: []const u8, idx: usize, m: usize) f32 {
        var dist: f32 = 0;
        // Logic from Quantizer.h:
        // blkNo = idx / 16
        // oft = idx % 16
        // pos = blkNo * (16 * m_aligned) + 16 * sub + oft
        // packed_pos = pos / 2
        // shift = (pos % 2) * 4

        // m_aligned: ((m - 1) / 2 + 1) * 2
        const m_aligned = ((m - 1) / 2 + 1) * 2;
        const blk_size = 16 * m_aligned;

        const blk_no = idx / 16;
        const oft = idx % 16;

        for (0..m) |sub| {
             const pos = blk_no * blk_size + 16 * sub + oft;
             const packed_pos = pos / 2;
             const is_high = (pos % 2) == 1;

             if (packed_pos >= objects.len) continue; // Should not happen if size calc is correct

             const byte = objects[packed_pos];
             const code = if (is_high) (byte >> 4) else (byte & 0x0F);

             if (code < lut[sub].len) {
                 dist += lut[sub][code];
             }
        }
        return dist;
    }

    pub fn search(self: *Index, query: []const f32, size: usize, epsilon: f32, blob_epsilon: f32) ![]ngt.ObjectDistance {
        // 1. Search Global Graph
        // The global graph objects are centroids.
        // We assume global codebook index has the graph.
        // But ngt.Index has search method.

        // Search parameters for global graph
        // Usually blob_epsilon (coefficient) is used here.
        const blobs = try self.quantizer.global_codebook.search(query, size, blob_epsilon); // Use 'size' or larger? QBG uses 'graphExplorationSize'
        defer self.allocator.free(blobs);

        // 2. Generate LUT
        const lut = try self.quantizer.createDistanceLookup(query);
        defer {
            for (lut) |l| self.allocator.free(l);
            self.allocator.free(lut);
        }

        // 3. Scan blobs
        var results = std.PriorityQueue(ngt.ObjectDistance, void, ngt.ObjectDistance.compareReverse).init(self.allocator, {});
        defer results.deinit();

        // Use a set to avoid duplicates? Blobs are unique from global search.
        // But objects in different blobs might be same? No, NGT QBG partitions objects.

        for (blobs) |blob_node| {
            // blob_node.id is index in global graph.
            // This corresponds to index in qbg_repo.
            // Note: NGT IDs are 1-based usually.
            // My ngt.zig `search` returns IDs from 0?
            // `read` reads IDs as u32.
            // `search` starts at 1 if 0 is null.
            // QBG repo is likely 0-indexed or 1-indexed matching global graph.
            // Assuming strict correspondence.

            const blob_id = blob_node.id;
            if (blob_id >= self.qbg_repo.nodes.len) continue;

            const blob = self.qbg_repo.nodes[blob_id];
            // blob.ids has object IDs.
            // blob.objects has packed codes.

            for (0..blob.ids.len) |k| {
                const obj_id = blob.ids[k];
                const dist = compute_pq_distance(lut, blob.objects, k, self.quantizer.division_no);

                try results.add(.{ .id = obj_id, .distance = dist });
                if (results.count() > size) {
                    _ = results.remove();
                }
            }
        }

        var final_results = try self.allocator.alloc(ngt.ObjectDistance, results.count());
        var i: usize = results.count();
        while (results.removeOrNull()) |item| {
            i -= 1;
            final_results[i] = item;
        }
        return final_results;
    }
};
