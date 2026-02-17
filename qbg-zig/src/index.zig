const std = @import("std");
const builtin = @import("builtin");
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

    // Prefetch a slice of memory
    fn prefetch_slice(ptr: []const u8) void {
        var i: usize = 0;
        // Prefetch with cache line stride (64 bytes usually)
        while (i < ptr.len) : (i += 64) {
            @prefetch(&ptr[i], .{ .rw = .read, .locality = 3, .cache = .data });
        }
    }

    inline fn simd_lookup(lut: @Vector(16, u8), indices: @Vector(16, u8)) @Vector(16, u8) {
        if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
             return asm (
                 "vpshufb %[idx], %[lut], %[res]"
                 : [res] "=x" (-> @Vector(16, u8)),
                 : [idx] "x" (indices),
                   [lut] "x" (lut)
             );
        } else if (builtin.cpu.arch == .aarch64 and std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon)) {
             return asm (
                 "tbl %[res].16b, {%[lut].16b}, %[idx].16b"
                 : [res] "=w" (-> @Vector(16, u8)),
                 : [lut] "w" (lut),
                   [idx] "w" (indices)
             );
        } else {
             var res: @Vector(16, u8) = undefined;
             const l_arr: [16]u8 = @bitCast(lut);
             const i_arr: [16]u8 = @bitCast(indices);
             inline for (0..16) |i| {
                 res[i] = l_arr[i_arr[i] & 0x0F];
             }
             return res;
        }
    }

    fn compute_pq_distance(
        lut: *quantizer_mod.DistanceLookupTableUint8,
        objects: []const u8,
        n_objects: usize,
        m: usize,
        distances: []f32
    ) void {
        const m_aligned = ((m - 1) / 2 + 1) * 2;
        const blk_size = 16 * m_aligned;

        var idx: usize = 0;
        while (idx < n_objects) : (idx += 16) {
            const blk_no = idx / 16;
            const remaining = n_objects - idx;

            var acc_u16 = @Vector(16, u16){ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            for (0..m) |sub| {
                var lut_vec: @Vector(16, u8) = undefined;
                @memcpy(&lut_vec, lut.lut[sub][0..16]);

                const byte_offset = (blk_no * blk_size + 16 * sub) / 2;

                var codes_8: @Vector(8, u8) = undefined;
                if (byte_offset + 8 <= objects.len) {
                    @memcpy(&codes_8, objects[byte_offset..][0..8]);
                } else {
                    @memset(&codes_8, 0);
                }

                const low_nibbles = codes_8 & @as(@Vector(8, u8), @splat(0x0F));
                const high_nibbles = codes_8 >> @as(@Vector(8, u8), @splat(4));

                const indices = @shuffle(u8, low_nibbles, high_nibbles,
                    @Vector(16, i32){0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}
                );

                const values_u8 = simd_lookup(lut_vec, indices);
                const values_u16 = @as(@Vector(16, u16), values_u8);
                acc_u16 += values_u16;
            }

            const acc0_u16 = @shuffle(u16, acc_u16, undefined, @Vector(8, i32){0, 1, 2, 3, 4, 5, 6, 7});
            const acc1_u16 = @shuffle(u16, acc_u16, undefined, @Vector(8, i32){8, 9, 10, 11, 12, 13, 14, 15});

            const acc0_f = convertU16ToF32(acc0_u16);
            const acc1_f = convertU16ToF32(acc1_u16);

            const v_scale = @as(@Vector(8, f32), @splat(lut.scale));
            const v_total_offset = @as(@Vector(8, f32), @splat(lut.total_offset));

            var res0 = acc0_f * v_scale + v_total_offset;
            var res1 = acc1_f * v_scale + v_total_offset;

            res0 = @sqrt(res0);
            res1 = @sqrt(res1);

            const store_len = @min(16, remaining);
            var res: [16]f32 = undefined;
            const p0: *align(1) @Vector(8, f32) = @ptrCast(&res[0]);
            p0.* = res0;
            const p1: *align(1) @Vector(8, f32) = @ptrCast(&res[8]);
            p1.* = res1;

            @memcpy(distances[idx..idx+store_len], res[0..store_len]);
        }
    }

    fn convertU16ToF32(v: @Vector(8, u16)) @Vector(8, f32) {
        return @Vector(8, f32){
            @floatFromInt(v[0]), @floatFromInt(v[1]), @floatFromInt(v[2]), @floatFromInt(v[3]),
            @floatFromInt(v[4]), @floatFromInt(v[5]), @floatFromInt(v[6]), @floatFromInt(v[7])
        };
    }

    pub fn search(self: *Index, query: []const f32, size: usize, epsilon: f32, blob_epsilon: f32) ![]ngt.ObjectDistance {
        const blobs = try self.quantizer.global_codebook.search(query, size, blob_epsilon);
        defer self.allocator.free(blobs);

        var lut = try self.quantizer.createDistanceLookupUint8(query);
        defer lut.deinit();

        var results = std.PriorityQueue(ngt.ObjectDistance, void, ngt.ObjectDistance.compareReverse).init(self.allocator, {});
        defer results.deinit();

        var dist_buffer = std.ArrayList(f32).init(self.allocator);
        defer dist_buffer.deinit();

        for (blobs) |blob_node| {
            const blob_id = blob_node.id;
            if (blob_id >= self.qbg_repo.nodes.len) continue;

            const blob = self.qbg_repo.nodes[blob_id];
            const n_obj = blob.ids.len;

            // Prefetch the blob's objects
            prefetch_slice(blob.objects);

            if (dist_buffer.capacity < n_obj) {
            }
            try dist_buffer.resize(n_obj);

            compute_pq_distance(&lut, blob.objects, n_obj, self.quantizer.division_no, dist_buffer.items);

            for (0..n_obj) |k| {
                const obj_id = blob.ids[k];
                const dist = dist_buffer.items[k];
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
