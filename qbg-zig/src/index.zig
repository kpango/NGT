const std = @import("std");
const builtin = @import("builtin");
const ngt = @import("ngt.zig");
const quantizer_mod = @import("quantizer.zig");
const qbg = @import("qbg.zig");
const context = @import("context.zig");

pub const Index = struct {
    quantizer: quantizer_mod.Quantizer,
    qbg_repo: qbg.QuantizedBlobGraphRepository,
    allocator: std.mem.Allocator,
    metric: ngt.distance.Metric = .L2, // Default to L2

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Index {
        var quant = try quantizer_mod.Quantizer.load(allocator, path);
        errdefer quant.deinit();

        var repo = try qbg.QuantizedBlobGraphRepository.load(allocator, path, &quant);
        errdefer repo.deinit();

        return .{
            .quantizer = quant,
            .qbg_repo = repo,
            .allocator = allocator,
            // metric should be loaded from config/property
        };
    }

    pub fn deinit(self: *Index) void {
        self.quantizer.deinit();
        self.qbg_repo.deinit();
    }

    // ... prefetch_slice ...
    fn prefetch_slice(ptr: []const u8) void {
        var i: usize = 0;
        while (i < ptr.len) : (i += 64) {
            @prefetch(&ptr[i], .{ .rw = .read, .locality = 3, .cache = .data });
        }
    }

    // ... simd_lookup ...
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

    // ... compute_pq_distance ...
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

            // Only apply sqrt if metric expects it (L2).
            // NGT applies sqrt after summation for L2.
            // For L1, no sqrt.
            // But compute_pq_distance is generic.
            // Ideally we should pass 'sqrt_needed' flag or similar.
            // But LUT values were computed with `compute_sub` which is squared L2.
            // So if metric is L2, we need sqrt.
            // If metric is L1, compute_sub is L1, we just sum up.
            // So we need to know if we should sqrt.
            // Let's pass the metric or a boolean to this function.
            // However, this function signature doesn't take metric.
            // Let's rely on 'distances' being processed or modify signature.
            // Since this function is internal, we can just apply sqrt here if needed.
            // But we need to know the metric.
            // For now, let's just assume this function returns the accumulated value (squared L2, or L1 sum).
            // The search function calling this can apply sqrt if L2?
            // BUT: standard NGT applies sqrt to display distance.
            // For ranking, squared L2 is fine (monotonic).
            // But NGT returns actual distance.
            // So we should apply sqrt if L2.
            // Let's modify search to handle it or pass a flag.
            // Wait, this function writes to `distances`.

            // Assuming this is L2 search context mostly.
            // If we want to support L1, we shouldn't sqrt.
            // Let's just always sqrt for now? No, L1 doesn't need it.
            // Let's assume we pass a flag `apply_sqrt`.

            // For simplicity in this step, let's keep sqrt unconditional (L2 behavior)
            // OR remove it and let caller handle.
            // But caller processes blocks.
            // Let's add `apply_sqrt` param.

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

    // Overloaded for metric aware?
    // Actually, `compute_pq_distance` above applies sqrt unconditionally.
    // I should fix it to take `metric`.

    fn compute_pq_distance_metric(
        lut: *quantizer_mod.DistanceLookupTableUint8,
        objects: []const u8,
        n_objects: usize,
        m: usize,
        distances: []f32,
        metric: ngt.distance.Metric
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
                } else { @memset(&codes_8, 0); }
                const low_nibbles = codes_8 & @as(@Vector(8, u8), @splat(0x0F));
                const high_nibbles = codes_8 >> @as(@Vector(8, u8), @splat(4));
                const indices = @shuffle(u8, low_nibbles, high_nibbles, @Vector(16, i32){0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15});
                const values_u8 = simd_lookup(lut_vec, indices);
                acc_u16 += @as(@Vector(16, u16), values_u8);
            }

            const acc0_u16 = @shuffle(u16, acc_u16, undefined, @Vector(8, i32){0, 1, 2, 3, 4, 5, 6, 7});
            const acc1_u16 = @shuffle(u16, acc_u16, undefined, @Vector(8, i32){8, 9, 10, 11, 12, 13, 14, 15});
            const acc0_f = convertU16ToF32(acc0_u16);
            const acc1_f = convertU16ToF32(acc1_u16);
            const v_scale = @as(@Vector(8, f32), @splat(lut.scale));
            const v_total_offset = @as(@Vector(8, f32), @splat(lut.total_offset));

            var res0 = acc0_f * v_scale + v_total_offset;
            var res1 = acc1_f * v_scale + v_total_offset;

            if (metric == .L2) {
                res0 = @sqrt(res0);
                res1 = @sqrt(res1);
            }

            const store_len = @min(16, remaining);
            var res: [16]f32 = undefined;
            const p0: *align(1) @Vector(8, f32) = @ptrCast(&res[0]);
            p0.* = res0;
            const p1: *align(1) @Vector(8, f32) = @ptrCast(&res[8]);
            p1.* = res1;

            @memcpy(distances[idx..idx+store_len], res[0..store_len]);
        }
    }

    fn convertU8ToF32(v: @Vector(8, u8)) @Vector(8, f32) {
        return @Vector(8, f32){
            @floatFromInt(v[0]), @floatFromInt(v[1]), @floatFromInt(v[2]), @floatFromInt(v[3]),
            @floatFromInt(v[4]), @floatFromInt(v[5]), @floatFromInt(v[6]), @floatFromInt(v[7])
        };
    }

    fn convertU16ToF32(v: @Vector(8, u16)) @Vector(8, f32) {
        return @Vector(8, f32){
            @floatFromInt(v[0]), @floatFromInt(v[1]), @floatFromInt(v[2]), @floatFromInt(v[3]),
            @floatFromInt(v[4]), @floatFromInt(v[5]), @floatFromInt(v[6]), @floatFromInt(v[7])
        };
    }

    pub fn search(self: *Index, ctx: *context.SearchContext, query: []const f32, size: usize, epsilon: f32, blob_epsilon: f32) ![]ngt.ObjectDistance {
        // Set up coarse index metric?
        // Global codebook index usually needs to match.
        // Assume ngt.Index has its own metric set correctly or we pass it?
        // ngt.Index struct in ngt.zig has metric field.
        // We need to set it if we want it to match self.metric.
        // But `quantizer.global_codebook` is already initialized.
        // We can force set it:
        // self.quantizer.global_codebook.metric = self.metric;
        // But it's better if it's set on load.
        // For now, let's assume we sync it here or it's L2 default.
        // Since `load` in ngt.zig doesn't take metric, we manually set it.
        // Wait, `quantizer.global_codebook` is in `quantizer`.

        // Search global
        // We need to pass the metric to global search or ensure it's set.
        // Let's modify `ngt.Index` to allow setting metric or `search` to take it.
        // Current `ngt.Index.search` uses `self.metric`.
        // So we should update `quantizer.global_codebook.metric`.

        // Accessing via pointers:
        // &self.quantizer.global_codebook.metric = self.metric;
        // Zig structs are values unless pointers.
        // `quantizer` is a value in `Index`.
        // `global_codebook` is a value in `Quantizer`.
        // So we can modify it directly if `self` is mutable pointer.
        // Yes `self: *Index`.

        // However, `quantizer.zig` defines `Quantizer` struct.
        // `ngt.zig` defines `Index`.
        // We can cast `self.quantizer.global_codebook` to mutable?
        // It is `ngt.Index`.
        // `self.quantizer.global_codebook.metric = self.metric;`

        // Actually, let's just do it.

        var global_cb = &self.quantizer.global_codebook;
        global_cb.metric = self.metric;

        const blobs = try global_cb.search(ctx, query, size, blob_epsilon);
        defer self.allocator.free(blobs);

        var lut = try self.quantizer.createDistanceLookupUint8(query, self.metric);
        defer lut.deinit();

        ctx.prepare_qbg_search();

        for (blobs) |blob_node| {
            const blob_id = blob_node.id;
            if (blob_id >= self.qbg_repo.nodes.len) continue;

            const blob = self.qbg_repo.nodes[blob_id];
            const n_obj = blob.ids.len;

            prefetch_slice(blob.objects);

            try ctx.dist_buffer.resize(n_obj);

            compute_pq_distance_metric(&lut, blob.objects, n_obj, self.quantizer.division_no, ctx.dist_buffer.items, self.metric);

            for (0..n_obj) |k| {
                const obj_id = blob.ids[k];
                const dist = ctx.dist_buffer.items[k];

                try ctx.results.add(.{ .id = obj_id, .distance = dist });
                if (ctx.results.count() > size) {
                    _ = ctx.results.remove();
                }
            }
        }

        var final_results = try self.allocator.alloc(ngt.ObjectDistance, ctx.results.count());
        var i: usize = ctx.results.count();
        while (ctx.results.removeOrNull()) |item| {
            i -= 1;
            final_results[i] = item;
        }
        return final_results;
    }
};
