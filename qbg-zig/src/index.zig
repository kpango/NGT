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

    // SIMD Optimized search
    // Using inline assembly for x86_64 AVX2 shuffle
    fn compute_pq_distance_simd(
        lut: *quantizer_mod.DistanceLookupTableUint8,
        objects: []const u8,
        n_objects: usize,
        m: usize,
        distances: []f32
    ) void {
        // Only optimized for m >= 2 and multiples of 2 (NGTQ_BATCH_SIZE)
        // Only optimized for AVX2 (x86_64)
        if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
             compute_pq_distance_avx2(lut, objects, n_objects, m, distances);
        } else {
             // Fallback to scalar
             compute_pq_distance_scalar(lut, objects, n_objects, m, distances);
        }
    }

    fn compute_pq_distance_scalar(
        lut: *quantizer_mod.DistanceLookupTableUint8,
        objects: []const u8,
        n_objects: usize,
        m: usize,
        distances: []f32
    ) void {
        const m_aligned = ((m - 1) / 2 + 1) * 2;
        const blk_size = 16 * m_aligned;

        for (0..n_objects) |idx| {
            var dist_u32: u32 = 0;
            const blk_no = idx / 16;
            const oft = idx % 16;

            for (0..m) |sub| {
                 const pos = blk_no * blk_size + 16 * sub + oft;
                 const packed_pos = pos / 2;
                 const is_high = (pos % 2) == 1;

                 const byte = objects[packed_pos];
                 const code = if (is_high) (byte >> 4) else (byte & 0x0F);

                 dist_u32 += lut.lut[sub][code];
            }
            // Approximation: sum(dist) * scale + offset
            // NGT implementation sums scaled values separately if scales differ.
            // But usually QBG uses `totalOffset` and unified scale?
            // In Quantizer.h `QuantizedObjectDistanceUint8` -> `operator()` uses `distance += distanceLUT.getDistance(...)`
            // Wait, standard NGT QBG uses float LUT if exact approximation is needed, or uint8 LUT.
            // If uint8 LUT:
            // dist = sqrt(sum(lut[code]) * scale + offset)
            // This assumes unified scale.
            // If individual scales, it sums `lut[code] * scale[sub]`.
            // Our `createDistanceLookupUint8` computes individual scales.
            // But if we want fast SIMD, we assume unified or we do vector madd.

            // Let's implement correct scalar logic matching our LUT generation:
            var dist_f: f32 = 0;
            for (0..m) |sub| {
                 const pos = blk_no * blk_size + 16 * sub + oft;
                 const packed_pos = pos / 2;
                 const is_high = (pos % 2) == 1;
                 const byte = objects[packed_pos];
                 const code = if (is_high) (byte >> 4) else (byte & 0x0F);

                 dist_f += @as(f32, @floatFromInt(lut.lut[sub][code])) * lut.scales[sub] + lut.offsets[sub];
            }
            distances[idx] = std.math.sqrt(dist_f);
        }
    }

    fn compute_pq_distance_avx2(
        lut: *quantizer_mod.DistanceLookupTableUint8,
        objects: []const u8,
        n_objects: usize,
        m: usize,
        distances: []f32
    ) void {
        const m_aligned = ((m - 1) / 2 + 1) * 2;
        const blk_size = 16 * m_aligned; // bytes per block of 16 objects

        // Process in blocks of 16 objects
        var idx: usize = 0;
        while (idx < n_objects) : (idx += 16) {
            const blk_no = idx / 16;
            const remaining = n_objects - idx;
            const current_blk_n = @min(16, remaining); // Should be 16 unless last block

            // We accumulate 16 distances (for objects idx..idx+15) in parallel.
            // AVX2 registers are 256-bit (32 bytes).
            // We can store 8 floats in one YMM register.
            // So we need 2 YMM registers for 16 objects: acc0 (0..7), acc1 (8..15).

            var acc0 = @Vector(8, f32){ 0, 0, 0, 0, 0, 0, 0, 0 };
            var acc1 = @Vector(8, f32){ 0, 0, 0, 0, 0, 0, 0, 0 };

            for (0..m) |sub| {
                // Load LUT for this subspace (16 bytes -> 128 bit)
                const lut_sub = lut.lut[sub];
                // Need to load into XMM.
                // Assuming lut_sub has 16 elements (or padded).
                var lut_vec: @Vector(16, u8) = undefined;
                @memcpy(&lut_vec, lut_sub[0..16]); // Safe if 16 centroids

                // Broadcast LUT to YMM (duplicate 128-bit to both lanes) for vpshufb
                // Zig @shuffle can do this?
                // const lut_ymm = @shuffle(u8, lut_vec, lut_vec, ...);
                // But we need to use it in asm.

                // Load Packed Codes for 16 objects in this subspace.
                // In interleaved format, the 16 codes for 16 objects of subspace `sub`
                // are stored contiguously in 16 nibbles -> 8 bytes.
                // Offset: blk_no * blk_size + 16 * sub (in nibbles).
                // Byte offset: (blk_no * blk_size + 16 * sub) / 2.
                // Wait, logic check:
                // arrange: stream[blkNo * alignedBlockSize + 16 * sub + oft] = byte (not packed)
                // compress: stream[idx] is byte. Packs into uint4.
                // idx iterates stream linearly.
                // Stream order: Block 0 -> Sub 0 (16 bytes) -> Sub 1 (16 bytes).
                // Packed order: Block 0 -> Sub 0 (8 bytes) -> Sub 1 (8 bytes).

                const byte_offset = (blk_no * blk_size + 16 * sub) / 2;
                const codes_packed_ptr = objects.ptr + byte_offset;

                // We need to expand 8 bytes to 16 bytes (nibbles).
                // Load 8 bytes (64 bit) -> broadcast to XMM or load?
                // Actually, we can load 128-bit (16 bytes) which covers sub and sub+1?
                // Or just load 64-bit and unpack.

                // Asm implementation for unpacking and lookup:
                // 1. Load 8 bytes (codes) into XMM.
                // 2. Unpack to 16 bytes:
                //    - Lo nibbles: AND 0x0F
                //    - Hi nibbles: Shift Right 4.
                //    Wait, packing: `uint4Objects[idx / 2] |= (stream[idx] << 4)` (idx odd).
                //    So High Nibble is Obj(odd). Low Nibble is Obj(even).
                //    Obj0 (low), Obj1 (high).
                //    We need to separate them.
                //    Byte: [H: Obj1, L: Obj0]
                //    We want: [Obj0, Obj1, Obj2, ...]
                //    So we unpack 8 bytes to 16 bytes.
                //    XMM: [B0, B1, ... B7, 0...0]
                //    Unpack:
                //      Low:  [B0&F, B1&F... B7&F] -> indices for even objects?
                //      High: [B0>>4, B1>>4...] -> indices for odd objects?
                //    BUT shuffle needs them in order 0..15.
                //    So we need to interleave.
                //    Bytes: [O1 O0], [O3 O2], ...
                //    We want [O0, O1, O2, O3...].
                //    So Low(B0), High(B0), Low(B1), High(B1)...

                // AVX2 instructions:
                // vpmovzxbw (byte to word)? No.
                // We can use bit masking and shifting.

                // Let's try to do this accumulation in standard Zig vector operations if possible,
                // relying on compiler optimization, or fallback to asm.

                // Pure Zig implementation of the inner loop logic (vectorized manually):
                const codes_8 = @as(*const @Vector(8, u8), @ptrCast(@alignCast(codes_packed_ptr))).*;

                // Expand to 16 bytes
                var indices: @Vector(16, u8) = undefined;
                // indices[0] = codes_8[0] & 0xF;
                // indices[1] = codes_8[0] >> 4;
                // ...
                // This is essentially interleave.
                // Zig 0.11/0.12+ supports extensive vector ops.

                // Low nibbles
                const low_nibbles = codes_8 & @as(@Vector(8, u8), @splat(0x0F));
                // High nibbles
                const high_nibbles = codes_8 >> @as(@Vector(8, u8), @splat(4));

                // Interleave
                const indices_low = @shuffle(u8, low_nibbles, high_nibbles, @Vector(16, i32){0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8});
                // Note: shuffle mask: positive index selects from first arg, negative (-1-index) from second.
                // 0 -> low[0]. -1 -> high[0].

                indices = indices_low;

                // Lookup
                // We need to look up 16 values from lut_sub.
                // Vector indexing? `lut_vec[indices]`.
                // Zig doesn't support vector indexing with vector yet?
                // We can use `@shuffle` if indices are comptime, but they are runtime.
                // So we need `pshufb`.

                // Use Asm for pshufb
                var values: @Vector(16, u8) = undefined;
                const lut_vec_16: @Vector(16, u8) = lut_vec;

                if (builtin.cpu.arch == .x86_64) {
                    values = asm (
                        "vpshufb %[idx], %[lut], %[res]"
                        : [res] "=x" (-> @Vector(16, u8)),
                        : [idx] "x" (indices),
                          [lut] "x" (lut_vec_16)
                    );
                } else {
                    // Fallback
                    for (0..16) |k| {
                        values[k] = lut_sub[indices[k]];
                    }
                }

                // Accumulate to floats
                const scale = lut.scales[sub];
                const offset = lut.offsets[sub];

                // Convert u8 values to f32 and madd
                // Split 16 values into 2x8 floats
                const vals_low: @Vector(8, u8) = @shuffle(u8, values, undefined, @Vector(8, i32){0, 1, 2, 3, 4, 5, 6, 7});
                const vals_high: @Vector(8, u8) = @shuffle(u8, values, undefined, @Vector(8, i32){8, 9, 10, 11, 12, 13, 14, 15});

                const f_low = convertU8ToF32(vals_low);
                const f_high = convertU8ToF32(vals_high);

                const v_scale = @as(@Vector(8, f32), @splat(scale));
                const v_offset = @as(@Vector(8, f32), @splat(offset));

                acc0 += f_low * v_scale + v_offset;
                acc1 += f_high * v_scale + v_offset;
            }

            // Sqrt and Store
            acc0 = @sqrt(acc0);
            acc1 = @sqrt(acc1);

            // Store results
            // Be careful with bounds if remaining < 16
            if (remaining >= 16) {
                @as(*@Vector(8, f32), @ptrCast(&distances[idx])).*;
                @as(*@Vector(8, f32), @ptrCast(&distances[idx+8])).* = acc1;
                // Wait, previous line statement has no effect. Assignment needed.
                const dest0 = distances[idx..idx+8];
                @as(*@Vector(8, f32), @ptrCast(dest0.ptr)).* = acc0;

                const dest1 = distances[idx+8..idx+16];
                @as(*@Vector(8, f32), @ptrCast(dest1.ptr)).* = acc1;
            } else {
                // Partial store
                var res: [16]f32 = undefined;
                @as(*@Vector(8, f32), @ptrCast(&res[0])).* = acc0;
                @as(*@Vector(8, f32), @ptrCast(&res[8])).* = acc1;
                for (0..remaining) |k| {
                    distances[idx + k] = res[k];
                }
            }
        }
    }

    fn convertU8ToF32(v: @Vector(8, u8)) @Vector(8, f32) {
        return @Vector(8, f32){
            @floatFromInt(v[0]), @floatFromInt(v[1]), @floatFromInt(v[2]), @floatFromInt(v[3]),
            @floatFromInt(v[4]), @floatFromInt(v[5]), @floatFromInt(v[6]), @floatFromInt(v[7])
        };
    }

    pub fn search(self: *Index, query: []const f32, size: usize, epsilon: f32, blob_epsilon: f32) ![]ngt.ObjectDistance {
        const blobs = try self.quantizer.global_codebook.search(query, size, blob_epsilon);
        defer self.allocator.free(blobs);

        // Use Uint8 LUT
        var lut = try self.quantizer.createDistanceLookupUint8(query);
        defer lut.deinit();

        var results = std.PriorityQueue(ngt.ObjectDistance, void, ngt.ObjectDistance.compareReverse).init(self.allocator, {});
        defer results.deinit();

        // Reusable buffer for distances
        var dist_buffer = std.ArrayList(f32).init(self.allocator);
        defer dist_buffer.deinit();

        for (blobs) |blob_node| {
            const blob_id = blob_node.id;
            if (blob_id >= self.qbg_repo.nodes.len) continue;

            const blob = self.qbg_repo.nodes[blob_id];
            const n_obj = blob.ids.len;

            try dist_buffer.resize(n_obj);

            compute_pq_distance_simd(&lut, blob.objects, n_obj, self.quantizer.division_no, dist_buffer.items);

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
