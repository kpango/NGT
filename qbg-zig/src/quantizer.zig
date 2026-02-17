const std = @import("std");
const ngt = @import("ngt.zig");
const serializer = @import("serializer.zig");

pub const NGTQ_SIMD_BLOCK_SIZE = 16;
pub const NGTQ_BATCH_SIZE = 2;

pub const Quantizer = struct {
    global_codebook: ngt.Index,
    local_codebooks: []ngt.ObjectRepository,
    rotation: ?[]f32,
    dimension: u32,
    division_no: u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Quantizer) void {
        self.global_codebook.deinit();
        for (self.local_codebooks) |*repo| {
            repo.deinit();
        }
        self.allocator.free(self.local_codebooks);
        if (self.rotation) |rot| {
            self.allocator.free(rot);
        }
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Quantizer {
        // Load global codebook (NGT Index)
        var global_path = try std.fs.path.join(allocator, &[_][]const u8{ path, "global" });
        defer allocator.free(global_path);

        const global_codebook = try ngt.Index.load(allocator, global_path);

        // Load local codebooks (NGT Object Repositories)
        var local_repos = std.ArrayList(ngt.ObjectRepository).init(allocator);
        defer local_repos.deinit();

        var i: u32 = 0;
        while (true) : (i += 1) {
            var name_buf: [32]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "local{}", .{i}) catch break;
            var local_dir_path = try std.fs.path.join(allocator, &[_][]const u8{ path, name });
            defer allocator.free(local_dir_path);

            const file = std.fs.cwd().openDir(local_dir_path, .{}) catch |err| {
                if (err == error.FileNotFound) break;
                return err;
            };
            file.close();

            var obj_path = try std.fs.path.join(allocator, &[_][]const u8{ local_dir_path, "obj" });
            defer allocator.free(obj_path);

            const obj_file = try std.fs.cwd().openFile(obj_path, .{});
            defer obj_file.close();

            var obj_reader = obj_file.reader();
            var obj_ser = serializer.Serializer.init(allocator, obj_reader.any());

            const repo = try ngt.ObjectRepository.read(allocator, &obj_ser);
            try local_repos.append(repo);
        }

        const local_codebooks = try local_repos.toOwnedSlice();
        const division_no: u32 = @intCast(local_codebooks.len);

        // Load Rotation "R"
        var rotation: ?[]f32 = null;
        var rotation_path = try std.fs.path.join(allocator, &[_][]const u8{ path, "R" });
        defer allocator.free(rotation_path);

        if (std.fs.cwd().openFile(rotation_path, .{})) |rot_file| {
            defer rot_file.close();
            var reader = rot_file.reader();
            var floats = std.ArrayList(f32).init(allocator);
            defer floats.deinit();

            var buf: [4096]u8 = undefined;
            while (try reader.readUntilDelimiterOrEof(&buf, '\n')) |line| {
                var it = std.mem.tokenizeAny(u8, line, " \t\r");
                while (it.next()) |token| {
                     const val = std.fmt.parseFloat(f32, token) catch continue;
                     try floats.append(val);
                }
            }
            rotation = try floats.toOwnedSlice();
        } else |err| {
            // Ignore if not found
        }

        var dimension: u32 = 0;
        if (global_codebook.objects.objects.len > 0) {
             for (global_codebook.objects.objects) |obj_opt| {
                 if (obj_opt) |obj| {
                     dimension = @intCast(obj.len);
                     break;
                 }
             }
        }

        return .{
            .global_codebook = global_codebook,
            .local_codebooks = local_codebooks,
            .rotation = rotation,
            .dimension = dimension,
            .division_no = division_no,
            .allocator = allocator,
        };
    }

    // Returns byte size for quantized stream of n objects (4-bit PQ)
    pub fn getUint4StreamSize(self: *Quantizer, n: u64) usize {
        const batch_size = NGTQ_BATCH_SIZE; // 2
        const block_size = NGTQ_SIMD_BLOCK_SIZE; // 16

        const m_aligned = ((self.division_no - 1) / batch_size + 1) * batch_size;
        const n_aligned = ((n - 1) / block_size + 1) * block_size;

        const stream_size = n_aligned * m_aligned;
        return stream_size / 2;
    }

    pub fn createDistanceLookup(self: *Quantizer, query: []const f32) ![][]f32 {
        if (self.local_codebooks.len == 0) return error.NoLocalCodebooks;

        var rotated_buf: ?[]f32 = null;
        var query_vec: []const f32 = query;
        defer if (rotated_buf) |buf| self.allocator.free(buf);

        if (self.rotation) |rot| {
             const dim = query.len;
             if (rot.len == dim * dim) {
                 const buf = try self.allocator.alloc(f32, dim);
                 rotated_buf = buf;
                 // R * q
                 for (0..dim) |r| {
                     var sum: f32 = 0;
                     for (0..dim) |c| {
                         sum += rot[r * dim + c] * query[c];
                     }
                     buf[r] = sum;
                 }
                 query_vec = buf;
             }
        }

        const lut = try self.allocator.alloc([]f32, self.division_no);
        errdefer {
             for (lut) |l| if (l.len > 0) self.allocator.free(l);
             self.allocator.free(lut);
        }

        const sub_dim = query_vec.len / self.division_no;
        if (sub_dim == 0) return error.InvalidDimension;

        for (0..self.division_no) |d| {
            const centroids = self.local_codebooks[d];
            const n_centroids = centroids.objects.len;
            lut[d] = try self.allocator.alloc(f32, n_centroids);

            const query_sub = query_vec[d * sub_dim .. (d + 1) * sub_dim];

            for (0..n_centroids) |c| {
                if (centroids.objects[c]) |centroid| {
                    if (centroid.len != sub_dim) return error.DimensionMismatch;
                    var dist: f32 = 0;
                    for (0..sub_dim) |k| {
                        const diff = query_sub[k] - centroid[k];
                        dist += diff * diff;
                    }
                    lut[d][c] = dist;
                } else {
                    lut[d][c] = std.math.inf(f32);
                }
            }
        }
        return lut;
    }
};
