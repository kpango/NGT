const std = @import("std");
const serializer = @import("serializer.zig");
const quantizer_mod = @import("quantizer.zig");

pub const QuantizedNode = struct {
    subspace_id: u32,
    ids: []u32,
    objects: []u8, // Packed 4-bit PQ codes

    pub fn deinit(self: *QuantizedNode, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.objects);
    }
};

pub const QuantizedBlobGraphRepository = struct {
    nodes: []QuantizedNode,
    num_subspaces: u64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *QuantizedBlobGraphRepository) void {
        for (self.nodes) |*node| {
            node.deinit(self.allocator);
        }
        self.allocator.free(self.nodes);
    }

    pub fn read(allocator: std.mem.Allocator, ser: *serializer.Serializer, quantizer: *quantizer_mod.Quantizer) !QuantizedBlobGraphRepository {
        const num_subspaces = try ser.readInt(u64);
        const size_u64 = try ser.readInt(u64);
        const size = std.math.cast(usize, size_u64) orelse return error.IndexTooLarge;

        const nodes = try allocator.alloc(QuantizedNode, size);
        errdefer allocator.free(nodes);

        // Initialize nodes to empty to ensure safe cleanup on error?
        // Zig's alloc does not zero initialize structs unless fields are default.
        // If we error in the loop, we need to clean up already allocated nodes.
        // Simplest way is to clean up in errdefer using a counter.
        var nodes_loaded: usize = 0;
        errdefer {
             for (0..nodes_loaded) |k| {
                 nodes[k].deinit(allocator);
             }
        }

        for (0..size) |i| {
            const subspace_id = try ser.readInt(u32);

            const ids_size_u32 = try ser.readInt(u32);
            const ids_size = std.math.cast(usize, ids_size_u32) orelse return error.IndexTooLarge;

            const ids = try allocator.alloc(u32, ids_size);
            errdefer allocator.free(ids);

            for (0..ids_size) |j| {
                ids[j] = try ser.readInt(u32);
            }

            const stream_size = quantizer.getUint4StreamSize(ids_size_u32);

            const objects = try allocator.alloc(u8, stream_size);
            errdefer allocator.free(objects);

            try ser.readBytes(objects);

            nodes[i] = .{
                .subspace_id = subspace_id,
                .ids = ids,
                .objects = objects,
            };
            nodes_loaded += 1;
        }

        return .{
            .nodes = nodes,
            .num_subspaces = num_subspaces,
            .allocator = allocator,
        };
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8, quantizer: *quantizer_mod.Quantizer) !QuantizedBlobGraphRepository {
        // Load "path/qg/grp"
        var grp_path = try std.fs.path.join(allocator, &[_][]const u8{ path, "qg", "grp" });
        defer allocator.free(grp_path);

        const file = try std.fs.cwd().openFile(grp_path, .{});
        defer file.close();

        var reader = file.reader();
        var ser = serializer.Serializer.init(allocator, reader.any());

        return QuantizedBlobGraphRepository.read(allocator, &ser, quantizer);
    }
};
