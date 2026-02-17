const std = @import("std");
const ngt = @import("ngt.zig");

pub const SearchContext = struct {
    visited: std.DynamicBitSet,
    unchecked: std.PriorityQueue(ngt.ObjectDistance, void, ngt.ObjectDistance.compare),
    results: std.PriorityQueue(ngt.ObjectDistance, void, ngt.ObjectDistance.compareReverse),
    candidates: std.ArrayList(u32),
    dist_buffer: std.ArrayList(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SearchContext {
        return .{
            .visited = std.DynamicBitSet.initEmpty(allocator, 0) catch unreachable,
            .unchecked = std.PriorityQueue(ngt.ObjectDistance, void, ngt.ObjectDistance.compare).init(allocator, {}),
            .results = std.PriorityQueue(ngt.ObjectDistance, void, ngt.ObjectDistance.compareReverse).init(allocator, {}),
            .candidates = std.ArrayList(u32).init(allocator),
            .dist_buffer = std.ArrayList(f32).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SearchContext) void {
        self.visited.deinit();
        self.unchecked.deinit();
        self.results.deinit();
        self.candidates.deinit();
        self.dist_buffer.deinit();
    }

    pub fn prepare_graph_search(self: *SearchContext, num_objects: usize) !void {
        // DynamicBitSet.capacity() returns current capacity (number of bits it can hold without realloc).
        // Since Zig 0.11+, it has capacity() method?
        // Actually, DynamicBitSet uses `unmanaged` inside.
        // It exposes `capacity()`. Wait, docs say it manages bit_length.
        // Let's assume standard resize behavior: if we need more bits, resize.
        // To reuse efficiently without shrinking, we check if current capacity is enough.
        // `DynamicBitSet.capacity()` exists in newer Zig.
        // If not available, we can rely on `resize` to be efficient (it usually is).
        // But `resize` changes `bit_length`.
        // We want `bit_length` to match `num_objects`? Yes, we need to track visited status for all objects.

        // Just force resize. If it's already big enough, it might shrink?
        // DynamicBitSet implementation usually reallocates if size changes significantly or implementation dependent.
        // To be safe and simple: always resize.
        try self.visited.resize(num_objects, false);

        // Clear all bits efficiently
        self.visited.setRangeValue(.{ .start = 0, .end = num_objects }, false);

        while (self.unchecked.removeOrNull()) |_| {}
        while (self.results.removeOrNull()) |_| {}
        self.candidates.clearRetainingCapacity();
    }

    pub fn prepare_qbg_search(self: *SearchContext) void {
        while (self.results.removeOrNull()) |_| {}
    }
};
