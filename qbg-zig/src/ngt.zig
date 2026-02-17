const std = @import("std");
const serializer = @import("serializer.zig");
const context = @import("context.zig");
const distance = @import("distance.zig");

pub const ObjectDistance = struct {
    id: u32,
    distance: f32,

    pub fn read(ser: *serializer.Serializer) !ObjectDistance {
        const id = try ser.readInt(u32);
        const distance_val = try ser.readFloat(f32);
        return .{ .id = id, .distance = distance_val };
    }

    pub fn compare(_: void, a: ObjectDistance, b: ObjectDistance) std.math.Order {
        if (a.distance < b.distance) return .lt;
        if (a.distance > b.distance) return .gt;
        return .eq;
    }

    pub fn compareReverse(_: void, a: ObjectDistance, b: ObjectDistance) std.math.Order {
        if (a.distance > b.distance) return .lt;
        if (a.distance < b.distance) return .gt;
        return .eq;
    }
};

pub const GraphNode = []ObjectDistance;

pub const GraphRepository = struct {
    nodes: []?GraphNode,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GraphRepository) void {
        for (self.nodes) |node_opt| {
            if (node_opt) |node| {
                self.allocator.free(node);
            }
        }
        self.allocator.free(self.nodes);
    }

    pub fn read(allocator: std.mem.Allocator, ser: *serializer.Serializer) !GraphRepository {
        const size = try ser.readInt(u64);
        const nodes = try allocator.alloc(?GraphNode, size);
        errdefer allocator.free(nodes);

        for (0..size) |i| {
            const type_char = try ser.readByte();
            if (type_char == '-') {
                nodes[i] = null;
            } else if (type_char == '+') {
                const node_size = try ser.readInt(u32);
                const node = try allocator.alloc(ObjectDistance, node_size);
                errdefer allocator.free(node);

                for (0..node_size) |j| {
                     node[j] = try ObjectDistance.read(ser);
                }
                nodes[i] = node;
            } else {
                return error.InvalidFormat;
            }
        }

        return .{
            .nodes = nodes,
            .allocator = allocator,
        };
    }
};

pub const Object = []f32;

pub const ObjectRepository = struct {
    objects: []?Object,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ObjectRepository) void {
        for (self.objects) |obj_opt| {
            if (obj_opt) |obj| {
                self.allocator.free(obj);
            }
        }
        self.allocator.free(self.objects);
    }

    pub fn read(allocator: std.mem.Allocator, ser: *serializer.Serializer) !ObjectRepository {
        const size = try ser.readInt(u64);
        const objects = try allocator.alloc(?Object, size);
        errdefer allocator.free(objects);

        for (0..size) |i| {
            const type_char = try ser.readByte();
            if (type_char == '-') {
                objects[i] = null;
            } else if (type_char == '+') {
                const obj_size = try ser.readInt(u32);
                const obj = try allocator.alloc(f32, obj_size);
                errdefer allocator.free(obj);
                for (0..obj_size) |j| {
                    obj[j] = try ser.readFloat(f32);
                }
                objects[i] = obj;
            } else {
                return error.InvalidFormat;
            }
        }

        return .{
            .objects = objects,
            .allocator = allocator,
        };
    }
};

fn prefetch_object(obj: []const f32) void {
    if (obj.len > 0) {
        @prefetch(&obj[0], .{ .rw = .read, .locality = 3, .cache = .data });
    }
}

pub const Index = struct {
    graph: GraphRepository,
    objects: ObjectRepository,
    allocator: std.mem.Allocator,
    metric: distance.Metric = .L2,

    pub fn deinit(self: *Index) void {
        self.graph.deinit();
        self.objects.deinit();
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Index {
        var graph_path = try std.fs.path.join(allocator, &[_][]const u8{ path, "grp" });
        defer allocator.free(graph_path);

        const graph_file = try std.fs.cwd().openFile(graph_path, .{});
        defer graph_file.close();
        var graph_reader = graph_file.reader();
        var graph_ser = serializer.Serializer.init(allocator, graph_reader.any());
        const graph = try GraphRepository.read(allocator, &graph_ser);

        var obj_path = try std.fs.path.join(allocator, &[_][]const u8{ path, "obj" });
        defer allocator.free(obj_path);

        const obj_file = try std.fs.cwd().openFile(obj_path, .{});
        defer obj_file.close();
        var obj_reader = obj_file.reader();
        var obj_ser = serializer.Serializer.init(allocator, obj_reader.any());
        const objects = try ObjectRepository.read(allocator, &obj_ser);

        return .{
            .graph = graph,
            .objects = objects,
            .allocator = allocator,
            // Metric could be loaded from property file
        };
    }

    pub fn search(self: *Index, ctx: *context.SearchContext, query: []const f32, size: usize, epsilon: f32) ![]ObjectDistance {
        try ctx.prepare_graph_search(self.objects.objects.len);

        var start_node: u32 = 1;
        if (self.graph.nodes.len > 1) {
            for (1..self.graph.nodes.len) |i| {
                if (self.graph.nodes[i] != null) {
                    start_node = @intCast(i);
                    break;
                }
            }
        }

        if (self.objects.objects.len <= start_node or self.objects.objects[start_node] == null) {
             return &[_]ObjectDistance{};
        }

        const start_dist = distance.compute(self.metric, query, self.objects.objects[start_node].?);
        const start_obj = ObjectDistance{ .id = start_node, .distance = start_dist };

        try ctx.unchecked.add(start_obj);
        try ctx.results.add(start_obj);
        ctx.visited.set(start_node);

        var exploration_radius = start_dist * (1.0 + epsilon);

        while (ctx.unchecked.removeOrNull()) |target| {
            if (target.distance > exploration_radius) break;

            const node_opt = if (target.id < self.graph.nodes.len) self.graph.nodes[target.id] else null;
            if (node_opt) |neighbors| {
                ctx.candidates.clearRetainingCapacity();

                for (neighbors) |neighbor_ref| {
                    const neighbor_id = neighbor_ref.id;
                    if (neighbor_id >= ctx.visited.capacity) continue;

                    if (ctx.visited.isSet(neighbor_id)) continue;
                    ctx.visited.set(neighbor_id);

                    if (self.objects.objects[neighbor_id] != null) {
                        try ctx.candidates.append(neighbor_id);
                    }
                }

                for (ctx.candidates.items) |cid| {
                    if (self.objects.objects[cid]) |obj| {
                        prefetch_object(obj);
                    }
                }

                for (ctx.candidates.items) |cid| {
                    if (self.objects.objects[cid]) |neighbor_obj| {
                        const dist = distance.compute(self.metric, query, neighbor_obj);
                        const obj = ObjectDistance{ .id = cid, .distance = dist };

                        try ctx.unchecked.add(obj);
                        try ctx.results.add(obj);

                        if (ctx.results.count() > size) {
                            _ = ctx.results.remove();
                            exploration_radius = ctx.results.peek().?.distance * (1.0 + epsilon);
                        }
                    }
                }
            }
        }

        var final_results = try self.allocator.alloc(ObjectDistance, ctx.results.count());
        var i: usize = ctx.results.count();
        while (ctx.results.removeOrNull()) |item| {
            i -= 1;
            final_results[i] = item;
        }
        return final_results;
    }
};
