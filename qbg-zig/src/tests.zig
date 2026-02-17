const std = @import("std");
const index_mod = @import("index.zig");
const ngt_mod = @import("ngt.zig");
const qbg_mod = @import("qbg.zig");
const quantizer_mod = @import("quantizer.zig");
const serializer_mod = @import("serializer.zig");
const context = @import("context.zig");
const distance = @import("distance.zig");

test "end-to-end load and search" {
    const allocator = std.testing.allocator;

    // Create directory "test_index" in CWD
    const test_idx = "test_index";
    std.fs.cwd().deleteTree(test_idx) catch {};
    try std.fs.cwd().makeDir(test_idx);
    var index_dir = try std.fs.cwd().openDir(test_idx, .{});
    defer index_dir.close();

    // 1. Create Global Codebook (NGT Index)
    try index_dir.makeDir("global");
    var global_dir = try index_dir.openDir("global", .{});

    // global/grp
    {
        const grp_file = try global_dir.createFile("grp", .{});
        defer grp_file.close();
        var writer = grp_file.writer();
        try writer.writeInt(u64, 2, .little); // size
        try writer.writeByte('-'); // 0
        try writer.writeByte('+'); // 1
        try writer.writeInt(u32, 0, .little); // 0 neighbors
    }

    // global/obj
    {
        const obj_file = try global_dir.createFile("obj", .{});
        defer obj_file.close();
        var writer = obj_file.writer();
        try writer.writeInt(u64, 2, .little); // size
        try writer.writeByte('-'); // 0
        try writer.writeByte('+'); // 1
        try writer.writeInt(u32, 2, .little); // dim
        try writer.writeInt(u32, @bitCast(@as(f32, 1.0)), .little);
        try writer.writeInt(u32, @bitCast(@as(f32, 0.0)), .little);
    }
    global_dir.close();

    // 2. Create Local Codebooks
    try index_dir.makeDir("local0");
    var local0_dir = try index_dir.openDir("local0", .{});
    {
        const obj_file = try local0_dir.createFile("obj", .{});
        defer obj_file.close();
        var writer = obj_file.writer();
        try writer.writeInt(u64, 2, .little); // size
        try writer.writeByte('-'); // 0
        try writer.writeByte('+'); // 1
        try writer.writeInt(u32, 1, .little); // dim
        try writer.writeInt(u32, @bitCast(@as(f32, 0.5)), .little);
    }
    local0_dir.close();

    try index_dir.makeDir("local1");
    var local1_dir = try index_dir.openDir("local1", .{});
    {
        const obj_file = try local1_dir.createFile("obj", .{});
        defer obj_file.close();
        var writer = obj_file.writer();
        try writer.writeInt(u64, 2, .little); // size
        try writer.writeByte('-'); // 0
        try writer.writeByte('+'); // 1
        try writer.writeInt(u32, 1, .little); // dim
        try writer.writeInt(u32, @bitCast(@as(f32, 0.5)), .little);
    }
    local1_dir.close();

    // 3. Create QBG Repo (qg/grp)
    try index_dir.makeDir("qg");
    var qg_dir = try index_dir.openDir("qg", .{});
    {
        const grp_file = try qg_dir.createFile("grp", .{});
        defer grp_file.close();
        var writer = grp_file.writer();

        try writer.writeInt(u64, 2, .little); // num_subspaces
        try writer.writeInt(u64, 2, .little); // size (blobs)

        // Blob 0: Subspace 0, ID 1, Objects [packed]
        try writer.writeInt(u32, 0, .little); // subspace_id
        try writer.writeInt(u32, 1, .little); // ids size
        try writer.writeInt(u32, 1, .little); // ID 1

        var objects = try allocator.alloc(u8, 16);
        defer allocator.free(objects);
        @memset(objects, 0);
        objects[0] = 0x01; // sub0=1 (pos 0)
        objects[8] = 0x01; // sub1=1 (pos 16 / 2 = 8)

        try writer.writeAll(objects);

        // Blob 1: ID 2
        try writer.writeInt(u32, 0, .little); // subspace_id
        try writer.writeInt(u32, 1, .little); // ids size
        try writer.writeInt(u32, 2, .little); // ID 2
        try writer.writeAll(objects); // Same codes
    }
    qg_dir.close();

    // Load
    var cwd_buf: [1024]u8 = undefined;
    const cwd_path = try std.fs.cwd().realpath(".", &cwd_buf);
    const abs_idx_path = try std.fs.path.join(allocator, &[_][]const u8{ cwd_path, test_idx });
    defer allocator.free(abs_idx_path);

    var index = try index_mod.Index.load(allocator, abs_idx_path);
    defer index.deinit();

    defer std.fs.cwd().deleteTree(test_idx) catch {};

    const query = try allocator.dupe(f32, &[_]f32{ 1.0, 1.0 });
    defer allocator.free(query);

    var ctx = context.SearchContext.init(allocator);
    defer ctx.deinit();

    // 1. Test L2 (Default)
    index.metric = .L2;
    const results_l2 = try index.search(&ctx, query, 10, 0.1, 0.1);
    defer allocator.free(results_l2);

    try std.testing.expect(results_l2.len >= 1);
    // Sub0: (1.0 - 0.5)^2 = 0.25
    // Sub1: (1.0 - 0.5)^2 = 0.25
    // Total L2 sq = 0.5. L2 = sqrt(0.5) ~ 0.7071
    if (results_l2.len > 0) {
        try std.testing.expectEqual(@as(u32, 1), results_l2[0].id);
        try std.testing.expectApproxEqAbs(@as(f32, 0.7071), results_l2[0].distance, 0.01);
    }

    // 2. Test L1
    // Update metric
    index.metric = .L1;
    const results_l1 = try index.search(&ctx, query, 10, 0.1, 0.1);
    defer allocator.free(results_l1);

    try std.testing.expect(results_l1.len >= 1);
    // Sub0: |1.0 - 0.5| = 0.5
    // Sub1: |1.0 - 0.5| = 0.5
    // Total L1 = 1.0
    if (results_l1.len > 0) {
        try std.testing.expectEqual(@as(u32, 1), results_l1[0].id);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), results_l1[0].distance, 0.01);
    }
}
