const std = @import("std");
const index_mod = @import("index.zig");
const ngt_mod = @import("ngt.zig");
const qbg_mod = @import("qbg.zig");
const quantizer_mod = @import("quantizer.zig");
const serializer_mod = @import("serializer.zig");

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

        var objects = try allocator.alloc(u8, 32);
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
    // Use full path to avoid relative path issues if test runs in different CWD
    var cwd_buf: [1024]u8 = undefined;
    const cwd_path = try std.fs.cwd().realpath(".", &cwd_buf);
    const abs_idx_path = try std.fs.path.join(allocator, &[_][]const u8{ cwd_path, test_idx });
    defer allocator.free(abs_idx_path);

    var index = try index_mod.Index.load(allocator, abs_idx_path);
    defer index.deinit();

    // Clean up
    // We defer deleteTree at top level scope (after creating dir) usually,
    // but here we want to ensure it stays during test.
    // We can just rely on defer deleteTree(test_idx) at end of function.
    // Wait, deleteTree requires the dir to be closed? `index` holds open files?
    // `Index.load` reads files and closes them immediately.
    // So deleteTree should work.

    defer std.fs.cwd().deleteTree(test_idx) catch {};

    // Search
    const query = try allocator.dupe(f32, &[_]f32{ 1.0, 1.0 });
    defer allocator.free(query);

    const results = try index.search(query, 10, 0.1, 0.1);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);

    var found = false;
    for (results) |res| {
        if (res.id == 1) {
            found = true;
            // Dist calculation:
            // Sub0: (1.0 - 0.5)^2 = 0.25
            // Sub1: (1.0 - 0.5)^2 = 0.25
            // Total = 0.5
            try std.testing.expectApproxEqAbs(@as(f32, 0.5), res.distance, 0.001);
            break;
        }
    }
    try std.testing.expect(found);
}
