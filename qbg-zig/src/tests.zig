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

        // Objects: 1 object, 2 subvectors.
        // align_blk = 32. n=1. s=32.
        // bytes = 32 * 2 / 2 = 32? Wait.
        // batch=2, blk=16. align_blk = 16*2? No.
        // align_blk = NGTQ_SIMD_BLOCK_SIZE * NGTQ_BATCH_SIZE is used in NGT for something else?
        // Let's re-verify logic.
        // Quantizer.h: getNumOfAlignedObjects = ((n-1)/16+1)*16.
        // streamSize = n_aligned * m_aligned.
        // byteSize = streamSize / 2.
        // Here n=1 -> n_aligned=16. m=2 -> m_aligned=2.
        // streamSize = 16 * 2 = 32.
        // byteSize = 16 bytes.
        // So I should write 16 bytes.

        var objects = try allocator.alloc(u8, 16);
        defer allocator.free(objects);
        @memset(objects, 0);

        // Interleaved layout:
        // Block 0 (objects 0..15).
        // Sub 0: 16 nibbles -> 8 bytes.
        // Sub 1: 16 nibbles -> 8 bytes.
        // Total 16 bytes.

        // Obj 0 is at index 0.
        // Sub 0: byte 0, low nibble (bits 0-3).
        // Sub 1: byte 8, low nibble.

        // Set code 1 for sub 0 (pos 0) -> 0x01
        objects[0] = 0x01;
        // Set code 1 for sub 1 (pos 16) -> 0x01 at byte 8
        objects[8] = 0x01;

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
            try std.testing.expectApproxEqAbs(@as(f32, 0.7071), res.distance, 0.01);
            break;
        }
    }
    try std.testing.expect(found);
}
