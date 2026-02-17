const std = @import("std");

pub const Serializer = struct {
    allocator: std.mem.Allocator,
    reader: std.io.AnyReader,

    pub fn init(allocator: std.mem.Allocator, reader: std.io.AnyReader) Serializer {
        return .{
            .allocator = allocator,
            .reader = reader,
        };
    }

    pub fn readInt(self: *Serializer, comptime T: type) !T {
        return self.reader.readInt(T, .little);
    }

    pub fn readFloat(self: *Serializer, comptime T: type) !T {
        const IntType = std.meta.Int(.unsigned, @bitSizeOf(T));
        const int_val = try self.reader.readInt(IntType, .little);
        return @bitCast(int_val);
    }

    pub fn readByte(self: *Serializer) !u8 {
        return self.reader.readByte();
    }

    pub fn readBytes(self: *Serializer, buffer: []u8) !void {
        try self.reader.readNoEof(buffer);
    }

    pub fn read(self: *Serializer, comptime T: type) !T {
        switch (@typeInfo(T)) {
            .Int => return self.readInt(T),
            .Float => return self.readFloat(T),
            .Struct => {
                if (@hasDecl(T, "read")) {
                    return T.read(self);
                } else {
                    @compileError("Struct " ++ @typeName(T) ++ " must implement 'read(*Serializer)'");
                }
            },
            else => @compileError("Unsupported type for read: " ++ @typeName(T)),
        }
    }

    pub fn readVector(self: *Serializer, comptime T: type) ![]T {
        const size = try self.readInt(u32);
        const vec = try self.allocator.alloc(T, size);
        errdefer self.allocator.free(vec);

        for (0..size) |i| {
            vec[i] = try self.read(T);
        }
        return vec;
    }
};

test "serializer readAny" {
    const TestStruct = struct {
        val: u32,
        pub fn read(ser: *Serializer) !@This() {
            return .{ .val = try ser.read(u32) };
        }
    };

    var buffer = std.ArrayList(u8).init(std.testing.allocator);
    defer buffer.deinit();

    var writer = buffer.writer();
    // TestStruct vector: size 1, val 123
    try writer.writeInt(u32, 1, .little);
    try writer.writeInt(u32, 123, .little);

    var fbs = std.io.fixedBufferStream(buffer.items);
    var ser = Serializer.init(std.testing.allocator, fbs.reader().any());

    const v = try ser.readVector(TestStruct);
    defer std.testing.allocator.free(v);

    try std.testing.expectEqual(@as(usize, 1), v.len);
    try std.testing.expectEqual(@as(u32, 123), v[0].val);
}
