const std = @import("std");
const builtin = @import("builtin");

pub const Metric = enum {
    L1,
    L2,
    Cosine,
    Hamming,
};

pub fn l2_distance_sq(a: []const f32, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    var i: usize = 0;
    var sum: f32 = 0;

    // SIMD loop
    const vec_len = 16; // 512-bit if available, or 8 for 256-bit. Zig handles generic vector length well.
    // Try 16 (64 bytes)
    if (len >= vec_len) {
        var v_sum = @Vector(vec_len, f32){ 0 } ** vec_len;
        while (i + vec_len <= len) : (i += vec_len) {
            const v_a: @Vector(vec_len, f32) = a[i..][0..vec_len].*;
            const v_b: @Vector(vec_len, f32) = b[i..][0..vec_len].*;
            const diff = v_a - v_b;
            v_sum += diff * diff;
        }
        sum += @reduce(.Add, v_sum);
    }

    // Scalar tail
    while (i < len) : (i += 1) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

pub fn l2_distance(a: []const f32, b: []const f32) f32 {
    return std.math.sqrt(l2_distance_sq(a, b));
}

pub fn l1_distance(a: []const f32, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    var i: usize = 0;
    var sum: f32 = 0;

    const vec_len = 16;
    if (len >= vec_len) {
        var v_sum = @Vector(vec_len, f32){ 0 } ** vec_len;
        while (i + vec_len <= len) : (i += vec_len) {
            const v_a: @Vector(vec_len, f32) = a[i..][0..vec_len].*;
            const v_b: @Vector(vec_len, f32) = b[i..][0..vec_len].*;
            const diff = @abs(v_a - v_b);
            v_sum += diff;
        }
        sum += @reduce(.Add, v_sum);
    }

    while (i < len) : (i += 1) {
        sum += @abs(a[i] - b[i]);
    }

    return sum;
}

pub fn cosine_distance(a: []const f32, b: []const f32) f32 {
    // 1 - (dot(a, b) / (norm(a) * norm(b)))
    // Assuming inputs are normalized? NGT typically normalizes for Cosine.
    // If normalized, distance = 1 - dot.
    // Let's implement full cosine just in case.

    const len = @min(a.len, b.len);
    var i: usize = 0;
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    const vec_len = 16;
    if (len >= vec_len) {
        var v_dot = @Vector(vec_len, f32){ 0 } ** vec_len;
        var v_norm_a = @Vector(vec_len, f32){ 0 } ** vec_len;
        var v_norm_b = @Vector(vec_len, f32){ 0 } ** vec_len;

        while (i + vec_len <= len) : (i += vec_len) {
            const v_a: @Vector(vec_len, f32) = a[i..][0..vec_len].*;
            const v_b: @Vector(vec_len, f32) = b[i..][0..vec_len].*;
            v_dot += v_a * v_b;
            v_norm_a += v_a * v_a;
            v_norm_b += v_b * v_b;
        }
        dot += @reduce(.Add, v_dot);
        norm_a += @reduce(.Add, v_norm_a);
        norm_b += @reduce(.Add, v_norm_b);
    }

    while (i < len) : (i += 1) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0 or norm_b == 0) return 1.0;

    return 1.0 - (dot / (std.math.sqrt(norm_a) * std.math.sqrt(norm_b)));
}

pub fn compute(metric: Metric, a: []const f32, b: []const f32) f32 {
    return switch (metric) {
        .L1 => l1_distance(a, b),
        .L2 => l2_distance(a, b),
        .Cosine => cosine_distance(a, b),
        .Hamming => @panic("Hamming not supported for float vectors"),
    };
}

// For LUT creation (subvector distance)
// L2: returns squared distance (to be additive)
// L1: returns L1 distance
// Cosine: usually returns dot product (if normalized) or some additive component?
// In PQ, Cosine is often approximated by decomposing dot product.
// dist(q, c) = 1 - dot(q, c). dot(q, c) = sum(dot(sub_q, sub_c)).
// So we want dot product for subvectors.
// NGT creates lookup table.
// If Metric is Cosine, we might want to return `dot(sub_q, sub_c)`?
// But `index.zig` accumulates and returns a value.
// If we return dot product, then `acc` will be dot product.
// Then `search` needs to convert acc to distance?
// Or we store `distance` in LUT?
// NGTQ uses `createFloatL2DistanceLookup` or `createFloatDotProductLookup`.
// For Cosine, NGTQ usually minimizes `L2(normalized_q, normalized_c)` which corresponds to maximizing dot product.
// L2^2 = |q|^2 + |c|^2 - 2qc = 2 - 2qc.
// So minimizing L2 is same as maximizing dot product.
// So usually L2 is fine if vectors are normalized.
// Let's stick to L2 for now or generic `compute`.
// But for LUT, we often want squared L2 to sum up correctly.
// `l2_distance_sq` is better for LUT.

pub fn compute_sub(metric: Metric, a: []const f32, b: []const f32) f32 {
    return switch (metric) {
        .L1 => l1_distance(a, b),
        .L2 => l2_distance_sq(a, b), // Squared for additive property in PQ
        .Cosine => cosine_distance(a, b), // Not strictly additive in this form?
        // For Cosine PQ, we usually accumulate dot products.
        // But implementing full Cosine here for now.
        // NGT actually has separate lookup creation for DotProduct.
        // Let's assume L2 for now for "standard" usage or add DotProduct support later.
        // We will return l2_sq for L2.
        .Hamming => 0,
    };
}
