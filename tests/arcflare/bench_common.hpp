// tests/ngtaq/bench_common.hpp
// Shared utilities for all ANN-Benchmarks benchmark tools.
// Includes hdf5_io.h; adds timing, L2-norm, BenchResult.
#pragma once
#include "hdf5_io.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

static inline double bc_now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(
        steady_clock::now().time_since_epoch()).count();
}

struct BenchResult {
    double recall = 0.0;
    double qps    = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
};

/// L2-normalize vector v[0..D) in place. No-op if norm < 1e-12.
static inline void l2_normalize(float* v, int D) {
    float norm2 = 0.f;
    for (int d = 0; d < D; ++d) norm2 += v[d] * v[d];
    if (norm2 > 1e-12f) {
        float inv = 1.f / std::sqrtf(norm2);
        for (int d = 0; d < D; ++d) v[d] *= inv;
    }
}

static const double EPSILON_GRID[] = {
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0
};
static const int EPSILON_GRID_SIZE = 13;
