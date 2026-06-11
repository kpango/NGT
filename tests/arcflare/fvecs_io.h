// tests/arcflare/fvecs_io.h
// Header-only .fvecs and .ivecs binary format readers for ANN benchmark datasets.
//
// .fvecs: [int32 dim][float32 × dim] repeated per vector
// .ivecs: [int32 k][int32 × k] repeated per entry
//
// Ground truth IDs in SIFT-1M ivecs are 0-indexed (= ArcFlare node IDs directly).
#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ArcFlare {

inline std::vector<std::vector<float>> loadFvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("loadFvecs: cannot open " + path);
    std::vector<std::vector<float>> vecs;
    int32_t dim = 0;
    while (f.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        if (dim <= 0 || dim > 65536)
            throw std::runtime_error("loadFvecs: implausible dim=" + std::to_string(dim));
        std::vector<float> v(static_cast<size_t>(dim));
        if (!f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(static_cast<size_t>(dim) * sizeof(float))))
            throw std::runtime_error("loadFvecs: truncated data at vector " + std::to_string(vecs.size()));
        vecs.push_back(std::move(v));
    }
    return vecs;
}

inline std::vector<std::vector<int32_t>> loadIvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("loadIvecs: cannot open " + path);
    std::vector<std::vector<int32_t>> vecs;
    int32_t k = 0;
    while (f.read(reinterpret_cast<char*>(&k), sizeof(k))) {
        if (k <= 0 || k > 65536)
            throw std::runtime_error("loadIvecs: implausible k=" + std::to_string(k));
        std::vector<int32_t> v(static_cast<size_t>(k));
        if (!f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(static_cast<size_t>(k) * sizeof(int32_t))))
            throw std::runtime_error("loadIvecs: truncated data at vector " + std::to_string(vecs.size()));
        vecs.push_back(std::move(v));
    }
    return vecs;
}

} // namespace ArcFlare
