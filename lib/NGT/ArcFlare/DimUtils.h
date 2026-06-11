// lib/NGT/ArcFlare/DimUtils.h
#pragma once
#include <stdexcept>

namespace NGT { namespace ArcFlare {

/// Returns the smallest integer >= D that is (a) a power of 2 AND (b) divisible by 64.
/// D=100 → 128,  D=200 → 256,  D=256 → 256,  D=960 → 1024,  D=784 → 1024
inline int pad_dim_for_v2(int D) {
    if (D <= 0) throw std::invalid_argument("pad_dim_for_v2: D must be > 0");
    int p = 64;
    while (p < D) p <<= 1;
    return p;  // all powers of 2 >= 64 are divisible by 64
}

}} // NGT::ArcFlare
