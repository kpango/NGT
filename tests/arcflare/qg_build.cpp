// tests/arcflare/qg_build.cpp
// Build ANNG + quantize to QG (D_sub=4) and QSG (D_sub=1, D_sub=2).
// Usage: qg_build <hdf5_path> <out_base> [metric=l2|angular] [edge_size=100] [threads=8]
// Produces:
//   <out_base>_anng  — raw ANNG (reusable base)
//   <out_base>_qg4   — NGTQG quantized, D_sub=4
//   <out_base>_qsg1  — NGTQG quantized, D_sub=1 (scalar)
//   <out_base>_qsg2  — NGTQG quantized, D_sub=2
#include "bench_common.hpp"
#include "NGT/Index.h"
#include "NGT/NGTQ/QuantizedGraph.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
namespace fs = std::filesystem;

static double elapsed_s(double t0) { return (bc_now_us() - t0) / 1e6; }

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <hdf5_path> <out_base> [metric=l2|angular] [edge_size=100] [threads=8]\n",
            argv[0]);
        return 1;
    }
    const char* hdf5_path  = argv[1];
    const char* out_base   = argv[2];
    const char* metric_str = (argc > 3) ? argv[3] : "l2";
    int edge_size          = (argc > 4) ? std::stoi(argv[4]) : 100;
    int n_threads          = (argc > 5) ? std::stoi(argv[5]) : 8;
    const bool is_angular  = (strcmp(metric_str, "angular") == 0 ||
                               strcmp(metric_str, "cosine")  == 0);

    const std::string anng_path = std::string(out_base) + "_anng";

    // ── Step 1: Read train vectors ─────────────────────────────────────────
    fprintf(stderr, "[Load] HDF5 train from: %s\n", hdf5_path);
    H5FloatDataset train = h5_read_float(hdf5_path, "train");
    const int N = train.n_rows, D = train.n_cols;
    fprintf(stderr, "  N=%d  D=%d  metric=%s  edge_size=%d\n",
            N, D, metric_str, edge_size);

    // ── Step 2: Normalize if angular ──────────────────────────────────────
    if (is_angular) {
        for (int i = 0; i < N; ++i)
            l2_normalize(train.data.data() + (size_t)i * D, D);
        fprintf(stderr, "[Prep] L2-normalized %d vectors\n", N);
    }

    // ── Step 3: Build ANNG (skip if already exists) ───────────────────────
    if (fs::exists(anng_path)) {
        fprintf(stderr, "[Skip] ANNG already exists: %s\n", anng_path.c_str());
    } else {
        const double t0 = bc_now_us();
        NGT::Property prop;
        prop.dimension            = D;
        prop.objectType           = NGT::ObjectSpace::ObjectType::Float;
        prop.distanceType         = NGT::Index::Property::DistanceType::DistanceTypeL2;
        prop.edgeSizeForCreation  = edge_size;

        fprintf(stderr, "[NGT] Creating ANNG (D=%d, edge_size=%d) ...\n", D, edge_size);
        NGT::Index::create(anng_path, prop);
        {
            NGT::Index ngt(anng_path);
            for (int i = 0; i < N; ++i) {
                std::vector<float> v(train.data.data() + (size_t)i * D,
                                     train.data.data() + (size_t)(i+1) * D);
                ngt.insert(v);
                if ((i+1) % 100000 == 0)
                    fprintf(stderr, "  inserted %d/%d (%.1fs)\n", i+1, N, elapsed_s(t0));
            }
            fprintf(stderr, "[NGT] Building graph (threads=%d)...\n", n_threads);
            ngt.createIndex(n_threads);
            fprintf(stderr, "[NGT] ANNG done (%.1fs)\n", elapsed_s(t0));
            ngt.save();
        }
        fprintf(stderr, "[Done] Saved ANNG to: %s\n", anng_path.c_str());
    }

    // ── Step 4: Quantize to QG and QSG variants ───────────────────────────
    struct Variant { int d_sub; const char* suffix; };
    static const Variant VARIANTS[] = {
        {4, "_qg4"}, {1, "_qsg1"}, {2, "_qsg2"}
    };
    for (auto& v : VARIANTS) {
        std::string dst = std::string(out_base) + v.suffix;
        if (fs::exists(dst)) {
            fprintf(stderr, "[Skip] %s already exists\n", dst.c_str());
            continue;
        }
        fprintf(stderr, "[Copy] %s -> %s\n", anng_path.c_str(), dst.c_str());
        fs::copy(anng_path, dst, fs::copy_options::recursive);

        fprintf(stderr, "[QG] Quantizing D_sub=%d -> %s\n", v.d_sub, dst.c_str());
        const double tq = bc_now_us();
        NGTQG::Index::quantize(dst, (size_t)v.d_sub, (size_t)edge_size, /*verbose=*/true);
        fprintf(stderr, "[QG] Done (%.1fs) -> %s\n", elapsed_s(tq), dst.c_str());
    }

    fprintf(stderr, "[All done]\n");
    return 0;
}
