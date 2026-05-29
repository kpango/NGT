// tests/ngtaq/hdf5_io.h
// ANN-Benchmarks HDF5 dataset reader.
// Format: each file has datasets:
//   "train"     float32 [N, D]    — training vectors
//   "test"      float32 [nq, D]   — query vectors
//   "neighbors" int32   [nq, k]   — ground-truth nearest neighbor IDs (0-based)
//   "distances" float32 [nq, k]   — ground-truth distances
#pragma once
#include <hdf5.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

struct H5FloatDataset {
    std::vector<float>   data;
    int n_rows = 0;
    int n_cols = 0;

    const float* row(int i) const { return data.data() + (size_t)i * n_cols; }
};

struct H5IntDataset {
    std::vector<int32_t> data;
    int n_rows = 0;
    int n_cols = 0;

    const int32_t* row(int i) const { return data.data() + (size_t)i * n_cols; }
};

inline H5FloatDataset h5_read_float(const std::string& path, const std::string& dset_name) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("h5_read_float: cannot open " + path);
    hid_t dset = H5Dopen2(file, dset_name.c_str(), H5P_DEFAULT);
    if (dset < 0) {
        H5Fclose(file);
        throw std::runtime_error("h5_read_float: dataset not found: " + dset_name + " in " + path);
    }
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2] = {0, 0};
    int ndims = H5Sget_simple_extent_dims(space, dims, nullptr);
    H5FloatDataset r;
    r.n_rows = (int)dims[0];
    r.n_cols = (ndims > 1) ? (int)dims[1] : 1;
    r.data.resize((size_t)r.n_rows * r.n_cols);
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, r.data.data());
    H5Sclose(space); H5Dclose(dset); H5Fclose(file);
    return r;
}

inline H5IntDataset h5_read_int(const std::string& path, const std::string& dset_name) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("h5_read_int: cannot open " + path);
    hid_t dset = H5Dopen2(file, dset_name.c_str(), H5P_DEFAULT);
    if (dset < 0) {
        H5Fclose(file);
        throw std::runtime_error("h5_read_int: dataset not found: " + dset_name + " in " + path);
    }
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    H5IntDataset r;
    r.n_rows = (int)dims[0];
    r.n_cols = (int)dims[1];
    r.data.resize((size_t)r.n_rows * r.n_cols);
    H5Dread(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, r.data.data());
    H5Sclose(space); H5Dclose(dset); H5Fclose(file);
    return r;
}

/// Compute recall@k.
/// results[qi] = list of predicted IDs (0-based), length >= k.
/// gt has shape [nq][gt_k], storing ground-truth IDs.
inline double compute_recall_k(const H5IntDataset& gt,
                                 const std::vector<std::vector<int>>& results,
                                 int k)
{
    const int nq   = gt.n_rows;
    const int gt_k = gt.n_cols;
    double total = 0.0;
    for (int qi = 0; qi < nq; ++qi) {
        int found = 0;
        const int res_k = (int)results[qi].size();
        for (int i = 0; i < k && i < res_k; ++i) {
            int pred = results[qi][i];
            for (int j = 0; j < std::min(k, gt_k); ++j) {
                if (gt.data[(size_t)qi * gt_k + j] == pred) { ++found; break; }
            }
        }
        total += (double)found / k;
    }
    return total / nq;
}
