// lib/NGT/NGTAQ/BinaryQuantizer.h
// BinaryQuantizer: PCA rotation + 2-bit sign/magnitude encoding + tau calibration.
//
// Pipeline: L2-normalize input → multiply by rotation matrix → threshold at 0 (sign)
//           and tau_raw = tau_normalized * sigma (magnitude).
//
// Plane A (sign):      bit i = 1 iff rotated[i] < 0
// Plane B (magnitude): bit i = 1 iff |rotated[i]| > tau_raw
#pragma once

#include "NGT/Common.h"
#include "NGT/ObjectSpace.h"
#include "NGT/NGTAQ/BQDistance.h"
#include "NGT/NGTQ/Quantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

namespace NGTAQ {

class BinaryQuantizer {
 public:
  BinaryQuantizer() : dim_(0), words_(0), tau_(0.0f), sigma_(1.0f), rotation_dim_(0) {}

  // Initialize for the given dimension. Must call before any other method.
  // dim must be a positive multiple of 64.
  void init(int dim) {
    assert(dim > 0 && dim % 64 == 0 && "Dimension must be a positive multiple of 64");
    dim_          = dim;
    words_        = dim / 64;
    tau_          = 0.0f;
    sigma_        = 1.0f;
    rotation_dim_ = 0;
    rotation_.clear();
  }

  // Use identity rotation (no rotation). Useful for testing.
  // WARNING: do NOT use with non-negative input vectors (e.g. SIFT histograms).
  // All sign bits will be 0, making BQ distance always 0.
  void setIdentityRotation() {
    rotation_dim_ = dim_;
    rotation_.assign(static_cast<size_t>(dim_) * dim_, 0.0f);
    for (int i = 0; i < dim_; i++) rotation_[i * dim_ + i] = 1.0f;
  }

  // Random sign flip (diagonal rotation with ±1 entries, seed-reproducible).
  // Each dimension is independently negated with probability 0.5.
  // NOTE: diagonal sign flips do NOT fix the non-negative input problem —
  // the sign XOR between two vectors is unaffected by a shared sign flip.
  // Use setRandomRotation() for inputs that are all non-negative (e.g. SIFT).
  void setRandomSignFlip(uint32_t seed = 42) {
    rotation_dim_ = dim_;
    rotation_.assign(static_cast<size_t>(dim_) * dim_, 0.0f);
    std::mt19937 rng(seed);
    std::bernoulli_distribution flip(0.5);
    for (int i = 0; i < dim_; i++)
      rotation_[i * static_cast<size_t>(dim_) + i] = flip(rng) ? 1.0f : -1.0f;
  }

  // Random orthogonal rotation via Gram-Schmidt on a random Gaussian matrix.
  // Mixes all dimensions, making sign bits informative even for non-negative
  // inputs (e.g. SIFT histograms after L2 normalization).
  // O(D³) to generate (once at index build time), O(D²) to apply per encode.
  void setRandomRotation(uint32_t seed = 42) {
    rotation_dim_ = dim_;
    const int D = dim_;
    // Fill with iid N(0,1) entries
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> M(static_cast<size_t>(D) * D);
    for (auto& v : M) v = nd(rng);
    // Gram-Schmidt orthogonalization (row-wise)
    rotation_.resize(static_cast<size_t>(D) * D);
    for (int i = 0; i < D; ++i) {
      // Copy row i from M
      float* ri = rotation_.data() + i * D;
      std::copy(M.begin() + i * D, M.begin() + (i + 1) * D, ri);
      // Subtract projections onto already-orthogonalized rows
      for (int j = 0; j < i; ++j) {
        const float* rj = rotation_.data() + j * D;
        float dot = 0.0f;
        for (int k = 0; k < D; ++k) dot += rj[k] * ri[k];
        for (int k = 0; k < D; ++k) ri[k] -= dot * rj[k];
      }
      // Normalize row i
      float norm2 = 0.0f;
      for (int k = 0; k < D; ++k) norm2 += ri[k] * ri[k];
      if (norm2 > 1e-12f) {
        float inv = 1.0f / std::sqrt(norm2);
        for (int k = 0; k < D; ++k) ri[k] *= inv;
      }
    }
  }

  // Set rotation from a pre-computed NGTQ::Rotation (row-major, dim x dim).
  void setRotation(const NGTQ::Rotation& rot) {
    rotation_.assign(rot.begin(), rot.end());
    rotation_dim_ = rot.dim;
  }

  void setTau(float tau_normalized) { tau_ = tau_normalized; }
  void setSigma(float sigma)        { sigma_ = sigma; }

  int   dim()   const { return dim_; }
  int   words() const { return words_; }
  float tau()   const { return tau_; }
  float sigma() const { return sigma_; }

  // Encode a raw float vector into BQ signature (interleaved layout).
  // Input:  raw float vector of length dim_.
  // Output: out_bq must point to 2*words_ uint64_t.
  //         out_bq[i*2]   = sign-plane word i
  //         out_bq[i*2+1] = magnitude-plane word i
  void encode(const float* raw, uint64_t* out_bq) const {
    thread_local std::vector<float> rotated;
    rotated.resize(dim_);
    applyNormalizeAndRotate(raw, rotated.data());

    const float tau_raw = tau_ * sigma_;
    std::memset(out_bq, 0, static_cast<size_t>(words_) * 2 * sizeof(uint64_t));
    for (int i = 0; i < dim_; i++) {
      float x = rotated[i];
      if (x < 0.0f)
        out_bq[i / 64 * 2]     |= (uint64_t(1) << (i % 64));  // sign word
      if (std::abs(x) > tau_raw)
        out_bq[i / 64 * 2 + 1] |= (uint64_t(1) << (i % 64));  // mag word
    }
  }

  // Calibrate tau empirically from a sample of vector pairs.
  // metric: DistanceTypeInnerProduct (cosine) or DistanceTypeL2.
  // Sets sigma_ from component distribution, then tau_ to 99th-percentile of
  // |δ_BQ - δ_true| errors measured with an initial tau=0.5 probe.
  void calibrateTau(
      const std::vector<const float*>& vecs,
      int n_sample_pairs,
      NGT::ObjectSpace::DistanceType metric) {
    if (static_cast<int>(vecs.size()) < 2 || n_sample_pairs <= 0) return;
    n_sample_pairs = std::min(
        n_sample_pairs,
        static_cast<int>(vecs.size()) * (static_cast<int>(vecs.size()) - 1) / 2);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(vecs.size()) - 1);

    // --- Estimate sigma from sample of rotated components ---
    thread_local std::vector<float> rotated_buf;
    rotated_buf.resize(dim_);

    int n_vecs_sample = std::min(1000, static_cast<int>(vecs.size()));
    std::vector<float> all_abs_components;
    all_abs_components.reserve(static_cast<size_t>(n_vecs_sample) * dim_);
    for (int s = 0; s < n_vecs_sample; s++) {
      int idx = pick(rng);
      applyNormalizeAndRotate(vecs[idx], rotated_buf.data());
      for (int i = 0; i < dim_; i++)
        all_abs_components.push_back(std::abs(rotated_buf[i]));
    }
    float sum_abs = 0.0f;
    for (float v : all_abs_components) sum_abs += v;
    // E[|N(0,σ²)|] = σ·sqrt(2/π)  →  σ = mean(|x|) / sqrt(2/π)
    sigma_ = (sum_abs / static_cast<float>(all_abs_components.size()))
             / std::sqrt(2.0f / static_cast<float>(M_PI));

    // --- Collect |δ_BQ - δ_true| with probe tau=0.5 ---
    tau_ = 0.5f;

    std::vector<float> errors;
    errors.reserve(n_sample_pairs);

    std::vector<uint64_t> bqP(static_cast<size_t>(words_) * 2);
    std::vector<uint64_t> bqQ(static_cast<size_t>(words_) * 2);

    for (int s = 0; s < n_sample_pairs; s++) {
      int i = pick(rng);
      int j = pick(rng);
      if (i == j) j = (j + 1) % static_cast<int>(vecs.size());

      encode(vecs[i], bqP.data());
      encode(vecs[j], bqQ.data());

      float delta_bq   = bqDistance(bqP.data(), bqQ.data(), words_, dim_);
      float delta_true = computeNormalizedDistance(vecs[i], vecs[j], dim_, metric);

      errors.push_back(std::abs(delta_bq - delta_true));
    }

    // tau = 99th percentile of error distribution
    std::sort(errors.begin(), errors.end());
    int p99_idx = static_cast<int>(static_cast<float>(errors.size()) * 0.99f);
    p99_idx = std::min(p99_idx, static_cast<int>(errors.size()) - 1);
    tau_ = errors[p99_idx];
  }

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&dim_),   sizeof(dim_));
    os.write(reinterpret_cast<const char*>(&tau_),   sizeof(tau_));
    os.write(reinterpret_cast<const char*>(&sigma_), sizeof(sigma_));
    uint32_t rot_size = static_cast<uint32_t>(rotation_.size());
    os.write(reinterpret_cast<const char*>(&rot_size), sizeof(rot_size));
    if (rot_size > 0)
      os.write(reinterpret_cast<const char*>(rotation_.data()), rot_size * sizeof(float));
  }

  void deserialize(std::istream& is) {
    is.read(reinterpret_cast<char*>(&dim_),   sizeof(dim_));
    is.read(reinterpret_cast<char*>(&tau_),   sizeof(tau_));
    is.read(reinterpret_cast<char*>(&sigma_), sizeof(sigma_));
    if (!is) throw std::runtime_error("BinaryQuantizer::deserialize: stream error");
    words_ = dim_ / 64;
    uint32_t rot_size = 0;
    is.read(reinterpret_cast<char*>(&rot_size), sizeof(rot_size));
    rotation_.resize(rot_size);
    if (rot_size > 0)
      is.read(reinterpret_cast<char*>(rotation_.data()), rot_size * sizeof(float));
    if (!is) {
      rotation_.clear();
      dim_ = 0; words_ = 0; tau_ = 0.0f; sigma_ = 1.0f; rotation_dim_ = 0;
      throw std::runtime_error("BinaryQuantizer::deserialize: stream error");
    }
    rotation_dim_ = dim_;
  }

 private:
  int   dim_;
  int   words_;
  float tau_;
  float sigma_;
  int   rotation_dim_;
  std::vector<float> rotation_;  // Row-major rotation matrix [dim_ × dim_]

  void applyNormalizeAndRotate(const float* raw, float* out_rotated) const {
    assert(!rotation_.empty() && "rotation not initialized; call setRotation or setIdentityRotation first");
    thread_local std::vector<float> normalized;
    normalized.resize(dim_);
    float norm = 0.0f;
    for (int i = 0; i < dim_; i++) norm += raw[i] * raw[i];
    norm = std::sqrt(norm + 1e-12f);
    for (int i = 0; i < dim_; i++) normalized[i] = raw[i] / norm;

    for (int i = 0; i < dim_; i++) {
      float s = 0.0f;
      const float* row = rotation_.data() + i * dim_;
      for (int j = 0; j < dim_; j++) s += row[j] * normalized[j];
      out_rotated[i] = s;
    }
  }

  static float computeNormalizedDistance(
      const float* p, const float* q, int D,
      NGT::ObjectSpace::DistanceType metric) {
    float np = 0.0f, nq = 0.0f;
    for (int i = 0; i < D; i++) { np += p[i] * p[i]; nq += q[i] * q[i]; }
    np = std::sqrt(np + 1e-12f);
    nq = std::sqrt(nq + 1e-12f);

    if (metric == NGT::ObjectSpace::DistanceTypeInnerProduct) {
      // Cosine distance normalized to [0,1]: (1 - dot(p_unit, q_unit)) / 2
      float dot = 0.0f;
      for (int i = 0; i < D; i++) dot += (p[i] / np) * (q[i] / nq);
      dot = std::max(-1.0f, std::min(1.0f, dot));
      return (1.0f - dot) / 2.0f;
    } else {
      // Normalized L2²: ||p_unit - q_unit||² / 4  (max = 4 for unit vectors → [0,1])
      float dist2 = 0.0f;
      for (int i = 0; i < D; i++) {
        float d = p[i] / np - q[i] / nq;
        dist2 += d * d;
      }
      return dist2 / 4.0f;
    }
  }
};

}  // namespace NGTAQ
