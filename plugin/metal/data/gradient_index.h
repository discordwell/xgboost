/*!
 * Copyright 2024 by Contributors
 * \file gradient_index.h
 * \brief Quantized feature histogram index matrix for Metal GPU plugin.
 */
#ifndef PLUGIN_METAL_DATA_GRADIENT_INDEX_H_
#define PLUGIN_METAL_DATA_GRADIENT_INDEX_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../src/common/hist_util.h"
#include "../data.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "xgboost/data.h"
#pragma GCC diagnostic pop
#include "xgboost/context.h"

namespace xgboost {
namespace metal {

/*!
 * \brief Preprocessed global index matrix stored in a Metal shared-memory buffer.
 *
 * Floating-point feature values are quantized into histogram bin indices using
 * HistogramCuts.  The quantized indices are stored row-major in a MetalVector
 * so they are accessible from both CPU and Metal GPU kernels.
 */
struct GHistIndexMatrix {
  /*! \brief Quantized bin indices, row-major with stride = row_stride. */
  MetalVector<uint32_t> index;

  /*! \brief Hit count per bin (used for feature importance / pruning). */
  std::vector<size_t> hit_count;

  /*! \brief The histogram cut points used for quantization. */
  xgboost::common::HistogramCuts cut;

  /*! \brief Maximum number of bins requested by the user. */
  size_t max_num_bins{0};

  /*! \brief Total number of bins across all features. */
  size_t nbins{0};

  /*! \brief Number of features in the dataset. */
  size_t nfeatures{0};

  /*! \brief Number of entries per row (= nfeatures when dense). */
  size_t row_stride{0};

  GHistIndexMatrix() : cut(0) {}

  /*!
   * \brief Build the quantized index matrix from a DMatrix.
   * \param ctx    XGBoost context (device, nthread, etc.).
   * \param dmat   The input data matrix.
   * \param max_bins  Maximum number of histogram bins.
   */
  void Init(Context const* ctx, DMatrix* dmat, int max_bins);

  /*! \return true if the underlying DMatrix is dense. */
  bool IsDense() const { return is_dense_; }

  /*! \brief Aggregate hit counts into per-feature counts. */
  void GetFeatureCounts(size_t* counts) const {
    auto nfeature = cut.cut_ptrs_.Size() - 1;
    for (size_t fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut.cut_ptrs_.ConstHostVector()[fid];
      auto iend   = cut.cut_ptrs_.ConstHostVector()[fid + 1];
      for (auto i = ibegin; i < iend; ++i) {
        counts[fid] += hit_count[i];
      }
    }
  }

 private:
  bool is_dense_{false};
};

}  // namespace metal
}  // namespace xgboost

#endif  // PLUGIN_METAL_DATA_GRADIENT_INDEX_H_
