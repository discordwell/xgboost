/*!
 * Copyright 2024-2026 by Contributors
 * \file hist_util.h
 * \brief Histogram management utilities for Metal tree updater.
 *
 * Provides a HistCollection that maps node ids to histogram buffers
 * stored in Metal shared memory, plus helper functions for zeroing,
 * copying, and computing the subtraction trick.
 */
#ifndef PLUGIN_METAL_COMMON_HIST_UTIL_H_
#define PLUGIN_METAL_COMMON_HIST_UTIL_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <xgboost/base.h>

#include "../data.h"
#include "row_set.h"

namespace xgboost {
namespace metal {
namespace common {

/*!
 * \brief A single histogram row -- a MetalVector of GradientPairInternal<float>.
 *
 * Each element corresponds to one quantized bin and stores (grad_sum, hess_sum).
 */
using GradientPairT = xgboost::detail::GradientPairInternal<float>;
using GHistRow = MetalVector<GradientPairT>;

/*!
 * \brief Fill a histogram with zeros.
 */
void InitHist(GHistRow* hist, size_t size);

/*!
 * \brief Copy histogram: dst = src (element-wise).
 */
void CopyHist(GHistRow* dst, const GHistRow& src, size_t size);

/*!
 * \brief Subtraction trick: dst = src1 - src2 (element-wise).
 */
void SubtractionHist(GHistRow* dst,
                     const GHistRow& src1,
                     const GHistRow& src2,
                     size_t size);

/*!
 * \brief Collection of gradient histograms, one per tree node.
 *
 * Histograms are stored in Metal shared-memory buffers so they can
 * be written by GPU compute kernels and read back on the CPU.
 */
class HistCollection {
 public:
  GHistRow& operator[](bst_uint nid) {
    return *(data_.at(nid));
  }

  const GHistRow& operator[](bst_uint nid) const {
    return *(data_.at(nid));
  }

  void Init(uint32_t nbins) {
    if (nbins_ != nbins) {
      nbins_ = nbins;
      data_.clear();
    }
  }

  /*!
   * \brief Create (or resize) a zeroed histogram for the given node.
   */
  void AddHistRow(bst_uint nid) {
    if (data_.count(nid) == 0) {
      auto row = std::make_shared<GHistRow>(nbins_, GradientPairT(0.0f, 0.0f));
      data_[nid] = row;
    } else {
      data_[nid]->Resize(nbins_);
      InitHist(data_[nid].get(), nbins_);
    }
  }

 private:
  uint32_t nbins_{0};
  std::unordered_map<uint32_t, std::shared_ptr<GHistRow>> data_;
};

}  // namespace common
}  // namespace metal
}  // namespace xgboost

#endif  // PLUGIN_METAL_COMMON_HIST_UTIL_H_
