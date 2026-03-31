/*!
 * Copyright 2024-2026 by Contributors
 * \file hist_util.mm
 * \brief Histogram management implementation for Metal tree updater.
 *
 * InitHist, CopyHist, and SubtractionHist operate on Metal shared-memory
 * buffers.  Because StorageModeShared is coherent on Apple Silicon, simple
 * CPU-side loops are correct and efficient for these small, per-node
 * operations.  For larger workloads the histogram *building* itself is
 * dispatched as a Metal compute kernel (see hist_updater.mm).
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>

#include "hist_util.h"
#include "../device_manager.h"

namespace xgboost {
namespace metal {
namespace common {

// ---------------------------------------------------------------------------
// InitHist -- zero a histogram
// ---------------------------------------------------------------------------

void InitHist(GHistRow* hist, size_t size) {
  CHECK(hist);
  if (size == 0) return;
  GradientPairT* data = hist->Data();
  CHECK(data);
  std::memset(data, 0, size * sizeof(GradientPairT));
}

// ---------------------------------------------------------------------------
// CopyHist -- element-wise copy
// ---------------------------------------------------------------------------

void CopyHist(GHistRow* dst, const GHistRow& src, size_t size) {
  CHECK(dst);
  CHECK_GE(dst->Size(), size);
  CHECK_GE(src.Size(), size);
  std::memcpy(dst->Data(), src.DataConst(), size * sizeof(GradientPairT));
}

// ---------------------------------------------------------------------------
// SubtractionHist -- dst = src1 - src2
// ---------------------------------------------------------------------------

void SubtractionHist(GHistRow* dst,
                     const GHistRow& src1,
                     const GHistRow& src2,
                     size_t size) {
  CHECK(dst);
  CHECK_GE(dst->Size(), size);
  CHECK_GE(src1.Size(), size);
  CHECK_GE(src2.Size(), size);

  GradientPairT* pdst       = dst->Data();
  const GradientPairT* psrc1 = src1.DataConst();
  const GradientPairT* psrc2 = src2.DataConst();

  // Element-wise subtraction of (grad, hess) pairs.
  // Operates on the raw float components for simplicity.
  const float* s1 = reinterpret_cast<const float*>(psrc1);
  const float* s2 = reinterpret_cast<const float*>(psrc2);
  float* d        = reinterpret_cast<float*>(pdst);
  const size_t n  = size * 2;  // two floats per GradientPairT
  for (size_t i = 0; i < n; ++i) {
    d[i] = s1[i] - s2[i];
  }
}

}  // namespace common
}  // namespace metal
}  // namespace xgboost
