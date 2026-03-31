/*!
 * Copyright 2024 by Contributors
 * \file gradient_index.mm
 * \brief Implementation of GHistIndexMatrix::Init for Metal GPU plugin.
 *
 * Quantization is performed on the CPU and stored in a Metal shared buffer
 * (StorageModeShared), making the result accessible to both CPU and GPU.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "gradient_index.h"

#include "../../src/common/hist_util.h"
#include "../../src/common/threading_utils.h"

namespace xgboost {
namespace metal {

// ---------------------------------------------------------------------------
// SearchBin — find the histogram bin for a single (feature, value) pair
// ---------------------------------------------------------------------------

static uint32_t SearchBin(const float* cut_values,
                          const uint32_t* cut_ptrs,
                          Entry const& e) {
  auto beg = cut_ptrs[e.index];
  auto end = cut_ptrs[e.index + 1];
  auto it = std::upper_bound(cut_values + beg, cut_values + end, e.fvalue);
  uint32_t idx = static_cast<uint32_t>(it - cut_values);
  if (idx == end) {
    idx -= 1;
  }
  return idx;
}

// ---------------------------------------------------------------------------
// GHistIndexMatrix::Init
// ---------------------------------------------------------------------------

void GHistIndexMatrix::Init(Context const* ctx,
                            DMatrix* dmat,
                            int max_bins) {
  nfeatures = dmat->Info().num_col_;

  // 1. Build histogram cut points on CPU.
  cut = xgboost::common::SketchOnDMatrix(ctx, dmat, max_bins);

  max_num_bins = static_cast<size_t>(max_bins);
  nbins = cut.Ptrs().back();

  // 2. Per-bin hit counts, zero-initialized.
  hit_count.assign(nbins, 0);

  const bool dense = dmat->IsDense();
  is_dense_ = dense;

  // 3. Determine row_stride (max non-zero entries per row).
  row_stride = 0;
  size_t n_rows = 0;
  if (!dense) {
    for (const auto& batch : dmat->GetBatches<SparsePage>()) {
      const auto& row_offset = batch.offset.ConstHostVector();
      n_rows += batch.Size();
      for (size_t i = 1; i < row_offset.size(); ++i) {
        row_stride = std::max(
            row_stride,
            static_cast<size_t>(row_offset[i] - row_offset[i - 1]));
      }
    }
  } else {
    row_stride = nfeatures;
    n_rows = dmat->Info().num_row_;
  }

  // 4. Allocate the quantized index buffer in Metal shared memory.
  const size_t n_index = n_rows * row_stride;
  index.Resize(n_index);

  if (nbins == 0) return;

  CHECK_GT(cut.cut_values_.Size(), 0U);

  const float* cut_values = cut.Values().data();
  const uint32_t* cut_ptrs = cut.Ptrs().data();
  uint32_t* index_data = index.Data();

  // 5. Populate the quantized indices, batch by batch.
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    const xgboost::Entry* data_ptr = batch.data.ConstHostVector().data();
    const bst_idx_t* offset_vec = batch.offset.ConstHostVector().data();
    const size_t batch_size = batch.Size();
    const auto base_rowid = batch.base_rowid;

    for (size_t i = 0; i < batch_size; ++i) {
      const size_t ibegin = offset_vec[i];
      const size_t iend = offset_vec[i + 1];
      const size_t size = iend - ibegin;
      const size_t start = (i + base_rowid) * row_stride;

      for (size_t j = 0; j < size; ++j) {
        const Entry& e = data_ptr[ibegin + j];
        uint32_t idx = SearchBin(cut_values, cut_ptrs, e);

        if (dense) {
          // Dense: store offset relative to feature's first bin.
          index_data[start + j] = idx - cut_ptrs[e.index];
        } else {
          // Sparse: store absolute bin index.
          index_data[start + j] = idx;
        }

        CHECK_LT(idx, nbins);
        hit_count[idx] += 1;
      }

      if (!dense) {
        // Sort the sparse row's entries by bin index.
        std::sort(index_data + start, index_data + start + size);
        // Pad remaining columns with sentinel value (= nbins).
        for (size_t j = size; j < row_stride; ++j) {
          index_data[start + j] = static_cast<uint32_t>(nbins);
        }
      }
    }
  }
}

}  // namespace metal
}  // namespace xgboost
