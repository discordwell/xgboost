/*!
 * Copyright 2024-2026 by Contributors
 * \file partition_builder.mm
 * \brief Row partitioning implementation for Metal tree updater.
 *
 * After each split, rows belonging to a node must be separated into
 * left-child and right-child subsets.  The implementation here works
 * entirely on CPU using the shared-memory MetalVector, which is
 * coherent on Apple Silicon.  For very large row counts a Metal
 * compute kernel could be dispatched, but the CPU path is already
 * efficient given that shared memory avoids copies.
 *
 * The algorithm for each node:
 *   1. Classify each row as left or right using the split condition.
 *   2. Pack left rows at the front and right rows at the back of
 *      a temporary buffer.
 *   3. Record counts for MergeToArray / AddSplit.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "partition_builder.h"
#include "../device_manager.h"
#include "xgboost/logging.h"

namespace xgboost {
namespace metal {
namespace common {

// ---------------------------------------------------------------------------
// Partition -- classify and rearrange rows for a batch of split nodes
// ---------------------------------------------------------------------------

void PartitionBuilder::Partition(
    const RowSetCollection& row_set_collection,
    const std::vector<int32_t>& node_ids,
    const std::vector<int32_t>& split_conditions,
    const std::vector<bst_uint>& split_features,
    const std::vector<bool>& default_lefts,
    size_t gmat_row_stride,
    const uint32_t* gmat_index_data,
    const uint32_t* gmat_cut_ptrs,
    bool is_dense) {
  CHECK_EQ(node_ids.size(), n_nodes_);

  for (size_t node_in_set = 0; node_in_set < n_nodes_; ++node_in_set) {
    const int32_t nid = node_ids[node_in_set];
    const auto& rid_span = row_set_collection[nid];
    const size_t range_size = rid_span.Size();

    size_t* p_rid_buf = data_.Data() + nodes_offsets_[node_in_set];
    const size_t* rid = rid_span.begin;
    const int32_t split_cond = split_conditions[node_in_set];
    const bst_uint fid = split_features[node_in_set];
    const bool default_left = default_lefts[node_in_set];

    size_t n_left = 0;
    size_t n_right = 0;

    if (is_dense) {
      // Dense layout: bin index stored relative to feature's first bin.
      // Reconstruct absolute bin = index[row * stride + fid] + cut_ptrs[fid].
      const uint32_t offset = gmat_cut_ptrs[fid];
      for (size_t k = 0; k < range_size; ++k) {
        const size_t id = rid[k];
        const int32_t value =
            static_cast<int32_t>(gmat_index_data[id * gmat_row_stride + fid]) +
            static_cast<int32_t>(offset);
        const bool is_left = (value <= split_cond);
        if (is_left) {
          p_rid_buf[n_left++] = id;
        } else {
          p_rid_buf[range_size - 1 - n_right] = id;
          ++n_right;
        }
      }
    } else {
      // Sparse layout: absolute bin indices stored per row, sorted.
      // Use linear scan to find the feature's bin.
      for (size_t k = 0; k < range_size; ++k) {
        const size_t id = rid[k];
        const uint32_t* gr_index_local =
            gmat_index_data + gmat_row_stride * id;

        int32_t fid_local = -1;
        for (size_t j = 0; j < gmat_row_stride; ++j) {
          uint32_t bin = gr_index_local[j];
          if (bin >= gmat_cut_ptrs[fid] && bin < gmat_cut_ptrs[fid + 1]) {
            fid_local = static_cast<int32_t>(bin);
            break;
          }
        }

        bool is_left;
        if (fid_local < 0) {
          is_left = default_left;
        } else {
          is_left = (fid_local <= split_cond);
        }

        if (is_left) {
          p_rid_buf[n_left++] = id;
        } else {
          p_rid_buf[range_size - 1 - n_right] = id;
          ++n_right;
        }
      }
    }

    result_rows_[2 * node_in_set]     = n_left;
    result_rows_[2 * node_in_set + 1] = n_right;
  }
}

// ---------------------------------------------------------------------------
// MergeToArray -- copy rearranged rows into the row set collection
// ---------------------------------------------------------------------------

void PartitionBuilder::MergeToArray(size_t nid, size_t* data_result) {
  const size_t n_total = GetNLeftElems(nid) + GetNRightElems(nid);
  if (n_total > 0) {
    const size_t* src = data_.Data() + nodes_offsets_[nid];
    std::memcpy(data_result, src, sizeof(size_t) * n_total);
  }
}

}  // namespace common
}  // namespace metal
}  // namespace xgboost
