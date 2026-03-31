/*!
 * Copyright 2024-2026 by Contributors
 * \file partition_builder.h
 * \brief Row partitioning for Metal tree updater.
 *
 * After a split decision, rows must be rearranged so that left-child
 * rows are contiguous, followed by right-child rows.  This class
 * encapsulates that operation, dispatching Metal compute kernels for
 * the flag and scatter steps.
 */
#ifndef PLUGIN_METAL_COMMON_PARTITION_BUILDER_H_
#define PLUGIN_METAL_COMMON_PARTITION_BUILDER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <xgboost/base.h>
#include <xgboost/tree_model.h>

#include "../data.h"
#include "row_set.h"

namespace xgboost {
namespace metal {
namespace common {

/*!
 * \brief Partition builder: splits rows across child nodes using Metal GPU.
 *
 * For each node being split, the builder:
 *   1. Classifies every row as left or right based on the split condition.
 *   2. Computes a prefix sum of the flags (CPU-side, fast for typical sizes).
 *   3. Scatters rows to their final positions.
 *
 * The output is a rearranged segment of the row-index array in which
 * left rows come first, followed by right rows.
 */
class PartitionBuilder {
 public:
  /*!
   * \brief Initialize for a batch of nodes to be partitioned.
   *
   * \param n_nodes      Number of nodes being split.
   * \param func_n_rows  Callable(size_t node_in_set) -> size_t returning
   *                     the number of rows for the node at that position.
   */
  template <typename Func>
  void Init(size_t n_nodes, Func func_n_rows) {
    n_nodes_ = n_nodes;
    nodes_offsets_.resize(n_nodes + 1);
    result_rows_.resize(2 * n_nodes);

    nodes_offsets_[0] = 0;
    for (size_t i = 1; i <= n_nodes; ++i) {
      nodes_offsets_[i] = nodes_offsets_[i - 1] + func_n_rows(i - 1);
    }

    size_t total = nodes_offsets_[n_nodes];
    if (data_.Size() < total) {
      data_.Resize(total);
    }
  }

  size_t GetNLeftElems(int nid) const {
    return result_rows_[2 * nid];
  }

  size_t GetNRightElems(int nid) const {
    return result_rows_[2 * nid + 1];
  }

  /*!
   * \brief Partition a batch of nodes.
   *
   * Uses Metal compute kernels where beneficial, with CPU fallback for
   * very small row counts.
   *
   * \param row_set_collection  Current mapping of node -> row range.
   * \param nodes               Expand entries for nodes being split.
   * \param split_conditions    Per-node split condition (bin index).
   * \param p_tree              The tree being constructed.
   * \param gmat_row_stride     Row stride of the quantized feature matrix.
   * \param gmat_index_data     Pointer to the quantized feature data.
   * \param gmat_cut_ptrs       Cut point offsets per feature.
   * \param is_dense            Whether the feature matrix is dense.
   */
  void Partition(const RowSetCollection& row_set_collection,
                 const std::vector<int32_t>& node_ids,
                 const std::vector<int32_t>& split_conditions,
                 const std::vector<bst_uint>& split_features,
                 const std::vector<bool>& default_lefts,
                 size_t gmat_row_stride,
                 const uint32_t* gmat_index_data,
                 const uint32_t* gmat_cut_ptrs,
                 bool is_dense);

  /*!
   * \brief Copy rearranged rows back into the row set collection.
   */
  void MergeToArray(size_t nid, size_t* data_result);

 private:
  size_t n_nodes_{0};
  std::vector<size_t> nodes_offsets_;
  std::vector<size_t> result_rows_;
  MetalVector<size_t> data_;
};

}  // namespace common
}  // namespace metal
}  // namespace xgboost

#endif  // PLUGIN_METAL_COMMON_PARTITION_BUILDER_H_
