/*!
 * Copyright 2024-2026 by Contributors
 * \file row_set.h
 * \brief Row set collection for Metal tree updater.
 *
 * Tracks which rows (instances) belong to which tree node during
 * tree construction.  Backed by MetalBuffer so row indices live
 * in unified (shared) memory accessible from both CPU and GPU.
 */
#ifndef PLUGIN_METAL_COMMON_ROW_SET_H_
#define PLUGIN_METAL_COMMON_ROW_SET_H_

#include <algorithm>
#include <cstddef>
#include <vector>

#include <xgboost/data.h>

#include "../data.h"

namespace xgboost {
namespace metal {
namespace common {

/*!
 * \brief Collection of row sets stored in Metal shared memory.
 *
 * Each element maps a tree node id to a contiguous range of row indices
 * inside a single MetalVector<size_t>.
 */
class RowSetCollection {
 public:
  /*! \brief A contiguous subset of rows associated with a single tree node. */
  struct Elem {
    size_t* begin{nullptr};
    size_t* end{nullptr};
    bst_node_t node_id{-1};

    Elem() = default;
    Elem(size_t* begin, size_t* end, bst_node_t node_id = -1)
        : begin(begin), end(end), node_id(node_id) {}

    inline size_t Size() const {
      return end - begin;
    }
  };

  inline size_t Size() const {
    return elem_of_each_node_.size();
  }

  /*! \brief Return the element set for a given node_id (const). */
  inline const Elem& operator[](unsigned node_id) const {
    const Elem& e = elem_of_each_node_[node_id];
    CHECK(e.begin != nullptr)
        << "access element that is not in the set";
    return e;
  }

  /*! \brief Return the element set for a given node_id (mutable). */
  inline Elem& operator[](unsigned node_id) {
    return elem_of_each_node_[node_id];
  }

  inline void Clear() {
    elem_of_each_node_.clear();
  }

  /*! \brief Initialize: node 0 owns all rows. */
  inline void Init() {
    CHECK_EQ(elem_of_each_node_.size(), 0U);
    size_t* begin = row_indices_.Begin();
    size_t* end   = row_indices_.End();
    elem_of_each_node_.emplace_back(Elem(begin, end, 0));
  }

  MetalVector<size_t>& Data() { return row_indices_; }

  /*! \brief Split a node's row set into left and right children. */
  inline void AddSplit(unsigned node_id,
                       unsigned left_node_id,
                       unsigned right_node_id,
                       size_t n_left,
                       size_t n_right) {
    const Elem e = elem_of_each_node_[node_id];
    CHECK(e.begin != nullptr);
    size_t* all_begin = row_indices_.Begin();
    size_t* begin = all_begin + (e.begin - all_begin);

    CHECK_EQ(n_left + n_right, e.Size());
    CHECK_LE(begin + n_left, e.end);
    CHECK_EQ(begin + n_left + n_right, e.end);

    if (left_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize(left_node_id + 1, Elem(nullptr, nullptr, -1));
    }
    if (right_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize(right_node_id + 1, Elem(nullptr, nullptr, -1));
    }

    elem_of_each_node_[left_node_id]  = Elem(begin, begin + n_left, left_node_id);
    elem_of_each_node_[right_node_id] = Elem(begin + n_left, e.end, right_node_id);
    elem_of_each_node_[node_id]       = Elem(nullptr, nullptr, -1);
  }

 private:
  MetalVector<size_t> row_indices_;
  std::vector<Elem> elem_of_each_node_;
};

}  // namespace common
}  // namespace metal
}  // namespace xgboost

#endif  // PLUGIN_METAL_COMMON_ROW_SET_H_
