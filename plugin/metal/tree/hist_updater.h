/*!
 * Copyright 2024-2026 by Contributors
 * \file hist_updater.h
 * \brief Core histogram-based tree updater for Apple Metal GPU.
 *
 * MetalHistUpdater drives the depth-wise and loss-guided tree growing
 * algorithms.  It mirrors the SYCL HistUpdater but targets Metal
 * compute shaders for histogram building and split evaluation, with
 * FP32 precision only (Metal does not support FP64).
 */
#ifndef PLUGIN_METAL_TREE_HIST_UPDATER_H_
#define PLUGIN_METAL_TREE_HIST_UPDATER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <vector>

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/linalg.h>
#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>

#include "../../src/common/random.h"
#include "../../src/tree/constraints.h"
#include "../../src/tree/hist/expand_entry.h"
#include "../../src/tree/param.h"

#include "../common/hist_util.h"
#include "../common/partition_builder.h"
#include "../common/row_set.h"
#include "../data.h"
#include "../data/gradient_index.h"

namespace xgboost {
namespace metal {
namespace tree {

// ---------------------------------------------------------------------------
// GradStats / SplitEntry / NodeEntry — simplified FP32-only versions
// ---------------------------------------------------------------------------

using GradientSumT = float;
using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
using GradStats = xgboost::detail::GradientPairInternal<GradientSumT>;

/*!
 * \brief Split entry: records the best split found so far for a node.
 */
struct SplitEntry {
  bst_float loss_chg{0.0f};
  bst_feature_t sindex{0};
  bst_float split_value{0.0f};
  GradStats left_sum;
  GradStats right_sum;

  SplitEntry() = default;

  bst_feature_t SplitIndex() const { return sindex & ((1U << 31) - 1U); }
  bool DefaultLeft() const { return (sindex >> 31) != 0; }

  bool NeedReplace(bst_float new_loss_chg, unsigned split_index) const {
    if (std::isinf(new_loss_chg)) return false;
    if (this->SplitIndex() <= split_index) {
      return new_loss_chg > this->loss_chg;
    } else {
      return !(this->loss_chg > new_loss_chg);
    }
  }

  bool Update(const SplitEntry& e) {
    if (this->NeedReplace(e.loss_chg, e.SplitIndex())) {
      *this = e;
      return true;
    }
    return false;
  }

  bool Update(bst_float new_loss_chg, unsigned split_index,
              bst_float new_split_value, bool default_left,
              const GradStats& left, const GradStats& right) {
    if (this->NeedReplace(new_loss_chg, split_index)) {
      this->loss_chg = new_loss_chg;
      if (default_left) split_index |= (1U << 31);
      this->sindex = split_index;
      this->split_value = new_split_value;
      this->left_sum = left;
      this->right_sum = right;
      return true;
    }
    return false;
  }
};

/*!
 * \brief Per-node bookkeeping during tree construction.
 */
struct NodeEntry {
  GradStats stats;
  GradientSumT root_gain{0.0f};
  GradientSumT weight{0.0f};
  SplitEntry best;

  explicit NodeEntry(const xgboost::tree::TrainParam&)
      : root_gain(0.0f), weight(0.0f) {}
  NodeEntry() : root_gain(0.0f), weight(0.0f) {}
};

// ---------------------------------------------------------------------------
// ExpandEntry — tree growing policy node
// ---------------------------------------------------------------------------

struct ExpandEntry {
  static constexpr bst_node_t kRootNid = 0;

  bst_node_t nid;
  int depth;
  xgboost::tree::SplitEntry split;

  ExpandEntry(int nid, int depth) : nid(nid), depth(depth) {}

  bst_float GetLossChange() const { return split.loss_chg; }
  bst_node_t GetNodeId() const { return nid; }

  bool IsValid(const xgboost::tree::TrainParam& param,
               int32_t num_leaves) const {
    if (split.loss_chg <= kRtEps) return false;
    if (split.loss_chg < param.min_split_loss) return false;
    if (param.max_depth > 0 && depth == param.max_depth) return false;
    if (param.max_leaves > 0 && num_leaves == param.max_leaves) return false;
    return true;
  }
};

// ---------------------------------------------------------------------------
// Lightweight split evaluator (FP32 only, no monotone constraints yet)
// ---------------------------------------------------------------------------

struct SplitEvaluator {
  GradientSumT reg_lambda{1.0f};
  GradientSumT reg_alpha{0.0f};
  GradientSumT min_child_weight{1.0f};
  GradientSumT max_delta_step{0.0f};

  void Init(const xgboost::tree::TrainParam& p) {
    reg_lambda = p.reg_lambda;
    reg_alpha = p.reg_alpha;
    min_child_weight = p.min_child_weight;
    max_delta_step = p.max_delta_step;
  }

  static GradientSumT ThresholdL1(GradientSumT w, float alpha) {
    if (w > +alpha) return w - alpha;
    if (w < -alpha) return w + alpha;
    return 0.0f;
  }

  GradientSumT CalcWeight(const GradStats& stats) const {
    if (stats.GetHess() < min_child_weight || stats.GetHess() <= 0.0f) {
      return 0.0f;
    }
    GradientSumT dw =
        -ThresholdL1(stats.GetGrad(), reg_alpha) /
        (stats.GetHess() + reg_lambda);
    if (max_delta_step != 0.0f && std::abs(dw) > max_delta_step) {
      dw = std::copysign(max_delta_step, dw);
    }
    return dw;
  }

  GradientSumT CalcGainGivenWeight(const GradStats& stats,
                                    GradientSumT w) const {
    if (stats.GetHess() <= 0.0f) return 0.0f;
    if (max_delta_step == 0.0f) {
      GradientSumT thr = ThresholdL1(stats.GetGrad(), reg_alpha);
      return (thr * thr) / (stats.GetHess() + reg_lambda);
    }
    return -(2.0f * stats.GetGrad() * w +
             (stats.GetHess() + reg_lambda) * w * w);
  }

  GradientSumT CalcGain(const GradStats& stats) const {
    return CalcGainGivenWeight(stats, CalcWeight(stats));
  }

  GradientSumT CalcSplitGain(const GradStats& left,
                              const GradStats& right) const {
    return CalcGainGivenWeight(left, CalcWeight(left)) +
           CalcGainGivenWeight(right, CalcWeight(right));
  }
};

// ---------------------------------------------------------------------------
// MetalHistUpdater — the main tree-building class
// ---------------------------------------------------------------------------

class MetalHistUpdater {
 public:
  MetalHistUpdater(const Context* ctx,
                   const xgboost::tree::TrainParam& param,
                   DMatrix const* fmat);

  ~MetalHistUpdater();

  /*! \brief Grow one tree. */
  void Update(xgboost::tree::TrainParam const* param,
              const HostDeviceVector<GradientPair>& gpair,
              DMatrix* p_fmat,
              xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
              RegTree* p_tree);

  bool UpdatePredictionCache(const DMatrix* data,
                             ::xgboost::linalg::MatrixView<float> out_preds);

 private:
  // ---- Lifecycle helpers ----

  void InitData(const HostDeviceVector<GradientPair>& gpair,
                const DMatrix& fmat, const RegTree& tree);

  void InitGHistIndex(DMatrix* dmat);

  // ---- Depth-wise expansion ----

  void ExpandWithDepthWise(RegTree* p_tree,
                           const HostDeviceVector<GradientPair>& gpair);

  void BuildNodeStats(RegTree* p_tree,
                      const HostDeviceVector<GradientPair>& gpair);

  void EvaluateAndApplySplits(RegTree* p_tree, int* num_leaves, int depth,
                              std::vector<ExpandEntry>* temp_qexpand_depth);

  void AddSplitsToTree(RegTree* p_tree, int* num_leaves, int depth,
                       std::vector<ExpandEntry>* nodes_for_apply_split,
                       std::vector<ExpandEntry>* temp_qexpand_depth);

  // ---- Loss-guided expansion ----

  void ExpandWithLossGuide(RegTree* p_tree,
                           const HostDeviceVector<GradientPair>& gpair);

  void BuildHistogramsLossGuide(ExpandEntry entry, RegTree* p_tree,
                                const HostDeviceVector<GradientPair>& gpair);

  // ---- Histogram construction ----

  void BuildLocalHistograms(RegTree* p_tree,
                            const HostDeviceVector<GradientPair>& gpair);

  void BuildHistGPU(const HostDeviceVector<GradientPair>& gpair,
                    const common::RowSetCollection::Elem& row_indices,
                    common::GHistRow* hist);

  // ---- Sibling split / subtraction trick ----

  void SplitSiblings(const std::vector<ExpandEntry>& nodes,
                     std::vector<ExpandEntry>* small_siblings,
                     std::vector<ExpandEntry>* big_siblings,
                     RegTree* p_tree);

  // ---- Split evaluation ----

  void EvaluateSplits(const std::vector<ExpandEntry>& nodes_set,
                      const RegTree& tree);

  void EnumerateSplit(const common::GHistRow& hist,
                      const NodeEntry& snode,
                      SplitEntry* p_best,
                      bst_uint fid);

  // ---- Row partitioning ----

  void ApplySplit(const std::vector<ExpandEntry>& nodes, RegTree* p_tree);

  void AddSplitsToRowSet(const std::vector<ExpandEntry>& nodes,
                         RegTree* p_tree);

  // ---- Node initialisation ----

  void InitNewNode(int nid,
                   const HostDeviceVector<GradientPair>& gpair,
                   const RegTree& tree);

  // ---- Metal kernel management ----

  void LoadMetalKernels();

  // ---- Data members ----

  const Context* ctx_;
  xgboost::tree::TrainParam param_;
  DMatrix const* p_last_fmat_;
  const RegTree* p_last_tree_{nullptr};

  // Quantised feature matrix (built once per DMatrix, reused across rounds).
  GHistIndexMatrix gmat_;
  bool gmat_initialized_{false};

  // Histogram collection and evaluator.
  common::HistCollection hist_;
  SplitEvaluator evaluator_;

  // Row sets and partition builder.
  common::RowSetCollection row_set_collection_;
  common::PartitionBuilder partition_builder_;

  // Per-node bookkeeping.
  std::vector<NodeEntry> snode_host_;

  // Column sampler.
  std::shared_ptr<xgboost::common::ColumnSampler> column_sampler_;
  FeatureInteractionConstraintHost interaction_constraints_;

  // Depth-wise queue.
  std::vector<ExpandEntry> qexpand_depth_wise_;

  // Nodes selected for explicit histogram building vs subtraction trick.
  std::vector<ExpandEntry> nodes_for_explicit_hist_build_;
  std::vector<ExpandEntry> nodes_for_subtraction_trick_;

  // Loss-guided queue.
  using ExpandQueue =
      std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                          std::function<bool(ExpandEntry, ExpandEntry)>>;
  std::unique_ptr<ExpandQueue> qexpand_loss_guided_;

  // Feature with the least bins (for dense root gradient sum trick).
  uint32_t fid_least_bins_{0};

  enum DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
  DataLayout data_layout_{kSparseData};

  // Metal kernel pipeline states (void* wrapping id<MTLComputePipelineState>).
  void* build_hist_pipeline_{nullptr};       // threadgroup-local, for nbins <= 4096
  void* build_hist_large_pipeline_{nullptr}; // device atomics, for nbins > 4096

  // Cached Metal buffers to avoid per-dispatch allocation
  void* cached_gpair_buf_{nullptr};     // gradient pairs (updated once per iteration)
  size_t cached_gpair_size_{0};         // current gpair buffer size
  void* cached_cut_ptrs_buf_{nullptr};  // cut point offsets (set once per DMatrix)

  // Metal library (void* wrapping id<MTLLibrary>).
  void* metal_library_{nullptr};
};

}  // namespace tree
}  // namespace metal
}  // namespace xgboost

#endif  // PLUGIN_METAL_TREE_HIST_UPDATER_H_
