/*!
 * Copyright 2024-2026 by Contributors
 * \file updater_hist.h
 * \brief Quantile-histogram tree updater for Apple Metal GPU.
 *
 * This is the thin facade that plugs into XGBoost's tree updater
 * registry.  The heavy lifting is delegated to MetalHistUpdater (pimpl).
 * Only FP32 histograms are supported because Metal lacks FP64.
 */
#ifndef PLUGIN_METAL_TREE_UPDATER_HIST_H_
#define PLUGIN_METAL_TREE_UPDATER_HIST_H_

#include <memory>
#include <vector>

#include <xgboost/tree_updater.h>
#include <xgboost/json.h>

#include "hist_updater.h"
#include "xgboost/data.h"

namespace xgboost {
namespace metal {
namespace tree {

/*!
 * \brief Construct a tree using quantized feature values with Metal GPU.
 *
 * Metal does not support FP64, so histograms are always built in FP32.
 */
class MetalQuantileHistMaker : public TreeUpdater {
 public:
  MetalQuantileHistMaker(Context const* ctx, ObjInfo const* task)
      : TreeUpdater(ctx), task_{task} {}

  void Configure(const Args& args) override;

  void Update(xgboost::tree::TrainParam const* param,
              GradientContainer* in_gpair,
              DMatrix* dmat,
              xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(
      const DMatrix* data,
      xgboost::common::Span<HostDeviceVector<bst_node_t>>,
      ::xgboost::linalg::MatrixView<float> out_preds) override;

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
  }

  char const* Name() const override {
    return "grow_quantile_histmaker_metal";
  }

 private:
  xgboost::tree::TrainParam param_;
  DMatrix const* p_last_dmat_{nullptr};
  bool is_gmat_initialized_{false};

  std::unique_ptr<MetalHistUpdater> pimpl_;
  ObjInfo const* task_{nullptr};
};

}  // namespace tree
}  // namespace metal
}  // namespace xgboost

#endif  // PLUGIN_METAL_TREE_UPDATER_HIST_H_
