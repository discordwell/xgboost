/*!
 * Copyright 2024-2026 by Contributors
 * \file updater_hist.mm
 * \brief Implementation of MetalQuantileHistMaker — the facade that
 *        registers the Metal histogram tree updater with XGBoost.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <memory>
#include <vector>

#include <xgboost/tree_updater.h>
#include <xgboost/logging.h>
#include <xgboost/gradient.h>

#include "updater_hist.h"
#include "hist_updater.h"
#include "../device_manager.h"

namespace xgboost {
namespace metal {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_hist_metal);

// ---------------------------------------------------------------------------
// Configure
// ---------------------------------------------------------------------------

void MetalQuantileHistMaker::Configure(const Args& args) {
  param_.UpdateAllowUnknown(args);

  // Verify we can obtain a Metal device.
  void* device_ptr = DeviceManager::GetDevice();
  CHECK(device_ptr) << "Metal is not available on this system.";

  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    LOG(INFO) << "MetalQuantileHistMaker: using device "
              << [[device name] UTF8String]
              << " (unified memory: "
              << ([device hasUnifiedMemory] ? "yes" : "no") << ")";
  }
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

void MetalQuantileHistMaker::Update(
    xgboost::tree::TrainParam const* param,
    GradientContainer* in_gpair,
    DMatrix* dmat,
    xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
    const std::vector<RegTree*>& trees) {
  auto gpair = in_gpair->FullGradOnly();

  // Lazily create or reconfigure the pimpl.
  if (!pimpl_) {
    pimpl_.reset(new MetalHistUpdater(ctx_, param_, dmat));
  }

  // Rescale learning rate by number of trees (boosting convention).
  float lr = param_.learning_rate;
  param_.learning_rate = lr / static_cast<float>(trees.size());

  for (auto* tree : trees) {
    pimpl_->Update(&param_, *(gpair->Data()), dmat, out_position, tree);
  }

  param_.learning_rate = lr;
  p_last_dmat_ = dmat;
}

// ---------------------------------------------------------------------------
// UpdatePredictionCache
// ---------------------------------------------------------------------------

bool MetalQuantileHistMaker::UpdatePredictionCache(
    const DMatrix* data,
    xgboost::common::Span<HostDeviceVector<bst_node_t>>,
    ::xgboost::linalg::MatrixView<float> out_preds) {
  if (param_.subsample < 1.0f) return false;
  if (pimpl_) {
    return pimpl_->UpdatePredictionCache(data, out_preds);
  }
  return false;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

XGBOOST_REGISTER_TREE_UPDATER(MetalQuantileHistMaker,
                               "grow_quantile_histmaker_metal")
    .describe("Grow tree using quantized histogram on Apple Metal GPU.")
    .set_body([](Context const* ctx, ObjInfo const* task) {
      return new MetalQuantileHistMaker(ctx, task);
    });

}  // namespace tree
}  // namespace metal
}  // namespace xgboost
