/*!
 * Copyright 2024-2026 by Contributors
 * \file hist_updater.mm
 * \brief Core histogram-based tree updater for Apple Metal GPU.
 *
 * This file implements the full depth-wise and loss-guided tree growing
 * algorithms, ported from the SYCL plugin to Apple Metal.  Histogram
 * construction is accelerated via Metal compute shaders operating on
 * shared-memory MetalBuffers, while split evaluation and tree bookkeeping
 * run on the CPU.
 *
 * Key differences from SYCL:
 *   - FP32 only (Metal has no FP64 support).
 *   - Metal shared memory is CPU-coherent on Apple Silicon, so no
 *     explicit host <-> device copies are needed for data in
 *     MetalBuffer/MetalVector.
 *   - Histogram building dispatches a Metal compute kernel.
 *   - Split evaluation is done on CPU (Metal lacks sub-group scans).
 *   - Row partitioning uses a CPU loop over shared memory.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/logging.h>

#include "../../src/common/hist_util.h"
#include "../../src/tree/common_row_partitioner.h"
#include "../../src/tree/param.h"

#include "hist_updater.h"
#include "../device_manager.h"

namespace xgboost {
namespace metal {
namespace tree {

// ============================================================================
// Construction / destruction
// ============================================================================

MetalHistUpdater::MetalHistUpdater(const Context* ctx,
                                   const xgboost::tree::TrainParam& param,
                                   DMatrix const* fmat)
    : ctx_(ctx),
      param_(param),
      p_last_fmat_(fmat),
      column_sampler_(std::make_shared<xgboost::common::ColumnSampler>()) {
  evaluator_.Init(param);
  LoadMetalKernels();
}

MetalHistUpdater::~MetalHistUpdater() {
  @autoreleasepool {
    if (build_hist_pipeline_) {
      (void)(__bridge_transfer id<MTLComputePipelineState>)build_hist_pipeline_;
      build_hist_pipeline_ = nullptr;
    }
    if (metal_library_) {
      (void)(__bridge_transfer id<MTLLibrary>)metal_library_;
      metal_library_ = nullptr;
    }
    if (cached_gpair_buf_) {
      (void)(__bridge_transfer id<MTLBuffer>)cached_gpair_buf_;
      cached_gpair_buf_ = nullptr;
    }
    if (cached_cut_ptrs_buf_) {
      (void)(__bridge_transfer id<MTLBuffer>)cached_cut_ptrs_buf_;
      cached_cut_ptrs_buf_ = nullptr;
    }
  }
}

// ============================================================================
// LoadMetalKernels — compile or load the .metallib for histogram kernels
// ============================================================================

void MetalHistUpdater::LoadMetalKernels() {
  @autoreleasepool {
    id<MTLDevice> device =
        (__bridge id<MTLDevice>)DeviceManager::GetDevice();

    NSError* error = nil;
    id<MTLLibrary> library = nil;

    // Try to load a pre-compiled metallib from next to the process executable.
    NSString* execPath =
        [[[NSProcessInfo processInfo] arguments] firstObject];
    if (execPath) {
      NSString* dir = [execPath stringByDeletingLastPathComponent];
      NSString* libPath =
          [dir stringByAppendingPathComponent:@"xgboost.metallib"];
      if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        library = [device newLibraryWithURL:url error:&error];
        if (library) {
          LOG(INFO) << "Loaded Metal library from "
                    << [libPath UTF8String];
        }
      }
    }

    // Fallback: build from embedded MSL source string.
    // The GHistIndexMatrix stores uint32_t bin indices.  For dense data the
    // stored value is the feature-local bin offset; for sparse data it is the
    // absolute bin index.  The kernel here handles the dense case: it adds
    // cut_ptrs[f] to reconstruct the absolute bin, then atomically accumulates
    // grad/hess into the histogram.
    if (!library) {
      NSString* source = @R"(
#include <metal_stdlib>
using namespace metal;

struct GradPair {
    float grad;
    float hess;
};

// CAS-loop atomic float add for threadgroup memory
inline void atomic_add_f(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint next;
    for (int i = 0; i < 14; i++) {
        next = as_type<uint>(as_type<float>(expected) + val);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed)) return;
    }
    do {
        next = as_type<uint>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed));
}

// Histogram kernel with threadgroup-local accumulation.
// Each threadgroup builds a local histogram, then flushes to global output.
// Threadgroup memory limit: 32KB → max ~4K bins (4K * 8 bytes = 32KB).
kernel void build_histogram(
    const device GradPair*  gpair        [[buffer(0)]],
    const device uint*      gmat_index   [[buffer(1)]],
    const device ulong*     row_indices  [[buffer(2)]],
    const device uint*      cut_ptrs     [[buffer(3)]],
    device float*           hist_out     [[buffer(4)]],
    constant uint&          num_rows     [[buffer(5)]],
    constant uint&          row_stride   [[buffer(6)]],
    constant uint&          num_features [[buffer(7)]],
    constant uint&          nbins        [[buffer(8)]],
    uint                    tid          [[thread_position_in_grid]],
    uint                    ltid         [[thread_position_in_threadgroup]],
    uint                    tg_size      [[threads_per_threadgroup]],
    uint                    gid          [[threadgroup_position_in_grid]])
{
    // Allocate threadgroup-local histogram (grad + hess per bin as atomic_uint)
    threadgroup atomic_uint local_hist[8192]; // 4096 bins * 2 = 32KB max

    // Zero local histogram
    for (uint i = ltid; i < nbins * 2; i += tg_size) {
        atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate into threadgroup-local histogram
    if (tid < num_rows) {
        uint row_id = row_indices[tid];
        float g = gpair[row_id].grad;
        float h = gpair[row_id].hess;

        const device uint* row = gmat_index + row_id * row_stride;
        for (uint f = 0; f < num_features; ++f) {
            uint bin = row[f] + cut_ptrs[f];
            if (bin < nbins) {
                atomic_add_f(&local_hist[2 * bin],     g);
                atomic_add_f(&local_hist[2 * bin + 1], h);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Flush local histogram to global output using CAS-loop device atomics
    for (uint i = ltid; i < nbins * 2; i += tg_size) {
        float val = as_type<float>(atomic_load_explicit(&local_hist[i], memory_order_relaxed));
        if (val != 0.0f) {
            device atomic_uint* addr = (device atomic_uint*)&hist_out[i];
            uint old = atomic_load_explicit(addr, memory_order_relaxed);
            uint next;
            do {
                next = as_type<uint>(as_type<float>(old) + val);
            } while (!atomic_compare_exchange_weak_explicit(
                addr, &old, next,
                memory_order_relaxed, memory_order_relaxed));
        }
    }
}
)";

      MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
      if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeFast;
      } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
      }

      library = [device newLibraryWithSource:source
                                     options:opts
                                       error:&error];
      if (!library) {
        LOG(FATAL) << "Failed to compile Metal histogram kernel: "
                   << (error ? [[error localizedDescription] UTF8String]
                             : "unknown error");
      }
      LOG(INFO) << "Compiled Metal histogram kernel from embedded source.";
    }

    metal_library_ = (__bridge_retained void*)library;

    // Build pipeline state for build_histogram kernel.
    id<MTLFunction> build_hist_fn =
        [library newFunctionWithName:@"build_histogram"];
    CHECK(build_hist_fn)
        << "Metal function 'build_histogram' not found in library.";

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:build_hist_fn
                                              error:&error];
    CHECK(pipeline)
        << "Failed to create build_histogram pipeline: "
        << (error ? [[error localizedDescription] UTF8String]
                  : "unknown error");

    build_hist_pipeline_ = (__bridge_retained void*)pipeline;
  }
}

// ============================================================================
// InitGHistIndex — build the quantized feature matrix (once per DMatrix)
// ============================================================================

void MetalHistUpdater::InitGHistIndex(DMatrix* dmat) {
  if (gmat_initialized_) return;

  gmat_.Init(ctx_, dmat, static_cast<int>(param_.max_bin));
  gmat_initialized_ = true;

  LOG(INFO) << "Metal GHistIndex: "
            << dmat->Info().num_row_ << " rows, "
            << gmat_.nfeatures << " features, "
            << gmat_.nbins << " bins, "
            << (gmat_.IsDense() ? "dense" : "sparse");
}

// ============================================================================
// Update — main entry point for growing one tree
// ============================================================================

void MetalHistUpdater::Update(
    xgboost::tree::TrainParam const* param,
    const HostDeviceVector<GradientPair>& gpair,
    DMatrix* p_fmat,
    xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
    RegTree* p_tree) {
  param_ = *param;
  evaluator_.Init(param_);
  interaction_constraints_.Reset();

  InitGHistIndex(p_fmat);
  InitData(gpair, *p_fmat, *p_tree);

  // Copy gradient pairs to cached Metal buffer once per tree iteration.
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)DeviceManager::GetDevice();
    size_t gpair_bytes = gpair.Size() * sizeof(GradientPair);
    if (!cached_gpair_buf_ || cached_gpair_size_ != gpair.Size()) {
      if (cached_gpair_buf_) {
        (void)(__bridge_transfer id<MTLBuffer>)cached_gpair_buf_;
      }
      id<MTLBuffer> buf = [device newBufferWithLength:gpair_bytes
                                              options:MTLResourceStorageModeShared];
      cached_gpair_buf_ = (__bridge_retained void*)buf;
      cached_gpair_size_ = gpair.Size();
    }
    id<MTLBuffer> gpairBuf = (__bridge id<MTLBuffer>)cached_gpair_buf_;
    std::memcpy([gpairBuf contents], gpair.ConstHostVector().data(), gpair_bytes);
  }

  if (param_.grow_policy == xgboost::tree::TrainParam::kLossGuide) {
    ExpandWithLossGuide(p_tree, gpair);
  } else {
    ExpandWithDepthWise(p_tree, gpair);
  }

  // Write final node stats.
  for (int nid = 0; nid < p_tree->NumNodes(); ++nid) {
    p_tree->Stat(nid).loss_chg = snode_host_[nid].best.loss_chg;
    p_tree->Stat(nid).base_weight = snode_host_[nid].weight;
    p_tree->Stat(nid).sum_hess =
        static_cast<float>(snode_host_[nid].stats.GetHess());
  }
}

// ============================================================================
// UpdatePredictionCache
// ============================================================================

bool MetalHistUpdater::UpdatePredictionCache(
    const DMatrix* data, ::xgboost::linalg::MatrixView<float> out_preds) {
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
    return false;
  }
  if (out_preds.Size() == 0) {
    return false;
  }

  auto sc_tree = p_last_tree_->HostScView();
  size_t n_nodes = row_set_collection_.Size();
  for (size_t node = 0; node < n_nodes; ++node) {
    const auto& rowset = row_set_collection_[node];
    if (rowset.begin != nullptr && rowset.end != nullptr &&
        rowset.Size() != 0) {
      int nid = rowset.node_id;
      if (sc_tree.IsDeleted(nid)) {
        while (sc_tree.IsDeleted(nid)) {
          nid = sc_tree.Parent(nid);
        }
        CHECK(sc_tree.IsLeaf(nid));
      }
      bst_float leaf_value = sc_tree.LeafValue(nid);
      const size_t* rid = rowset.begin;
      const size_t num_rows = rowset.Size();
      for (size_t i = 0; i < num_rows; ++i) {
        float& val = const_cast<float&>(out_preds(rid[i]));
        val += leaf_value;
      }
    }
  }
  return true;
}

// ============================================================================
// InitData — set up row sets, allocate histograms
// ============================================================================

void MetalHistUpdater::InitData(
    const HostDeviceVector<GradientPair>& gpair,
    const DMatrix& fmat,
    const RegTree& tree) {
  CHECK(param_.max_depth > 0 || param_.max_leaves > 0)
      << "max_depth or max_leaves cannot be both 0 (unlimited).";
  if (param_.grow_policy == xgboost::tree::TrainParam::kDepthWise) {
    CHECK(param_.max_depth > 0) << "max_depth cannot be 0 when depthwise.";
  }

  const auto& info = fmat.Info();

  // Initialise row set collection.
  row_set_collection_.Clear();
  hist_.Init(static_cast<uint32_t>(gmat_.nbins));

  auto& row_indices = row_set_collection_.Data();
  row_indices.Resize(info.num_row_);
  size_t* p_row_indices = row_indices.Data();

  // Populate row indices, filtering out rows with negative hessians.
  const GradientPair* gpair_ptr = gpair.ConstHostVector().data();
  size_t count = 0;
  for (size_t i = 0; i < info.num_row_; ++i) {
    if (gpair_ptr[i].GetHess() >= 0.0f) {
      p_row_indices[count++] = i;
    }
  }
  row_indices.Resize(count);
  row_set_collection_.Init();

  // Determine data layout.
  {
    const size_t nrow = info.num_row_;
    const size_t ncol = info.num_col_;
    const size_t nnz = info.num_nonzero_;
    if (nrow * ncol == nnz) {
      data_layout_ = kDenseDataZeroBased;
    } else {
      data_layout_ = kSparseData;
    }
  }

  p_last_tree_ = &tree;
  column_sampler_->Init(ctx_, info.num_col_, info.feature_weights,
                        param_.colsample_bynode, param_.colsample_bylevel,
                        param_.colsample_bytree);

  // Find the feature with the least bins (for dense root sum trick).
  if (data_layout_ != kSparseData) {
    const auto& cut_ptrs = gmat_.cut.Ptrs();
    uint32_t min_nbins = 0;
    for (bst_uint i = 0; i < gmat_.nfeatures; ++i) {
      uint32_t nb = cut_ptrs[i + 1] - cut_ptrs[i];
      if (nb > 0 && (min_nbins == 0 || nb < min_nbins)) {
        min_nbins = nb;
        fid_least_bins_ = i;
      }
    }
    CHECK_GT(min_nbins, 0U);
  }

  snode_host_.clear();
  snode_host_.resize(1, NodeEntry());

  if (param_.grow_policy == xgboost::tree::TrainParam::kLossGuide) {
    auto cmp = [](ExpandEntry lhs, ExpandEntry rhs) {
      if (lhs.GetLossChange() == rhs.GetLossChange()) {
        return lhs.GetNodeId() > rhs.GetNodeId();
      }
      return lhs.GetLossChange() < rhs.GetLossChange();
    };
    qexpand_loss_guided_.reset(new ExpandQueue(cmp));
  } else {
    qexpand_depth_wise_.clear();
  }
}

// ============================================================================
// BuildHistGPU — dispatch Metal compute kernel for one node
// ============================================================================

void MetalHistUpdater::BuildHistGPU(
    const HostDeviceVector<GradientPair>& gpair,
    const common::RowSetCollection::Elem& row_indices,
    common::GHistRow* hist) {
  const size_t num_rows = row_indices.Size();
  if (num_rows == 0) {
    common::InitHist(hist, gmat_.nbins);
    return;
  }

  // Zero the output histogram.
  common::InitHist(hist, gmat_.nbins);

  @autoreleasepool {
    id<MTLDevice> device =
        (__bridge id<MTLDevice>)DeviceManager::GetDevice();
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)DeviceManager::GetQueue();
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)build_hist_pipeline_;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    CHECK(cmdBuf) << "Failed to create Metal command buffer.";

    id<MTLComputeCommandEncoder> encoder =
        [cmdBuf computeCommandEncoder];
    CHECK(encoder) << "Failed to create Metal compute encoder.";

    [encoder setComputePipelineState:pipeline];

    // Buffer 0: gradient pairs — cached buffer, updated once per iteration in Update().
    CHECK(cached_gpair_buf_) << "Gradient pairs buffer not initialized";
    [encoder setBuffer:(__bridge id<MTLBuffer>)cached_gpair_buf_ offset:0 atIndex:0];

    // Buffer 1: quantized feature index (already in Metal buffer).
    id<MTLBuffer> gmatBuf =
        (__bridge id<MTLBuffer>)gmat_.index.GetMTLBuffer();
    [encoder setBuffer:gmatBuf offset:0 atIndex:1];

    // Buffer 2: row indices for this node (subset of row_set_collection).
    id<MTLBuffer> rowIdxBuf =
        (__bridge id<MTLBuffer>)row_set_collection_.Data().GetMTLBuffer();
    size_t rowIdxOffset =
        reinterpret_cast<const uint8_t*>(row_indices.begin) -
        reinterpret_cast<const uint8_t*>(
            row_set_collection_.Data().DataConst());
    [encoder setBuffer:rowIdxBuf offset:rowIdxOffset atIndex:2];

    // Buffer 3: cut point offsets — cached, set once per DMatrix.
    if (!cached_cut_ptrs_buf_) {
      const auto& cut_ptrs_vec = gmat_.cut.Ptrs();
      id<MTLBuffer> buf = [device newBufferWithBytes:cut_ptrs_vec.data()
                                              length:cut_ptrs_vec.size() * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
      cached_cut_ptrs_buf_ = (__bridge_retained void*)buf;
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)cached_cut_ptrs_buf_
                offset:0 atIndex:3];

    // Buffer 4: output histogram.
    id<MTLBuffer> histBuf =
        (__bridge id<MTLBuffer>)hist->GetMTLBuffer();
    [encoder setBuffer:histBuf offset:0 atIndex:4];

    // Constant buffers 5-8.
    uint32_t num_rows_u = static_cast<uint32_t>(num_rows);
    uint32_t row_stride_u = static_cast<uint32_t>(gmat_.row_stride);
    uint32_t num_features_u = static_cast<uint32_t>(gmat_.nfeatures);
    uint32_t nbins_u = static_cast<uint32_t>(gmat_.nbins);

    [encoder setBytes:&num_rows_u length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&row_stride_u length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&num_features_u length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&nbins_u length:sizeof(uint32_t) atIndex:8];

    // Dispatch.
    NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > 256) threadGroupSize = 256;
    MTLSize gridSize = MTLSizeMake(num_rows, 1, 1);
    MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreads:gridSize
        threadsPerThreadgroup:groupSize];

    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
  }
}

// ============================================================================
// BuildLocalHistograms — build histograms for explicitly selected nodes
// ============================================================================

void MetalHistUpdater::BuildLocalHistograms(
    RegTree* p_tree,
    const HostDeviceVector<GradientPair>& gpair) {
  const size_t n_nodes = nodes_for_explicit_hist_build_.size();
  for (size_t i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes_for_explicit_hist_build_[i].nid;
    hist_.AddHistRow(nid);
    if (row_set_collection_[nid].Size() > 0) {
      BuildHistGPU(gpair, row_set_collection_[nid], &(hist_[nid]));
    } else {
      common::InitHist(&(hist_[nid]), gmat_.nbins);
    }
  }

  // Subtraction trick for big siblings.
  for (size_t i = 0; i < nodes_for_subtraction_trick_.size(); ++i) {
    const int32_t nid = nodes_for_subtraction_trick_[i].nid;
    hist_.AddHistRow(nid);

    auto sc_tree = p_tree->HostScView();
    const int32_t parent_id = sc_tree.Parent(nid);
    const int32_t sibling_id =
        sc_tree.IsLeftChild(nid) ? sc_tree.RightChild(parent_id)
                                 : sc_tree.LeftChild(parent_id);

    common::SubtractionHist(&(hist_[nid]), hist_[parent_id],
                            hist_[sibling_id], gmat_.nbins);
  }
}

// ============================================================================
// SplitSiblings — decide which child gets explicit hist vs subtraction trick
// ============================================================================

void MetalHistUpdater::SplitSiblings(
    const std::vector<ExpandEntry>& nodes,
    std::vector<ExpandEntry>* small_siblings,
    std::vector<ExpandEntry>* big_siblings,
    RegTree* p_tree) {
  for (auto const& entry : nodes) {
    int nid = entry.nid;
    RegTree::Node& node = (*p_tree)[nid];
    if (node.IsRoot()) {
      small_siblings->push_back(entry);
    } else {
      const int32_t left_id = (*p_tree)[node.Parent()].LeftChild();
      const int32_t right_id = (*p_tree)[node.Parent()].RightChild();

      if (nid == left_id &&
          row_set_collection_[left_id].Size() <
              row_set_collection_[right_id].Size()) {
        small_siblings->push_back(entry);
      } else if (nid == right_id &&
                 row_set_collection_[right_id].Size() <=
                     row_set_collection_[left_id].Size()) {
        small_siblings->push_back(entry);
      } else {
        big_siblings->push_back(entry);
      }
    }
  }
}

// ============================================================================
// InitNewNode — compute gradient sums and weights for a new tree node
// ============================================================================

void MetalHistUpdater::InitNewNode(
    int nid,
    const HostDeviceVector<GradientPair>& gpair,
    const RegTree& tree) {
  snode_host_.resize(tree.NumNodes(), NodeEntry());
  auto sc_tree = tree.HostScView();

  if (sc_tree.IsRoot(nid)) {
    GradStats grad_stat(0.0f, 0.0f);

    if (data_layout_ != kSparseData) {
      // Dense: sum histogram bins of the feature with least bins.
      const auto& cut_ptrs = gmat_.cut.Ptrs();
      const uint32_t ibegin = cut_ptrs[fid_least_bins_];
      const uint32_t iend = cut_ptrs[fid_least_bins_ + 1];
      const auto* hist = hist_[nid].DataConst();
      for (uint32_t i = ibegin; i < iend; ++i) {
        grad_stat.Add(hist[i].GetGrad(), hist[i].GetHess());
      }
    } else {
      // Sparse: iterate over all rows.
      const auto& e = row_set_collection_[nid];
      const size_t* row_idxs = e.begin;
      const size_t size = e.Size();
      const GradientPair* gpair_ptr = gpair.ConstHostVector().data();
      for (size_t i = 0; i < size; ++i) {
        size_t row_idx = row_idxs[i];
        grad_stat.Add(gpair_ptr[row_idx].GetGrad(),
                      gpair_ptr[row_idx].GetHess());
      }
    }
    snode_host_[nid].stats = grad_stat;
  } else {
    int parent_id = sc_tree.Parent(nid);
    if (sc_tree.IsLeftChild(nid)) {
      snode_host_[nid].stats = snode_host_[parent_id].best.left_sum;
    } else {
      snode_host_[nid].stats = snode_host_[parent_id].best.right_sum;
    }
  }

  // Compute weight and root gain.
  snode_host_[nid].weight = evaluator_.CalcWeight(snode_host_[nid].stats);
  snode_host_[nid].root_gain = evaluator_.CalcGain(snode_host_[nid].stats);
}

// ============================================================================
// EnumerateSplit — scan one feature's histogram bins for the best split
// ============================================================================

void MetalHistUpdater::EnumerateSplit(
    const common::GHistRow& hist,
    const NodeEntry& snode,
    SplitEntry* p_best,
    bst_uint fid) {
  const auto& cut_ptrs = gmat_.cut.Ptrs();
  const uint32_t ibegin = cut_ptrs[fid];
  const uint32_t iend = cut_ptrs[fid + 1];
  const auto* hist_data = hist.DataConst();
  const float* cut_val = gmat_.cut.Values().data();
  const float min_child_weight = evaluator_.min_child_weight;

  // Forward scan: missing values go right (default_left = false).
  GradStats sum(0.0f, 0.0f);
  for (uint32_t i = ibegin; i < iend; ++i) {
    sum.Add(hist_data[i].GetGrad(), hist_data[i].GetHess());
    if (sum.GetHess() >= min_child_weight) {
      GradStats c(snode.stats.GetGrad() - sum.GetGrad(),
                  snode.stats.GetHess() - sum.GetHess());
      if (c.GetHess() >= min_child_weight) {
        bst_float loss_chg =
            evaluator_.CalcSplitGain(sum, c) - snode.root_gain;
        bst_float split_pt = cut_val[i];
        p_best->Update(loss_chg, fid, split_pt, /*default_left=*/false,
                       sum, c);
      }
    }
  }

  // Reverse scan: missing values go left (default_left = true).
  GradStats rsum(0.0f, 0.0f);
  for (int32_t i = static_cast<int32_t>(iend) - 1;
       i >= static_cast<int32_t>(ibegin); --i) {
    rsum.Add(hist_data[i].GetGrad(), hist_data[i].GetHess());
    if (rsum.GetHess() >= min_child_weight) {
      GradStats c(snode.stats.GetGrad() - rsum.GetGrad(),
                  snode.stats.GetHess() - rsum.GetHess());
      if (c.GetHess() >= min_child_weight) {
        bst_float loss_chg =
            evaluator_.CalcSplitGain(c, rsum) - snode.root_gain;
        bst_float split_pt =
            (i > static_cast<int32_t>(ibegin)) ? cut_val[i - 1]
                                                : cut_val[i] - 1.0f;
        p_best->Update(loss_chg, fid, split_pt, /*default_left=*/true,
                       c, rsum);
      }
    }
  }
}

// ============================================================================
// EvaluateSplits — find best split for each node in a set
// ============================================================================

void MetalHistUpdater::EvaluateSplits(
    const std::vector<ExpandEntry>& nodes_set,
    const RegTree& tree) {
  const size_t n_nodes = nodes_set.size();

  for (size_t nid_in_set = 0; nid_in_set < n_nodes; ++nid_in_set) {
    const bst_node_t nid = nodes_set[nid_in_set].nid;
    auto features_set =
        column_sampler_->GetFeatureSet(ctx_, tree.GetDepth(nid));
    const auto& hist = hist_[nid];

    for (size_t idx = 0; idx < features_set->Size(); ++idx) {
      const bst_feature_t fid = features_set->ConstHostVector()[idx];
      if (interaction_constraints_.Query(nid, fid)) {
        EnumerateSplit(hist, snode_host_[nid], &snode_host_[nid].best, fid);
      }
    }
  }
}

// ============================================================================
// BuildNodeStats — initialise nodes in the depth-wise queue
// ============================================================================

void MetalHistUpdater::BuildNodeStats(
    RegTree* p_tree,
    const HostDeviceVector<GradientPair>& gpair) {
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;
    this->InitNewNode(nid, gpair, *p_tree);
    if (!(*p_tree)[nid].IsLeftChild() && !(*p_tree)[nid].IsRoot()) {
      auto parent_id = (*p_tree)[nid].Parent();
      auto left_sibling_id = (*p_tree)[parent_id].LeftChild();
      auto parent_split_fid = snode_host_[parent_id].best.SplitIndex();
      interaction_constraints_.Split(parent_id, parent_split_fid,
                                     left_sibling_id, nid);
    }
  }
}

// ============================================================================
// AddSplitsToTree — expand the tree with new children
// ============================================================================

void MetalHistUpdater::AddSplitsToTree(
    RegTree* p_tree, int* num_leaves, int depth,
    std::vector<ExpandEntry>* nodes_for_apply_split,
    std::vector<ExpandEntry>* temp_qexpand_depth) {
  const auto lr = param_.learning_rate;

  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;

    if (snode_host_[nid].best.loss_chg < kRtEps ||
        (param_.max_depth > 0 && depth == param_.max_depth) ||
        (param_.max_leaves > 0 && (*num_leaves) == param_.max_leaves)) {
      (*p_tree)[nid].SetLeaf(snode_host_[nid].weight * lr);
    } else {
      nodes_for_apply_split->push_back(entry);

      NodeEntry& e = snode_host_[nid];
      bst_float left_leaf_weight =
          evaluator_.CalcWeight(GradStats{e.best.left_sum}) * lr;
      bst_float right_leaf_weight =
          evaluator_.CalcWeight(GradStats{e.best.right_sum}) * lr;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg,
                         e.stats.GetHess(), e.best.left_sum.GetHess(),
                         e.best.right_sum.GetHess());

      int left_id = (*p_tree)[nid].LeftChild();
      int right_id = (*p_tree)[nid].RightChild();
      temp_qexpand_depth->push_back(
          ExpandEntry(left_id, p_tree->GetDepth(left_id)));
      temp_qexpand_depth->push_back(
          ExpandEntry(right_id, p_tree->GetDepth(right_id)));
      (*num_leaves)++;
    }
  }
}

// ============================================================================
// EvaluateAndApplySplits
// ============================================================================

void MetalHistUpdater::EvaluateAndApplySplits(
    RegTree* p_tree, int* num_leaves, int depth,
    std::vector<ExpandEntry>* temp_qexpand_depth) {
  EvaluateSplits(qexpand_depth_wise_, *p_tree);

  std::vector<ExpandEntry> nodes_for_apply_split;
  AddSplitsToTree(p_tree, num_leaves, depth, &nodes_for_apply_split,
                  temp_qexpand_depth);
  ApplySplit(nodes_for_apply_split, p_tree);
}

// ============================================================================
// ApplySplit — partition rows for each split node
// ============================================================================

void MetalHistUpdater::ApplySplit(
    const std::vector<ExpandEntry>& nodes, RegTree* p_tree) {
  if (nodes.empty()) return;

  const size_t n_nodes = nodes.size();

  // Gather split conditions.
  std::vector<int32_t> node_ids(n_nodes);
  std::vector<int32_t> split_conditions(n_nodes);
  std::vector<bst_uint> split_features(n_nodes);
  std::vector<bool> default_lefts(n_nodes);

  const auto& cut_ptrs = gmat_.cut.Ptrs();
  const float* cut_vals = gmat_.cut.Values().data();

  for (size_t i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes[i].nid;
    node_ids[i] = nid;
    const auto& node = (*p_tree)[nid];
    split_features[i] = node.SplitIndex();
    default_lefts[i] = node.DefaultLeft();

    // Compute split condition: the maximum absolute bin index that goes left.
    bst_feature_t fid = node.SplitIndex();
    bst_float split_pt = node.SplitCond();
    const uint32_t begin = cut_ptrs[fid];
    const uint32_t end = cut_ptrs[fid + 1];
    int32_t split_cond = static_cast<int32_t>(end) - 1;
    for (uint32_t j = begin; j < end; ++j) {
      if (cut_vals[j] > split_pt) {
        split_cond = static_cast<int32_t>(j) - 1;
        break;
      }
    }
    split_conditions[i] = split_cond;
  }

  partition_builder_.Init(n_nodes, [&](size_t node_in_set) {
    return row_set_collection_[node_ids[node_in_set]].Size();
  });

  partition_builder_.Partition(
      row_set_collection_, node_ids, split_conditions, split_features,
      default_lefts, gmat_.row_stride, gmat_.index.DataConst(),
      cut_ptrs.data(), gmat_.IsDense());

  // Merge partitioned rows back into the row set collection.
  for (size_t i = 0; i < n_nodes; ++i) {
    const int32_t nid = node_ids[i];
    size_t* data_result =
        const_cast<size_t*>(row_set_collection_[nid].begin);
    partition_builder_.MergeToArray(i, data_result);
  }

  AddSplitsToRowSet(nodes, p_tree);
}

// ============================================================================
// AddSplitsToRowSet
// ============================================================================

void MetalHistUpdater::AddSplitsToRowSet(
    const std::vector<ExpandEntry>& nodes, RegTree* p_tree) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    const int32_t nid = nodes[i].nid;
    const size_t n_left = partition_builder_.GetNLeftElems(i);
    const size_t n_right = partition_builder_.GetNRightElems(i);
    row_set_collection_.AddSplit(
        nid, (*p_tree)[nid].LeftChild(), (*p_tree)[nid].RightChild(),
        n_left, n_right);
  }
}

// ============================================================================
// ExpandWithDepthWise — level-by-level tree growing
// ============================================================================

void MetalHistUpdater::ExpandWithDepthWise(
    RegTree* p_tree,
    const HostDeviceVector<GradientPair>& gpair) {
  int num_leaves = 0;

  qexpand_depth_wise_.emplace_back(ExpandEntry::kRootNid,
                                   p_tree->GetDepth(ExpandEntry::kRootNid));
  ++num_leaves;

  for (int depth = 0; depth < param_.max_depth + 1; ++depth) {
    std::vector<ExpandEntry> temp_qexpand_depth;

    SplitSiblings(qexpand_depth_wise_, &nodes_for_explicit_hist_build_,
                  &nodes_for_subtraction_trick_, p_tree);

    // Allocate histograms for subtraction-trick nodes (parent needed).
    for (auto const& entry : nodes_for_subtraction_trick_) {
      hist_.AddHistRow(entry.nid);
    }

    BuildLocalHistograms(p_tree, gpair);
    BuildNodeStats(p_tree, gpair);
    EvaluateAndApplySplits(p_tree, &num_leaves, depth, &temp_qexpand_depth);

    // Clean up for the next level.
    qexpand_depth_wise_.clear();
    nodes_for_subtraction_trick_.clear();
    nodes_for_explicit_hist_build_.clear();

    if (temp_qexpand_depth.empty()) {
      break;
    } else {
      qexpand_depth_wise_ = std::move(temp_qexpand_depth);
    }
  }
}

// ============================================================================
// BuildHistogramsLossGuide — for loss-guided growth
// ============================================================================

void MetalHistUpdater::BuildHistogramsLossGuide(
    ExpandEntry entry, RegTree* p_tree,
    const HostDeviceVector<GradientPair>& gpair) {
  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(entry);

  auto sc_tree = p_tree->HostScView();
  if (!sc_tree.IsRoot(entry.nid)) {
    auto parent_id = sc_tree.Parent(entry.nid);
    auto sibling_id =
        sc_tree.IsLeftChild(entry.nid) ? sc_tree.RightChild(parent_id)
                                       : sc_tree.LeftChild(parent_id);
    nodes_for_subtraction_trick_.emplace_back(sibling_id,
                                              p_tree->GetDepth(sibling_id));
    hist_.AddHistRow(sibling_id);
  }

  BuildLocalHistograms(p_tree, gpair);
}

// ============================================================================
// ExpandWithLossGuide — best-first (leaf-wise) tree growing
// ============================================================================

void MetalHistUpdater::ExpandWithLossGuide(
    RegTree* p_tree,
    const HostDeviceVector<GradientPair>& gpair) {
  int num_leaves = 0;
  const auto lr = param_.learning_rate;

  ExpandEntry root(ExpandEntry::kRootNid,
                   p_tree->GetDepth(ExpandEntry::kRootNid));

  // Build and evaluate the root node.
  BuildHistogramsLossGuide(root, p_tree, gpair);
  InitNewNode(ExpandEntry::kRootNid, gpair, *p_tree);
  EvaluateSplits({root}, *p_tree);
  root.split.loss_chg = snode_host_[ExpandEntry::kRootNid].best.loss_chg;

  qexpand_loss_guided_->push(root);
  ++num_leaves;

  while (!qexpand_loss_guided_->empty()) {
    const ExpandEntry candidate = qexpand_loss_guided_->top();
    const int nid = candidate.nid;
    qexpand_loss_guided_->pop();

    if (!candidate.IsValid(param_, num_leaves)) {
      (*p_tree)[nid].SetLeaf(snode_host_[nid].weight * lr);
    } else {
      NodeEntry& e = snode_host_[nid];
      bst_float left_leaf_weight =
          evaluator_.CalcWeight(GradStats{e.best.left_sum}) * lr;
      bst_float right_leaf_weight =
          evaluator_.CalcWeight(GradStats{e.best.right_sum}) * lr;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg,
                         e.stats.GetHess(), e.best.left_sum.GetHess(),
                         e.best.right_sum.GetHess());

      ApplySplit({candidate}, p_tree);

      const int cleft = (*p_tree)[nid].LeftChild();
      const int cright = (*p_tree)[nid].RightChild();

      ExpandEntry left_node(cleft, p_tree->GetDepth(cleft));
      ExpandEntry right_node(cright, p_tree->GetDepth(cright));

      // Build histogram for the smaller child; subtraction trick for larger.
      if (row_set_collection_[cleft].Size() <
          row_set_collection_[cright].Size()) {
        BuildHistogramsLossGuide(left_node, p_tree, gpair);
      } else {
        BuildHistogramsLossGuide(right_node, p_tree, gpair);
      }

      InitNewNode(cleft, gpair, *p_tree);
      InitNewNode(cright, gpair, *p_tree);

      bst_uint featureid = snode_host_[nid].best.SplitIndex();
      interaction_constraints_.Split(nid, featureid, cleft, cright);

      EvaluateSplits({left_node, right_node}, *p_tree);

      left_node.split.loss_chg = snode_host_[cleft].best.loss_chg;
      right_node.split.loss_chg = snode_host_[cright].best.loss_chg;

      qexpand_loss_guided_->push(left_node);
      qexpand_loss_guided_->push(right_node);

      ++num_leaves;
    }
  }
}

}  // namespace tree
}  // namespace metal
}  // namespace xgboost
