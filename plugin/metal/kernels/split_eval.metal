/*!
 * Copyright 2024-2026 by XGBoost Contributors
 * \file split_eval.metal
 * \brief Metal compute kernel for split evaluation.
 *
 * For each (node, feature) pair, scans histogram bins left-to-right,
 * accumulating gradient and hessian sums.  At each bin boundary, computes
 * the XGBoost split gain and tracks the best split that satisfies the
 * min_child_weight constraint.
 *
 * Gain formula:
 *   gain = 0.5 * ( left_G^2 / (left_H + lambda)
 *                 + right_G^2 / (right_H + lambda)
 *                 - parent_G^2 / (parent_H + lambda) )
 *        - gamma
 *
 * Design:
 *   - One thread per feature for the simple scan-based evaluator.
 *   - A parallel reduction variant selects the globally best split
 *     across all features using threadgroup reduction.
 *   - L1 regularization (alpha) support via the Threshold-L1 function,
 *     matching the SYCL reference implementation.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Negative infinity used as sentinel for "no valid split found".
constant constexpr float NEG_INF = -HUGE_VALF;

/// Default threadgroup size for the reduction kernel.
constant constexpr uint SPLIT_EVAL_THREADGROUP_SIZE = 256;

// ---------------------------------------------------------------------------
// Helper: ThresholdL1
// ---------------------------------------------------------------------------

/// Apply L1 regularization threshold to a gradient sum.
///   if   w >  alpha: return w - alpha
///   elif w < -alpha: return w + alpha
///   else:            return 0
inline float threshold_l1(float w, float alpha) {
    if (w > +alpha) return w - alpha;
    if (w < -alpha) return w + alpha;
    return 0.0f;
}

// ---------------------------------------------------------------------------
// Helper: CalcWeight
// ---------------------------------------------------------------------------

/// Compute the optimal leaf weight given gradient/hessian sums and
/// regularization parameters.
///   w = -ThresholdL1(sum_grad, alpha) / (sum_hess + lambda)
///
/// Returns 0 if sum_hess < min_child_weight (invalid split).
inline float calc_weight(float sum_grad, float sum_hess,
                         float reg_lambda, float reg_alpha,
                         float min_child_weight, float max_delta_step) {
    if (sum_hess < min_child_weight || sum_hess <= 0.0f) {
        return 0.0f;
    }
    float w = -threshold_l1(sum_grad, reg_alpha) / (sum_hess + reg_lambda);
    if (max_delta_step != 0.0f && abs(w) > max_delta_step) {
        w = copysign(max_delta_step, w);
    }
    return w;
}

// ---------------------------------------------------------------------------
// Helper: CalcGainGivenWeight
// ---------------------------------------------------------------------------

/// Compute the gain contribution of a single child node, given its
/// gradient/hessian sums and the optimal weight.
///   gain = -(2 * sum_grad * w + (sum_hess + lambda) * w^2)
inline float calc_gain_given_weight(float sum_grad, float sum_hess,
                                     float w, float reg_lambda) {
    return -(2.0f * sum_grad * w + (sum_hess + reg_lambda) * w * w);
}

// ---------------------------------------------------------------------------
// Helper: CalcGain
// ---------------------------------------------------------------------------

/// Compute the gain for a child node.  Uses the simplified formula when
/// max_delta_step == 0 (avoids computing weight explicitly):
///   gain = ThresholdL1(G, alpha)^2 / (H + lambda)
inline float calc_gain(float sum_grad, float sum_hess,
                       float reg_lambda, float reg_alpha,
                       float min_child_weight, float max_delta_step) {
    if (sum_hess < min_child_weight || sum_hess <= 0.0f) {
        return 0.0f;
    }
    if (max_delta_step == 0.0f) {
        float tl1 = threshold_l1(sum_grad, reg_alpha);
        return (tl1 * tl1) / (sum_hess + reg_lambda);
    }
    float w = calc_weight(sum_grad, sum_hess, reg_lambda, reg_alpha,
                          min_child_weight, max_delta_step);
    return calc_gain_given_weight(sum_grad, sum_hess, w, reg_lambda);
}

// ---------------------------------------------------------------------------
// Helper: CalcSplitGain
// ---------------------------------------------------------------------------

/// Compute the full split gain:
///   gain = 0.5 * (gain_left + gain_right - gain_parent) - gamma
inline float calc_split_gain(float left_grad, float left_hess,
                              float right_grad, float right_hess,
                              float parent_grad, float parent_hess,
                              float reg_lambda, float reg_alpha, float gamma,
                              float min_child_weight, float max_delta_step) {
    float gain_left  = calc_gain(left_grad, left_hess, reg_lambda, reg_alpha,
                                  min_child_weight, max_delta_step);
    float gain_right = calc_gain(right_grad, right_hess, reg_lambda, reg_alpha,
                                  min_child_weight, max_delta_step);
    float gain_parent = calc_gain(parent_grad, parent_hess, reg_lambda, reg_alpha,
                                   min_child_weight, max_delta_step);
    return 0.5f * (gain_left + gain_right - gain_parent) - gamma;
}

// ---------------------------------------------------------------------------
// Kernel: evaluate_splits
// ---------------------------------------------------------------------------

/// Evaluate all possible split points for a single feature.
/// One thread is dispatched per feature.
///
/// For each feature, the kernel scans the histogram bins from left to right,
/// accumulating grad/hess.  At each bin boundary it computes the split gain
/// and tracks the best (highest gain) split that satisfies min_child_weight.
///
/// Output: best_splits[feature_id] = float4(feature_id, bin_id, gain, 0)
///
/// Buffer bindings:
///   0: hist             - histogram [grad, hess] per bin, all features concatenated
///   1: feature_segments - bin start index per feature (n_features + 1 entries)
///                         feature f's bins are hist[2*feature_segments[f] .. 2*feature_segments[f+1])
///   2: best_splits      - output: one float4 per feature
///   3: parent_grad       - total gradient sum for the node
///   4: parent_hess       - total hessian sum for the node
///   5: lambda            - L2 regularization parameter
///   6: gamma             - minimum split gain threshold
///   7: min_child_weight  - minimum hessian sum for a valid child
///   8: n_features        - number of features
///   9: reg_alpha         - L1 regularization parameter (optional, default 0)
///  10: max_delta_step    - max absolute leaf weight (optional, default 0)
kernel void evaluate_splits(
    device const float*   hist              [[buffer(0)]],
    device const uint*    feature_segments  [[buffer(1)]],
    device float4*        best_splits       [[buffer(2)]],
    constant float&       parent_grad       [[buffer(3)]],
    constant float&       parent_hess       [[buffer(4)]],
    constant float&       lambda            [[buffer(5)]],
    constant float&       gamma             [[buffer(6)]],
    constant float&       min_child_weight  [[buffer(7)]],
    constant uint&        n_features        [[buffer(8)]],
    constant float&       reg_alpha         [[buffer(9)]],
    constant float&       max_delta_step    [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_features) {
        return;
    }

    const uint fid = tid;
    const uint bin_start = feature_segments[fid];
    const uint bin_end   = feature_segments[fid + 1];
    const uint n_bins    = bin_end - bin_start;

    // Track best split for this feature
    float best_gain   = NEG_INF;
    uint  best_bin    = 0;

    // Running sums for left child (scan left-to-right)
    float left_grad = 0.0f;
    float left_hess = 0.0f;

    // Scan through bins, accumulating left child statistics.
    // We do NOT include the last bin in the scan because it would put
    // all data on the left side (no right child).
    for (uint b = 0; b < n_bins - 1; ++b) {
        const uint hist_idx = (bin_start + b) * 2;
        left_grad += hist[hist_idx + 0];
        left_hess += hist[hist_idx + 1];

        // Right child is the complement
        float right_grad = parent_grad - left_grad;
        float right_hess = parent_hess - left_hess;

        // Check min_child_weight constraint for both children
        if (left_hess < min_child_weight || right_hess < min_child_weight) {
            continue;
        }

        float gain = calc_split_gain(
            left_grad, left_hess,
            right_grad, right_hess,
            parent_grad, parent_hess,
            lambda, reg_alpha, gamma,
            min_child_weight, max_delta_step);

        if (gain > best_gain) {
            best_gain = gain;
            best_bin  = b;
        }
    }

    // Write result: (feature_id, bin_id, gain, 0)
    // If no valid split was found, gain remains NEG_INF.
    best_splits[fid] = float4(
        as_type<float>(fid),      // feature index (stored as float bits)
        as_type<float>(best_bin), // bin index (stored as float bits)
        best_gain,                // split gain
        0.0f                      // reserved / padding
    );
}

// ---------------------------------------------------------------------------
// Kernel: evaluate_splits_categorical
// ---------------------------------------------------------------------------

/// Split evaluation for categorical features using one-hot encoding.
/// Instead of accumulating left-to-right, each bin is independently
/// evaluated as a potential "go-left" category.
///
/// Output format is the same as evaluate_splits.
kernel void evaluate_splits_categorical(
    device const float*   hist              [[buffer(0)]],
    device const uint*    feature_segments  [[buffer(1)]],
    device float4*        best_splits       [[buffer(2)]],
    constant float&       parent_grad       [[buffer(3)]],
    constant float&       parent_hess       [[buffer(4)]],
    constant float&       lambda            [[buffer(5)]],
    constant float&       gamma             [[buffer(6)]],
    constant float&       min_child_weight  [[buffer(7)]],
    constant uint&        n_features        [[buffer(8)]],
    constant float&       reg_alpha         [[buffer(9)]],
    constant float&       max_delta_step    [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_features) {
        return;
    }

    const uint fid = tid;
    const uint bin_start = feature_segments[fid];
    const uint bin_end   = feature_segments[fid + 1];
    const uint n_bins    = bin_end - bin_start;

    float best_gain = NEG_INF;
    uint  best_bin  = 0;

    // For categorical features, each category (bin) is independently
    // considered as a potential "go-left" partition.
    for (uint b = 0; b < n_bins; ++b) {
        const uint hist_idx = (bin_start + b) * 2;
        float left_grad = hist[hist_idx + 0];
        float left_hess = hist[hist_idx + 1];

        float right_grad = parent_grad - left_grad;
        float right_hess = parent_hess - left_hess;

        if (left_hess < min_child_weight || right_hess < min_child_weight) {
            continue;
        }

        float gain = calc_split_gain(
            left_grad, left_hess,
            right_grad, right_hess,
            parent_grad, parent_hess,
            lambda, reg_alpha, gamma,
            min_child_weight, max_delta_step);

        if (gain > best_gain) {
            best_gain = gain;
            best_bin  = b;
        }
    }

    best_splits[fid] = float4(
        as_type<float>(fid),
        as_type<float>(best_bin),
        best_gain,
        0.0f
    );
}

// ---------------------------------------------------------------------------
// Kernel: find_best_split
// ---------------------------------------------------------------------------

/// Parallel reduction across per-feature best splits to find the single
/// globally best split for a node.
///
/// Input:  per_feature_splits[f] = float4(fid_bits, bin_bits, gain, 0)
/// Output: best_split[0]        = the float4 with the highest gain
///
/// Uses threadgroup shared memory for a standard parallel max-reduction.
kernel void find_best_split(
    device const float4*  per_feature_splits [[buffer(0)]],
    device float4*        best_split         [[buffer(1)]],
    constant uint&        n_features         [[buffer(2)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    threadgroup float4 shared_splits[SPLIT_EVAL_THREADGROUP_SIZE];

    // Each thread loads one candidate (or NEG_INF sentinel)
    float4 my_best = float4(0.0f, 0.0f, NEG_INF, 0.0f);

    // Stride over all features
    for (uint f = tid; f < n_features; f += tgs) {
        float4 candidate = per_feature_splits[f];
        if (candidate.z > my_best.z) {
            my_best = candidate;
        }
    }

    shared_splits[tid] = my_best;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float4 other = shared_splits[tid + stride];
            if (other.z > shared_splits[tid].z) {
                shared_splits[tid] = other;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the result
    if (tid == 0) {
        best_split[gid] = shared_splits[0];
    }
}

// ---------------------------------------------------------------------------
// Kernel: compute_leaf_weights
// ---------------------------------------------------------------------------

/// Compute optimal leaf weight for each node.
///   weight = -ThresholdL1(grad, alpha) / (hess + lambda)
///
/// Input:  node_sums[n] = float2(grad, hess)
/// Output: weights[n]   = float leaf weight
kernel void compute_leaf_weights(
    device const float2*  node_sums        [[buffer(0)]],
    device float*         weights          [[buffer(1)]],
    constant float&       lambda           [[buffer(2)]],
    constant float&       reg_alpha        [[buffer(3)]],
    constant float&       min_child_weight [[buffer(4)]],
    constant float&       max_delta_step   [[buffer(5)]],
    constant uint&        n_nodes          [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_nodes) {
        return;
    }

    float2 gh = node_sums[tid];
    weights[tid] = calc_weight(gh.x, gh.y, lambda, reg_alpha,
                               min_child_weight, max_delta_step);
}
