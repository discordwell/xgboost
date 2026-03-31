/*!
 * Copyright 2024-2026 by XGBoost Contributors
 * \file partition.metal
 * \brief Metal compute kernels for row partitioning after a split decision.
 *
 * After the best split is determined for a tree node, rows must be
 * partitioned into left and right children.  We use a two-pass approach:
 *
 *   Pass 1 (partition_flags):
 *     For each row, evaluate whether it goes left (flag=1) or right (flag=0)
 *     based on its quantized bin index vs. the split threshold.
 *
 *   Between passes (host-side or prefix_sum_scan kernel):
 *     Compute an exclusive prefix sum of the flags array.  This gives each
 *     left-going row its destination index.  The total sum gives n_left.
 *
 *   Pass 2 (partition_scatter):
 *     Using the prefix sum offsets, scatter row indices into separate
 *     left and right output arrays.
 *
 * An optional single-pass atomic variant (partition_atomic) is also provided
 * for small node sizes where atomic contention is tolerable.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Threadgroup size for the prefix sum scan kernel.
constant constexpr uint SCAN_THREADGROUP_SIZE = 256;

// ---------------------------------------------------------------------------
// Kernel: partition_flags
// ---------------------------------------------------------------------------

/// Pass 1: Compute a go-left flag for each row in the node.
///
/// For each row, loads the quantized bin index for the split feature and
/// compares it against the split threshold.  Rows with bin <= split_bin
/// go left (flag = 1); others go right (flag = 0).
///
/// This handles the default-left case for missing values: if the row's bin
/// index equals the special missing-value sentinel (0xFF for uint8, 0xFFFF
/// for uint16), the default_left flag determines the direction.
///
/// Buffer bindings:
///   0: gidx          - quantized feature bin indices, row-major
///   1: ridx          - row indices belonging to the current node
///   2: flags         - output: 1 = left, 0 = right (one per row)
///   3: split_feature - index of the feature used for splitting
///   4: split_bin     - bin threshold: left if bin_value <= split_bin
///   5: row_stride    - features per row in gidx
///   6: n_rows        - number of rows in ridx
///   7: default_left  - 1 if missing values go left, 0 otherwise
///   8: missing_bin   - sentinel value for missing data (e.g. 0xFF)
kernel void partition_flags(
    device const uint8_t*  gidx          [[buffer(0)]],
    device const uint*     ridx          [[buffer(1)]],
    device uint*           flags         [[buffer(2)]],
    constant uint&         split_feature [[buffer(3)]],
    constant uint&         split_bin     [[buffer(4)]],
    constant uint&         row_stride    [[buffer(5)]],
    constant uint&         n_rows        [[buffer(6)]],
    constant uint&         default_left  [[buffer(7)]],
    constant uint&         missing_bin   [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_rows) {
        return;
    }

    const uint row_id = ridx[tid];
    const uint8_t bin_value = gidx[row_id * row_stride + split_feature];

    uint go_left;
    if ((uint)bin_value == missing_bin) {
        // Missing value: use default direction
        go_left = default_left;
    } else {
        // Normal comparison: left if bin_value <= split_bin
        go_left = ((uint)bin_value <= split_bin) ? 1u : 0u;
    }

    flags[tid] = go_left;
}

// ---------------------------------------------------------------------------
// Kernel: partition_flags_uint16
// ---------------------------------------------------------------------------

/// Same as partition_flags but for uint16_t bin indices.
kernel void partition_flags_uint16(
    device const uint16_t* gidx          [[buffer(0)]],
    device const uint*     ridx          [[buffer(1)]],
    device uint*           flags         [[buffer(2)]],
    constant uint&         split_feature [[buffer(3)]],
    constant uint&         split_bin     [[buffer(4)]],
    constant uint&         row_stride    [[buffer(5)]],
    constant uint&         n_rows        [[buffer(6)]],
    constant uint&         default_left  [[buffer(7)]],
    constant uint&         missing_bin   [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_rows) {
        return;
    }

    const uint row_id = ridx[tid];
    const uint16_t bin_value = gidx[row_id * row_stride + split_feature];

    uint go_left;
    if ((uint)bin_value == missing_bin) {
        go_left = default_left;
    } else {
        go_left = ((uint)bin_value <= split_bin) ? 1u : 0u;
    }

    flags[tid] = go_left;
}

// ---------------------------------------------------------------------------
// Kernel: prefix_sum_scan
// ---------------------------------------------------------------------------

/// Compute an exclusive prefix sum of uint flags within threadgroup-sized
/// blocks.  For inputs larger than one threadgroup, a multi-level scan
/// is needed: this kernel produces per-block partial sums in block_sums,
/// and the host performs a second-level scan + offset propagation.
///
/// Uses the Blelloch (work-efficient) parallel scan algorithm.
///
/// Buffer bindings:
///   0: input       - input flags (uint, 0 or 1)
///   1: output      - exclusive prefix sum output
///   2: block_sums  - per-block total sums (for multi-block scan)
///   3: n_elements  - total number of elements
kernel void prefix_sum_scan(
    device const uint*  input       [[buffer(0)]],
    device uint*        output      [[buffer(1)]],
    device uint*        block_sums  [[buffer(2)]],
    constant uint&      n_elements  [[buffer(3)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    // Each threadgroup handles 2 * tgs elements (work-efficient scan)
    const uint block_size = 2 * tgs;
    const uint block_offset = gid * block_size;

    // Shared memory for the scan
    threadgroup uint shared_data[SCAN_THREADGROUP_SIZE * 2];

    // Load input into shared memory (two elements per thread)
    uint ai = tid;
    uint bi = tid + tgs;
    uint global_ai = block_offset + ai;
    uint global_bi = block_offset + bi;

    shared_data[ai] = (global_ai < n_elements) ? input[global_ai] : 0u;
    shared_data[bi] = (global_bi < n_elements) ? input[global_bi] : 0u;

    // Up-sweep (reduce) phase
    uint offset = 1;
    for (uint d = block_size >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint left  = offset * (2 * tid + 1) - 1;
            uint right = offset * (2 * tid + 2) - 1;
            shared_data[right] += shared_data[left];
        }
        offset <<= 1;
    }

    // Save block sum and clear the last element for down-sweep
    if (tid == 0) {
        if (block_sums) {
            block_sums[gid] = shared_data[block_size - 1];
        }
        shared_data[block_size - 1] = 0;
    }

    // Down-sweep phase
    for (uint d = 1; d < block_size; d <<= 1) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint left  = offset * (2 * tid + 1) - 1;
            uint right = offset * (2 * tid + 2) - 1;
            uint temp = shared_data[left];
            shared_data[left]  = shared_data[right];
            shared_data[right] += temp;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output
    if (global_ai < n_elements) output[global_ai] = shared_data[ai];
    if (global_bi < n_elements) output[global_bi] = shared_data[bi];
}

// ---------------------------------------------------------------------------
// Kernel: prefix_sum_add_block_offset
// ---------------------------------------------------------------------------

/// After prefix_sum_scan produces per-block prefix sums, and the host
/// computes a prefix sum of the block_sums, this kernel adds the
/// per-block offset to each element to produce the final global prefix sum.
kernel void prefix_sum_add_block_offset(
    device uint*        data        [[buffer(0)]],
    device const uint*  offsets     [[buffer(1)]],
    constant uint&      n_elements  [[buffer(2)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    const uint block_size = 2 * tgs;
    const uint block_offset_idx = gid * block_size;
    const uint offset_val = offsets[gid];

    uint ai = block_offset_idx + tid;
    uint bi = block_offset_idx + tid + tgs;

    if (ai < n_elements) data[ai] += offset_val;
    if (bi < n_elements) data[bi] += offset_val;
}

// ---------------------------------------------------------------------------
// Kernel: partition_scatter
// ---------------------------------------------------------------------------

/// Pass 2: Scatter row indices into left and right arrays using prefix
/// sum offsets computed in the previous step.
///
/// For each row i:
///   if flags[i] == 1: ridx_left[prefix_sum[i]]           = ridx_in[i]
///   if flags[i] == 0: ridx_right[i - prefix_sum[i] - ... ] = ridx_in[i]
///
/// The right-side index is computed as:
///   right_offset = (i - prefix_sum[i])   // number of right-going rows before i
///
/// Buffer bindings:
///   0: ridx_in     - input row indices (current node)
///   1: flags       - go-left flags (1 = left, 0 = right)
///   2: prefix_sum  - exclusive prefix sum of flags
///   3: ridx_left   - output: left child row indices
///   4: ridx_right  - output: right child row indices
///   5: n_rows      - number of rows in ridx_in
///   6: n_left      - total number of left-going rows (sum of all flags)
kernel void partition_scatter(
    device const uint*  ridx_in     [[buffer(0)]],
    device const uint*  flags       [[buffer(1)]],
    device const uint*  prefix_sum  [[buffer(2)]],
    device uint*        ridx_left   [[buffer(3)]],
    device uint*        ridx_right  [[buffer(4)]],
    constant uint&      n_rows      [[buffer(5)]],
    constant uint&      n_left      [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_rows) {
        return;
    }

    const uint row_id = ridx_in[tid];
    const uint flag   = flags[tid];
    const uint psum   = prefix_sum[tid];

    if (flag == 1u) {
        // Left child: destination index is the prefix sum value
        ridx_left[psum] = row_id;
    } else {
        // Right child: destination index = (position among right-going rows)
        // tid - psum gives the number of right-going rows at or before this position
        // (since psum counts left-going rows before tid, tid - psum counts right-going)
        const uint right_idx = tid - psum;
        ridx_right[right_idx] = row_id;
    }
}

// ---------------------------------------------------------------------------
// Kernel: partition_atomic
// ---------------------------------------------------------------------------

/// Single-pass atomic partition for small node sizes.
///
/// Instead of the two-pass prefix-sum approach, this kernel uses two
/// atomic counters to directly scatter rows.  This is simpler but incurs
/// atomic contention, so it's only efficient for small nodes (< ~10K rows).
///
/// Rows going left are placed at the front of ridx_out, and rows going
/// right are placed at the back (growing toward the front).
///
/// Buffer bindings:
///   0: gidx          - quantized feature bin indices
///   1: ridx_in       - input row indices
///   2: ridx_out      - output buffer (left from front, right from back)
///   3: counters      - [n_left_atomic, n_right_atomic] (must be zeroed first)
///   4: split_feature - feature index
///   5: split_bin     - split threshold
///   6: row_stride    - features per row in gidx
///   7: n_rows        - number of rows
///   8: default_left  - missing value direction
///   9: missing_bin   - sentinel value for missing data
kernel void partition_atomic(
    device const uint8_t*  gidx          [[buffer(0)]],
    device const uint*     ridx_in       [[buffer(1)]],
    device uint*           ridx_out      [[buffer(2)]],
    device atomic_uint*    counters      [[buffer(3)]],
    constant uint&         split_feature [[buffer(4)]],
    constant uint&         split_bin     [[buffer(5)]],
    constant uint&         row_stride    [[buffer(6)]],
    constant uint&         n_rows        [[buffer(7)]],
    constant uint&         default_left  [[buffer(8)]],
    constant uint&         missing_bin   [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_rows) {
        return;
    }

    const uint row_id = ridx_in[tid];
    const uint8_t bin_value = gidx[row_id * row_stride + split_feature];

    bool go_left;
    if ((uint)bin_value == missing_bin) {
        go_left = (default_left != 0u);
    } else {
        go_left = ((uint)bin_value <= split_bin);
    }

    if (go_left) {
        // Place at front, growing forward
        uint pos = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
        ridx_out[pos] = row_id;
    } else {
        // Place at back, growing backward
        uint pos = atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
        ridx_out[n_rows - 1 - pos] = row_id;
    }
}

// ---------------------------------------------------------------------------
// Kernel: partition_atomic_uint16
// ---------------------------------------------------------------------------

/// Single-pass atomic partition for uint16_t bin indices.
kernel void partition_atomic_uint16(
    device const uint16_t* gidx          [[buffer(0)]],
    device const uint*     ridx_in       [[buffer(1)]],
    device uint*           ridx_out      [[buffer(2)]],
    device atomic_uint*    counters      [[buffer(3)]],
    constant uint&         split_feature [[buffer(4)]],
    constant uint&         split_bin     [[buffer(5)]],
    constant uint&         row_stride    [[buffer(6)]],
    constant uint&         n_rows        [[buffer(7)]],
    constant uint&         default_left  [[buffer(8)]],
    constant uint&         missing_bin   [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_rows) {
        return;
    }

    const uint row_id = ridx_in[tid];
    const uint16_t bin_value = gidx[row_id * row_stride + split_feature];

    bool go_left;
    if ((uint)bin_value == missing_bin) {
        go_left = (default_left != 0u);
    } else {
        go_left = ((uint)bin_value <= split_bin);
    }

    if (go_left) {
        uint pos = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
        ridx_out[pos] = row_id;
    } else {
        uint pos = atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
        ridx_out[n_rows - 1 - pos] = row_id;
    }
}
