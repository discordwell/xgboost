/*!
 * Copyright 2024-2026 by XGBoost Contributors
 * \file histogram.metal
 * \brief Metal compute kernel for gradient histogram construction.
 *
 * For each row assigned to a tree node, loads the gradient pair (grad, hess),
 * looks up the quantized bin index for each feature in the assigned feature
 * group, and atomically accumulates grad/hess into the histogram bin.
 *
 * Design:
 *   - One threadgroup per (node, feature_group) pair.
 *   - Threadgroup memory for local histogram accumulation to minimize
 *     global memory contention.
 *   - CAS-loop float atomic add (no native float atomics in threadgroup memory).
 *   - Final flush from threadgroup histogram to global device histogram.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum bins supported in threadgroup memory.  With 2 floats (grad, hess)
/// per bin stored as atomic_uint, this is 2 * 1024 * 4 = 8 KB, well within
/// Metal's 32 KB threadgroup memory limit.
constant constexpr uint MAX_LOCAL_BINS = 1024;

// ---------------------------------------------------------------------------
// CAS-loop float atomic add into threadgroup memory
// ---------------------------------------------------------------------------

/// Atomically add \p val to the float stored at \p addr in threadgroup memory.
/// Metal does not provide native float atomics for threadgroup address space,
/// so we use a compare-and-swap loop over the raw uint bit pattern.
///
/// The unrolled fast-path (14 attempts) covers the common case where
/// contention is low and the CAS succeeds quickly; the trailing do-while
/// handles the rare high-contention case.
inline void atomic_add_float(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint next;
    float current;

    // Unrolled fast path -- avoids the branch overhead of the full loop
    // for the first 14 attempts, which is sufficient for typical contention
    // levels on Apple GPUs (32-wide SIMD groups).
    for (int attempt = 0; attempt < 14; ++attempt) {
        current = as_type<float>(expected);
        next = as_type<uint>(current + val);
        if (atomic_compare_exchange_weak_explicit(
                addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }

    // Full loop for extremely high contention
    do {
        current = as_type<float>(expected);
        next = as_type<uint>(current + val);
    } while (!atomic_compare_exchange_weak_explicit(
                addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed));
}

// ---------------------------------------------------------------------------
// CAS-loop float atomic add into device memory
// ---------------------------------------------------------------------------

/// Same as above, but targeting device address space for the global histogram
/// flush and for use when threadgroup memory is insufficient.
inline void atomic_add_float_device(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint next;
    float current;

    for (int attempt = 0; attempt < 14; ++attempt) {
        current = as_type<float>(expected);
        next = as_type<uint>(current + val);
        if (atomic_compare_exchange_weak_explicit(
                addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }

    do {
        current = as_type<float>(expected);
        next = as_type<uint>(current + val);
    } while (!atomic_compare_exchange_weak_explicit(
                addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed));
}

// ---------------------------------------------------------------------------
// Kernel: clear_histogram
// ---------------------------------------------------------------------------

/// Zero-initialize a region of the histogram buffer.
/// Dispatched as 1D grid with enough threads to cover n_values floats.
kernel void clear_histogram(
    device float*   hist       [[buffer(0)]],   // histogram buffer to clear
    constant uint&  n_values   [[buffer(1)]],   // total floats to zero
    uint tid [[thread_position_in_grid]])
{
    if (tid < n_values) {
        hist[tid] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Kernel: build_histogram
// ---------------------------------------------------------------------------

/// Build gradient/hessian histogram for a single (node, feature_group) pair.
///
/// Memory layout of the output histogram (interleaved grad/hess per bin):
///   hist[2*bin + 0] = sum of gradients  for bin
///   hist[2*bin + 1] = sum of hessians   for bin
///
/// The kernel is dispatched with one threadgroup.  Each thread iterates over
/// a strided subset of the node's rows, accumulates into the threadgroup-local
/// histogram, then flushes the local histogram to global memory.
///
/// Buffer bindings:
///   0: gpair        - gradient pairs for all rows, packed as float2(grad, hess)
///   1: gidx         - quantized feature bin indices, row-major [n_total_rows x row_stride]
///   2: ridx         - row indices belonging to the current node
///   3: hist         - output histogram in device memory (grad0, hess0, grad1, hess1, ...)
///   4: n_rows       - number of rows in ridx
///   5: row_stride   - number of features per row in the gidx matrix
///   6: n_bins_start - first bin index for this feature group (offset into gidx column space)
///   7: n_bins       - number of bins in this feature group
kernel void build_histogram(
    device const float2*   gpair        [[buffer(0)]],
    device const uint8_t*  gidx         [[buffer(1)]],
    device const ulong*    ridx         [[buffer(2)]],
    device float*          hist         [[buffer(3)]],
    constant uint&         n_rows       [[buffer(4)]],
    constant uint&         row_stride   [[buffer(5)]],
    constant uint&         n_bins_start [[buffer(6)]],
    constant uint&         n_bins       [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    // -----------------------------------------------------------------------
    // Allocate threadgroup histogram: 2 atomic_uint per bin (grad, hess).
    // MAX_LOCAL_BINS * 2 * sizeof(atomic_uint) = up to 8 KB.
    // -----------------------------------------------------------------------
    threadgroup atomic_uint local_hist[MAX_LOCAL_BINS * 2];

    // -----------------------------------------------------------------------
    // Phase 0: Zero the threadgroup histogram
    // -----------------------------------------------------------------------
    const uint hist_size = n_bins * 2;  // total entries (grad + hess interleaved)
    for (uint i = tid; i < hist_size; i += tgs) {
        atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Phase 1: Accumulate gradient pairs into local histogram
    //
    // Each thread processes rows in stride of threadgroup size.  For each row,
    // it iterates over every feature column in the feature group [n_bins_start,
    // n_bins_start + n_bins), reads the quantized bin index from gidx, and
    // atomically adds the row's grad and hess to the corresponding bin.
    // -----------------------------------------------------------------------
    const uint feat_start = n_bins_start;
    const uint feat_end   = n_bins_start + n_bins;

    for (uint r = tid; r < n_rows; r += tgs) {
        const uint row_id = ridx[r];
        const float2 gh = gpair[row_id];
        const float grad = gh.x;
        const float hess = gh.y;

        // Pointer to the start of this row's quantized indices
        device const uint8_t* row_gidx = gidx + (uint)row_id * row_stride;

        // Iterate over features in this feature group
        for (uint f = feat_start; f < feat_end; ++f) {
            const uint bin = (uint)row_gidx[f];

            // Bounds check: bin must be < n_bins (the number of bins in this
            // feature group).  The bin index from gidx is relative to the
            // feature's own bin range, so it's already in [0, n_bins).
            if (bin < n_bins) {
                atomic_add_float(&local_hist[2 * bin + 0], grad);
                atomic_add_float(&local_hist[2 * bin + 1], hess);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Phase 2: Flush threadgroup histogram to global memory
    //
    // When multiple threadgroups target the same histogram (multiple nodes
    // mapped to the same feature group), we use device atomics for the flush.
    // For single-threadgroup dispatch (gid == 0 and only one threadgroup),
    // a simple store suffices, but we use atomics uniformly for correctness.
    // -----------------------------------------------------------------------
    for (uint i = tid; i < hist_size; i += tgs) {
        float val = as_type<float>(
            atomic_load_explicit(&local_hist[i], memory_order_relaxed));
        atomic_add_float_device(
            (device atomic_uint*)&hist[i], val);
    }
}

// ---------------------------------------------------------------------------
// Kernel: build_histogram_dense
// ---------------------------------------------------------------------------

/// Optimized histogram kernel for dense data with feature-offset remapping.
///
/// Unlike build_histogram which iterates over a feature group's columns,
/// this kernel handles the common dense case where gidx stores per-feature
/// local bin indices and a separate offsets array maps (feature -> global bin).
///
/// Buffer bindings:
///   0: gpair       - gradient pairs for all rows, packed as float2(grad, hess)
///   1: gidx        - quantized feature bin indices, row-major [n_total_rows x n_features]
///   2: ridx        - row indices belonging to the current node
///   3: hist        - output histogram in device memory (grad0, hess0, grad1, hess1, ...)
///   4: offsets     - per-feature bin offset: global_bin = offsets[f] + gidx[row, f]
///   5: n_rows      - number of rows in ridx
///   6: n_features  - number of features (columns in gidx)
///   7: n_total_bins - total number of bins across all features
kernel void build_histogram_dense(
    device const float2*   gpair       [[buffer(0)]],
    device const uint8_t*  gidx        [[buffer(1)]],
    device const ulong*    ridx        [[buffer(2)]],
    device float*          hist        [[buffer(3)]],
    device const uint*     offsets     [[buffer(4)]],
    constant uint&         n_rows      [[buffer(5)]],
    constant uint&         n_features  [[buffer(6)]],
    constant uint&         n_total_bins [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    // -----------------------------------------------------------------------
    // Threadgroup histogram
    // -----------------------------------------------------------------------
    threadgroup atomic_uint local_hist[MAX_LOCAL_BINS * 2];

    // Clamp to available local memory
    const uint n_bins_clamped = min(n_total_bins, MAX_LOCAL_BINS);
    const uint hist_size = n_bins_clamped * 2;

    // Zero local histogram
    for (uint i = tid; i < hist_size; i += tgs) {
        atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Accumulate
    // -----------------------------------------------------------------------
    for (uint r = tid; r < n_rows; r += tgs) {
        const uint row_id = ridx[r];
        const float2 gh = gpair[row_id];
        const float grad = gh.x;
        const float hess = gh.y;

        device const uint8_t* row_gidx = gidx + (uint)row_id * n_features;

        for (uint f = 0; f < n_features; ++f) {
            const uint local_bin = (uint)row_gidx[f];
            const uint global_bin = offsets[f] + local_bin;

            if (global_bin < n_bins_clamped) {
                atomic_add_float(&local_hist[2 * global_bin + 0], grad);
                atomic_add_float(&local_hist[2 * global_bin + 1], hess);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Flush to global
    // -----------------------------------------------------------------------
    for (uint i = tid; i < hist_size; i += tgs) {
        float val = as_type<float>(
            atomic_load_explicit(&local_hist[i], memory_order_relaxed));
        atomic_add_float_device(
            (device atomic_uint*)&hist[i], val);
    }
}

// ---------------------------------------------------------------------------
// Kernel: build_histogram_uint16
// ---------------------------------------------------------------------------

/// Same as build_histogram but with uint16_t bin indices for datasets
/// with more than 256 bins per feature.
kernel void build_histogram_uint16(
    device const float2*    gpair        [[buffer(0)]],
    device const uint16_t*  gidx         [[buffer(1)]],
    device const uint*      ridx         [[buffer(2)]],
    device float*           hist         [[buffer(3)]],
    constant uint&          n_rows       [[buffer(4)]],
    constant uint&          row_stride   [[buffer(5)]],
    constant uint&          n_bins_start [[buffer(6)]],
    constant uint&          n_bins       [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint local_hist[MAX_LOCAL_BINS * 2];

    const uint hist_size = n_bins * 2;
    for (uint i = tid; i < hist_size; i += tgs) {
        atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint feat_start = n_bins_start;
    const uint feat_end   = n_bins_start + n_bins;

    for (uint r = tid; r < n_rows; r += tgs) {
        const uint row_id = ridx[r];
        const float2 gh = gpair[row_id];
        const float grad = gh.x;
        const float hess = gh.y;

        device const uint16_t* row_gidx = gidx + (uint)row_id * row_stride;

        for (uint f = feat_start; f < feat_end; ++f) {
            const uint bin = (uint)row_gidx[f];
            if (bin < n_bins) {
                atomic_add_float(&local_hist[2 * bin + 0], grad);
                atomic_add_float(&local_hist[2 * bin + 1], hess);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < hist_size; i += tgs) {
        float val = as_type<float>(
            atomic_load_explicit(&local_hist[i], memory_order_relaxed));
        atomic_add_float_device(
            (device atomic_uint*)&hist[i], val);
    }
}

// ---------------------------------------------------------------------------
// Kernel: subtract_histogram
// ---------------------------------------------------------------------------

/// Compute the sibling histogram via the subtraction trick:
///   dst[i] = parent[i] - sibling[i]
///
/// This avoids building the histogram for the larger child node, which is
/// the key optimization in the histogram-based tree builder.
kernel void subtract_histogram(
    device const float*  parent   [[buffer(0)]],
    device const float*  sibling  [[buffer(1)]],
    device float*        dst      [[buffer(2)]],
    constant uint&       n_values [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n_values) {
        dst[tid] = parent[tid] - sibling[tid];
    }
}

// ---------------------------------------------------------------------------
// Kernel: reduce_histograms
// ---------------------------------------------------------------------------

/// Reduce multiple sub-histograms (from multi-threadgroup dispatch) into
/// a single histogram.  Each sub-histogram has n_bins * 2 floats.
///
///   output[i] = sum over block j of sub_hists[j * stride + i]
kernel void reduce_histograms(
    device const float*  sub_hists  [[buffer(0)]],
    device float*        output     [[buffer(1)]],
    constant uint&       n_values   [[buffer(2)]],  // n_bins * 2
    constant uint&       n_blocks   [[buffer(3)]],  // number of sub-histograms
    uint tid [[thread_position_in_grid]])
{
    if (tid < n_values) {
        float sum = 0.0f;
        for (uint b = 0; b < n_blocks; ++b) {
            sum += sub_hists[b * n_values + tid];
        }
        output[tid] = sum;
    }
}
