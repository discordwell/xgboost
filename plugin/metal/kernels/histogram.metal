/*!
 * Copyright 2024-2026, XGBoost Contributors
 * \file histogram.metal
 * \brief Metal compute kernel for gradient histogram construction.
 *
 * Uses threadgroup-local accumulation with CAS-loop atomic float adds,
 * then flushes to global output. This avoids the high cost of global
 * device atomics.
 */

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
// Threadgroup memory limit: 32KB -> max ~4K bins (4K * 8 bytes = 32KB).
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
        ulong row_id = row_indices[tid];
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

// CAS-loop atomic float add for device memory
inline void atomic_add_f_device(device atomic_uint* addr, float val) {
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

// Fallback histogram kernel for large bin counts (>4096) that don't fit
// in threadgroup memory. Uses global device atomics directly — slower but
// correct for any number of bins.
kernel void build_histogram_large(
    const device GradPair*  gpair        [[buffer(0)]],
    const device uint*      gmat_index   [[buffer(1)]],
    const device ulong*     row_indices  [[buffer(2)]],
    const device uint*      cut_ptrs     [[buffer(3)]],
    device float*           hist_out     [[buffer(4)]],
    constant uint&          num_rows     [[buffer(5)]],
    constant uint&          row_stride   [[buffer(6)]],
    constant uint&          num_features [[buffer(7)]],
    constant uint&          nbins        [[buffer(8)]],
    uint                    tid          [[thread_position_in_grid]])
{
    if (tid >= num_rows) return;

    ulong row_id = row_indices[tid];
    float g = gpair[row_id].grad;
    float h = gpair[row_id].hess;

    const device uint* row = gmat_index + row_id * row_stride;
    for (uint f = 0; f < num_features; ++f) {
        uint bin = row[f] + cut_ptrs[f];
        if (bin < nbins) {
            atomic_add_f_device((device atomic_uint*)&hist_out[2 * bin],     g);
            atomic_add_f_device((device atomic_uint*)&hist_out[2 * bin + 1], h);
        }
    }
}
