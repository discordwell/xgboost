/*!
 * Copyright 2024 by Contributors
 * \file device_manager.mm
 * \brief Metal device and command queue management — Objective-C++ implementation.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "device_manager.h"
#include "data.h"
#include "xgboost/logging.h"

namespace xgboost {
namespace metal {

// ---------------------------------------------------------------------------
// DeviceManager
// ---------------------------------------------------------------------------

void* DeviceManager::GetDevice() {
  static id<MTLDevice> device = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    device = MTLCreateSystemDefaultDevice();
    CHECK(device) << "Metal is not supported on this system.";
    LOG(INFO) << "Metal device: " << [[device name] UTF8String];
  });
  return (__bridge void*)device;
}

void* DeviceManager::GetQueue() {
  static id<MTLCommandQueue> queue = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    id<MTLDevice> device = (__bridge id<MTLDevice>)GetDevice();
    queue = [device newCommandQueue];
    CHECK(queue) << "Failed to create Metal command queue.";
  });
  return (__bridge void*)queue;
}

// ---------------------------------------------------------------------------
// Buffer allocation helpers (declared in data.h, implemented here)
// ---------------------------------------------------------------------------

void* MetalAllocateBuffer(void* device_ptr, size_t bytes, void** contents) {
  CHECK(device_ptr) << "MetalAllocateBuffer called with null device.";
  CHECK_GT(bytes, 0) << "MetalAllocateBuffer: size must be > 0.";

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
  id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
  CHECK(buffer) << "Failed to allocate Metal buffer of " << bytes << " bytes.";

  if (contents) {
    *contents = [buffer contents];
  }
  // Retain via CFBridgingRetain so the caller owns the reference.
  return (void*)CFBridgingRetain(buffer);
}

void MetalReleaseBuffer(void* buffer) {
  if (buffer) {
    // Transfer ownership back to ARC and let it release.
    (void)CFBridgingRelease(buffer);
  }
}

void* MetalGetDefaultDevice() {
  return DeviceManager::GetDevice();
}

}  // namespace metal
}  // namespace xgboost
