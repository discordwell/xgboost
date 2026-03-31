/*!
 * Copyright 2024 by Contributors
 * \file device_manager.h
 * \brief Metal device and command queue management for Apple Silicon GPU.
 *
 * All Metal handles are exposed as void* so this header can be included from
 * pure C++ translation units (.cc) without requiring Objective-C.
 */
#ifndef PLUGIN_METAL_DEVICE_MANAGER_H_
#define PLUGIN_METAL_DEVICE_MANAGER_H_

namespace xgboost {
namespace metal {

class DeviceManager {
 public:
  /// Return the system default MTLDevice (as void*).
  /// The device is created on first call and cached for the process lifetime.
  static void* GetDevice();

  /// Return a MTLCommandQueue (as void*) associated with the default device.
  /// The queue is created on first call and cached for the process lifetime.
  static void* GetQueue();
};

}  // namespace metal
}  // namespace xgboost

#endif  // PLUGIN_METAL_DEVICE_MANAGER_H_
