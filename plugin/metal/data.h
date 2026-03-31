/*!
 * Copyright 2024 by Contributors
 * \file data.h
 * \brief Metal buffer and vector types for unified memory on Apple Silicon.
 *
 * This header is included from both .cc and .mm translation units. All Metal
 * object handles are stored as opaque void* so that Objective-C types do not
 * leak into pure C++ code.
 */
#ifndef PLUGIN_METAL_DATA_H_
#define PLUGIN_METAL_DATA_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/logging.h"

namespace xgboost {
namespace metal {

// ---------------------------------------------------------------------------
// Opaque helpers implemented in device_manager.mm
// ---------------------------------------------------------------------------

// Allocate a MTLBuffer with StorageModeShared.  Returns the raw MTLBuffer
// pointer (retained) and writes the CPU-visible base address to *contents.
void* MetalAllocateBuffer(void* device, size_t bytes, void** contents);

// Release (CFRelease) a MTLBuffer previously returned by MetalAllocateBuffer.
void MetalReleaseBuffer(void* buffer);

// Obtain the default Metal device (void* wrapping id<MTLDevice>).
void* MetalGetDefaultDevice();

// ---------------------------------------------------------------------------
// MetalBuffer<T> -- thin wrapper around a MTLBuffer (StorageModeShared)
// ---------------------------------------------------------------------------

template <typename T>
class MetalBuffer {
  static_assert(std::is_standard_layout<T>::value,
                "MetalBuffer admits only standard-layout types");

 public:
  MetalBuffer() : size_(0), capacity_(0), data_(nullptr), buffer_(nullptr) {}

  // Allocate a buffer that can hold `n` elements of type T.
  explicit MetalBuffer(size_t n)
      : size_(n), capacity_(n), data_(nullptr), buffer_(nullptr) {
    if (n > 0) {
      void* contents = nullptr;
      buffer_ = MetalAllocateBuffer(MetalGetDefaultDevice(),
                                    n * sizeof(T), &contents);
      data_ = static_cast<T*>(contents);
    }
  }

  // Construct from a host std::vector, copying data into the Metal buffer.
  explicit MetalBuffer(const std::vector<T>& vec)
      : size_(vec.size()), capacity_(vec.size()), data_(nullptr),
        buffer_(nullptr) {
    if (size_ > 0) {
      void* contents = nullptr;
      buffer_ = MetalAllocateBuffer(MetalGetDefaultDevice(),
                                    size_ * sizeof(T), &contents);
      data_ = static_cast<T*>(contents);
      std::memcpy(data_, vec.data(), size_ * sizeof(T));
    }
  }

  ~MetalBuffer() {
    if (buffer_) {
      MetalReleaseBuffer(buffer_);
    }
  }

  // Copy constructor -- deep copy.
  MetalBuffer(const MetalBuffer& other)
      : size_(other.size_), capacity_(other.size_), data_(nullptr),
        buffer_(nullptr) {
    if (size_ > 0) {
      void* contents = nullptr;
      buffer_ = MetalAllocateBuffer(MetalGetDefaultDevice(),
                                    size_ * sizeof(T), &contents);
      data_ = static_cast<T*>(contents);
      std::memcpy(data_, other.data_, size_ * sizeof(T));
    }
  }

  MetalBuffer& operator=(const MetalBuffer& other) {
    if (this != &other) {
      MetalBuffer tmp(other);
      Swap(tmp);
    }
    return *this;
  }

  // Move constructor.
  MetalBuffer(MetalBuffer&& other) noexcept
      : size_(other.size_), capacity_(other.capacity_), data_(other.data_),
        buffer_(other.buffer_) {
    other.size_ = 0;
    other.capacity_ = 0;
    other.data_ = nullptr;
    other.buffer_ = nullptr;
  }

  MetalBuffer& operator=(MetalBuffer&& other) noexcept {
    if (this != &other) {
      if (buffer_) {
        MetalReleaseBuffer(buffer_);
      }
      size_ = other.size_;
      capacity_ = other.capacity_;
      data_ = other.data_;
      buffer_ = other.buffer_;
      other.size_ = 0;
      other.capacity_ = 0;
      other.data_ = nullptr;
      other.buffer_ = nullptr;
    }
    return *this;
  }

  T* Data() { return data_; }
  const T* Data() const { return data_; }
  const T* DataConst() const { return data_; }

  size_t Size() const { return size_; }
  size_t Capacity() const { return capacity_; }
  bool Empty() const { return size_ == 0; }

  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }

  T* Begin() const { return data_; }
  T* End() const { return data_ + size_; }

  // Reallocate if the new size exceeds current capacity.
  // Existing data is preserved up to min(old_size, new_size).
  void Resize(size_t n) {
    if (n <= capacity_) {
      size_ = n;
      return;
    }
    void* contents = nullptr;
    void* new_buf = MetalAllocateBuffer(MetalGetDefaultDevice(),
                                        n * sizeof(T), &contents);
    T* new_data = static_cast<T*>(contents);
    if (size_ > 0 && data_) {
      std::memcpy(new_data, data_, size_ * sizeof(T));
    }
    if (buffer_) {
      MetalReleaseBuffer(buffer_);
    }
    buffer_ = new_buf;
    data_ = new_data;
    capacity_ = n;
    size_ = n;
  }

  // Resize without preserving existing data.
  void ResizeNoCopy(size_t n) {
    if (n <= capacity_) {
      size_ = n;
      return;
    }
    if (buffer_) {
      MetalReleaseBuffer(buffer_);
      buffer_ = nullptr;
      data_ = nullptr;
    }
    void* contents = nullptr;
    buffer_ = MetalAllocateBuffer(MetalGetDefaultDevice(),
                                  n * sizeof(T), &contents);
    data_ = static_cast<T*>(contents);
    capacity_ = n;
    size_ = n;
  }

  // Fill all elements with a given value.
  void Fill(T val) {
    if (size_ == 0 || !data_) return;
    std::fill(data_, data_ + size_, val);
  }

  void Clear() {
    if (buffer_) {
      MetalReleaseBuffer(buffer_);
    }
    buffer_ = nullptr;
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  // Return the underlying MTLBuffer handle (void*).
  void* GetMTLBuffer() { return buffer_; }
  const void* GetMTLBuffer() const { return buffer_; }

 private:
  void Swap(MetalBuffer& other) noexcept {
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
    std::swap(data_, other.data_);
    std::swap(buffer_, other.buffer_);
  }

  size_t size_;
  size_t capacity_;
  T* data_;
  void* buffer_;  // Opaque id<MTLBuffer>
};

// ---------------------------------------------------------------------------
// MetalVector<T> -- convenient std::vector-like wrapper backed by MetalBuffer
// ---------------------------------------------------------------------------

template <typename T>
class MetalVector {
  static_assert(std::is_standard_layout<T>::value,
                "MetalVector admits only standard-layout types");

 public:
  using value_type = T;  // NOLINT

  MetalVector() = default;

  explicit MetalVector(size_t n) : buf_(n) {}

  MetalVector(size_t n, T val) : buf_(n) { buf_.Fill(val); }

  explicit MetalVector(const std::vector<T>& vec) : buf_(vec) {}

  T* Data() { return buf_.Data(); }
  const T* Data() const { return buf_.Data(); }
  const T* DataConst() const { return buf_.DataConst(); }

  size_t Size() const { return buf_.Size(); }
  size_t Capacity() const { return buf_.Capacity(); }
  bool Empty() const { return buf_.Empty(); }

  T& operator[](size_t i) { return buf_[i]; }
  const T& operator[](size_t i) const { return buf_[i]; }

  T* Begin() const { return buf_.Begin(); }
  T* End() const { return buf_.End(); }

  void Resize(size_t n) { buf_.Resize(n); }
  void Resize(size_t n, T val) {
    size_t old_size = buf_.Size();
    buf_.Resize(n);
    if (n > old_size) {
      std::fill(buf_.Data() + old_size, buf_.Data() + n, val);
    }
  }
  void ResizeNoCopy(size_t n) { buf_.ResizeNoCopy(n); }
  void Fill(T val) { buf_.Fill(val); }
  void Clear() { buf_.Clear(); }

  void Init(const std::vector<T>& vec) {
    buf_ = MetalBuffer<T>(vec);
  }

  void* GetMTLBuffer() { return buf_.GetMTLBuffer(); }
  const void* GetMTLBuffer() const { return buf_.GetMTLBuffer(); }

 private:
  MetalBuffer<T> buf_;
};

}  // namespace metal
}  // namespace xgboost

#endif  // PLUGIN_METAL_DATA_H_
