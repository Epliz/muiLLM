#ifndef __MUILLM_KARRAY_HPP__
#define __MUILLM_KARRAY_HPP__

#include <cuda_fp16.h>

template<typename T, unsigned N>
struct __attribute__((packed)) karray {
  T data[N];

  static inline karray<T, N> __device__ load(const T* ptr) {
    return *((const karray<T, N>*)ptr);
  }

  static inline void __device__ store(T* ptr, const karray<T, N>& v) {
    for (unsigned i = 0; i < N; i++) {
      ((karray<T, N>*)ptr)->data[i] = v.data[i];
    }
  }

  inline karray<T, N>& __device__ operator*=(const karray<T, N>& s) {
    for (unsigned i = 0; i < N; i++) {
      data[i] *= s.data[i];
    }
    return *this;
  }

  inline const T& __device__ operator[](size_t i) const {
    return data[i];
  }

  inline T& __device__ operator[](size_t i) {
    return data[i];
  }
};

template<unsigned N>
struct __attribute__((packed)) karray<half, N> {
  half data[N];

  static inline karray<half, N> __device__ load(const half* ptr) {
    return *((const karray<half, N>*)ptr);
  }

  static inline void __device__ store(half* ptr, const karray<half, N>& v) {
    for (unsigned i = 0; i < N; i++) {
      ((karray<half, N>*)ptr)->data[i] = v.data[i];
    }
  }

  inline karray<half, N>& __device__ operator*=(const karray<half, N>& s) {
    for (unsigned i = 0; i < N; i++) {
      data[i] = __hmul(data[i], s.data[i]);
    }
    return *this;
  }

  inline const half& __device__ operator[](size_t i) const {
    return data[i];
  }

  inline half& __device__ operator[](size_t i) {
    return data[i];
  }
};

#endif // __MUILLM_KARRAY_HPP__