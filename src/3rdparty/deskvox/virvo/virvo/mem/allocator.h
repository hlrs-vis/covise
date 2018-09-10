#pragma once

#include "../vvmacros.h"

#include <new>
#include <stddef.h>

#include <virvo/math/simd/intrinsics.h> // VV_ARCH

#if VV_CXX_GCC || VV_CXX_CLANG
#if VV_ARCH == VV_ARCH_ARM

// TODO:!!!

#include <cstdlib>

namespace virvo
{

inline void* _mm_malloc(size_t s, size_t aln)
{
    return aligned_alloc(aln, s);
}

inline void _mm_free(void* ptr)
{
    free(ptr);
}

}

#else
#include <mm_malloc.h>
#endif
#else
#include <malloc.h>
#endif


namespace virvo
{
namespace mem
{

template <typename T, size_t A>
class aligned_allocator
{
public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  aligned_allocator()
  {
  }

  aligned_allocator(const aligned_allocator& /* rhs */)
  {
  }

  template <typename U>
  aligned_allocator(const aligned_allocator<U, A>& /* rhs */)
  {
  }

  template <typename U>
  struct rebind
  {
    typedef aligned_allocator<U, A> other;
  };

  pointer address(reference r) const
  {
    return &r;
  }

  const_pointer address(const_reference r) const
  {
    return &r;
  }

  pointer allocate(size_type n, void* /*hint */ = 0)
  {
    return (pointer)_mm_malloc(n * sizeof(T), A);
  }

  void deallocate(pointer p, size_type /* n */)
  {
    _mm_free(p);
  }

  size_t max_size() const
  {
    return static_cast<size_t>(-1) / sizeof(T);
  }

  void construct(pointer p, const_reference val)
  {
    new(static_cast<void*>(p)) T(val);
  }

  void destroy(pointer p)
  {
    p->T::~T();
  }

  bool operator==(const aligned_allocator& /* rhs */) const
  {
    return true;
  }

  bool operator!=(const aligned_allocator& rhs) const
  {
    return !(*this == rhs);
  }
};

} // mem
} // virvo

