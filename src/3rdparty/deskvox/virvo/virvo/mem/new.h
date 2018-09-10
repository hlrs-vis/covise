#pragma once

#ifdef __GNUC__
#include <mm_malloc.h>
#else
#include <malloc.h>
#endif

#include "../vvmacros.h"

void* operator new(size_t size) throw(std::bad_alloc)
{
  if (size == 0)
  {
    // new needs to return a valid pointer
    size = 1;
  }

  void* ptr = _mm_malloc(size, CACHE_LINE);
  if (ptr != NULL)
  {
    return ptr;
  }
  else
  {
    throw std::bad_alloc();
  }
}

void* operator new[](size_t size) throw(std::bad_alloc)
{
  if (size == 0)
  {
    // new needs to return a valid pointer
    size = 1;
  }

  void* ptr = _mm_malloc(size, CACHE_LINE);
  if (ptr != NULL)
  {
    return ptr;
  }
  else
  {
    throw std::bad_alloc();
  }
}


void operator delete[](void* ptr) throw()
{
  _mm_free(ptr);
}

