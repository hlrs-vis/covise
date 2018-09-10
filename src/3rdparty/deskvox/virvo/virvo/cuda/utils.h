// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_CUDAUTILS_H
#define VV_CUDAUTILS_H


#include <cuda_runtime.h>
 
#include "../vvexport.h"


namespace virvo
{
namespace cuda
{


// Returns the first block size for which the given kernel, specified by <attr>,
// can be executed on the current device.
VVAPI bool findConfig(cudaFuncAttributes const& attr, dim3 const*& begin, dim3 const* end, size_t dynamicSharedMem = 0);


// Returns the first block size for which the given kernel
// can be executed on the current device.
template<class T>
bool findConfig(T* func, dim3 const*& begin, dim3 const* end, size_t dynamicSharedMem = 0)
{
  cudaFuncAttributes attr;

#if CUDART_VERSION < 5000
  if (cudaSuccess == cudaFuncGetAttributes(&attr, (const char*)func))
#else
  if (cudaSuccess == cudaFuncGetAttributes(&attr, (const void*)func))
#endif
  {
    return findConfig(attr, begin, end, dynamicSharedMem);
  }

  return false;
}


// Returns whether the given kernel, can be executed on the current
// device using the given launch configuration.
template<class T>
bool isLaunchable(T* func, dim3 const& blockDim, size_t dynamicSharedMem = 0)
{
  dim3 const* begin = &blockDim;
  dim3 const* end   = &blockDim + 1;

  return findConfig(func, begin, end, dynamicSharedMem);
}


VVAPI bool checkError(bool *success, cudaError_t err, const char *msg = NULL, bool syncIfDebug = true);
VVAPI int deviceCount();
VVAPI bool init();
VVAPI bool initGlInterop();


} // namespace cuda
} // namespace virvo



#ifndef __CUDACC__

#include <cmath>

inline float rsqrtf(const float x)
{
  return 1.0f / sqrtf(x);
}

#ifdef _MSC_VER
inline float fminf(float x, float y)
{
  return x < y ? x : y;
}

inline float fmaxf(float x, float y)
{
  return x > y ? x : y;
}
#endif

#endif


inline __device__ __host__ float3 operator-(const float3& v)
{
  return make_float3(-v.x, -v.y, -v.z);
}

inline __device__ __host__ float3 operator+(const float3& v1, const float3& v2)
{
  return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline __device__ __host__ float4 operator+(const float4& v1, const float4& v2)
{
  return make_float4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

inline __device__ __host__ uchar4 operator+(const uchar4& v1, const uchar4& v2)
{
  return make_uchar4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

inline __device__ __host__ float3 operator-(const float3& v1, const float3& v2)
{
  return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline __device__ __host__ float4 operator-(const float4& v1, const float4& v2)
{
  return make_float4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}

inline __device__ __host__ float3 operator*(const float3& v1, const float3& v2)
{
  return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

inline __device__ __host__ float3 operator/(const float3& v1, const float3& v2)
{
  return make_float3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

inline __device__ __host__ float3 operator+(const float3& v, const float f)
{
  return make_float3(v.x + f, v.y + f, v.z + f);
}

inline __device__ __host__ float3 operator-(const float3& v, const float f)
{
  return make_float3(v.x - f, v.y - f, v.z - f);
}

inline __device__ __host__ float3 operator*(const float3& v, const float f)
{
  return make_float3(v.x * f, v.y * f, v.z * f);
}

inline __device__ __host__ float4 operator*(const float4& v, const float f)
{
  return make_float4(v.x * f, v.y * f, v.z * f, v.w * f);
}

inline __device__ __host__ float3 operator/(const float3& v, const float f)
{
  return v * (1.0f / f);
}

inline __device__ __host__ float3 operator*(const float3& v, const float m[16])
{
  float3 result = make_float3(m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * 1.0f,
                              m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7] * 1.0f,
                              m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11] * 1.0f);
  const float w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15] * 1.0f;

  if (w != 1.0f)
  {
    const float wInv = 1.0f / w;
    result.x *= wInv;
    result.y *= wInv;
    result.z *= wInv;
  }
  return result;
}

inline __device__ __host__ float3 make_float3(const float v)
{
  return make_float3(v, v, v);
}

inline __device__ __host__ float3 make_float3(const float4& v)
{
  return make_float3(v.x, v.y, v.z);
}

inline __device__ __host__ float4 make_float4(const float v)
{
  return make_float4(v, v, v, v);
}

inline __device__ __host__ float4 make_float4(const float3& v)
{
  return make_float4(v.x, v.y, v.z, 1.0f);
}

inline __device__ __host__ uchar4 make_uchar4(const unsigned char v)
{
  return make_uchar4(v, v, v, v);
}

inline __device__ __host__ void operator+=(float3& v, const float f)
{
  v.x += f;
  v.y += f;
  v.z += f;
}

inline __device__ __host__ void operator-=(float3& v, const float f)
{
  v.x -= f;
  v.y -= f;
  v.z -= f;
}

inline __device__ __host__ void operator*=(float3& v, const float f)
{
  v.x *= f;
  v.y *= f;
  v.z *= f;
}

inline __device__ __host__ void operator*=(float4& v, const float f)
{
  v.x *= f;
  v.y *= f;
  v.z *= f;
  v.w *= f;
}

inline __device__ __host__ void operator/=(float3& v, const float f)
{
  const float fInv = 1.0f / f;
  v.x *= fInv;
  v.y *= fInv;
  v.z *= fInv;
}

inline __device__ __host__ void operator+=(float3& v1, const float3& v2)
{
  v1.x += v2.x;
  v1.y += v2.y;
  v1.z += v2.z;
}

inline __device__ __host__ void operator-=(float3& v1, const float3& v2)
{
  v1.x -= v2.x;
  v1.y -= v2.y;
  v1.z -= v2.z;
}

inline __device__ __host__ void operator*=(float3& v1, const float3& v2)
{
  v1.x *= v2.x;
  v1.y *= v2.y;
  v1.z *= v2.z;
}

inline __device__ __host__ void operator/=(float3& v1, const float3& v2)
{
  v1.x /= v2.x;
  v1.y /= v2.y;
  v1.z /= v2.z;
}

inline __device__ __host__ void operator*=(float3& v, const float m[16])
{
  v = v * m;
}

inline __device__ __host__ float clamp(float f, float a = 0.0f, float b = 1.0f)
{
    return fmaxf(a, fminf(f, b));
}

inline __device__ __host__ float3 clamp(float3 v, float a = 0.0f, float b = 1.0f)
{
    return make_float3
    (
        clamp(v.x, a, b),
        clamp(v.y, a, b),
        clamp(v.z, a, b)
    );
}

inline __device__ __host__ float3 cross(const float3& v1, const float3& v2)
{
  return make_float3(v1.y * v2.z - v1.z * v2.y,
                     v1.z * v2.x - v1.x * v2.z,
                     v1.x * v2.y - v1.y * v2.x);
}

inline __device__ __host__ float dot(const float3& v1, const float3& v2)
{
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __device__ __host__ float dot(const float4& v1, const float4& v2)
{
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

inline __device__ __host__ float norm(const float3& v)
{
  return sqrtf(dot(v, v));
}

inline __device__ __host__ float3 normalize(const float3& v)
{
  const float lengthInv = rsqrtf(dot(v, v));
  return v * lengthInv;
}

inline __device__ __host__ void invert(float3& v)
{
  v = make_float3(1.0f / v.x, 1.0f / v.y, 1.0f / v.z);
}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
