#ifndef FLUID_SYSTEM_KERNELS_CUH
#define FLUID_SYSTEM_KERNELS_CUH

#include "Config.h"
#include "K_Common.cuh"

#ifdef SPHSIMLIB_VEC_TYPE_FLOAT4
static __inline__ __host__ __device__ float_vec make_vec(float_vec v)
{
	return v;
}

static __inline__ __host__ __device__ float_vec make_vec(float v)
{
	float_vec t; t.x = v; t.y = v; t.z = v; t.w = 0; return t;
}
static __inline__ __host__ __device__ float_vec make_vec(float3 v)
{
  float_vec t; t.x = v.x; t.y = v.y; t.z = v.z; t.w = 0; return t;
}
static __inline__ __host__ __device__ float_vec make_vec(float x, float y, float z)
{
  float_vec t; t.x = x; t.y = y; t.z = z; t.w = 0; return t;
}
static __inline__ __host__ __device__ float_vec make_vec(float x, float y, float z, float w)
{
  float_vec t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}
#else
static __inline__ __host__ __device__ float_vec make_vec(float x, float y, float z)
{
  float_vec t; t.x = x; t.y = y; t.z = z; return t;
}
#endif



#ifdef SPHSIMLIB_USE_TEXTURES
#ifndef __DEVICE_EMULATION__
#define USE_TEX
#endif
#else
#undef USE_TEX 
#endif

#ifdef USE_TEX
#define FETCH(a, t, i) tex1Dfetch(t##_tex, i)
#define FETCH_NOTEX(a, t, i) a.t[i]
#define FETCH_FLOAT3(a,t,i) make_float3(FETCH(a,t,i))
#define FETCH_MATRIX3(a,t,i) tex1DfetchMatrix3(t##_tex,i)
#define FETCH_MATRIX3_NOTEX(a,t,i) a.t[i]
#else
#define FETCH(a, t, i) a.t[i]
#define FETCH_NOTEX(a, t, i) a.t[i]
#define FETCH_FLOAT3(a,t,i) make_float3(FETCH(a,t,i))
#define FETCH_MATRIX3(a,t,i) a.t[i]
#define FETCH_MATRIX3_NOTEX(a,t,i) a.t[i]
//#define FETCH(a, t, i) (a + __mul24(i,sizeof(a)) + (void*)offsetof(a, t))
#endif


static __inline__ __host__ __device__ matrix3 make_matrix3(float x1, float y1, float z1, 
														   float x2, float y2, float z2, 
														   float x3, float y3, float z3)
{
	matrix3 t; 
	t.r1 = make_vec(x1,y1,z1);
	t.r2 = make_vec(x2,y2,z2);
	t.r3 = make_vec(x3,y3,z3);
	return t;
}
static __inline__ __host__ __device__ matrix3 make_matrix3(float_vec r1, float_vec r2, float_vec r3)
{
	matrix3 t; 
	t.r1 = r1;
	t.r2 = r2;
	t.r3 = r3;
	return t;
}

#if defined(__cplusplus) && defined(__CUDACC__)

#include <device_functions.h>

static __inline__ __device__ matrix3 tex1DfetchMatrix3(texture<float4, 1, cudaReadModeElementType> t, int x)
{
	float4 r1 = tex1Dfetch(t, x*3);
	float4 r2 = tex1Dfetch(t, x*3+1);
	float4 r3 = tex1Dfetch(t, x*3+2);

	return make_matrix3(r1,r2,r3);
}

// static __inline__ __device__ matrix3 tex1Dfetch(texture<matrix3, 1, cudaReadModeElementType> t, int x)
// {
// 	float_vec r1 = tex1Dfetch(t, x*3);
// 	float_vec r2 = tex1Dfetch(t, x*3+1);
// 	float_vec r3 = tex1Dfetch(t, x*3+2);
// 
// 	return make_matrix3(r1,r2,r3);
// }

#endif

#include "cutil_math.h"

inline __host__ __device__ matrix3 operator*(float a, matrix3 b)
{
	return make_matrix3(a*b.r1, a*b.r2, a*b.r3);
}

inline __host__ __device__ matrix3 operator/(float a, matrix3 b)
{
	return make_matrix3(a/b.r1, a/b.r2, a/b.r3);
}

inline __host__ __device__ matrix3 operator/(matrix3 a, float b)
{
	return make_matrix3(a.r1/b, a.r2/b, a.r3/b);
}

inline __host__ __device__ matrix3 operator+(matrix3 a, matrix3 b)
{
	return make_matrix3(a.r1+b.r1, a.r2+b.r2, a.r3+b.r3);
}

inline __host__ __device__ matrix3 operator-(matrix3 a, matrix3 b)
{
	return make_matrix3(a.r1-b.r1, a.r2-b.r2, a.r3-b.r3);
}

inline __host__ __device__ void operator*=(matrix3 &a, float b)
{
	a.r1 *= b;
	a.r2 *= b;
	a.r3 *= b;
}

inline __host__ __device__ void operator+=(matrix3 &a, matrix3 b)
{
	a.r1 += b.r1;
	a.r2 += b.r2;
	a.r3 += b.r3;
}

inline __host__ __device__ matrix3 outer(float u1, float u2, float u3, float v1, float v2, float v3)
{ 
	return make_matrix3(
		u1*v1, u1*v2, u1*v3,
		u2*v1, u2*v2, u2*v3,
		u3*v1, u3*v2 ,u3*v3
		);
}
// outer product a*b' (column vector*column vector transposed) ==> (column*row vector)
inline __host__ __device__ matrix3 outer(float3 a, float3 b)
{ 
	return outer(a.x, a.y, a.z, 
				 b.x, b.y, b.z);
}

//dot product, b should be a column vector..
inline __host__ __device__ float3 dot(matrix3 a, float3 b)
{ 
	return make_float3(
		a.r1.x*b.x + a.r1.y*b.y + a.r1.z*b.z, 
		a.r2.x*b.x + a.r2.y*b.y + a.r2.z*b.z,
		a.r3.x*b.x + a.r3.y*b.y + a.r3.z*b.z);
}

// outer product
inline __host__ __device__ matrix3 transpose(matrix3 t)
{ 
	return make_matrix3(
		t.r1.x, t.r2.x, t.r3.x,
		t.r1.y, t.r2.y, t.r3.y,
		t.r1.z, t.r2.z, t.r3.z
		);
}

// trace
inline __host__ __device__ float trace(matrix3 t)
{
	return t.r1.x+t.r2.y+t.r3.z;
}

#endif