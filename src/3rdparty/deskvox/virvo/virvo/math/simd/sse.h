
#pragma once

#ifndef VV_SIMD_SSE_H
#define VV_SIMD_SSE_H

#include "intrinsics.h"
#include "../vector.h"

#include <virvo/vvmacros.h>

#include <stdexcept>

#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE2)

namespace MATH_NAMESPACE
{
namespace simd
{

class float4
{
public:

    typedef __m128 value_type;
    __m128 value;

    VV_FORCE_INLINE float4()
    {
    }

    VV_FORCE_INLINE float4(float x, float y, float z, float w)
        : value(_mm_set_ps(w, z, y, x))
    {
    }

    VV_FORCE_INLINE float4(float const v[4])
        : value(_mm_load_ps(v))
    {
    }

    VV_FORCE_INLINE float4(float s)
        : value(_mm_set1_ps(s))
    {
    }

    VV_FORCE_INLINE float4(__m128i const& i)
        : value(_mm_cvtepi32_ps(i))
    {
    }

    VV_FORCE_INLINE float4(__m128 const& v)
        : value(v)
    {
    }

    VV_FORCE_INLINE operator __m128() const
    {
        return value;
    }
};

class int4
{
public:

    typedef __m128i value_type;
    __m128i value;

    VV_FORCE_INLINE int4()
    {
    }

    VV_FORCE_INLINE int4(int x, int y, int z, int w)
        : value(_mm_set_epi32(w, z, y, x))
    {
    }

    VV_FORCE_INLINE int4(int const v[4])
        : value(_mm_load_si128(reinterpret_cast< __m128i const* >(v)))
    {
    }

    VV_FORCE_INLINE int4(int s)
        : value(_mm_set1_epi32(s))
    {
    }

    VV_FORCE_INLINE int4(unsigned s)
        : value(_mm_set1_epi32(s))
    {
    }

    VV_FORCE_INLINE int4(float4 const& f)
        : value(_mm_cvtps_epi32(f))
    {
    }

    VV_FORCE_INLINE int4(__m128i const& v)
        : value(v)
    {
    }

    VV_FORCE_INLINE operator __m128i() const
    {
        return value;
    }
};

union mask4
{
public:

    __m128  f;
    __m128i i;

    mask4()
    {
    }

    VV_FORCE_INLINE mask4(__m128 m)
        : f(m)
    {
    }

    VV_FORCE_INLINE mask4(__m128i m)
        : i(m)
    {
    }

    VV_FORCE_INLINE mask4(float4 const& m)
        : f(m)
    {
    }

    VV_FORCE_INLINE operator float4() const
    {
        return f;
    }

};


inline float4 reinterpret_as_float(int4 const& a)
{
    return _mm_castsi128_ps(a);
}

inline int4 reinterpret_as_int(float4 const& a)
{
    return _mm_castps_si128(a);
}

VV_FORCE_INLINE float4 select(mask4 const& m, float4 const& a, float4 const& b)
{
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE4_1)
    return _mm_blendv_ps(b, a, m.f);
#else
    return _mm_or_ps(_mm_and_ps(m.f, a), _mm_andnot_ps(m.f, b));
#endif
}

VV_FORCE_INLINE int4 select(mask4 const& m, int4 const& a, int4 const& b)
{
    return reinterpret_as_int( select(m, reinterpret_as_float(a), reinterpret_as_float(b)) );
}

//-------------------------------------------------------------------------------------------------
// float4
//

VV_FORCE_INLINE void store(float dst[4], float4 const& v)
{
  _mm_store_ps(dst, v);
}

template <int U0, int U1, int V2, int V3>
VV_FORCE_INLINE float4 shuffle(float4 const& u, float4 const& v)
{
  return _mm_shuffle_ps(u, v, _MM_SHUFFLE(V3, V2, U1, U0));
}

template <int V0, int V1, int V2, int V3>
VV_FORCE_INLINE float4 shuffle(float4 const& v)
{
  return _mm_shuffle_ps(v, v, _MM_SHUFFLE(V3, V2, V1, V0));
}


/* operators */

VV_FORCE_INLINE float4 operator-(float4 const& v)
{
  return _mm_sub_ps(_mm_setzero_ps(), v);
}

VV_FORCE_INLINE float4 operator+(float4 const& u, float4 const& v)
{
  return _mm_add_ps(u, v);
}

VV_FORCE_INLINE float4 operator-(float4 const& u, float4 const& v)
{
  return _mm_sub_ps(u, v);
}

VV_FORCE_INLINE float4 operator*(float4 const& u, float4 const& v)
{
  return _mm_mul_ps(u, v);
}

VV_FORCE_INLINE float4 operator/(float4 const& u, float4 const& v)
{
  return _mm_div_ps(u, v);
}

VV_FORCE_INLINE float4& operator+=(float4& u, float4 const& v)
{
  u = u + v;
  return u;
}

VV_FORCE_INLINE float4& operator-=(float4& u, float4 const& v)
{
  u = u - v;
  return u;
}

VV_FORCE_INLINE float4& operator*=(float4& u, float4 const& v)
{
  u = u * v;
  return u;
}

VV_FORCE_INLINE float4& operator/=(float4& u, float4 const& v)
{
  u = u / v;
  return u;
}



/* vector math functions */

/*! \brief  returns a vector with each element {x|y|z|w} containing
 the result of the dot product
 */
VV_FORCE_INLINE float4 dot(float4 const& u, float4 const& v)
{
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE4_1)
  return _mm_dp_ps(u, v, 0xFF);
#else
  __m128 t1 = _mm_mul_ps(u, v);
  __m128 t2 = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2,3,0,1));
  __m128 t3 = _mm_add_ps(t1, t2);
  __m128 t4 = _mm_shuffle_ps(t3, t3, _MM_SHUFFLE(0,1,2,3));
  __m128 t5 = _mm_add_ps(t3, t4);

  return t5;
#endif
}


/*! \brief  Newton Raphson refinement
 */

template <unsigned N>
VV_FORCE_INLINE float4 rcp_step(float4 const& v)
{
  float4 t = v;

  for (unsigned i = 0; i < N; ++i)
  {
    t = (t + t) - (v * t * t);
  }

  return t;
}

template <unsigned N>
VV_FORCE_INLINE float4 rcp(float4 const& v)
{
  float4 x0 = _mm_rcp_ps(v);
  return rcp_step<N>(x0);
}

VV_FORCE_INLINE float4 rcp(float4 const& v)
{
  float4 x0 = _mm_rcp_ps(v);
  return rcp_step<1>(x0);
}

template <unsigned N>
VV_FORCE_INLINE float4 rsqrt_step(float4 const& v, float4 const& x0)
{
  float4 threehalf(1.5f);
  float4 vhalf = v * float4(0.5f);
  float4 t = x0;

  for (unsigned i = 0; i < N; ++i)
  {
    t = t * (threehalf - vhalf * t * t);
  }

  return t;
}

template <unsigned N>
VV_FORCE_INLINE float4 rsqrt(float4 const& v)
{
  float4 x0 = _mm_rsqrt_ps(v);
  return rsqrt_step<N>(v, x0);
}

VV_FORCE_INLINE float4 rsqrt(float4 const& v)
{
  float4 x0 = _mm_rsqrt_ps(v);
  return rsqrt_step<1>(v, x0);
}

// TODO: find a better place for this
VV_FORCE_INLINE vector< 4, float4 > transpose(vector< 4, float4 > const& v)
{

    float4 tmp0 = _mm_unpacklo_ps(v.x, v.y);
    float4 tmp1 = _mm_unpacklo_ps(v.z, v.w);
    float4 tmp2 = _mm_unpackhi_ps(v.x, v.y);
    float4 tmp3 = _mm_unpackhi_ps(v.z, v.w);

    return vector< 4, float4 >
    (
        _mm_movelh_ps(tmp0, tmp1),
        _mm_movehl_ps(tmp1, tmp0),
        _mm_movelh_ps(tmp2, tmp3),
        _mm_movehl_ps(tmp3, tmp2)
    );

}


//-------------------------------------------------------------------------------------------------
// int4
//

VV_FORCE_INLINE void store(int dst[4], int4 const& v)
{
  _mm_store_si128(reinterpret_cast<__m128i*>(dst), v);
}

template <int A0, int A1, int A2, int A3>
VV_FORCE_INLINE int4 shuffle(int4 const& a)
{
  return _mm_shuffle_epi32(a, _MM_SHUFFLE(A3, A2, A1, A0));
}

/* operators */

VV_FORCE_INLINE int4 operator-(int4 const& v)
{
  return _mm_sub_epi32(_mm_setzero_si128(), v);
}

VV_FORCE_INLINE int4 operator+(int4 const& u, int4 const& v)
{
  return _mm_add_epi32(u, v);
}

VV_FORCE_INLINE int4 operator-(int4 const& u, int4 const& v)
{
  return _mm_sub_epi32(u, v);
}

VV_FORCE_INLINE int4 operator*(int4 const& u, int4 const& v)
{
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE4_1)
  return _mm_mullo_epi32(u, v);
#else
  __m128i t0 = shuffle<1,0,3,0>(u);             // a1  ... a3  ...
  __m128i t1 = shuffle<1,0,3,0>(v);             // b1  ... b3  ...
  __m128i t2 = _mm_mul_epu32(u, v);             // ab0 ... ab2 ...
  __m128i t3 = _mm_mul_epu32(t0, t1);           // ab1 ... ab3 ...
  __m128i t4 = _mm_unpacklo_epi32(t2, t3);      // ab0 ab1 ... ...
  __m128i t5 = _mm_unpackhi_epi32(t2, t3);      // ab2 ab3 ... ...
  __m128i t6 = _mm_unpacklo_epi64(t4, t5);      // ab0 ab1 ab2 ab3

  return t6;
#endif
}

VV_FORCE_INLINE int4& operator+=(int4& u, int4 const& v)
{
  u = u + v;
  return u;
}

VV_FORCE_INLINE int4& operator-=(int4& u, int4 const& v)
{
  u = u - v;
  return u;
}

VV_FORCE_INLINE int4& operator*=(int4& u, int4 const& v)
{
  u = u * v;
  return u;
}

VV_FORCE_INLINE int4 operator&(int4 const& u, int4 const& v)
{
    return _mm_and_si128(u, v);
}

VV_FORCE_INLINE int4 operator|(int4 const& u, int4 const& v)
{
    return _mm_or_si128(u, v);
}

VV_FORCE_INLINE int4 operator^(int4 const& u, int4 const& v)
{
    return _mm_xor_si128(u, v);
}

VV_FORCE_INLINE int4 operator<<(int4 const& a, int count)
{
    return _mm_slli_epi32(a, count);
}

VV_FORCE_INLINE int4 operator>>(int4 const& a, int count)
{
    return _mm_srli_epi32(a, count);
}

VV_FORCE_INLINE int4& operator&=(int4& u, int4 const& v)
{
    u = u & v;
    return u;
}

VV_FORCE_INLINE int4& operator|=(int4& u, int4 const& v)
{
    u = u | v;
    return u;
}

VV_FORCE_INLINE int4& operator^=(int4& u, int4 const& v)
{
    u = u ^ v;
    return u;
}

VV_FORCE_INLINE int4& operator<<=(int4& a, int count)
{
    a << count;
    return a;
}

VV_FORCE_INLINE int4& operator>>=(int4& a, int count)
{
    a >> count;
    return a;
}

VV_FORCE_INLINE mask4 operator<(int4 const& u, int4 const& v)
{
  return _mm_cmplt_epi32(u, v);
}

VV_FORCE_INLINE mask4 operator>(int4 const& u, int4 const& v)
{
  return _mm_cmpgt_epi32(u, v);
}

VV_FORCE_INLINE mask4 operator<=(int4 const& u, int4 const& v)
{
  return _mm_or_si128(_mm_cmplt_epi32(u, v), _mm_cmpeq_epi32(u, v));
}

VV_FORCE_INLINE mask4 operator>=(int4 const& u, int4 const& v)
{
  return _mm_or_si128(_mm_cmpgt_epi32(u, v), _mm_cmpeq_epi32(u, v));
}

VV_FORCE_INLINE mask4 operator==(int4 const& u, int4 const& v)
{
  return _mm_cmpeq_epi32(u, v);
}

VV_FORCE_INLINE mask4 operator&&(int4 const& u, int4 const& v)
{
  return _mm_and_si128(u, v);
}


/* function analogs for cstdlib */

VV_FORCE_INLINE int4 min(int4 const& u, int4 const& v)
{
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE4_1)
  return _mm_min_epi32(u, v);
#else
  return select(mask4(u < v), u, v);
#endif
}

VV_FORCE_INLINE int4 max(int4 const& u, int4 const& v)
{
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE4_1)
  return _mm_max_epi32(u, v);
#else
  return select(mask4(u > v), u, v);
#endif
}


//-------------------------------------------------------------------------------------------------
// mask4
//

/* float4 */

VV_FORCE_INLINE mask4 operator<(float4 const& u, float4 const& v)
{
  return _mm_cmplt_ps(u, v);
}

VV_FORCE_INLINE mask4 operator>(float4 const& u, float4 const& v)
{
  return _mm_cmpgt_ps(u, v);
}

VV_FORCE_INLINE mask4 operator<=(float4 const& u, float4 const& v)
{
  return _mm_cmple_ps(u, v);
}

VV_FORCE_INLINE mask4 operator>=(float4 const& u, float4 const& v)
{
  return _mm_cmpge_ps(u, v);
}

VV_FORCE_INLINE mask4 operator==(float4 const& u, float4 const& v)
{
  return _mm_cmpeq_ps(u, v);
}

VV_FORCE_INLINE mask4 operator!=(float4 const& u, float4 const& v)
{
  return _mm_cmpneq_ps(u, v);
}

VV_FORCE_INLINE mask4 operator&&(float4 const& u, float4 const& v)
{
  return _mm_and_ps(u, v);
}

VV_FORCE_INLINE mask4 operator||(float4 const& u, float4 const& v)
{
  return _mm_and_ps(u, v);
}


VV_FORCE_INLINE bool any(mask4 const& m)
{
    return _mm_movemask_ps(m.f) != 0;
}

VV_FORCE_INLINE bool all(mask4 const& m)
{
    return _mm_movemask_ps(m.f) == 0xF;
}

VV_FORCE_INLINE mask4 operator!(mask4 const& a)
{
    return _mm_xor_si128(a.i, _mm_cmpeq_epi32(a.i, a.i));
}

VV_FORCE_INLINE mask4 operator&(mask4 const& a, mask4 const& b)
{
    return _mm_and_si128(a.i, b.i);
}

VV_FORCE_INLINE mask4 operator|(mask4 const& a, mask4 const& b)
{
    return _mm_or_si128(a.i, b.i);
}

VV_FORCE_INLINE mask4 operator^(mask4 const& a, mask4 const& b)
{
    return _mm_xor_si128(a.i, b.i);
}

VV_FORCE_INLINE mask4& operator&=(mask4& a, mask4 const& b)
{
    a = a & b;
    return a;
}

VV_FORCE_INLINE mask4& operator|=(mask4& a, mask4 const& b)
{
    a = a | b;
    return a;
}

VV_FORCE_INLINE mask4& operator^=(mask4& a, mask4 const& b)
{
    a = a ^ b;
    return a;
}

VV_FORCE_INLINE float4 neg(float4 const& v, mask4 const& mask)
{
    return select(mask, -v, 0.0f);
}

VV_FORCE_INLINE float4 add(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, u + v, 0.0f);
}

VV_FORCE_INLINE float sub(float u, float v, float /* mask */)
{
  return u - v;
}

VV_FORCE_INLINE float4 sub(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, u - v, 0.0f);
}

VV_FORCE_INLINE float4 mul(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, u * v, 0.0f);
}

VV_FORCE_INLINE float4 div(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, u / v, 0.0f);
}

VV_FORCE_INLINE float4 lt(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u < v).f, float4(0.0f));
}

VV_FORCE_INLINE float4 gt(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u > v).f, float4(0.0f));
}

VV_FORCE_INLINE float4 le(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u <= v).f, float4(0.0f));
}

VV_FORCE_INLINE float4 ge(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u >= v).f, float4(0.0f));
}

VV_FORCE_INLINE float4 eq(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u == v).f, float4(0.0f));
}

VV_FORCE_INLINE float4 neq(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u != v).f, float4(0.0f));
}

VV_FORCE_INLINE int4 add(int4 const& u, int4 const& v, mask4 const& mask)
{
    return select(mask, u + v, 0);
}

VV_FORCE_INLINE int4 sub(int4 const& u, int4 const& v, mask4 const& mask)
{
    return select(mask, u - v, 0);
}

VV_FORCE_INLINE int4 mul(int4 const& u, int4 const& v, mask4 const& mask)
{
    return select(mask, u * v, 0);
}

template < typename S, typename T >
VV_FORCE_INLINE void store(S dst[4], T const& v, mask4 const& mask)
{
    T old(dst);
    store( dst, select(mask, v, old) );
}

template < typename S, typename T >
VV_FORCE_INLINE void store(S dst[4], T const& v, mask4 const& mask, T const& old)
{
    store( dst, select(mask, v, old) );
}

VV_FORCE_INLINE float4 sqrt(float4 const& v)
{
    return _mm_sqrt_ps(v);
}

VV_FORCE_INLINE float4 approx_rsqrt(float4 const& v)
{
    return _mm_rsqrt_ps(v);
}


/* Vec3 */

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > neg(vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( neg(v.x, mask), neg(v.y, mask), neg(v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > add(vector< 3, T > const& u, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( add(u.x, v.x, mask), add(u.y, v.y, mask), add(u.z, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > sub(vector< 3, T > const& u, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( sub(u.x, v.x, mask), sub(u.y, v.y, mask), sub(u.z, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > mul(vector< 3, T > const& u, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( mul(u.x, v.x, mask), mul(u.y, v.y, mask), mul(u.z, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > div(vector< 3, T > const& u, vector< 3, T > const& v, M const& mask)
{
    return vector<3, T >( div(u.x, v.x, mask), div(u.y, v.y, mask), div(u.z, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > add(vector< 3, T > const& v, T const& s, M const& mask)
{
    return vector< 3, T >( add(v.x, s, mask), add(v.y, s, mask), add(v.z, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > sub(vector< 3, T > const& v, T const& s, M const& mask)
{
    return vector< 3, T >( sub(v.x, s, mask), sub(v.y, s, mask), sub(v.z, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > mul(vector< 3, T > const& v, T const& s, M const& mask)
{
    return vector< 3, T >( mul(v.x, s, mask), mul(v.y, s, mask), mul(v.z, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > div(vector< 3, T > const& v, T const& s, M const& mask)
{
    return vector< 3, T >( div(v.x, s, mask), div(v.y, s, mask), div(v.z, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > add(T const& s, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( add(s, v.x, mask), add(s, v.y, mask), add(s, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > sub(T const& s, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( sub(s, v.x, mask), sub(s, v.y, mask), sub(s, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > mul(T const& s, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( mul(s, v.x, mask), mul(s, v.y, mask), sub(s, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > div(T const& s, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( div(s, v.x, mask), div(s, v.y, mask), div(s, v.z, mask) );
}


/* Vec4 */

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > neg(vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( neg(v.x, mask), neg(v.y, mask), neg(v.z, mask), neg(v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > add(vector< 4, T > const& u, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( add(u.x, v.x, mask), add(u.y, v.y, mask), add(u.z, v.z, mask), add(u.w, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > sub(vector< 4, T > const& u, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( sub(u.x, v.x, mask), sub(u.y, v.y, mask), sub(u.z, v.z, mask), sub(u.w, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > mul(vector< 4, T > const& u, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( mul(u.x, v.x, mask), mul(u.y, v.y, mask), mul(u.z, v.z, mask), mul(u.w, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > div(vector< 4, T > const& u, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( div(u.x, v.x, mask), div(u.y, v.y, mask), div(u.z, v.z, mask), div(u.w, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > add(vector< 4, T > const& v, T const& s, M const& mask)
{
    return vector< 4, T >( add(v.x, s, mask), add(v.y, s, mask), add(v.z, s, mask), add(v.w, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > sub(vector< 4, T > const& v, T const& s, M const& mask)
{
    return vector< 4, T >( sub(v.x, s, mask), sub(v.y, s, mask), sub(v.z, s, mask), sub(v.w, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > mul(vector< 4, T > const& v, T const& s, M const& mask)
{
    return vector< 4, T >( mul(v.x, s, mask), mul(v.y, s, mask), mul(v.z, s, mask), mul(v.w, s, mask) );
}

VV_FORCE_INLINE vector< 4, float > mul(vector< 4, float > const& v, float s, float /* mask */)
{
    return v * s;
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > div(vector< 4, T > const& v, T const& s, M const& mask)
{
    return vector< 4, T >( div(v.x, s, mask), div(v.y, s, mask), div(v.z, s, mask), div(v.w, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > add(T const& s, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( add(s, v.x, mask), add(s, v.y, mask), add(s, v.z, mask), add(s, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > sub(T const& s, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( sub(s, v.x, mask), sub(s, v.y, mask), sub(s, v.z, mask), sub(s, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > mul(T const& s, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( mul(s, v.x, mask), mul(s, v.y, mask), mul(s, v.z, mask), mul(s, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > div(T const& s, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( div(s, v.x, mask), div(s, v.y, mask), div(s, v.z, mask), div(s, v.w, mask) );
}


//-------------------------------------------------------------------------------------------------
// cstdlib-like functions
//

VV_FORCE_INLINE float4 min(float4 const& u, float4 const& v)
{
  return _mm_min_ps(u, v);
}

VV_FORCE_INLINE float4 max(float4 const& u, float4 const& v)
{
  return _mm_max_ps(u, v);
}

VV_FORCE_INLINE float4 round(float4 const& v)
{
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE4_1)
  return _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT);
#else
  // Mask out the signbits of v
  __m128 s = _mm_and_ps(v, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));
  // Magic number: 2^23 with the signbits of v
  __m128 m = _mm_or_ps(s, _mm_castsi128_ps(_mm_set1_epi32(0x4B000000)));
  __m128 x = _mm_add_ps(v, m);
  __m128 y = _mm_sub_ps(x, m);

  return y;
#endif
}

VV_FORCE_INLINE float4 ceil(float4 const& v)
{
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE4_1)
  return _mm_ceil_ps(v);
#else
  // i = trunc(v)
  __m128 i = _mm_cvtepi32_ps(_mm_cvttps_epi32(v));
  // r = i < v ? i i + 1 : i
  __m128 t = _mm_cmplt_ps(i, v);
  __m128 d = _mm_cvtepi32_ps(_mm_castps_si128(t)); // mask to float: 0 -> 0.0f, 0xFFFFFFFF -> -1.0f
  __m128 r = _mm_sub_ps(i, d);

  return r;
#endif
}

VV_FORCE_INLINE float4 floor(float4 const& v)
{
#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE4_1)
  return _mm_floor_ps(v);
#else
  // i = trunc(v)
  __m128 i = _mm_cvtepi32_ps(_mm_cvttps_epi32(v));
  // r = i > v ? i - 1 : i
  __m128 t = _mm_cmpgt_ps(i, v);
  __m128 d = _mm_cvtepi32_ps(_mm_castps_si128(t)); // mask to float: 0 -> 0.0f, 0xFFFFFFFF -> -1.0f
  __m128 r = _mm_add_ps(i, d);

  return r;
#endif
}

} // simd
} // MATH_NAMESPACE

#endif // VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE2)

#endif // VV_SIMD_SSE_H
