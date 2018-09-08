#ifndef VV_SAMPLER_COMMON_H
#define VV_SAMPLER_COMMON_H


#include <virvo/vvmacros.h>

#include "math/math.h"


namespace virvo
{


template < typename T, typename FloatT >
VV_FORCE_INLINE T lerp(const T &a, const T &b, FloatT x)
{
    return a + x * (b - a);
}


namespace detail
{


template < typename T >
VV_FORCE_INLINE T point(T const* tex, ssize_t idx)
{
    return tex[idx];
}


template < typename T >
VV_FORCE_INLINE simd::float4 point(T const* tex, const simd::float4 &idx)
{

    VV_ALIGN(16) int indices[4];
    store(&indices[0], idx);
    return simd::float4
    (
        tex[indices[0]],
        tex[indices[1]],
        tex[indices[2]],
        tex[indices[3]]
    );

}


VV_FORCE_INLINE vector< 4, simd::float4 > point(vector< 4, float > const* tex, const simd::float4 &idx)
{

    // Special case: colors are AoS. Those can be obtained
    // without a context switch to GP registers by transposing
    // to SoA after memory lookup.

    simd::float4 iidx( idx * 4 );
    VV_ALIGN(16) int indices[4];
    store(&indices[0], iidx);

    float const* tmp = reinterpret_cast< float const* >(tex);

    vector< 4, simd::float4 > colors
    (
        &tmp[0] + indices[0],
        &tmp[0] + indices[1],
        &tmp[0] + indices[2],
        &tmp[0] + indices[3]
    );

    colors = transpose(colors);
    return colors;

}


namespace bspline
{

// weight functions for Mitchell - Netravalli B-Spline with B = 1 and C = 0

template < typename FloatT >
struct w0_func
{
    VV_FORCE_INLINE FloatT operator()( FloatT a )
    {
        return FloatT( (1.0 / 6.0) * (-(a * a * a) + 3.0 * a * a - 3.0 * a + 1.0) );
    }
};

template < typename FloatT >
struct w1_func
{
    VV_FORCE_INLINE FloatT operator()( FloatT a )
    {
        return FloatT( (1.0 / 6.0) * (3.0 * a * a * a - 6.0 * a * a + 4.0) );
    }
};

template < typename FloatT >
struct w2_func
{
    VV_FORCE_INLINE FloatT operator()( FloatT a )
    {
        return FloatT( (1.0 / 6.0) * (-3.0 * a * a * a + 3.0 * a * a + 3.0 * a + 1.0) );
    }
};

template < typename FloatT >
struct w3_func
{
    VV_FORCE_INLINE FloatT operator()( FloatT a )
    {
        return FloatT( (1.0 / 6.0) * (a * a * a) );
    }
};

} // bspline

namespace cspline
{

// weight functions for Catmull - Rom Cardinal Spline

template < typename FloatT >
struct w0_func
{
    VV_FORCE_INLINE FloatT operator()( FloatT a )
    {
        return FloatT( -0.5 * a * a * a + a * a - 0.5 * a );
    }
};

template < typename FloatT >
struct w1_func
{
    VV_FORCE_INLINE FloatT operator()( FloatT a )
    {
        return FloatT( 1.5 * a * a * a - 2.5 * a * a + 1.0 );
    }
};

template < typename FloatT >
struct w2_func
{
    VV_FORCE_INLINE FloatT operator()( FloatT a )
    {
        return FloatT( -1.5 * a * a * a + 2.0 * a * a + 0.5 * a );
    }
};

template < typename FloatT >
struct w3_func
{
    VV_FORCE_INLINE FloatT operator()( FloatT a )
    {
        return FloatT( 0.5 * a * a * a - 0.5 * a * a );
    }
};

} // cspline

// helper functions for cubic interpolation
template < typename FloatT >
VV_FORCE_INLINE FloatT g0( FloatT x )
{
    bspline::w0_func< FloatT > w0;
    bspline::w1_func< FloatT > w1;
    return w0(x) + w1(x);
}

template < typename FloatT >
VV_FORCE_INLINE FloatT g1( FloatT x )
{
    bspline::w2_func< FloatT > w2;
    bspline::w3_func< FloatT > w3;
    return w2(x) + w3(x);
}

template < typename FloatT >
VV_FORCE_INLINE FloatT h0( FloatT x )
{
    bspline::w0_func< FloatT > w0;
    bspline::w1_func< FloatT > w1;
    return ((floor( x ) - FloatT(1.0) + w1(x)) / (w0(x) + w1(x))) + x;
}

template < typename FloatT >
VV_FORCE_INLINE FloatT h1( FloatT x )
{
    bspline::w2_func< FloatT > w2;
    bspline::w3_func< FloatT > w3;
    return ((floor( x ) + FloatT(1.0) + w3(x)) / (w2(x) + w3(x))) - x;
}


} // detail


} // virvo


#endif // VV_SAMPLER_COMMON_H


