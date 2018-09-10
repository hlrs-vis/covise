#ifndef VV_TEXTURE_SAMPLER1D_H
#define VV_TEXTURE_SAMPLER1D_H


#include "sampler_common.h"
#include "texture_common.h"

#include "math/math.h"


namespace virvo
{


namespace detail
{


template
<
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
#ifndef _MSC_VER
VV_FORCE_INLINE
#endif
ReturnT nearest(VoxelT const* tex, FloatT coord, FloatT texsize)
{

    typedef FloatT float_type;

    float_type lo = floor(coord * texsize);
    lo = clamp(lo, float_type(0.0f), texsize - 1);
    return point(tex, lo);

}


template
<
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
#ifndef _MSC_VER
VV_FORCE_INLINE
#endif
ReturnT linear(VoxelT const* tex, FloatT coord, FloatT texsize)
{

    typedef FloatT float_type;
    typedef ReturnT return_type;

    float_type texcoordf( coord * texsize - float_type(0.5) );
    texcoordf = clamp( texcoordf, float_type(0.0), texsize - 1 );

    float_type lo = floor(texcoordf);
    float_type hi = ceil(texcoordf);


    // In virvo, the return type is a float4.
    // TODO: what if precision(return_type) < precision(float_type)?
    return_type samples[2] =
    {
        point(tex, lo),
        point(tex, hi)
    };

    float_type u = texcoordf - lo;

    return lerp( samples[0], samples[1], u );

}


template
<
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
#ifndef _MSC_VER
VV_FORCE_INLINE
#endif
ReturnT cubic2(VoxelT const* tex, FloatT coord, FloatT texsize)
{

    typedef FloatT float_type;
    typedef ReturnT return_type;

    bspline::w0_func< FloatT > w0;
    bspline::w1_func< FloatT > w1;
    bspline::w2_func< FloatT > w2;
    bspline::w3_func< FloatT > w3;

    float_type x = coord * texsize - float_type(0.5);
    float_type floorx = floor( x );
    float_type fracx  = x - floor( x );

    float_type tmp0 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    float_type h0   = ( floorx - float_type(0.5) + tmp0 ) / texsize;

    float_type tmp1 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    float_type h1   = ( floorx + float_type(1.5) + tmp1 ) / texsize;


    // In virvo, the return type is a float4.
    // TODO: what if precision(return_type) < precision(float_type)?
    return_type f_0 = linear< return_type >( tex, h0, texsize );
    return_type f_1 = linear< return_type >( tex, h1, texsize );

    return g0(fracx) * f_0 + g1(fracx) * f_1;

}


template
<
    typename ReturnT,
    typename W0,
    typename W1,
    typename W2,
    typename W3,
    typename FloatT,
    typename VoxelT
>
inline ReturnT cubic(VoxelT const* tex, FloatT coord, FloatT texsize, W0 w0, W1 w1, W2 w2, W3 w3)
{

    typedef FloatT float_type;

    float_type x = coord * texsize - float_type(0.5);
    float_type floorx = floor( x );
    float_type fracx = x - floor( x );

    float_type pos[4] =
    {
        float_type( floorx - 1 ),
        float_type( floorx ),
        float_type( floorx + 1 ),
        float_type( floorx + 2 )
    };

    using virvo::clamp;

    for (size_t i = 0; i < 4; ++i)
    {
        pos[i] = clamp(pos[i], float_type(0.0), texsize - float_type(1.0));
    }

#define TEX(x) (point(tex,x))
    return w0(fracx) * TEX(pos[0]) + w1(fracx) * TEX(pos[1]) + w2(fracx) * TEX(pos[2]) + w3(fracx) * TEX(pos[3]);
#undef TEX
}


template
<
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
VV_FORCE_INLINE ReturnT tex1D(texture< VoxelT, virvo::ElementType, 1 > const& tex, FloatT coord)
{

    FloatT texsize = tex.width();

    switch (tex.get_filter_mode())
    {

    default:
        // fall-through
    case virvo::Nearest:
        return nearest< ReturnT >( tex.data, coord, texsize );

    case virvo::Linear:
        return linear< ReturnT >( tex.data, coord, texsize );

    case virvo::BSpline:
        return cubic2< ReturnT >( tex.data, coord, texsize );

/*    case virvo::BSplineInterpol:
        return cubic< ReturnT >( tex.prefiltered_data, coord, texsize,
            bspline::w0_func< FloatT >(), bspline::w1_func< FloatT >(),
            bspline::w2_func< FloatT >(), bspline::w3_func< FloatT >() );*/


    case virvo::CardinalSpline:
        return cubic< ReturnT >( tex.data, coord, texsize,
            cspline::w0_func< FloatT >(), cspline::w1_func< FloatT >(),
            cspline::w2_func< FloatT >(), cspline::w3_func< FloatT >() );

    }

}


} // detail


} // virvo


#endif // VV_TEXTURE_SAMPLER1D_H


