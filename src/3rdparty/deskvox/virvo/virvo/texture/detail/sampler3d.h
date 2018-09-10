#ifndef VV_TEXTURE_SAMPLER3D_H
#define VV_TEXTURE_SAMPLER3D_H


#include "sampler_common.h"
#include "texture_common.h"

#include "math/math.h"


namespace virvo
{


namespace detail
{


template < typename T >
inline T index(T x, T y, T z, vector< 3, T > texsize)
{
    return z * texsize[0] * texsize[1] + y * texsize[0] + x;
}


template
<
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
inline ReturnT nearest(VoxelT const* tex, vector< 3, FloatT > coord, vector< 3, FloatT > texsize)
{

#if 1

    using virvo::clamp;

    typedef FloatT float_type;
    typedef vector< 3, float_type > float3;

    float3 lo
    (
        floor(coord.x * texsize.x),
        floor(coord.y * texsize.y),
        floor(coord.z * texsize.z)
    );

    lo[0] = clamp(lo[0], float_type(0.0f), texsize[0] - 1);
    lo[1] = clamp(lo[1], float_type(0.0f), texsize[1] - 1);
    lo[2] = clamp(lo[2], float_type(0.0f), texsize[2] - 1);

    float_type idx = index(lo[0], lo[1], lo[2], texsize);
    return point(tex, idx);

#else

    // TODO: should be done similar to the code below..

    Float3T texsizef(itof(texsize[0]), itof(texsize[1]), itof(texsize[2]));

    Float3T texcoordf(coord[0] * texsizef[0] - FloatT(0.5),
                      coorm[1] * texsizef[1] - FloatT(0.5),
                      coord[2] * texsizef[2] - FloatT(0.5));

    texcoordf[0] = clamp( texcoordf[0], FloatT(0.0), texsizef[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], FloatT(0.0), texsizef[1] - 1 );
    texcoordf[2] = clamp( texcoordf[2], FloatT(0.0), texsizef[2] - 1 );

    Float3T lof( floor(texcoordf[0]), floor(texcoordf[1]), floor(texcoordf[2]) );
    Float3T hif( ceil(texcoordf[0]),  ceil(texcoordf[1]),  ceil(texcoordf[2]) );
    Int3T   lo( ftoi(lof[0]), ftoi(lof[1]), ftoi(lof[2]) );
    Int3T   hi( ftoi(hif[0]), ftoi(hif[1]), ftoi(hif[2]) );

    Float3T uvw = texcoordf - uvw;

    IntT idx = index( uvw[0] < FloatT(0.5) ? lo[0] : hi[0],
                      uvw[1] < FloatT(0.5) ? lo[1] : hi[1],
                      uvw[2] < FloatT(0.5) ? lo[2] : hi[2],
                      texsize);

    return point(tex, idx);

#endif

}


template
<
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
inline ReturnT linear(VoxelT const* tex, vector< 3, FloatT > coord, vector< 3, FloatT > texsize)
{

    using virvo::clamp;

    typedef FloatT float_type;
    typedef vector< 3, float_type > float3;

    float3 texcoordf( coord * texsize - float_type(0.5) );

    texcoordf[0] = clamp( texcoordf[0], float_type(0.0), texsize[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], float_type(0.0), texsize[1] - 1 );
    texcoordf[2] = clamp( texcoordf[2], float_type(0.0), texsize[2] - 1 );

    float3 lo( floor(texcoordf[0]), floor(texcoordf[1]), floor(texcoordf[2]) );
    float3 hi( ceil(texcoordf[0]),  ceil(texcoordf[1]),  ceil(texcoordf[2]) );


    // Implicit cast from return type to float type.
    // TODO: what if return type is e.g. a float4?
    float_type samples[8] =
    {
        float_type( point(tex, index( lo.x, lo.y, lo.z, texsize )) ),
        float_type( point(tex, index( hi.x, lo.y, lo.z, texsize )) ),
        float_type( point(tex, index( lo.x, hi.y, lo.z, texsize )) ),
        float_type( point(tex, index( hi.x, hi.y, lo.z, texsize )) ),
        float_type( point(tex, index( lo.x, lo.y, hi.z, texsize )) ),
        float_type( point(tex, index( hi.x, lo.y, hi.z, texsize )) ),
        float_type( point(tex, index( lo.x, hi.y, hi.z, texsize )) ),
        float_type( point(tex, index( hi.x, hi.y, hi.z, texsize )) )
    };


    float3 uvw = texcoordf - lo;

    float_type p1  = lerp(samples[0], samples[1], uvw[0]);
    float_type p2  = lerp(samples[2], samples[3], uvw[0]);
    float_type p3  = lerp(samples[4], samples[5], uvw[0]);
    float_type p4  = lerp(samples[6], samples[7], uvw[0]);

    float_type p12 = lerp(p1, p2, uvw[1]);
    float_type p34 = lerp(p3, p4, uvw[1]);

    return lerp(p12, p34, uvw[2]);

}


template
<
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
inline ReturnT cubic8(VoxelT const* tex, vector< 3, FloatT > coord, vector< 3, FloatT > texsize)
{

    typedef FloatT float_type;
    typedef vector< 3, float_type > float3;

    bspline::w0_func< FloatT > w0;
    bspline::w1_func< FloatT > w1;
    bspline::w2_func< FloatT > w2;
    bspline::w3_func< FloatT > w3;

    float_type x = coord.x * texsize.x - float_type(0.5);
    float_type floorx = floor( x );
    float_type fracx  = x - floor( x );

    float_type y = coord.y * texsize.y - float_type(0.5);
    float_type floory = floor( y );
    float_type fracy  = y - floor( y );

    float_type z = coord.z * texsize.z - float_type(0.5);
    float_type floorz = floor( z );
    float_type fracz  = z - floor( z );


    float_type tmp000 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    float_type h_000  = ( floorx - float_type(0.5) + tmp000 ) / texsize.x;

    float_type tmp100 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    float_type h_100  = ( floorx + float_type(1.5) + tmp100 ) / texsize.x;

    float_type tmp010 = ( w1(fracy) ) / ( w0(fracy) + w1(fracy) );
    float_type h_010  = ( floory - float_type(0.5) + tmp010 ) / texsize.y;

    float_type tmp110 = ( w3(fracy) ) / ( w2(fracy) + w3(fracy) );
    float_type h_110  = ( floory + float_type(1.5) + tmp110 ) / texsize.y;

    float_type tmp001 = ( w1(fracz) ) / ( w0(fracz) + w1(fracz) );
    float_type h_001  = ( floorz - float_type(0.5) + tmp001 ) / texsize.z;

    float_type tmp101 = ( w3(fracz) ) / ( w2(fracz) + w3(fracz) );
    float_type h_101  = ( floorz + float_type(1.5) + tmp101 ) / texsize.z;


    // Implicit cast from return type to float type.
    // TODO: what if return type is e.g. a float4?
    float_type f_000 = linear< float_type >( tex, float3(h_000, h_010, h_001), texsize );
    float_type f_100 = linear< float_type >( tex, float3(h_100, h_010, h_001), texsize );
    float_type f_010 = linear< float_type >( tex, float3(h_000, h_110, h_001), texsize );
    float_type f_110 = linear< float_type >( tex, float3(h_100, h_110, h_001), texsize );

    float_type f_001 = linear< float_type >( tex, float3(h_000, h_010, h_101), texsize );
    float_type f_101 = linear< float_type >( tex, float3(h_100, h_010, h_101), texsize );
    float_type f_011 = linear< float_type >( tex, float3(h_000, h_110 ,h_101), texsize );
    float_type f_111 = linear< float_type >( tex, float3(h_100, h_110, h_101), texsize );

    float_type f_00  = g0(fracx) * f_000 + g1(fracx) * f_100;
    float_type f_10  = g0(fracx) * f_010 + g1(fracx) * f_110;
    float_type f_01  = g0(fracx) * f_001 + g1(fracx) * f_101;
    float_type f_11  = g0(fracx) * f_011 + g1(fracx) * f_111;

    float_type f_0   = g0(fracy) * f_00 + g1(fracy) * f_10;
    float_type f_1   = g0(fracy) * f_01 + g1(fracy) * f_11;

    return g0(fracz) * f_0 + g1(fracz) * f_1;

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
inline ReturnT cubic(VoxelT const* tex, vector< 3, FloatT > coord, vector< 3, FloatT > texsize, W0 w0, W1 w1, W2 w2, W3 w3)
{

    typedef FloatT float_type;
    typedef vector< 3, float_type > float3;

    float_type x = coord.x * texsize.x - float_type(0.5);
    float_type floorx = floor( x );
    float_type fracx  = x - floor( x );

    float_type y = coord.y * texsize.y - float_type(0.5);
    float_type floory = floor( y );
    float_type fracy  = y - floor( y );

    float_type z = coord.z * texsize.z - float_type(0.5);
    float_type floorz = floor( z );
    float_type fracz  = z - floor( z );

    float3 pos[4] =
    {
        float3( floorx - 1, floory - 1, floorz - 1 ),
        float3( floorx,     floory,     floorz ),
        float3( floorx + 1, floory + 1, floorz + 1 ),
        float3( floorx + 2, floory + 2, floorz + 2 )
    };

    using virvo::clamp;

    for (size_t i = 0; i < 4; ++i)
    {
        pos[i].x = clamp(pos[i].x, float_type(0.0f), texsize.x - 1);
        pos[i].y = clamp(pos[i].y, float_type(0.0f), texsize.y - 1);
        pos[i].z = clamp(pos[i].z, float_type(0.0f), texsize.z - 1);
    }

#define TEX(x,y,z) (point(tex, index(x,y,z,texsize)))
    float_type f00 = w0(fracx) * TEX(pos[0].x, pos[0].y, pos[0].z) + w1(fracx) * TEX(pos[1].x, pos[0].y, pos[0].z) + w2(fracx) * TEX(pos[2].x, pos[0].y, pos[0].z) + w3(fracx) * TEX(pos[3].x, pos[0].y, pos[0].z);
    float_type f01 = w0(fracx) * TEX(pos[0].x, pos[1].y, pos[0].z) + w1(fracx) * TEX(pos[1].x, pos[1].y, pos[0].z) + w2(fracx) * TEX(pos[2].x, pos[1].y, pos[0].z) + w3(fracx) * TEX(pos[3].x, pos[1].y, pos[0].z);
    float_type f02 = w0(fracx) * TEX(pos[0].x, pos[2].y, pos[0].z) + w1(fracx) * TEX(pos[1].x, pos[2].y, pos[0].z) + w2(fracx) * TEX(pos[2].x, pos[2].y, pos[0].z) + w3(fracx) * TEX(pos[3].x, pos[2].y, pos[0].z);
    float_type f03 = w0(fracx) * TEX(pos[0].x, pos[3].y, pos[0].z) + w1(fracx) * TEX(pos[1].x, pos[3].y, pos[0].z) + w2(fracx) * TEX(pos[2].x, pos[3].y, pos[0].z) + w3(fracx) * TEX(pos[3].x, pos[3].y, pos[0].z);

    float_type f04 = w0(fracx) * TEX(pos[0].x, pos[0].y, pos[1].z) + w1(fracx) * TEX(pos[1].x, pos[0].y, pos[1].z) + w2(fracx) * TEX(pos[2].x, pos[0].y, pos[1].z) + w3(fracx) * TEX(pos[3].x, pos[0].y, pos[1].z);
    float_type f05 = w0(fracx) * TEX(pos[0].x, pos[1].y, pos[1].z) + w1(fracx) * TEX(pos[1].x, pos[1].y, pos[1].z) + w2(fracx) * TEX(pos[2].x, pos[1].y, pos[1].z) + w3(fracx) * TEX(pos[3].x, pos[1].y, pos[1].z);
    float_type f06 = w0(fracx) * TEX(pos[0].x, pos[2].y, pos[1].z) + w1(fracx) * TEX(pos[1].x, pos[2].y, pos[1].z) + w2(fracx) * TEX(pos[2].x, pos[2].y, pos[1].z) + w3(fracx) * TEX(pos[3].x, pos[2].y, pos[1].z);
    float_type f07 = w0(fracx) * TEX(pos[0].x, pos[3].y, pos[1].z) + w1(fracx) * TEX(pos[1].x, pos[3].y, pos[1].z) + w2(fracx) * TEX(pos[2].x, pos[3].y, pos[1].z) + w3(fracx) * TEX(pos[3].x, pos[3].y, pos[1].z);

    float_type f08 = w0(fracx) * TEX(pos[0].x, pos[0].y, pos[2].z) + w1(fracx) * TEX(pos[1].x, pos[0].y, pos[2].z) + w2(fracx) * TEX(pos[2].x, pos[0].y, pos[2].z) + w3(fracx) * TEX(pos[3].x, pos[0].y, pos[2].z);
    float_type f09 = w0(fracx) * TEX(pos[0].x, pos[1].y, pos[2].z) + w1(fracx) * TEX(pos[1].x, pos[1].y, pos[2].z) + w2(fracx) * TEX(pos[2].x, pos[1].y, pos[2].z) + w3(fracx) * TEX(pos[3].x, pos[1].y, pos[2].z);
    float_type f10 = w0(fracx) * TEX(pos[0].x, pos[2].y, pos[2].z) + w1(fracx) * TEX(pos[1].x, pos[2].y, pos[2].z) + w2(fracx) * TEX(pos[2].x, pos[2].y, pos[2].z) + w3(fracx) * TEX(pos[3].x, pos[2].y, pos[2].z);
    float_type f11 = w0(fracx) * TEX(pos[0].x, pos[3].y, pos[2].z) + w1(fracx) * TEX(pos[1].x, pos[3].y, pos[2].z) + w2(fracx) * TEX(pos[2].x, pos[3].y, pos[2].z) + w3(fracx) * TEX(pos[3].x, pos[3].y, pos[2].z);

    float_type f12 = w0(fracx) * TEX(pos[0].x, pos[0].y, pos[3].z) + w1(fracx) * TEX(pos[1].x, pos[0].y, pos[3].z) + w2(fracx) * TEX(pos[2].x, pos[0].y, pos[3].z) + w3(fracx) * TEX(pos[3].x, pos[0].y, pos[3].z);
    float_type f13 = w0(fracx) * TEX(pos[0].x, pos[1].y, pos[3].z) + w1(fracx) * TEX(pos[1].x, pos[1].y, pos[3].z) + w2(fracx) * TEX(pos[2].x, pos[1].y, pos[3].z) + w3(fracx) * TEX(pos[3].x, pos[1].y, pos[3].z);
    float_type f14 = w0(fracx) * TEX(pos[0].x, pos[2].y, pos[3].z) + w1(fracx) * TEX(pos[1].x, pos[2].y, pos[3].z) + w2(fracx) * TEX(pos[2].x, pos[2].y, pos[3].z) + w3(fracx) * TEX(pos[3].x, pos[2].y, pos[3].z);
    float_type f15 = w0(fracx) * TEX(pos[0].x, pos[3].y, pos[3].z) + w1(fracx) * TEX(pos[1].x, pos[3].y, pos[3].z) + w2(fracx) * TEX(pos[2].x, pos[3].y, pos[3].z) + w3(fracx) * TEX(pos[3].x, pos[3].y, pos[3].z);
#undef TEX

    float_type f0 = w0(fracy) * f00 + w1(fracy) * f01 + w2(fracy) * f02 + w3(fracy) * f03;
    float_type f1 = w0(fracy) * f04 + w1(fracy) * f05 + w2(fracy) * f06 + w3(fracy) * f07;
    float_type f2 = w0(fracy) * f08 + w1(fracy) * f09 + w2(fracy) * f10 + w3(fracy) * f11;
    float_type f3 = w0(fracy) * f12 + w1(fracy) * f13 + w2(fracy) * f14 + w3(fracy) * f15;

    return w0(fracz) * f0 + w1(fracz) * f1 + w2(fracz) * f2 + w3(fracz) * f3;

}


template
<
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
inline ReturnT tex3D(texture< VoxelT, ElementType, 3 > const& tex, vector< 3, FloatT > coord)
{

    vector< 3, FloatT > texsize( tex.width(), tex.height(), tex.depth() );

    switch (tex.get_filter_mode())
    {

    default:
        // fall-through
    case virvo::Nearest:
        return nearest< ReturnT >( tex.data, coord, texsize );

    case virvo::Linear:
        return linear< ReturnT >( tex.data, coord, texsize );

    case virvo::BSpline:
        return cubic8< ReturnT >( tex.data, coord, texsize );

    case virvo::BSplineInterpol:
        return cubic< ReturnT >( tex.prefiltered_data, coord, texsize,
            bspline::w0_func< FloatT >(), bspline::w1_func< FloatT >(),
            bspline::w2_func< FloatT >(), bspline::w3_func< FloatT >() );


    case virvo::CardinalSpline:
        return cubic< ReturnT >( tex.data, coord, texsize,
            cspline::w0_func< FloatT >(), cspline::w1_func< FloatT >(),
            cspline::w2_func< FloatT >(), cspline::w3_func< FloatT >() );

    }

}


} // detail


} // virvo


#endif // VV_TEXTURE_SAMPLER3D_H


