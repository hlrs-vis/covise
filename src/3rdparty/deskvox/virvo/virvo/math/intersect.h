#ifndef VV_MATH_INTERSECT_H
#define VV_MATH_INTERSECT_H

#include "simd/sse.h"

#include "aabb.h"
#include "plane.h"
#include "triangle.h"
#include "vector.h"

namespace MATH_NAMESPACE
{

template < typename T1, typename T2 >
struct hit_record;


//-------------------------------------------------------------------------------------------------
// ray / plane
//

template < typename T >
struct hit_record< basic_ray< T >, basic_plane< 3, T > >
{

    typedef T value_type;

    hit_record() : hit(false) {}

    bool                    hit;
    value_type              t;
    vector< 3, value_type > pos;

};

template < typename T >
inline hit_record< basic_ray< T >, basic_plane< 3, T > > intersect
(
    basic_ray< T > const& ray, basic_plane< 3, T > const& p
)
{

    hit_record< basic_ray< T >, basic_plane< 3, T > > result;
    T s = dot(p.normal, ray.dir);

    if (s == T(0.0))
    {
        result.hit = false;
    }
    else
    {
        result.hit = true;
        result.t   = ( p.offset - dot(p.normal, ray.ori) ) / s;
        result.pos = ray.ori + result.t * ray.dir;
    }
    return result;

}


//-------------------------------------------------------------------------------------------------
// ray / aabb
//

template < typename T >
struct hit_record< basic_ray< T >, basic_aabb< T > >
{

    typedef T value_type;

    hit_record() : hit(false) {}

    bool            hit;
    value_type      tnear;
    value_type      tfar;

};

#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE2)

template < >
struct hit_record< basic_ray< simd::float4 >, basic_aabb< simd::float4 > >
{

    simd::mask4     hit;
    simd::float4    tnear;
    simd::float4    tfar;

};

#endif

template < typename T >
inline hit_record< basic_ray< T >, basic_aabb< T > > intersect
(
    basic_ray< T > const& ray, basic_aabb< T > const& aabb
)
{

    hit_record< basic_ray< T >, basic_aabb< T > > result;

    vector< 3, T > invr( T(1.0) / ray.dir.x, T(1.0) / ray.dir.y, T(1.0) / ray.dir.z );
    vector< 3, T > t1 = (aabb.min - ray.ori) * invr;
    vector< 3, T > t2 = (aabb.max - ray.ori) * invr;

    result.tnear = max( min(t1.z, t2.z), max( min(t1.y, t2.y), min(t1.x, t2.x) ) );
    result.tfar  = min( max(t1.z, t2.z), min( max(t1.y, t2.y), max(t1.x, t2.x) ) );
    result.hit   = result.tfar >= result.tnear && result.tfar >= T(0.0);

    return result;

}


//-------------------------------------------------------------------------------------------------
// ray triangle
//

template < typename T, typename U >
struct hit_record< basic_ray< T >, basic_triangle< 3, U, unsigned > >
{

    typedef T value_type;

    bool hit;
    unsigned prim_id;
    unsigned geom_id;
    value_type t;

    value_type u;
    value_type v;

};

#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE2)

template < typename U >
struct hit_record< simd::ray4, basic_triangle< 3, U, unsigned > >
{

    typedef simd::float4 value_type;

    simd::mask4 hit;
    simd::int4 prim_id;
    simd::int4 geom_id;
    value_type t;

    value_type u;
    value_type v;

};

#endif

template < typename T, typename U >
inline hit_record< basic_ray< T >, basic_triangle< 3, U, unsigned > > intersect
(
    basic_ray< T > const& ray, basic_triangle< 3, U, unsigned > const& tri
)
{

    typedef vector< 3, T > vec_type;

    hit_record< basic_ray< T >, basic_triangle< 3, U, unsigned > > result;
    result.t = T(-1.0);

    // case T != U
    vec_type v1(tri.v1);
    vec_type e1(tri.e1);
    vec_type e2(tri.e2);

    vec_type s1 = cross(ray.dir, e2);
    T div = dot(s1, e1);

    result.hit = ( div != T(0.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    T inv_div = T(1.0) / div;

    vec_type d = ray.ori - v1;
    T b1 = dot(d, s1) * inv_div;

    result.hit &= ( b1 >= T(0.0) && b1 <= T(1.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    vec_type s2 = cross(d, e1);
    T b2 = dot(ray.dir, s2) * inv_div;

    result.hit &= ( b2 >= T(0.0) && b1 + b2 <= T(1.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    result.prim_id = tri.prim_id;
    result.geom_id = tri.geom_id;
    result.t = dot(e2, s2) * inv_div;
    result.u = b1;
    result.v = b2;
    return result;

}


} // MATH_NAMESPACE

#endif // VV_MATH_INTERSECT_H


