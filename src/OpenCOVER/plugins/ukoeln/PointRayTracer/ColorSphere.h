#pragma once

#include <visionaray/detail/macros.h>
#include <visionaray/math/math.h>
#include <visionaray/bvh.h> // split_primitive() for basic_sphere

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// ColorSphere, tightly packed 16-byte data type using 12 bytes for position,
// 1 byte for sphere radius and 3 bytes for color
//

struct VSNRAY_ALIGN(16) ColorSphere
{
    vector<3, float> center;
    unorm<8> radius;
    vector<3, unorm<8>> color;
};

template <typename T>
struct ColorSphereHitRecord
{
    using M = typename simd::mask_type<T>::type;

    M hit = M(false);
    T t = numeric_limits<T>::max();
    vector<3, T> color;
    int u, v, geom_id, prim_id;
};


template <typename T, typename Cond>
VSNRAY_FUNC
void update_if(ColorSphereHitRecord<T>& dst, ColorSphereHitRecord<T> const& src, Cond const& cond)
{    
    dst.hit    |= cond;
    dst.t       = select( cond, src.t, dst.t );
    dst.color   = select (cond, src.color, dst.color );
}

VSNRAY_FUNC
inline basic_aabb<float> get_bounds(ColorSphere const& s)
{
    basic_aabb<float> bounds;

    float radius = s.radius;

    bounds.invalidate();
    bounds.insert(s.center - radius);
    bounds.insert(s.center + radius);

    return bounds;
}

inline void split_primitive(aabb& L, aabb& R, float plane, int axis, ColorSphere const& s)
{
    basic_sphere<float> bs;
    bs.center = s.center;
    bs.radius = s.radius;
    split_primitive(L, R, plane, axis, bs);
}

template <typename T>
VSNRAY_FUNC
inline ColorSphereHitRecord<T> intersect(basic_ray<T> const& ray, ColorSphere const& sphere)
{
    typedef basic_ray<T> ray_type;
    typedef vector<3, T> vec_type;

    ray_type r = ray;
    r.ori -= vec_type( sphere.center );
    float radius = sphere.radius;

    auto A = dot(r.dir, r.dir);
    auto B = dot(r.dir, r.ori) * T(2.0);
    auto C = dot(r.ori, r.ori) - radius * radius;

    // solve Ax**2 + Bx + C
    auto disc = B * B - T(4.0) * A * C;
    auto valid = disc >= T(0.0);

    auto root_disc = select(valid, sqrt(disc), disc);

    auto q = select( B < T(0.0), T(-0.5) * (B - root_disc), T(-0.5) * (B + root_disc) );

    auto t1 = q / A;
    auto t2 = C / q;

    ColorSphereHitRecord<T> result;
    result.hit      = valid;
    result.t        = select( valid, select( t1 > t2, t2, t1 ), T(-1.0) );
    result.color    = select( valid, vector<3, T>(sphere.color), result.color );
    //printf("color %f %f %f \n", (float)result.color.x, (float)result.color.y, (float)result.color.z);
    return result;
}

} // visionaray
