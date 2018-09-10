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

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

#include <GL/glew.h>

#ifdef VV_ARCH_CUDA
#include <thrust/device_vector.h>
#endif

#undef MATH_NAMESPACE

#include <visionaray/detail/pixel_access.h> // detail (TODO?)!
#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/material.h>
#include <visionaray/packet_traits.h>
#include <visionaray/pixel_format.h>
#include <visionaray/pixel_traits.h>
#include <visionaray/point_light.h>
#include <visionaray/render_target.h>
#include <visionaray/scheduler.h>
#include <visionaray/shade_record.h>
#include <visionaray/variant.h>

#ifdef VV_ARCH_CUDA
#include <visionaray/cuda/pixel_pack_buffer.h>
#endif

#undef MATH_NAMESPACE

#include "gl/util.h"
#include "vvcudarendertarget.h"
#include "vvraycaster.h"
#include "vvtextureutil.h"
#include "vvtoolshed.h"
#include "vvvoldesc.h"

#ifdef VV_ARCH_CUDA
#include "cuda/utils.h"
#endif

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Global typedefs
//

#if defined(VV_ARCH_CUDA)
using ray_type          = basic_ray<float>;
using sched_type        = cuda_sched<ray_type>;
using transfunc_type    = cuda_texture<vec4,      1>;
using volume8_type      = cuda_texture<unorm< 8>, 3>;
using volume16_type     = cuda_texture<unorm<16>, 3>;
using volume32_type     = cuda_texture<float,     3>;
#else
#if defined(VV_ARCH_SSE2) || defined(VV_ARCH_SSE4_1)
using ray_type = basic_ray<simd::float4>;
#elif defined(VV_ARCH_AVX) || defined(VV_ARCH_AVX2)
using ray_type = basic_ray<simd::float8>;
#else
using ray_type = basic_ray<float>;
#endif
using sched_type        = tiled_sched<ray_type>;
using transfunc_type    = texture<vec4,      1>;
using volume8_type      = texture<unorm< 8>, 3>;
using volume16_type     = texture<unorm<16>, 3>;
using volume32_type     = texture<float,     3>;
#endif

//-------------------------------------------------------------------------------------------------
// Ray type, depends upon target architecture
//



//-------------------------------------------------------------------------------------------------
// Misc. helpers
//

template <typename T, typename Tex>
VSNRAY_FUNC
inline vector<3, T> gradient(Tex const& tex, vector<3, T> tex_coord)
{
    vector<3, T> s1;
    vector<3, T> s2;

    float DELTA = 0.01f;

    s1.x = tex3D(tex, tex_coord + vector<3, T>(DELTA, 0.0f, 0.0f));
    s2.x = tex3D(tex, tex_coord - vector<3, T>(DELTA, 0.0f, 0.0f));
    // signs for y and z are swapped because of texture orientation
    s1.y = tex3D(tex, tex_coord - vector<3, T>(0.0f, DELTA, 0.0f));
    s2.y = tex3D(tex, tex_coord + vector<3, T>(0.0f, DELTA, 0.0f));
    s1.z = tex3D(tex, tex_coord - vector<3, T>(0.0f, 0.0f, DELTA));
    s2.z = tex3D(tex, tex_coord + vector<3, T>(0.0f, 0.0f, DELTA));

    return s2 - s1;
}

template <typename F, typename I>
VSNRAY_FUNC
inline F normalize_depth(I const& depth, pixel_format depth_format, F /* */)
{
    if (depth_format == PF_DEPTH24_STENCIL8)
    {
        auto d = (depth & 0xFFFFFF00) >> 8;
        return F(d) / 16777215.0f;
    }

    // Assume PF_DEPTH32F
    return reinterpret_as_float(depth);
}

template <typename I1, typename I2, typename Params>
VSNRAY_FUNC
inline void get_depth(I1 x, I1 y, I2& depth_raw, Params const& params)
{
    // Get depth value from visionaray buffer
    // dst format equals src format because our implementation
    // takes care of the conversion itself in the rendering kernel
    if (params.depth_format == PF_DEPTH24_STENCIL8)
    {
        detail::pixel_access::get( // detail (TODO?)!
                pixel_format_constant<PF_DEPTH24_STENCIL8>{},   // dst format
                pixel_format_constant<PF_DEPTH24_STENCIL8>{},   // src format
                x,
                y,
                params.viewport.w,
                params.viewport.h,
                depth_raw,
                params.depth_buffer
                );
    }
    else
    {
        // Assume PF_DEPTH32F
        detail::pixel_access::get( // detail (TODO?)!
                pixel_format_constant<PF_DEPTH32F>{},           // dst format
                pixel_format_constant<PF_DEPTH32F>{},           // src format
                x,
                y,
                params.viewport.w,
                params.viewport.h,
                depth_raw,
                params.depth_buffer
                );
    }
}

VSNRAY_FUNC
inline vec3 gatherv(vec3 const* base_addr, int index)
{
    return base_addr[index];
}

template <
    typename T,
    typename I,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
VSNRAY_CPU_FUNC
inline vector<3, T> gatherv(vector<3, T> const* base_addr, I const& index)
{
    // basically like visionaray::simd::gather, but
    // base_addr points to vec3's of simd-vectors

    typename simd::aligned_array<I>::type indices;
    store(indices, index);

    array<vector<3, float>, simd::num_elements<T>::value> arr;

    for (int i = 0; i < simd::num_elements<T>::value; ++i)
    {
        auto vecs = unpack(base_addr[indices[i]]);
        arr[i] = vecs[i];
    }

    return simd::pack(arr);
}


//-------------------------------------------------------------------------------------------------
// Clip sphere, hit_record stores both tnear and tfar (in contrast to basic_sphere)!
//

struct clip_sphere : basic_sphere<float>
{
};

template <typename T>
struct clip_sphere_hit_record
{
    using M = typename simd::mask_type<T>::type;

    M hit   = M(false);
    T tnear =  numeric_limits<T>::max();
    T tfar  = -numeric_limits<T>::max();
};

template <typename T>
VSNRAY_FUNC
inline clip_sphere_hit_record<T> intersect(basic_ray<T> const& ray, clip_sphere const& sphere)
{

    typedef basic_ray<T> ray_type;
    typedef vector<3, T> vec_type;

    ray_type r = ray;
    r.ori -= vec_type( sphere.center );

    auto A = dot(r.dir, r.dir);
    auto B = dot(r.dir, r.ori) * T(2.0);
    auto C = dot(r.ori, r.ori) - sphere.radius * sphere.radius;

    // solve Ax**2 + Bx + C
    auto disc = B * B - T(4.0) * A * C;
    auto valid = disc >= T(0.0);

    auto root_disc = select(valid, sqrt(disc), disc);

    auto q = select( B < T(0.0), T(-0.5) * (B - root_disc), T(-0.5) * (B + root_disc) );

    auto t1 = q / A;
    auto t2 = C / q;

    clip_sphere_hit_record<T> result;
    result.hit = valid;
    result.tnear   = select( valid, select( t1 > t2, t2, t1 ), T(-1.0) );
    result.tfar    = select( valid, select( t1 > t2, t1, t2 ), T(-1.0) );
    return result;
}


//-------------------------------------------------------------------------------------------------
// Clip clone
//

struct clip_cone
{
    vec3 tip;       // position of the cone's tip
    vec3 axis;      // unit vector pointing from tip into the cone
    float theta;    // *half* angle between axis and cone surface
};

template <typename T>
struct clip_cone_hit_record : clip_sphere_hit_record<T>
{
};

template <typename T>
VSNRAY_FUNC
inline clip_cone_hit_record<T> intersect(basic_ray<T> const& ray, clip_cone const& cone)
{
    using R = basic_ray<T>;
    using V = vector<3, T>;

    R r = ray;
    r.ori -= V(cone.tip);

    T cos2_theta(cos(cone.theta) * cos(cone.theta));

    auto A = dot(r.dir, V(cone.axis)) * dot(r.dir, V(cone.axis)) - cos2_theta;
    auto B = T(2.0) * (dot(r.dir, V(cone.axis)) * dot(r.ori, V(cone.axis)) - dot(r.dir, r.ori) * cos2_theta);
    auto C = dot(r.ori, V(cone.axis)) * dot(r.ori, V(cone.axis)) - dot(r.ori, r.ori) * cos2_theta;

    // solve Ax**2 + Bx + C
    auto disc = B * B - T(4.0) * A * C;
    auto valid = disc >= T(0.0);

    auto root_disc = select(valid, sqrt(disc), disc);

    auto q = select( B < T(0.0), T(-0.5) * (B - root_disc), T(-0.5) * (B + root_disc) );

    auto t1 = q / A;
    auto t2 = C / q;

    auto isect_pos1 = V(ray.ori) + V(ray.dir) * t1;
    auto hits_shadow_cone1 = dot(isect_pos1 - V(cone.tip), V(cone.axis)) > T(0.0);

    auto isect_pos2 = V(ray.ori) + V(ray.dir) * t2;
    auto hits_shadow_cone2 = dot(isect_pos2 - V(cone.tip), V(cone.axis)) > T(0.0);

    t1 = select(hits_shadow_cone1, T(-1.0), t1);
    t2 = select(hits_shadow_cone2, T(-1.0), t2);

    valid &= dot(ray.dir, V(cone.axis)) >= T(0.0);

    clip_cone_hit_record<T> result;
    result.hit = valid;
    result.tnear   = select( valid, select( t1 > t2, t2, t1 ), T(-1.0) );
    result.tfar    = select( valid, select( t1 > t2, t1, t2 ), T(-1.0) );
    return result;
}


//-------------------------------------------------------------------------------------------------
// Clip box, basically an aabb, but intersect() returns a hit record containing the
// plane normal of the box' side where the ray entered
//

struct clip_box : basic_aabb<float>
{
    using base_type = basic_aabb<float>;

    clip_box() = default;
    VSNRAY_FUNC clip_box(vec3 const& min, vec3 const& max)
        : base_type(min, max)
    {
    }
};

template <typename T>
struct clip_box_hit_record : hit_record<basic_ray<T>, basic_aabb<float>>
{
    vector<3, T> normal;
};

template <typename T>
VSNRAY_FUNC
inline clip_box_hit_record<T> intersect(basic_ray<T> const& ray, clip_box const& box)
{
    auto hr = intersect(ray, static_cast<clip_box::base_type>(box));

    // calculate normal
    vector<3, float> normals[6] {
            {  1.0f,  0.0f,  0.0f },
            { -1.0f,  0.0f,  0.0f },
            {  0.0f,  1.0f,  0.0f },
            {  0.0f, -1.0f,  0.0f },
            {  0.0f,  0.0f,  1.0f },
            {  0.0f,  0.0f, -1.0f }
            };

    auto isect_pos = ray.ori + ray.dir * hr.tnear;
    auto dir = normalize(isect_pos - vector<3, T>(box.center()));
    auto cosa = dot(dir, vector<3, T>(normals[0]));

    vector<3, T> normal(normals[0]);

    for (int i = 1; i < 6; ++i)
    {
        T dp    = dot(dir, vector<3, T>(normals[i]));
        normal  = select(dp > cosa, normals[i], normal);
        cosa    = select(dp > cosa, dp, cosa);
    }

    clip_box_hit_record<T> result;
    result.hit    = hr.hit;
    result.tnear  = hr.tnear;
    result.tfar   = hr.tfar;
    result.normal = normal;
    return result;
}


//-------------------------------------------------------------------------------------------------
// Clip plane (just another name for plane)
//

using clip_plane = basic_plane<3, float>;


//-------------------------------------------------------------------------------------------------
// Create clip intervals and deduce clip normals from primitive list
//

template <typename T>
struct clip_object_visitor
{
public:

    enum { MAX_INTERVALS = 64 };

    struct RT
    {
        int num_intervals;
        vector<2, T> intervals[MAX_INTERVALS];
        vector<3, T> normal;
    };

    using return_type = RT;

public:

    // Create with ray and tnear / tfar obtained from ray / bbox intersection
    VSNRAY_FUNC
    clip_object_visitor(basic_ray<T> const& ray, T const& tnear, T const& tfar)
        : ray_(ray)
        , tnear_(tnear)
        , tfar_(tfar)
    {
    }

    // Clip plane
    VSNRAY_FUNC
    return_type operator()(clip_plane const& ref) const
    {
        auto hit_rec = intersect(ray_, ref);
        auto ndotd = dot(ray_.dir, vector<3, T>(ref.normal));

        return_type result;
        result.num_intervals = 1;
        result.intervals[0].x = select(ndotd >  0.0f, hit_rec.t, tnear_);
        result.intervals[0].y = select(ndotd <= 0.0f, hit_rec.t, tfar_);
        result.normal     = ref.normal;
        return result;
    }

    // Clip sphere
    VSNRAY_FUNC
    return_type operator()(clip_sphere const& ref) const
    {
        using V = vector<3, T>;

        auto hit_rec = intersect(ray_, ref);

        return_type result;
        result.num_intervals = 1;
        result.intervals[0].x = select(hit_rec.tnear > tnear_, hit_rec.tnear, tnear_);
        result.intervals[0].y = select(hit_rec.tfar  < tfar_,  hit_rec.tfar,  tfar_);

        // normal at tfar, pointing inwards
        V isect_pos = ray_.ori + result.intervals[0].y * ray_.dir;
        result.normal = -(isect_pos - V(ref.center)) / T(ref.radius);

        return result;
    }

    // Clip cone
    VSNRAY_FUNC
    return_type operator()(clip_cone const& ref) const
    {
        using V = vector<3, T>;

        auto hit_rec = intersect(ray_, ref);

        return_type result;
        result.num_intervals = 1;
        result.intervals[0].x = select(hit_rec.tnear > tnear_, hit_rec.tnear, tnear_);
        result.intervals[0].y = select(hit_rec.tfar  < tfar_,  hit_rec.tfar,  tfar_);

        // normal at tfar, pointing inwards
        V isect_pos = ray_.ori + result.intervals[0].y * ray_.dir;
        V tmp = isect_pos - V(ref.tip);
        result.normal = normalize(tmp * dot(V(ref.axis), tmp) / dot(tmp, tmp) - V(ref.axis));

        return result;
    }

private:

    basic_ray<T>    ray_;
    T               tnear_;
    T               tfar_;
};


//-------------------------------------------------------------------------------------------------
// Wrapper that either uses CUDA/GL interop or simple CPU <- GPU transfer to make the
// OpenGL depth buffer available to the Visionaray kernel
//

#ifdef VV_ARCH_CUDA

struct depth_buffer_type : cuda::pixel_pack_buffer
{
    unsigned const* data() const
    {
        return static_cast<unsigned const*>(cuda::pixel_pack_buffer::data());
    }
};

#else

struct depth_buffer_type
{
    void map(recti viewport, pixel_format format)
    {
        auto info = map_pixel_format(format);

        buffer.resize((viewport.w - viewport.x) * (viewport.h - viewport.y));

        glReadPixels(
                viewport.x,
                viewport.y,
                viewport.w,
                viewport.h,
                info.format,
                info.type,
                buffer.data()
                );
    }

    void unmap()
    {
    }

    unsigned const* data() const
    {
        return buffer.data();
    }

    aligned_vector<unsigned> buffer;
};

#endif

//-------------------------------------------------------------------------------------------------
// Wrapper to consolidate virvo and Visionaray render targets
//

class virvo_render_target
{
public:

    static const pixel_format CF = PF_RGBA32F;
    static const pixel_format DF = PF_UNSPECIFIED;

    using color_type = typename pixel_traits<CF>::type;
    using depth_type = typename pixel_traits<DF>::type;

    using ref_type = render_target_ref<CF, DF>;

public:

    virvo_render_target(int w, int h, color_type* c, depth_type* d)
        : width_(w)
        , height_(h)
        , color_(c)
        , depth_(d)
    {
    }

    int width() const { return width_; }
    int height() const { return height_; }

    color_type* color() { return color_; }
    depth_type* depth() { return depth_; }

    color_type const* color() const { return color_; }
    depth_type const* depth() const { return depth_; }

    ref_type ref() { return { color(), depth(), width(), height() }; }

    void begin_frame() {}
    void end_frame() {}

    int width_;
    int height_;

    color_type* color_;
    depth_type* depth_;
};


//-------------------------------------------------------------------------------------------------
// Volume kernel params
//

struct volume_kernel_params
{
    enum projection_mode
    {
        AlphaCompositing,
        MaxIntensity,
        MinIntensity,
        DRR
    };

    using clip_object    = variant<clip_plane, clip_sphere, clip_cone>;
    using transfunc_ref  = typename transfunc_type::ref_type;

    clip_box                    bbox;
    float                       delta;
    int                         num_channels;
    transfunc_ref const*        transfuncs;
    vec2 const*                 ranges;
    unsigned const*             depth_buffer;
    pixel_format                depth_format;
    projection_mode             mode;
    bool                        depth_test;
    bool                        opacity_correction;
    bool                        early_ray_termination;
    bool                        local_shading;
    mat4                        camera_matrix_inv;
    recti                       viewport;
    point_light<float>          light;

    struct
    {
        clip_object const*      begin;
        clip_object const*      end;
    } clip_objects;
};


//-------------------------------------------------------------------------------------------------
// Visionaray volume rendering kernel
//

template <typename Volume>
struct volume_kernel
{
    using Params = volume_kernel_params;
    using VolRef = typename Volume::ref_type;

    VSNRAY_FUNC
    explicit volume_kernel(Params const& p, VolRef const* vols)
        : params(p)
        , volumes(vols)
    {
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> operator()(R ray, int x, int y) const
    {
        using S    = typename R::scalar_type;
        using I    = typename simd::int_type<S>::type;
        using Mask = typename simd::mask_type<S>::type;
        using Mat4 = matrix<4, 4, S>;
        using C    = vector<4, S>;

        result_record<S> result;
        result.color = C(0.0);

        auto hit_rec = intersect(ray, params.bbox);
        auto tmax = hit_rec.tfar;

        // convert depth buffer(x,y) to "t" coordinates
        if (params.depth_test)
        {
            // unproject (win to obj)
            I depth_raw(0);
            get_depth(x, y, depth_raw, params);
            S depth = normalize_depth(depth_raw, params.depth_format, S{});

            vector<3, S> win(expand_pixel<S>().x(x), expand_pixel<S>().y(y), depth);
            vector<4, S> u(
                    S(2.0 * (win[0] - params.viewport[0]) / params.viewport[2] - 1.0),
                    S(2.0 * (win[1] - params.viewport[1]) / params.viewport[3] - 1.0),
                    S(2.0 * win[2] - 1.0),
                    S(1.0)
                    );

            vector<4, S> v = Mat4(params.camera_matrix_inv) * u;
            vector<3, S> obj = v.xyz() / v.w;

            // convert to "t" coordinates
            tmax = length(obj - ray.ori);
        }


        auto t = max(S(0.0f), hit_rec.tnear);
        tmax = min(hit_rec.tfar, tmax);


        // calculate intervals clipped by planes, spheres, etc., along with the
        // normals of the farthest intersection in view direction
        const int MaxClipIntervals = 64;
        vector<2, S> clip_intervals[MaxClipIntervals];
        vector<3, S> clip_normals[MaxClipIntervals];

        auto num_clip_objects = min(
                MaxClipIntervals - 1, // room for bbox normal, which is the last clip object
                static_cast<int>(params.clip_objects.end - params.clip_objects.begin)
                );

        for (auto it = params.clip_objects.begin; it != params.clip_objects.end; ++it)
        {
            clip_object_visitor<S> visitor(ray, t, tmax);
            auto clip_data = apply_visitor(visitor, *it);

            clip_intervals[it - params.clip_objects.begin] = clip_data.intervals[0];
            clip_normals[it - params.clip_objects.begin] = clip_data.normal;
        }

        // treat the bbox entry plane as a clip
        // object that contributes a shading normal
        clip_normals[num_clip_objects] = hit_rec.normal;


        // calculate the volume rendering integral
        while (visionaray::any(t < tmax))
        {
            Mask clipped(false);

            S tnext = t + params.delta;
            for (int i = 0; i < num_clip_objects; ++i)
            {
                clipped |= t >= clip_intervals[i].x && t <= clip_intervals[i].y;
                tnext = select(
                        t >= clip_intervals[i].x && t <= clip_intervals[i].y && tnext < clip_intervals[i].y,
                        clip_intervals[i].y,
                        tnext
                        );
            }

            if (!visionaray::all(clipped))
            {
                auto pos = ray.ori + ray.dir * t;
                auto tex_coord = vector<3, S>(
                        ( pos.x + (params.bbox.size().x / 2) ) / params.bbox.size().x,
                        (-pos.y + (params.bbox.size().y / 2) ) / params.bbox.size().y,
                        (-pos.z + (params.bbox.size().z / 2) ) / params.bbox.size().z
                        );

                C color(0.0);

                for (int i = 0; i < params.num_channels; ++i)
                {
                    S voxel  = tex3D(volumes[i], tex_coord);
                    C colori = tex1D(params.transfuncs[i], voxel);

                    auto do_shade = params.local_shading && colori.w >= 0.1f;

                    if (visionaray::any(do_shade))
                    {
                        // TODO: make this modifiable
                        plastic<S> mat;
                        mat.ca() = from_rgb(vector<3, S>(0.3f, 0.3f, 0.3f));
                        mat.cd() = from_rgb(vector<3, S>(0.8f, 0.8f, 0.8f));
                        mat.cs() = from_rgb(vector<3, S>(0.8f, 0.8f, 0.8f));
                        mat.ka() = 1.0f;
                        mat.kd() = 1.0f;
                        mat.ks() = 1.0f;
                        mat.specular_exp() = 1000.0f;


                        // calculate shading
                        auto grad = gradient(volumes[i], tex_coord);
                        auto normal = normalize(grad);

                        auto float_eq = [&](S const& a, S const& b) { return abs(a - b) < params.delta * S(0.5); };

                        Mask at_boundary = float_eq(t, hit_rec.tnear);
                        I clip_normal_index = select(
                                at_boundary,
                                I(num_clip_objects), // bbox normal is stored at last position in the list
                                I(0)
                                );

                        for (int i = 0; i < num_clip_objects; ++i)
                        {
                            Mask hit = float_eq(t, clip_intervals[i].y + params.delta); // TODO: understand why +delta
                            clip_normal_index = select(hit, I(i), clip_normal_index);
                            at_boundary |= hit;
                        }

                        if (visionaray::any(at_boundary))
                        {
                            auto boundary_normal = gatherv(clip_normals, clip_normal_index);
                            normal = select(
                                    at_boundary,
                                    boundary_normal * colori.w + normal * (S(1.0) - colori.w),
                                    normal
                                    );
                        }

                        do_shade &= length(grad) != 0.0f;

                        shade_record<S> sr;
                        sr.normal = normal;
                        sr.geometric_normal = normal;
                        sr.view_dir = -ray.dir;
                        sr.tex_color = vector<3, S>(1.0);
                        sr.light_dir = normalize(params.light.position());
                        sr.light_intensity = params.light.intensity(pos);

                        auto shaded_clr = mat.shade(sr);

                        colori.xyz() = mul(
                                colori.xyz(),
                                to_rgb(shaded_clr),
                                do_shade,
                                colori.xyz()
                                );
                    }

                    if (params.opacity_correction)
                    {
                        colori.w = 1.0f - pow(1.0f - colori.w, params.delta);
                    }

                    // premultiplied alpha
                    colori.xyz() *= colori.w;

                    color += colori;
                }


                // compositing
                if (params.mode == Params::AlphaCompositing)
                {
                    result.color += select(
                            t < tmax && !clipped,
                            color * (1.0f - result.color.w),
                            C(0.0)
                            );

                    // early-ray termination - don't traverse w/o a contribution
                    if (params.early_ray_termination && visionaray::all(result.color.w >= 0.999f))
                    {
                        break;
                    }
                }
                else if (params.mode == Params::MaxIntensity)
                {
                    result.color = select(
                            t < tmax && !clipped,
                            max(color, result.color),
                            result.color
                            );
                }
                else if (params.mode == Params::MinIntensity)
                {
                    result.color = select(
                            t < tmax && !clipped,
                            min(color, result.color),
                            result.color
                            );
                }
                else if (params.mode == Params::DRR)
                {
                    result.color += select(
                            t < tmax && !clipped,
                            color,
                            C(0.0)
                            );
                }
            }

            // step on
            t = tnext;
        }

        result.hit = hit_rec.hit;
        return result;
    }

    Params params;
    VolRef const* volumes;
};


//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct vvRayCaster::Impl
{
#if defined(VV_ARCH_CUDA)
    Impl()
        : sched(8, 8)
    {
    }
#else
    Impl()
        : sched(vvToolshed::getNumProcessors())
    {
        char* num_threads = getenv("VV_NUM_THREADS");
        if (num_threads != nullptr)
        {
            std::string str(num_threads);
            sched.reset(std::stoi(str));
        }
    }
#endif

    using params_type = volume_kernel_params;

    sched_type                      sched;
    params_type                     params;
    std::vector<volume8_type>       volumes8;
    std::vector<volume16_type>      volumes16;
    std::vector<volume32_type>      volumes32;
    std::vector<transfunc_type>     transfuncs;
    depth_buffer_type               depth_buffer;

    // Internal storage format for textures
    virvo::PixelFormat              texture_format = virvo::PF_R8;

    void updateVolumeTextures(vvVolDesc* vd, vvRenderer* renderer);
    void updateTransfuncTexture(vvVolDesc* vd, vvRenderer* renderer);

    template <typename Volumes>
    void updateVolumeTexturesImpl(vvVolDesc* vd, vvRenderer* renderer, Volumes& volume);
};


void vvRayCaster::Impl::updateVolumeTextures(vvVolDesc* vd, vvRenderer* renderer)
{
    if (texture_format == virvo::PF_R8)
    {
        updateVolumeTexturesImpl(vd, renderer, volumes8);
    }
    else if (texture_format == virvo::PF_R16UI)
    {
        updateVolumeTexturesImpl(vd, renderer, volumes16);
    }
    else if (texture_format == virvo::PF_R32F)
    {
        updateVolumeTexturesImpl(vd, renderer, volumes32);
    }
}

void vvRayCaster::Impl::updateTransfuncTexture(vvVolDesc* vd, vvRenderer* /*renderer*/)
{
    transfuncs.resize(vd->tf.size());
    for (size_t i = 0; i < vd->tf.size(); ++i)
    {
        aligned_vector<vec4> tf(256 * 1 * 1);
        vd->computeTFTexture(i, 256, 1, 1, reinterpret_cast<float*>(tf.data()));

        transfuncs[i] = transfunc_type(tf.size());
        transfuncs[i].reset(tf.data());
        transfuncs[i].set_address_mode(Clamp);
        transfuncs[i].set_filter_mode(Nearest);
    }
}

template <typename Volumes>
void vvRayCaster::Impl::updateVolumeTexturesImpl(vvVolDesc* vd, vvRenderer* renderer, Volumes& volumes)
{
    using Volume = typename Volumes::value_type;

    tex_filter_mode filter_mode = renderer->getParameter(vvRenderer::VV_SLICEINT).asInt() == virvo::Linear ? Linear : Nearest;
    tex_address_mode address_mode = Clamp;

    volumes.resize(vd->frames * vd->getChan());

    virvo::TextureUtil tu(vd);
    for (size_t f = 0; f < vd->frames; ++f)
    {
        for (int c = 0; c < vd->getChan(); ++c)
        {
            virvo::TextureUtil::Pointer tex_data = nullptr;

            virvo::TextureUtil::Channels channelbits = 1ULL << c;

            tex_data = tu.getTexture(virvo::vec3i(0),
                virvo::vec3i(vd->vox),
                texture_format,
                channelbits,
                f);

            size_t index = f * vd->getChan() + c;

            volumes[index] = Volume(vd->vox[0], vd->vox[1], vd->vox[2]);
            volumes[index].reset(reinterpret_cast<typename Volume::value_type const*>(tex_data));
            volumes[index].set_address_mode(address_mode);
            volumes[index].set_filter_mode(filter_mode);
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Public interface
//

vvRayCaster::vvRayCaster(vvVolDesc* vd, vvRenderState renderState)
    : vvRenderer(vd, renderState)
    , impl_(new Impl)
{
    rendererType = RAYREND;

    glewInit();

#if defined(VV_ARCH_CUDA)
    virvo::cuda::initGlInterop();

    virvo::RenderTarget* rt = virvo::PixelUnpackBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);

    // no direct rendering
    if (rt == NULL)
    {
        rt = virvo::DeviceBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);
    }
    setRenderTarget(rt);
#else
    setRenderTarget(virvo::HostBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED));
#endif

    updateVolumeData();
    updateTransferFunction();
}

vvRayCaster::~vvRayCaster()
{
}

void vvRayCaster::renderVolumeGL()
{
    mat4 view_matrix;
    mat4 proj_matrix;
    recti viewport;

    glGetFloatv(GL_MODELVIEW_MATRIX, view_matrix.data());
    glGetFloatv(GL_PROJECTION_MATRIX, proj_matrix.data());
    glGetIntegerv(GL_VIEWPORT, viewport.data());

    virvo::RenderTarget* rt = getRenderTarget();

    assert(rt);

    virvo_render_target virvo_rt(
        rt->width(),
        rt->height(),
        static_cast<virvo_render_target::color_type*>(rt->deviceColor()),
        static_cast<virvo_render_target::depth_type*>(rt->deviceDepth())
        );

    auto sparams = make_sched_params(
        view_matrix,
        proj_matrix,
        virvo_rt
        );

    // determine ray integration step size (aka delta)
    int axis = 0;
    if (vd->getSize()[1] / vd->vox[1] < vd->getSize()[axis] / vd->vox[axis])
    {
        axis = 1;
    }
    if (vd->getSize()[2] / vd->vox[2] < vd->getSize()[axis] / vd->vox[axis])
    {
        axis = 2;
    }

    float delta = (vd->getSize()[axis] / vd->vox[axis]) / _quality;

    auto bbox = vd->getBoundingBox();

    // Get OpenGL depth buffer to clip against
    pixel_format depth_format = PF_UNSPECIFIED;

    bool depth_test = glIsEnabled(GL_DEPTH_TEST);

    if (depth_test)
    {
        GLint depth_bits = 0;
        glGetFramebufferAttachmentParameteriv(
                GL_FRAMEBUFFER,
                GL_DEPTH,
                GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE,
                &depth_bits
                );

        GLint stencil_bits = 0;
        glGetFramebufferAttachmentParameteriv(
                GL_FRAMEBUFFER,
                GL_STENCIL,
                GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE,
                &stencil_bits
                );


        // TODO: make this more general
        // 24-bit depth buffer and 8-bit stencil buffer
        // is however a quite common case
        depth_format = (depth_bits == 24 && stencil_bits == 8) ? PF_DEPTH24_STENCIL8 : PF_DEPTH32F;

#ifdef __APPLE__
        // PIXEL_PACK_BUFFER with unsigned does not work
        // on Mac OS X, default to 32-bit floating point
        // depth buffer
        depth_format = PF_DEPTH32F;
#endif

        impl_->depth_buffer.map(viewport, depth_format);
        depth_test = true;
    }

    // assemble clip objects
    aligned_vector<typename Impl::params_type::clip_object> clip_objects;

#if 0
    // OpenGL clip planes
    for (int i = 0; i < GL_MAX_CLIP_PLANES; ++i)
    {
        if (!glIsEnabled(GL_CLIP_PLANE0 + i))
        {
            continue;
        }

        GLdouble eq[4] = { 0, 0, 0, 0 };
        glGetClipPlane(GL_CLIP_PLANE0 + i, eq);

        clip_plane pl;
        pl.normal = vec3(eq[0], eq[1], eq[2]);
        pl.offset = eq[3];
        clip_objects.push_back(pl);
    }
#else
/*    auto s0 = vvClipSphere::create();
    s0->center = virvo::vec3(0, 0, 50);
    s0->radius = 50.0f;
    setParameter(VV_CLIP_OBJ0, s0);
    setParameter(VV_CLIP_OBJ_ACTIVE0, true);*/

/*    auto c0 = vvClipCone::create();
    c0->tip = virvo::vec3(0, 0, 0);
    c0->axis = virvo::vec3(0, 0, -1);
    c0->theta = 40.0f * constants::degrees_to_radians<float>();
    setParameter(VV_CLIP_OBJ0, c0);
    setParameter(VV_CLIP_OBJ_ACTIVE0, true);*/

    typedef vvRenderState::ParameterType PT;
    PT act_id = VV_CLIP_OBJ_ACTIVE0;
    PT obj_id = VV_CLIP_OBJ0;

    for ( ; act_id != VV_CLIP_OBJ_ACTIVE_LAST && obj_id != VV_CLIP_OBJ_LAST
          ; act_id = PT(act_id + 1), obj_id = PT(obj_id + 1))
    {
        if (getParameter(act_id))
        {
            auto obj = getParameter(obj_id).asClipObj();

            if (auto plane = boost::dynamic_pointer_cast<vvClipPlane>(obj))
            {
                clip_plane pl;
                pl.normal = vec3(plane->normal.x, plane->normal.y, plane->normal.z);
                pl.offset = plane->offset;
                clip_objects.push_back(pl);
            }
            else if (auto sphere = boost::dynamic_pointer_cast<vvClipSphere>(obj))
            {
                clip_sphere sp;
                sp.center = vec3(sphere->center.x, sphere->center.y, sphere->center.z);
                sp.radius = sphere->radius;
                clip_objects.push_back(sp);
            }
            else if (auto cone = boost::dynamic_pointer_cast<vvClipCone>(obj))
            {
                clip_cone co;
                co.tip = vec3(cone->tip.x, cone->tip.y, cone->tip.z);
                co.axis = vec3(cone->axis.x, cone->axis.y, cone->axis.z);
                co.theta = cone->theta;
                clip_objects.push_back(co);
            }
        }
    }
#endif


    // Lights
    point_light<float> light;

    if (getParameter(VV_LIGHTING))
    {
        assert( glIsEnabled(GL_LIGHTING) );
        auto l = virvo::gl::getLight(GL_LIGHT0);
        vec4 lpos(l.position.x, l.position.y, l.position.z, l.position.w);

        light.set_position( (inverse(view_matrix) * lpos).xyz() );
        light.set_cl(vec3(l.diffuse.x, l.diffuse.y, l.diffuse.z));
        light.set_kl(l.diffuse.w);
        light.set_constant_attenuation(l.constant_attenuation);
        light.set_linear_attenuation(l.linear_attenuation);
        light.set_quadratic_attenuation(l.quadratic_attenuation);
    }


#ifdef VV_ARCH_CUDA
    // TODO: consolidate!
    thrust::device_vector<typename volume8_type::ref_type>  device_volumes8;
    auto volumes8_data = [&]()
    {
        device_volumes8.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            device_volumes8[c] = typename volume8_type::ref_type(impl_->volumes8[vd->getCurrentFrame() + c]);
        }
        return thrust::raw_pointer_cast(device_volumes8.data());
    };

    thrust::device_vector<typename volume16_type::ref_type> device_volumes16;
    auto volumes16_data = [&]()
    {
        device_volumes16.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            device_volumes16[c] = typename volume16_type::ref_type(impl_->volumes16[vd->getCurrentFrame() + c]);
        }
        return thrust::raw_pointer_cast(device_volumes16.data());
    };

    thrust::device_vector<typename volume32_type::ref_type> device_volumes32;
    auto volumes32_data = [&]()
    {
        device_volumes32.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            device_volumes32[c] = typename volume32_type::ref_type(impl_->volumes32[vd->getCurrentFrame() + c]);
        }
        return thrust::raw_pointer_cast(device_volumes32.data());
    };

    std::vector<typename transfunc_type::ref_type> trefs;
    for (const auto &tf : impl_->transfuncs)
        trefs.push_back(tf);
    thrust::device_vector<typename transfunc_type::ref_type> device_transfuncs(trefs);

    auto transfuncs_data = [&]()
    {
        return thrust::raw_pointer_cast(device_transfuncs.data());
    };

    thrust::device_vector<vec2> device_ranges;
    auto ranges_data = [&]()
    {
        for (int c = 0; c < vd->getChan(); ++c)
        {
            device_ranges.push_back(vec2(vd->range(c).x, vd->range(c).y));
        }

        return thrust::raw_pointer_cast(device_ranges.data());
    };

    thrust::device_vector<typename Impl::params_type::clip_object> device_objects(clip_objects);
    auto clip_objects_begin = [&]()
    {
        return thrust::raw_pointer_cast(device_objects.data());
    };

    auto clip_objects_end = [&]()
    {
        return clip_objects_begin() + device_objects.size();
    };
#else
    aligned_vector<typename volume8_type::ref_type>  host_volumes8;
    auto volumes8_data = [&]()
    {
        host_volumes8.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            host_volumes8[c] = typename volume8_type::ref_type(impl_->volumes8[vd->getCurrentFrame() + c]);
        }
        return host_volumes8.data();
    };

    aligned_vector<typename volume16_type::ref_type> host_volumes16;
    auto volumes16_data = [&]()
    {
        host_volumes16.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            host_volumes16[c] = typename volume16_type::ref_type(impl_->volumes16[vd->getCurrentFrame() + c]);
        }
        return host_volumes16.data();
    };

    aligned_vector<typename volume32_type::ref_type> host_volumes32;
    auto volumes32_data = [&]()
    {
        host_volumes32.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            host_volumes32[c] = typename volume32_type::ref_type(impl_->volumes32[vd->getCurrentFrame() + c]);
        }
        return host_volumes32.data();
    };

    aligned_vector<typename transfunc_type::ref_type> host_transfuncs(impl_->transfuncs.size());
    auto transfuncs_data = [&]()
    {
        for (size_t i = 0; i < impl_->transfuncs.size(); ++i)
        {
            host_transfuncs[i] = typename transfunc_type::ref_type(impl_->transfuncs[i]);
        }
        return host_transfuncs.data();
    };

    aligned_vector<vec2> host_ranges;
    auto ranges_data = [&]()
    {
        for (int c = 0; c < vd->getChan(); ++c)
        {
            host_ranges.push_back(vec2(vd->range(c).x, vd->range(c).y));
        }

        return host_ranges.data();
    };

    auto clip_objects_begin = [&]()
    {
        return clip_objects.data();
    };

    auto clip_objects_end = [&]()
    {
        return clip_objects.data() + clip_objects.size();
    };
#endif


    // assemble volume kernel params and call kernel
    impl_->params.bbox                      = clip_box( vec3(bbox.min.data()), vec3(bbox.max.data()) );
    impl_->params.delta                     = delta;
    impl_->params.num_channels              = vd->getChan();
    impl_->params.transfuncs                = transfuncs_data();
    impl_->params.ranges                    = ranges_data();
    impl_->params.depth_buffer              = impl_->depth_buffer.data();
    impl_->params.depth_format              = depth_format;
    impl_->params.mode                      = Impl::params_type::projection_mode(getParameter(VV_MIP_MODE).asInt());
    impl_->params.depth_test                = depth_test;
    impl_->params.opacity_correction        = getParameter(VV_OPCORR);
    impl_->params.early_ray_termination     = getParameter(VV_TERMINATEEARLY);
    impl_->params.local_shading             = getParameter(VV_LIGHTING);
    impl_->params.camera_matrix_inv         = inverse(proj_matrix * view_matrix);
    impl_->params.viewport                  = viewport;
    impl_->params.light                     = light;
    impl_->params.clip_objects.begin        = clip_objects_begin();
    impl_->params.clip_objects.end          = clip_objects_end();

    if (impl_->texture_format == virvo::PF_R8)
    {
        volume_kernel<volume8_type> kernel(impl_->params, volumes8_data());
        impl_->sched.frame(kernel, sparams);
    }
    else if (impl_->texture_format == virvo::PF_R16UI)
    {
        volume_kernel<volume16_type> kernel(impl_->params, volumes16_data());
        impl_->sched.frame(kernel, sparams);
    }
    else if (impl_->texture_format == virvo::PF_R32F)
    {
        volume_kernel<volume32_type> kernel(impl_->params, volumes32_data());
        impl_->sched.frame(kernel, sparams);
    }

    if (depth_test)
    {
        impl_->depth_buffer.unmap();
    }
}

void vvRayCaster::updateTransferFunction()
{
    impl_->updateTransfuncTexture(vd, this);
}

void vvRayCaster::updateVolumeData()
{
    impl_->updateVolumeTextures(vd, this);
}

bool vvRayCaster::checkParameter(ParameterType param, vvParam const& value) const
{
    switch (param)
    {
    case VV_SLICEINT:
        {
            virvo::tex_filter_mode mode = static_cast< virvo::tex_filter_mode >(value.asInt());

            if (mode == virvo::Nearest || mode == virvo::Linear)
            {
                return true;
            }
        }
        return false;

    case VV_CLIP_OBJ0:
    case VV_CLIP_OBJ1:
    case VV_CLIP_OBJ2:
    case VV_CLIP_OBJ3:
    case VV_CLIP_OBJ4:
    case VV_CLIP_OBJ5:
    case VV_CLIP_OBJ6:
    case VV_CLIP_OBJ7:
        return true;

    default:
        return vvRenderer::checkParameter(param, value);
    }
}

void vvRayCaster::setParameter(ParameterType param, vvParam const& value)
{
    switch (param)
    {
    case VV_SLICEINT:
        {
            if (_interpolation != static_cast< virvo::tex_filter_mode >(value.asInt()))
            {
                _interpolation = static_cast< virvo::tex_filter_mode >(value.asInt());
                tex_filter_mode filter_mode = _interpolation == virvo::Linear ? Linear : Nearest;

                for (auto& tex : impl_->volumes8)
                {
                    tex.set_filter_mode(filter_mode);
                }

                for (auto& tex : impl_->volumes16)
                {
                    tex.set_filter_mode(filter_mode);
                }

                for (auto& tex : impl_->volumes32)
                {
                    tex.set_filter_mode(filter_mode);
                }
            }
        }
        break;

    default:
        vvRenderer::setParameter(param, value);
        break;
    }
}

bool vvRayCaster::instantClassification() const
{
    return true;
}

vvRenderer* createRayCaster(vvVolDesc* vd, vvRenderState const& rs)
{
    return new vvRayCaster(vd, rs);
}
