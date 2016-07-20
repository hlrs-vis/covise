/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef POINT_RAY_TRACER_KERNEL_H
#define POINT_RAY_TRACER_KERNEL_H

#include <visionaray/get_color.h>

#include "PointRayTracerGlobals.h"

// kernel with ray tracing logic
template <typename BVHs, typename Colors>
struct Kernel
{
    using R = ray_type;
    using S = R::scalar_type;
    using C = visionaray::vector<4, S>;

    Kernel(BVHs bvhs_begin, BVHs bvhs_end, Colors colors)
        : m_bvhs_begin(bvhs_begin)
        , m_bvhs_end(bvhs_end)
        , m_colors(colors)
    {
    }

    VSNRAY_FUNC
    visionaray::result_record<S> operator()(R ray)
    {
        const auto bgcolor = visionaray::vec3(0.2,0.2,0.2);

        visionaray::result_record<S> result;
        result.color = C(bgcolor, 1.0f);

        auto hit_rec = visionaray::closest_hit(
                ray,
                m_bvhs_begin,
                m_bvhs_end
                );

        result.hit = hit_rec.hit;
        result.isect_pos = ray.ori + ray.dir * hit_rec.t;

        auto color = get_color(m_colors,hit_rec,bvh_ref(),visionaray::colors_per_face_binding());

        result.color = select(
                hit_rec.hit,
                C(visionaray::vector<3, S>(color), S(1.0)),
                result.color
                );

        return result;
    }



    BVHs m_bvhs_begin;
    BVHs m_bvhs_end;
    Colors m_colors;
    /*const bvh_ref    *m_bvhs_begin;
    const bvh_ref    *m_bvhs_end;
    const color_type *m_colors;*/
};

#endif // POINT_RAY_TRACER_KERNEL_H
