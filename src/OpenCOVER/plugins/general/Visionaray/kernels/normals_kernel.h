/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifndef VSNRAY_PLUGIN_NORMALS_KERNEL_H
#define VSNRAY_PLUGIN_NORMALS_KERNEL_H 1

#include <visionaray/get_surface.h>
#include <visionaray/result_record.h>
#include <visionaray/traverse.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Normals debug kernel
//

template <typename Params>
struct normals_kernel
{
    enum normal_type
    {
        GeometricNormals,
        ShadingNormals
    };


    VSNRAY_FUNC normals_kernel(Params const& p, normal_type t)
        : params(p)
        , type(t)
    {
    }

    template <typename R>
    VSNRAY_FUNC result_record<typename R::scalar_type> operator()(R ray) const
    {
        using S = typename R::scalar_type;
        using C = vector<4, S>;


        result_record<S> result;
        auto hit_rec        = closest_hit(ray, params.prims.begin, params.prims.end);
        auto surf           = get_surface(hit_rec, params);
        auto normal         = type == GeometricNormals ? surf.geometric_normal : surf.shading_normal;
        result.hit          = hit_rec.hit;
        result.color        = select( hit_rec.hit, C((normal + S(1.0)) / S(2.0), S(1.0)), C(0.0) );
        result.isect_pos    = ray.ori + ray.dir * hit_rec.t;
        return result;
    }

    Params params;
    normal_type type;
};

} // namespace visionaray

#endif // VSNRAY_PLUGIN_NORMALS_KERNEL_H
