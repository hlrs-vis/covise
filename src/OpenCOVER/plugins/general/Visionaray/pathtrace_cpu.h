/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifndef VSNRAY_PLUGIN_PATHTRACE_CPU_H
#define VSNRAY_PLUGIN_PATHTRACE_CPU_H 1

#include <visionaray/math/math.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/area_light.h>
#include <visionaray/scheduler.h>

#include "common.h"
#include "two_array_ref.h"

namespace visionaray
{
    class renderer;
}

void pathtrace_cpu(
        const visionaray::index_bvh<visionaray::basic_triangle<3, float>>::bvh_ref* pbegin,
        const visionaray::index_bvh<visionaray::basic_triangle<3, float>>::bvh_ref* pend,
        two_array_ref<visionaray::aligned_vector<visionaray::vec3>> const& normals,
        two_array_ref<visionaray::aligned_vector<visionaray::vec2>> const& tex_coords,
        two_array_ref<material_list> const& materials,
        two_array_ref<visionaray::aligned_vector<visionaray::vec3>> const& colors,
        two_array_ref<texture_list> const& texture_refs,
        const visionaray::area_light<float, visionaray::basic_triangle<3, float>>* lbegin,
        const visionaray::area_light<float, visionaray::basic_triangle<3, float>>* lend,
        unsigned bounces,
        float epsilon,
        visionaray::vec4 clear_color,
        visionaray::vec4 ambient,
        visionaray::tiled_sched<host_ray_type>& sched,
        visionaray::mat4 view_matrix,
        visionaray::mat4 proj_matrix,
        visionaray::renderer* rend,
        unsigned& frame_num
        );

#endif // VSNRAY_PLUGIN_PATHTRACE_CPU_H
