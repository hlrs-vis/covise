/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifndef VSNRAY_PLUGIN_PATHTRACE_GPU_H
#define VSNRAY_PLUGIN_PATHTRACE_GPU_H 1

#include <thrust/device_vector.h>

#include <visionaray/math/math.h>
#include <visionaray/area_light.h>
#include <visionaray/scheduler.h>

#include "common.h"
#include "two_array_ref.h"

namespace visionaray
{
    class renderer;
}

void pathtrace_gpu(
        const visionaray::cuda_index_bvh<visionaray::basic_triangle<3, float>>::bvh_ref* pbegin,
        const visionaray::cuda_index_bvh<visionaray::basic_triangle<3, float>>::bvh_ref* pend,
        two_array_ref<thrust::device_vector<visionaray::vec3>> const& normals,
        two_array_ref<thrust::device_vector<visionaray::vec2>> const& tex_coords,
        two_array_ref<device_material_list> const& materials,
        two_array_ref<thrust::device_vector<visionaray::vec3>> const& colors,
        two_array_ref<device_texture_list> const& texture_refs,
        const visionaray::area_light<float, visionaray::basic_triangle<3, float>>* lbegin,
        const visionaray::area_light<float, visionaray::basic_triangle<3, float>>* lend,
        unsigned bounces,
        float epsilon,
        visionaray::vec4 clear_color,
        visionaray::vec4 ambient,
        visionaray::cuda_sched<device_ray_type>& sched,
        visionaray::mat4 view_matrix,
        visionaray::mat4 proj_matrix,
        visionaray::renderer* rend,
        unsigned& frame_num
        );

#endif // VSNRAY_PLUGIN_PATHTRACE_GPU_H
