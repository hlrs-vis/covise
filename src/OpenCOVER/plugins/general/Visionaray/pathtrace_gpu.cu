/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/Geometry>

#include <visionaray/kernels.h>

#include "mask_intersector.h"
#include "pathtrace_gpu.h"
#include "renderer.h"

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
        )
{
    using namespace visionaray;

    auto kparams = make_kernel_params(
        normals_per_vertex_binding{},
        colors_per_vertex_binding{},
        pbegin,
        pend,
        normals,
        tex_coords,
        materials,
        colors,
        texture_refs,
        lbegin,
        lend,
        bounces,
        epsilon,
        clear_color,
        vec4(1.0f));

    mask_intersector<
        two_array_ref<device_tex_coord_list>,
        two_array_ref<device_texture_list>> intersector;

    intersector.tex_coords = kparams.tex_coords;
    intersector.textures = kparams.textures;


    auto sparams = make_sched_params(pixel_sampler::jittered_blend_type{},
                                     view_matrix,
                                     proj_matrix,
                                     *rend,
                                     intersector);

    pathtracing::kernel<decltype(kparams)> k;
    k.params = kparams;
    sched.frame(k, sparams, ++frame_num);
}
