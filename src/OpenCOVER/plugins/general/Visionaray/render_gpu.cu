/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/Geometry>

#include <visionaray/kernels.h>

#include "kernels/bvh_costs_kernel.h"
#include "kernels/normals_kernel.h"
#include "kernels/tex_coords_kernel.h"
#include "mask_intersector.h"
#include "render_gpu.h"
#include "renderer.h"

void render_gpu(
        const visionaray::cuda_index_bvh<visionaray::basic_triangle<3, float>>::bvh_ref* pbegin,
        const visionaray::cuda_index_bvh<visionaray::basic_triangle<3, float>>::bvh_ref* pend,
        two_array_ref<thrust::device_vector<visionaray::vec3>> const& normals,
        two_array_ref<thrust::device_vector<visionaray::vec2>> const& tex_coords,
        two_array_ref<device_material_list> const& materials,
        two_array_ref<thrust::device_vector<visionaray::vec3>> const& colors,
        two_array_ref<device_texture_list> const& texture_refs,
        const visionaray::spot_light<float>* lbegin,
        const visionaray::spot_light<float>* lend,
        unsigned bounces,
        float epsilon,
        visionaray::vec4 clear_color,
        visionaray::vec4 ambient,
        visionaray::cuda_sched<device_ray_type>& sched,
        visionaray::mat4 view_matrix,
        visionaray::mat4 proj_matrix,
        visionaray::renderer* rend,
        std::shared_ptr<visionaray::render_state> state,
        std::shared_ptr<visionaray::debug_state> dev_state
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


    auto sparams = make_sched_params(view_matrix,
                                     proj_matrix,
                                     *rend);

    auto sparams_isect = make_sched_params(view_matrix,
                                     proj_matrix,
                                     *rend,
                                     intersector);

    using KParams = decltype(kparams);

    // debug kernels
    if (dev_state->debug_mode && dev_state->show_bvh_costs)
    {
        // sparams w/o intersector
        bvh_costs_kernel<KParams> k(kparams);
        sched.frame(k, sparams);
    }
    else if (dev_state->debug_mode && dev_state->show_geometric_normals)
    {
        normals_kernel<KParams> k(kparams, normals_kernel<KParams>::GeometricNormals);
        sched.frame(k, sparams);
    }
    else if (dev_state->debug_mode && dev_state->show_shading_normals)
    {
        normals_kernel<KParams> k(kparams, normals_kernel<KParams>::ShadingNormals);
        sched.frame(k, sparams);
    }
    else if (dev_state->debug_mode && dev_state->show_tex_coords)
    {
        tex_coords_kernel<KParams> k(kparams);
        sched.frame(k, sparams);
    }

    // non-debug kernels
    else if (state->algo == Simple)
    {
        simple::kernel<KParams> k;
        k.params = kparams;
        sched.frame(k, sparams_isect);
    }
    else if (state->algo == Whitted)
    {
        whitted::kernel<KParams> k;
        k.params = kparams;
        sched.frame(k, sparams_isect);
    }
}
