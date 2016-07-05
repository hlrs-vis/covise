#ifndef POINT_RAY_TRACER_GLOBALS_H
#define POINT_RAY_TRACER_GLOBALS_H

#define USEOLD 0

#include <visionaray/array_ref.h>
#include <visionaray/bvh.h>
#include <visionaray/camera.h>
#include <visionaray/scheduler.h>
#include <visionaray/traverse.h>
#include <visionaray/cpu_buffer_rt.h>

//Visionaray helpers
template <typename P>
using array_ref_bvh = visionaray::index_bvh_t<visionaray::array_ref<P>, visionaray::aligned_vector<visionaray::bvh_node>, visionaray::aligned_vector<unsigned>>;
using host_ray_type = visionaray::basic_ray<visionaray::simd::float4>;
using sphere_type   = visionaray::basic_sphere<float>;
using color_type    = visionaray::vector<3, visionaray::unorm<8>>;
using host_bvh_type = array_ref_bvh<sphere_type>;
using bvh_ref       = host_bvh_type::bvh_ref;
using host_render_target_type = visionaray::cpu_buffer_rt<visionaray::PF_RGBA8, visionaray::PF_DEPTH24_STENCIL8>;

enum eye
{
    Left,
    Right
};

struct viewing_params {
#if(USEOLD)
    host_render_target_type host_rt;
#endif
    visionaray::mat4 view_matrix;
    visionaray::mat4 proj_matrix;
    int width;
    int height;
};


#endif //POINT_RAY_TRACER_GLOBALS_H
