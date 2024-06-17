#ifndef POINT_RAY_TRACER_GLOBALS_H
#define POINT_RAY_TRACER_GLOBALS_H

#include <visionaray/array_ref.h>
#include <visionaray/bvh.h>
#include <visionaray/scheduler.h>
#include <visionaray/traverse.h>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

#include "ColorSphere.h"

//Visionaray helpers
template <typename P>
using array_ref_bvh       = visionaray::index_bvh_t<visionaray::array_ref<P>, visionaray::aligned_vector<visionaray::bvh_node>, visionaray::aligned_vector<unsigned>>;
using sphere_type         = visionaray::ColorSphere;
//using color_type          = visionaray::vector<3, visionaray::unorm<8>>;
using host_bvh_type       = visionaray::index_bvh<sphere_type>; // TODO
using point_vector        = visionaray::aligned_vector<sphere_type, 32>;
//using color_vector        = visionaray::aligned_vector<color_type, 32>;

#ifdef __CUDACC__
using ray_type            = visionaray::basic_ray<float>;
using sched_type          = visionaray::cuda_sched<ray_type>;
using device_bvh_type     = visionaray::cuda_index_bvh<sphere_type>;
using bvh_ref             = device_bvh_type::bvh_ref;
//using device_color_vector = thrust::device_vector<color_type>;
#else
using ray_type            = visionaray::basic_ray<visionaray::simd::float4>;
using sched_type          = visionaray::tiled_sched<ray_type>;
using bvh_ref             = host_bvh_type::bvh_ref;
#endif

enum eye
{
    Left,
    Right
};

struct viewing_params {
    visionaray::mat4 view_matrix;
    visionaray::mat4 proj_matrix;
    int width;
    int height;
};


#endif //POINT_RAY_TRACER_GLOBALS_H
