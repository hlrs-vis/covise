// #ifdef __CUDACC__
// #include <thrust/device_vector.h>
// #include <visionaray/pixel_unpack_buffer_rt.h>
// #endif

// #include "ColorSphere.h"
# include "exa/mat4.h"


// //Visionaray helpers
// template <typename P>
// using array_ref_bvh       = visionaray::index_bvh_t<visionaray::array_ref<P>, visionaray::aligned_vector<visionaray::bvh_node>, visionaray::aligned_vector<unsigned>>;
// using sphere_type         = visionaray::ColorSphere;
// //using color_type          = visionaray::vector<3, visionaray::unorm<8>>;
// using host_bvh_type       = visionaray::index_bvh<sphere_type>; // TODO
// using point_vector        = visionaray::aligned_vector<sphere_type, 32>;
// //using color_vector        = visionaray::aligned_vector<color_type, 32>;

// #ifdef __CUDACC__
// using ray_type            = visionaray::basic_ray<float>;
// using sched_type          = visionaray::cuda_sched<ray_type>;
// using render_target_type  = visionaray::pixel_unpack_buffer_rt<visionaray::PF_RGBA8, visionaray::PF_DEPTH24_STENCIL8>;
// using device_bvh_type     = visionaray::cuda_index_bvh<sphere_type>;
// using bvh_ref             = device_bvh_type::bvh_ref;
// //using device_color_vector = thrust::device_vector<color_type>;
// #else
// using ray_type            = visionaray::basic_ray<visionaray::simd::float4>;
// using sched_type          = visionaray::tiled_sched<ray_type>;
// using render_target_type  = visionaray::cpu_buffer_rt<visionaray::PF_RGBA8, visionaray::PF_DEPTH24_STENCIL8>;
// using bvh_ref             = host_bvh_type::bvh_ref;
// #endif
namespace opencover {
// enum eye
// {
//     Left,
//     Right
// };

struct viewing_params {
math::mat4f view_matrix;
math::mat4f proj_matrix;
int width;
int height;
};
}