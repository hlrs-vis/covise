/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <algorithm>
#include <map>
#include <string>

#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Texture>

#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/bvh.h>
#include <visionaray/generic_material.h>
#include <visionaray/scheduler.h>
#include <visionaray/spot_light.h>


//-------------------------------------------------------------------------------------------------
// Type definitions
//

using triangle_type = visionaray::basic_triangle<3, float>;
using triangle_list = visionaray::aligned_vector<triangle_type>;
using normal_list = visionaray::aligned_vector<visionaray::vec3>;
using tex_coord_list = visionaray::aligned_vector<visionaray::vec2>;
using material_type = visionaray::generic_material<visionaray::matte<float>,
                                                   visionaray::plastic<float>,
                                                   visionaray::glass<float>,
                                                   visionaray::emissive<float> >;
using material_list = visionaray::aligned_vector<material_type>;
using vertex_color_type = visionaray::vector<3, float>;
using color_list = visionaray::aligned_vector<vertex_color_type>;
using light_type = visionaray::spot_light<float>;
using light_list = visionaray::aligned_vector<light_type>;
using node_mask_map = std::map<osg::ref_ptr<osg::Node>, osg::Node::NodeMask>;

using host_tex_type = visionaray::texture<visionaray::vector<4, visionaray::unorm<8> >, 2>;
using host_tex_ref_type = typename host_tex_type::ref_type;
using texture_list = visionaray::aligned_vector<host_tex_ref_type>;
using texture_map = std::map<std::string, host_tex_type>;

using host_ray_type = visionaray::basic_ray<visionaray::simd::float4>;
using host_bvh_type = visionaray::index_bvh<triangle_type>;
using host_sched_type = visionaray::tiled_sched<host_ray_type>;

#ifdef __CUDACC__
using device_normal_list = thrust::device_vector<visionaray::vec3>;
using device_tex_coord_list = thrust::device_vector<visionaray::vec2>;
using device_material_list = thrust::device_vector<material_type>;
using device_color_list = thrust::device_vector<vertex_color_type>;
using device_tex_type = visionaray::cuda_texture<visionaray::vector<4, visionaray::unorm<8> >, 2>;
using device_tex_ref_type = typename device_tex_type::ref_type;
using device_texture_list = thrust::device_vector<device_tex_ref_type>;
using device_texture_map = std::map<std::string, device_tex_type>;
using device_ray_type = visionaray::basic_ray<float>;
using device_bvh_type = visionaray::cuda_index_bvh<triangle_type>;
using device_sched_type = visionaray::cuda_sched<device_ray_type>;
#endif


//-------------------------------------------------------------------------------------------------
// Conversion between osg and visionaray
//

inline visionaray::vec2 osg_cast(osg::Vec2 const &v)
{
    return visionaray::vec2(v.x(), v.y());
}

inline visionaray::vec3 osg_cast(osg::Vec3 const &v)
{
    return visionaray::vec3(v.x(), v.y(), v.z());
}

inline visionaray::vec4 osg_cast(osg::Vec4 const &v)
{
    return visionaray::vec4(v.x(), v.y(), v.z(), v.w());
}

inline visionaray::mat4 osg_cast(osg::Matrixd const &m)
{
    float arr[16];
    std::copy(m.ptr(), m.ptr() + 16, arr);
    return visionaray::mat4(arr);
}

inline visionaray::tex_address_mode osg_cast(osg::Texture::WrapMode mode)
{
    switch (mode)
    {

    default:
    // fall-through
    case osg::Texture::CLAMP:
        return visionaray::Clamp;

    case osg::Texture::REPEAT:
        return visionaray::Wrap;

    case osg::Texture::MIRROR:
        return visionaray::Mirror;
    }
}

inline material_type osg_cast(osg::Material const *mat, bool specular = true)
{
    using namespace visionaray;

    auto ca = mat->getAmbient(osg::Material::Face::FRONT);
    auto cd = mat->getDiffuse(osg::Material::Face::FRONT);
    auto cs = mat->getSpecular(osg::Material::Face::FRONT);
    auto ce = mat->getEmission(osg::Material::Face::FRONT);

    auto alpha = std::min(ca.a(), std::min(cd.a(), std::min(cs.a(), 1.0f)));

    if (alpha < 1.0f)
    {
        glass<float> vsnray_mat;
        vsnray_mat.ct() = from_rgb(osg_cast(cd).xyz());
        vsnray_mat.cr() = from_rgb(osg_cast(cs).xyz());
        vsnray_mat.kt() = alpha;
        vsnray_mat.kr() = 1.0f - alpha;
        vsnray_mat.ior() = spectrum<float>(1.3f); // TODO
        return material_type(vsnray_mat);
    }
    else if (ce[0] > 0.0f || ce[1] > 0.0f || ce[2] > 0.0f)
    {
        emissive<float> vsnray_mat;
        vsnray_mat.ce() = from_rgb(osg_cast(ce).xyz());
        vsnray_mat.ls() = 1.0f;
        return material_type(vsnray_mat);
    }
    else if ((cs[0] == 0.0f && cs[1] == 0.0f && cs[2] == 0.0f) || !specular)
    {
        matte<float> vsnray_mat;
        vsnray_mat.ca() = from_rgb(osg_cast(ca).xyz());
        vsnray_mat.cd() = from_rgb(osg_cast(cd).xyz());
        vsnray_mat.ka() = 1.0f;
        vsnray_mat.kd() = 1.0f;
        return material_type(vsnray_mat);
    }
    else
    {
        plastic<float> vsnray_mat;
        vsnray_mat.ca() = from_rgb(osg_cast(ca).xyz());
        vsnray_mat.cd() = from_rgb(osg_cast(cd).xyz());
        vsnray_mat.cs() = from_rgb(osg_cast(cs).xyz());
        vsnray_mat.ka() = 1.0f;
        vsnray_mat.kd() = 1.0f;
        vsnray_mat.ks() = 1.0f;
        vsnray_mat.specular_exp() = mat->getShininess(osg::Material::Face::FRONT);
        return material_type(vsnray_mat);
    }
}

inline visionaray::tex_filter_mode osg_cast(osg::Texture::FilterMode mode)
{
    switch (mode)
    {

    default:
    // fall-through
    case osg::Texture::LINEAR:
    case osg::Texture::LINEAR_MIPMAP_LINEAR:
    case osg::Texture::LINEAR_MIPMAP_NEAREST:
        return visionaray::Linear;

    case osg::Texture::NEAREST:
    case osg::Texture::NEAREST_MIPMAP_LINEAR:
    case osg::Texture::NEAREST_MIPMAP_NEAREST:
        return visionaray::Nearest;
    }
}
