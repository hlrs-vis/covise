// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_DATASET_H_
#define REN_DATASET_H_

#include <vector>
#include <fstream>
#include <sstream>

#include <lamure/utils.h>
#include <lamure/types.h>
#include <lamure/ren/platform.h>
#include <lamure/ren/bvh.h>
#include <lamure/ren/config.h>
#include <scm/gl_core/primitives/box.h>

namespace lamure {
namespace ren {

class model_database;

class dataset
{

public:

    struct serialized_surfel {
      float x, y, z;
      uint8_t r, g, b, fake;
      float size;
      float nx, ny, nz;
    };
    struct serialized_triangle {
      float va_x_, va_y_, va_z_;
      float n0_x_, n0_y_, n0_z_;
      float c0_x_, c0_y_;
      float vb_x_, vb_y_, vb_z_;
      float n1_x_, n1_y_, n1_z_;
      float c1_x_, c1_y_;
      float vc_x_, vc_y_, vc_z_;
      float n2_x_, n2_y_, n2_z_;
      float c2_x_, c2_y_;
    };
    struct serialized_vertex {
      float v_x_, v_y_, v_z_;
      float n_x_, n_y_, n_z_;
      float c_x_, c_y_;
    };
    struct serialized_surfel_qz {
      uint16_t x, y, z; // quantized pos between node extents
      uint16_t n_enum;  // enumerated point on unit cube
      uint16_t rgb_565; // default 565 color quantization
      uint16_t size;    // quantized radius between avg rad and max deviation
    };


                        dataset() {};
                        dataset(const std::string& filename);
    virtual             ~dataset();

    const model_t       model_id() const { return model_id_; };
    const scm::gl::boxf& aabb() const { return aabb_; };
    const bool          is_loaded() const { return is_loaded_; };
    const bvh*          get_bvh() const { return bvh_; };
    
    void                set_transform(const scm::math::mat4f& transform) { transform_ = transform; };
    const scm::math::mat4f transform() const { return transform_; };

protected:
    void                load(const std::string& filename);

    friend class        model_database;
    model_t             model_id_;

private:
    scm::gl::boxf       aabb_;
    bool                is_loaded_;
    bvh*                bvh_;

    scm::math::mat4f    transform_;

};


} } // namespace lamure


#endif // REN_LOD_POINT_CLOUD_H_
