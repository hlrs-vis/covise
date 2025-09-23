// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef PROV_AUX_H_
#define PROV_AUX_H_

#include <lamure/types.h>
#include <lamure/platform.h>

#include <string>
#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>

namespace lamure {
namespace prov {

class octree;

class aux
{
  public:

    struct feature {
      uint32_t camera_id_;
      uint32_t using_count_;
      scm::math::vec2f coords_;
      scm::math::vec2f error_;
    };

    struct sparse_point {
      scm::math::vec3f pos_;
      uint8_t r_;
      uint8_t g_;
      uint8_t b_;
      uint8_t a_;
      std::vector<feature> features_;
    };

    struct view {
      uint32_t camera_id_;
      scm::math::vec3f position_;
      scm::math::mat4f transform_; //trans + rot
      float focal_length_;
      float distortion_;
      uint32_t image_width_;
      uint32_t image_height_;
      uint32_t atlas_tile_id_;
      std::string image_file_;
    };

    struct atlas {
      uint32_t num_atlas_tiles_{0};
      uint32_t atlas_width_{0};
      uint32_t atlas_height_{0};
      uint32_t rotated_{0};
    };

    struct atlas_tile {
      uint32_t atlas_tile_id_{0};
      uint32_t x_{0};
      uint32_t y_{0};
      uint32_t width_{0};
      uint32_t height_{0};
    };

                        aux();
                        aux(const std::string& filename);
    virtual             ~aux() {}

    const std::string   get_filename() const { return filename_; }
    const uint32_t      get_num_views() const { return views_.size(); }
    const uint64_t      get_num_sparse_points() const { return sparse_points_.size(); }
    const uint32_t      get_num_atlas_tiles() const { return atlas_tiles_.size(); }

    const view&         get_view(uint32_t id) const;
    const sparse_point& get_sparse_point(uint64_t id) const;
    const atlas_tile&   get_atlas_tile(uint32_t id) const;
    
    void                add_view(const view& view);
    void                add_sparse_point(const sparse_point& point);
    void                add_atlas_tile(const atlas_tile& tile);

    void                set_octree(const std::shared_ptr<octree> _octree);
    const std::shared_ptr<octree> get_octree() const;

    void                set_atlas(const atlas& atlas);
    const atlas&        get_atlas() const;

    void                write_aux_file(const std::string& filename);

    std::vector<sparse_point>& get_sparse_points() { return sparse_points_; };

protected:

    void                load_aux_file(const std::string& filename);

private:

    std::vector<view> views_;
    std::vector<sparse_point> sparse_points_;
    std::vector<atlas_tile> atlas_tiles_;
    std::shared_ptr<octree> octree_;
    atlas atlas_;
    std::string filename_;

};


} } // namespace lamure


#endif // PROV_AUX_H_

