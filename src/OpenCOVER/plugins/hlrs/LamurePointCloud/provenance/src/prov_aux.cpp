// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/prov/prov_aux.h>

#include <limits>

#include <sys/stat.h>
#include <fcntl.h>

#include <lamure/prov/aux_stream.h>
#include <lamure/prov/octree.h>

namespace lamure {
namespace prov {


aux::
aux()
: filename_("") {


} 

aux::
aux(const std::string& filename)
: filename_("") {

    std::string extension = filename.substr(filename.find_last_of(".") + 1);

    if (extension.compare("aux") == 0) {
       load_aux_file(filename);
    }
    else {
       throw std::runtime_error(
          "lamure: aux::Invalid file extension encountered.");
    }

};

void aux::
load_aux_file(const std::string& filename) {

    filename_ = filename;

    aux_stream aux_stream;
    aux_stream.read_aux(filename, *this);
}


void aux::
write_aux_file(const std::string& filename) {
    
    filename_ = filename;

    aux_stream aux_stream;
    aux_stream.write_aux(filename, *this);

}

const aux::view& aux::
get_view(const uint32_t view_id) const {
    assert(view_id >= 0 && view_id < views_.size());
    return views_[view_id];
}


const aux::sparse_point& aux::
get_sparse_point(const uint64_t point_id) const {
    assert(point_id >= 0 && point_id < sparse_points_.size());
    return sparse_points_[point_id];
}

const aux::atlas_tile& aux::
get_atlas_tile(const uint32_t tile_id) const {
    assert(tile_id >= 0 && tile_id < atlas_tiles_.size());
    return atlas_tiles_[tile_id];
}

void aux::
add_view(const aux::view& view) {
    views_.push_back(view);
}

void aux::
add_sparse_point(const aux::sparse_point& point) {
    sparse_points_.push_back(point);
}

void aux::
add_atlas_tile(const aux::atlas_tile& tile) {
    atlas_tiles_.push_back(tile);
}

void aux::set_octree(const std::shared_ptr<octree> _octree) {
  octree_ = _octree;
}

const std::shared_ptr<octree> aux::
get_octree() const {
  return octree_;
}

void aux::set_atlas(const atlas& _atlas) {
  atlas_ = _atlas;
}

const aux::atlas& aux::
get_atlas() const {
  return atlas_;
}



} } // namespace lamure

