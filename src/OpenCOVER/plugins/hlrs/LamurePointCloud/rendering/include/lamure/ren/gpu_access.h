// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_REN_GPU_ACCESS_H
#define LAMURE_REN_GPU_ACCESS_H

#include <lamure/ren/bvh.h>
#include <lamure/ren/config.h>
#include <lamure/types.h>

#include <boost/assign/list_of.hpp>
#include <lamure/ren/data_provenance.h>
#include <memory>
#include <scm/core.h>
#include <scm/gl_core.h>

namespace lamure
{
namespace ren
{
class gpu_access
{
  public:
    gpu_access(scm::gl::render_device_ptr device, const slot_t num_slots, const uint32_t num_surfels_per_node, bool create_layout = true);
    gpu_access(scm::gl::render_device_ptr device, const slot_t num_slots, const uint32_t num_surfels_per_node, Data_Provenance const &data_provenance, bool create_layout = true);
    ~gpu_access();

    const slot_t num_slots() const { return num_slots_; };
    const size_t size_of_surfel() const { return size_of_surfel_; };
    const size_t size_of_slot() const { return size_of_slot_; };

    char *map(scm::gl::render_device_ptr const &device);
    void unmap(scm::gl::render_device_ptr const &device);
    char *map_provenance(scm::gl::render_device_ptr const &device);
    void unmap_provenance(scm::gl::render_device_ptr const &device);
    const bool is_mapped() const { return is_mapped_; };
    const bool has_layout() const { return has_layout_; };

    scm::gl::buffer_ptr get_buffer() { return buffer_; };
    scm::gl::buffer_ptr get_buffer_provenance() { return buffer_provenance_; };

    scm::gl::vertex_array_ptr get_memory(bvh::primitive_type type);

    static const size_t query_video_memory_in_mb(scm::gl::render_device_ptr const &device);

  private:
    slot_t num_slots_;
    size_t size_of_slot_;
    size_t size_of_slot_provenance_;
    size_t size_of_surfel_;
    size_t size_of_surfel_qz_;
    
    bool is_mapped_;
    bool is_mapped_provenance_;
    bool has_layout_;


    scm::gl::vertex_array_ptr pcl_memory_; // vertex layout for uncompressed point clouds
    scm::gl::vertex_array_ptr pcl_qz_memory_; // vertex layout for quantized point clouds
    scm::gl::vertex_array_ptr tri_memory_; // vertex layout for uncompressed tri mesh

    scm::gl::buffer_ptr buffer_;
    scm::gl::buffer_ptr buffer_provenance_;
};
}
}

#endif
