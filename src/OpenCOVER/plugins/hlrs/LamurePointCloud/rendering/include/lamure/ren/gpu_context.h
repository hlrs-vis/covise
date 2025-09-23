// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_REN_GPU_CONTEXT_H
#define LAMURE_REN_GPU_CONTEXT_H

#include <lamure/ren/data_provenance.h>
#include <lamure/ren/bvh.h>
#include <lamure/ren/cut_database_record.h>
#include <lamure/ren/gpu_access.h>
#include <lamure/types.h>

namespace lamure
{
namespace ren
{
class gpu_context
{
  public:
    gpu_context(const context_t context_id);
    ~gpu_context();

    struct temporary_storages
    {
        temporary_storages(char *storage_a, char *storage_b) : storage_a_(storage_a), storage_b_(storage_b){};

        char *storage_a_;
        char *storage_b_;
    };
    struct fix_struct
    {
        char *fix_buffer_;
        char *fix_buffer_provenance_;
    };
    const context_t context_id() const { return context_id_; };
    const bool is_created() const { return is_created_; };

    temporary_storages get_temporary_storages() { return temporary_storages_; };
    temporary_storages get_temporary_storages_provenance() { return temporary_storages_provenance_; };

    scm::gl::buffer_ptr get_context_buffer(scm::gl::render_device_ptr device);
    scm::gl::buffer_ptr get_context_buffer(scm::gl::render_device_ptr device, Data_Provenance const &data_provenance);
    scm::gl::vertex_array_ptr get_context_memory(bvh::primitive_type type, scm::gl::render_device_ptr device);
    scm::gl::vertex_array_ptr get_context_memory(bvh::primitive_type type, scm::gl::render_device_ptr device, Data_Provenance const &data_provenance);

    const node_t upload_budget_in_nodes() const { return upload_budget_in_nodes_; };
    const node_t render_budget_in_nodes() const { return render_budget_in_nodes_; };

    void map_temporary_storage(const cut_database_record::temporary_buffer &buffer, scm::gl::render_device_ptr device);
    void map_temporary_storage(const cut_database_record::temporary_buffer &buffer, scm::gl::render_device_ptr device, Data_Provenance const &data_provenance);
    void unmap_temporary_storage(const cut_database_record::temporary_buffer &buffer, scm::gl::render_device_ptr device);
    void unmap_temporary_storage(const cut_database_record::temporary_buffer &buffer, scm::gl::render_device_ptr device, Data_Provenance const &data_provenance);
    bool update_primary_buffer(const cut_database_record::temporary_buffer &from_buffer, scm::gl::render_device_ptr device_);
    bool update_primary_buffer_fix(const cut_database_record::temporary_buffer &from_buffer, scm::gl::render_device_ptr device, Data_Provenance const &data_provenance);

    fix_struct get_fix_a() { return fix_a_; };
    fix_struct get_fix_b() { return fix_b_; };

    void create(scm::gl::render_device_ptr device);
    void create(scm::gl::render_device_ptr device, Data_Provenance const &data_provenance);

  private:
    void test_video_memory(scm::gl::render_device_ptr device);
    void test_video_memory(scm::gl::render_device_ptr device, Data_Provenance const &data_provenance);

    context_t context_id_;

    bool is_created_;

    gpu_access *temp_buffer_a_;
    gpu_access *temp_buffer_b_;

    fix_struct fix_a_;
    fix_struct fix_b_;

    gpu_access *primary_buffer_;

    temporary_storages temporary_storages_;
    temporary_storages temporary_storages_provenance_;
    node_t upload_budget_in_nodes_;
    node_t render_budget_in_nodes_;
};
}
}

#endif
