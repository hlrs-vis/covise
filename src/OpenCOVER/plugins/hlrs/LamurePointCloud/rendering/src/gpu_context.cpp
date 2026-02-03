// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/config.h>
#include <lamure/ren/cut_database.h>
#include <lamure/ren/gpu_context.h>
#include <lamure/ren/model_database.h>
#include <lamure/ren/policy.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <algorithm>
#include <iostream>

namespace lamure
{
namespace ren
{
gpu_context::gpu_context(const context_t context_id)
    : context_id_(context_id), is_created_(false), temp_buffer_a_(nullptr), temp_buffer_b_(nullptr), primary_buffer_(nullptr), temporary_storages_(temporary_storages(nullptr, nullptr)),
      temporary_storages_provenance_(temporary_storages(nullptr, nullptr)), upload_budget_in_nodes_(0), render_budget_in_nodes_(0)
{
}

gpu_context::~gpu_context()
{
    temporary_storages_ = temporary_storages(nullptr, nullptr);
    temporary_storages_provenance_ = temporary_storages(nullptr, nullptr);

    if(temp_buffer_a_)
    {
        delete temp_buffer_a_;
        temp_buffer_a_ = nullptr;
    }

    if(temp_buffer_b_)
    {
        delete temp_buffer_b_;
        temp_buffer_b_ = nullptr;
    }

    if(primary_buffer_)
    {
        delete primary_buffer_;
        primary_buffer_ = nullptr;
    }
}

void gpu_context::set_contexts(uint32_t contexts)
{
    contexts_ = contexts > 0 ? contexts : 1u;
}

void gpu_context::create(scm::gl::render_device_ptr device)
{
    assert(device);
    if(is_created_)
    {
        return;
    }
    is_created_ = true;

    test_video_memory(device);

    model_database *database = model_database::get_instance();

    assert(database);

    temp_buffer_a_ = new gpu_access(device, upload_budget_in_nodes_, database->get_primitives_per_node(), false);
    temp_buffer_b_ = new gpu_access(device, upload_budget_in_nodes_, database->get_primitives_per_node(), false);
    primary_buffer_ = new gpu_access(device, render_budget_in_nodes_, database->get_primitives_per_node(), true);

    map_temporary_storage(cut_database_record::temporary_buffer::BUFFER_A, device);
    map_temporary_storage(cut_database_record::temporary_buffer::BUFFER_B, device);
}

void gpu_context::create(scm::gl::render_device_ptr device, Data_Provenance const &data_provenance)
{
    assert(device);
    if(is_created_)
    {
        return;
    }
    is_created_ = true;

    test_video_memory(device, data_provenance);

    model_database *database = model_database::get_instance();

    fix_a_.fix_buffer_ = new char[8 * sizeof(float) * database->get_primitives_per_node() * upload_budget_in_nodes_];
    fix_a_.fix_buffer_provenance_ = new char[data_provenance.get_size_in_bytes() * database->get_primitives_per_node() * upload_budget_in_nodes_];

    fix_b_.fix_buffer_ = new char[8 * sizeof(float) * database->get_primitives_per_node() * upload_budget_in_nodes_];
    fix_b_.fix_buffer_provenance_ = new char[data_provenance.get_size_in_bytes() * database->get_primitives_per_node() * upload_budget_in_nodes_];


    primary_buffer_ = new gpu_access(device, render_budget_in_nodes_, database->get_primitives_per_node(), data_provenance, true);


}

void gpu_context::test_video_memory(scm::gl::render_device_ptr device)
{
    model_database *database = model_database::get_instance();
    policy *policy = policy::get_instance();
    size_t render_budget_in_mb = policy->render_budget_in_mb();

    float safety = 0.75f;
    size_t video_ram_free_in_mb = gpu_access::query_video_memory_in_mb(device) * safety;

    if (policy->out_of_core_budget_in_mb() == 0 || video_ram_free_in_mb < render_budget_in_mb)
    {
        if (policy->out_of_core_budget_in_mb() > 0) {
            std::cout << "##### The specified render budget is too large! " << video_ram_free_in_mb << " MB will be used for the render budget #####" << std::endl;
        }
        render_budget_in_mb = video_ram_free_in_mb;
    }
    else
    {
        std::cout << "##### " << render_budget_in_mb << " MB will be used for the render budget #####" << std::endl;
    }

    if (contexts_ > 1) {
        render_budget_in_mb = std::max<size_t>(1, render_budget_in_mb / contexts_);
    }

    long node_size_total = database->get_slot_size();
    if(node_size_total==0)
        render_budget_in_nodes_ = 0;
    else
        render_budget_in_nodes_ = (render_budget_in_mb * 1024 * 1024) / node_size_total;

    size_t max_upload_budget_in_mb = policy->max_upload_budget_in_mb();
    
    if(node_size_total==0)
        upload_budget_in_nodes_ = 0;
    else
        upload_budget_in_nodes_ = (max_upload_budget_in_mb * 1024u * 1024u) / node_size_total;

    std::cout << "[Lamure] ctx=" << context_id_
              << " contexts=" << contexts_
              << " render_budget_mb=" << render_budget_in_mb
              << " upload_budget_mb=" << max_upload_budget_in_mb
              << std::endl;
}

void gpu_context::test_video_memory(scm::gl::render_device_ptr device, Data_Provenance const &data_provenance)
{
    model_database *database = model_database::get_instance();
    policy *policy = policy::get_instance();
    size_t render_budget_in_mb = policy->render_budget_in_mb();

    float safety = 0.75f;
    size_t video_ram_free_in_mb = gpu_access::query_video_memory_in_mb(device) * safety;

    if (policy->out_of_core_budget_in_mb() == 0 || video_ram_free_in_mb < render_budget_in_mb)
    {
        if (policy->out_of_core_budget_in_mb() > 0) {
            std::cout << "##### The specified render budget is too large! " << video_ram_free_in_mb << " MB will be used for the render budget #####" << std::endl;
        }
        render_budget_in_mb = video_ram_free_in_mb;
    }
    else
    {
        std::cout << "##### " << render_budget_in_mb << " MB will be used for the render budget #####" << std::endl;
    }

    long node_size_total = database->get_primitives_per_node() * data_provenance.get_size_in_bytes() + database->get_slot_size();
    if (contexts_ > 1) {
        render_budget_in_mb = std::max<size_t>(1, render_budget_in_mb / contexts_);
    }
    render_budget_in_nodes_ = (render_budget_in_mb * 1024 * 1024) / node_size_total;

    size_t max_upload_budget_in_mb = policy->max_upload_budget_in_mb();

    upload_budget_in_nodes_ = (max_upload_budget_in_mb * 1024u * 1024u) / node_size_total;

    std::cout << "[Lamure] ctx=" << context_id_
              << " contexts=" << contexts_
              << " render_budget_mb=" << render_budget_in_mb
              << " upload_budget_mb=" << max_upload_budget_in_mb
              << " (provenance bytes=" << data_provenance.get_size_in_bytes() << ")"
              << std::endl;

#ifdef LAMURE_ENABLE_INFO
    std::cout << "lamure: context " << context_id_ << " render budget (MB): " << render_budget_in_mb << std::endl;
    std::cout << "lamure: context " << context_id_ << " upload budget (MB): " << max_upload_budget_in_mb << std::endl;
#endif
}

scm::gl::buffer_ptr gpu_context::get_context_buffer(scm::gl::render_device_ptr device)
{
    if(!is_created_)
        create(device);

    assert(device);
    return primary_buffer_->get_buffer();
}

scm::gl::buffer_ptr gpu_context::get_context_buffer(scm::gl::render_device_ptr device, Data_Provenance const &data_provenance)
{
    if(!is_created_)
        create(device, data_provenance);

    assert(device);

    return primary_buffer_->get_buffer();
}

scm::gl::vertex_array_ptr gpu_context::get_context_memory(bvh::primitive_type type, scm::gl::render_device_ptr device)
{
    if(!is_created_)
        create(device);

    assert(device);

    return primary_buffer_->get_memory(type);
}

scm::gl::vertex_array_ptr gpu_context::get_context_memory(bvh::primitive_type type, scm::gl::render_device_ptr device, Data_Provenance const &data_provenance)
{
    if(!is_created_)
        create(device, data_provenance);

    assert(device);

    return primary_buffer_->get_memory(type);
}

void gpu_context::map_temporary_storage(const cut_database_record::temporary_buffer &buffer, scm::gl::render_device_ptr device)
{
    if(!is_created_)
        create(device);

    assert(device);

    switch(buffer)
    {
    case cut_database_record::temporary_buffer::BUFFER_A:
        if(!temp_buffer_a_->is_mapped())
        {
            temporary_storages_.storage_a_ = temp_buffer_a_->map(device);
        }
        return;
        break;

    case cut_database_record::temporary_buffer::BUFFER_B:
        if(!temp_buffer_b_->is_mapped())
        {
            temporary_storages_.storage_b_ = temp_buffer_b_->map(device);
        }
        return;
        break;

    default:
        break;
    }

    throw std::runtime_error("lamure: Failed to map temporary buffer on context: " + context_id_);
}

void gpu_context::map_temporary_storage(const cut_database_record::temporary_buffer &buffer, scm::gl::render_device_ptr device, Data_Provenance const &data_provenance)
{
    if(!is_created_)
        create(device, data_provenance);

    assert(device);

    int first_error = 5;

    switch(buffer)
    {
    case cut_database_record::temporary_buffer::BUFFER_A:
        if(!temp_buffer_a_->is_mapped())
        {
            temporary_storages_.storage_a_ = temp_buffer_a_->map(device);
        }

        return;
        break;

    case cut_database_record::temporary_buffer::BUFFER_B:
        if(!temp_buffer_b_->is_mapped())
        {
            temporary_storages_.storage_b_ = temp_buffer_b_->map(device);
        }

        return;
        break;

    default:
        break;
    }

    throw std::runtime_error("lamure: Failed to map temporary buffer on context: " + context_id_);
}

void gpu_context::unmap_temporary_storage(const cut_database_record::temporary_buffer &buffer, scm::gl::render_device_ptr device)
{
    if(!is_created_)
        create(device);

    assert(device);

    switch(buffer)
    {
    case cut_database_record::temporary_buffer::BUFFER_A:
        if(temp_buffer_a_->is_mapped())
        {
            temp_buffer_a_->unmap(device);
        }
        break;

    case cut_database_record::temporary_buffer::BUFFER_B:
        if(temp_buffer_b_->is_mapped())
        {
            temp_buffer_b_->unmap(device);
        }
        break;

    default:
        break;
    }
}

void gpu_context::unmap_temporary_storage(const cut_database_record::temporary_buffer &buffer, scm::gl::render_device_ptr device, Data_Provenance const &data_provenance)
{
    if(!is_created_)
        create(device, data_provenance);

    assert(device);

    switch(buffer)
    {
    case cut_database_record::temporary_buffer::BUFFER_A:
        if(temp_buffer_a_->is_mapped())
        {
           temp_buffer_a_->unmap_provenance(device);
        }
        break;

    case cut_database_record::temporary_buffer::BUFFER_B:
        if(temp_buffer_b_->is_mapped())
        {
           temp_buffer_b_->unmap_provenance(device);
        }
        break;

    default:
        break;
    }
}

// returns true if any node has been uploaded; false otherwise
bool gpu_context::update_primary_buffer(const cut_database_record::temporary_buffer &from_buffer, scm::gl::render_device_ptr device)
{
    if(!is_created_)
        create(device);

    assert(device);

    model_database *database = model_database::get_instance();

    cut_database *cuts = cut_database::get_instance();

    size_t uploaded_nodes = 0;

    switch(from_buffer)
    {
    case cut_database_record::temporary_buffer::BUFFER_A:
    {
        if(temp_buffer_a_->is_mapped())
        {
            throw std::runtime_error("lamure: gpu_context::Failed to transfer nodes into main memory on context: " + context_id_);
        }
        std::vector<cut_database_record::slot_update_desc> &transfer_descr_list = cuts->get_updated_set(context_id_);
        if(!transfer_descr_list.empty())
        {
            uploaded_nodes += transfer_descr_list.size();

            for(const auto &transfer_desc : transfer_descr_list)
            {
                size_t offset_in_temp_VBO = transfer_desc.src_ * database->get_slot_size();
                size_t offset_in_render_VBO = transfer_desc.dst_ * database->get_slot_size();
                device->main_context()->copy_buffer_data(primary_buffer_->get_buffer(), temp_buffer_a_->get_buffer(), offset_in_render_VBO, offset_in_temp_VBO, database->get_slot_size());
            }
        }
        break;
    }

    case cut_database_record::temporary_buffer::BUFFER_B:
    {
        if(temp_buffer_b_->is_mapped())
        {
            throw std::runtime_error("lamure: gpu_context::Failed to transfer nodes into main memory on context: " + context_id_);
        }
        std::vector<cut_database_record::slot_update_desc> &transfer_descr_list = cuts->get_updated_set(context_id_);
        if(!transfer_descr_list.empty())
        {
            uploaded_nodes += transfer_descr_list.size();

            for(const auto &transfer_desc : transfer_descr_list)
            {
                size_t offset_in_temp_VBO = transfer_desc.src_ * database->get_slot_size();
                size_t offset_in_render_VBO = transfer_desc.dst_ * database->get_slot_size();
                device->main_context()->copy_buffer_data(primary_buffer_->get_buffer(), temp_buffer_b_->get_buffer(), offset_in_render_VBO, offset_in_temp_VBO, database->get_slot_size());
            }
        }
        break;
    }
    default:
        break;
    }

    return uploaded_nodes != 0;
}

bool gpu_context::update_primary_buffer_fix(const cut_database_record::temporary_buffer &from_buffer, scm::gl::render_device_ptr device, Data_Provenance const &data_provenance)
{
    if(!is_created_)
        create(device, data_provenance);

    assert(device);

    model_database *database = model_database::get_instance();

    cut_database *cuts = cut_database::get_instance();

    size_t uploaded_nodes = 0;

    switch(from_buffer)
    {
    case cut_database_record::temporary_buffer::BUFFER_A:
    {
        std::vector<cut_database_record::slot_update_desc> &transfer_descr_list = cuts->get_updated_set(context_id_);
        if(!transfer_descr_list.empty())
        {
            uploaded_nodes += transfer_descr_list.size();

            device->opengl_api().glBindBuffer(GL_ARRAY_BUFFER, primary_buffer_->get_buffer()->object_id());
            for(const auto &transfer_desc : transfer_descr_list)
            {
                size_t offset_in_temp_VBO = transfer_desc.src_ * database->get_slot_size();
                size_t offset_in_render_VBO = transfer_desc.dst_ * database->get_slot_size();
                device->opengl_api().glBufferSubData(GL_ARRAY_BUFFER, offset_in_render_VBO, database->get_slot_size(), fix_a_.fix_buffer_ + offset_in_temp_VBO);
            }

            device->opengl_api().glBindBuffer(GL_ARRAY_BUFFER, primary_buffer_->get_buffer_provenance()->object_id());
            for(const auto &transfer_desc : transfer_descr_list)
            {
                size_t offset_in_temp_VBO_provenance = transfer_desc.src_ * database->get_primitives_per_node() * data_provenance.get_size_in_bytes();
                size_t offset_in_render_VBO_provenance = transfer_desc.dst_ * database->get_primitives_per_node() * data_provenance.get_size_in_bytes();
                size_t provenance_slot_size = database->get_primitives_per_node() * data_provenance.get_size_in_bytes();
                device->opengl_api().glBufferSubData(GL_ARRAY_BUFFER, offset_in_render_VBO_provenance, provenance_slot_size, fix_a_.fix_buffer_provenance_ + offset_in_temp_VBO_provenance);
            }
        }
        break;
    }

    case cut_database_record::temporary_buffer::BUFFER_B:
    {
        std::vector<cut_database_record::slot_update_desc> &transfer_descr_list = cuts->get_updated_set(context_id_);
        if(!transfer_descr_list.empty())
        {
            uploaded_nodes += transfer_descr_list.size();

            device->opengl_api().glBindBuffer(GL_ARRAY_BUFFER, primary_buffer_->get_buffer()->object_id());
            for(const auto &transfer_desc : transfer_descr_list)
            {
                size_t offset_in_temp_VBO = transfer_desc.src_ * database->get_slot_size();
                size_t offset_in_render_VBO = transfer_desc.dst_ * database->get_slot_size();
                device->opengl_api().glBufferSubData(GL_ARRAY_BUFFER, offset_in_render_VBO, database->get_slot_size(), fix_b_.fix_buffer_ + offset_in_temp_VBO);
            }

            device->opengl_api().glBindBuffer(GL_ARRAY_BUFFER, primary_buffer_->get_buffer_provenance()->object_id());
            for(const auto &transfer_desc : transfer_descr_list)
            {
                size_t offset_in_temp_VBO_provenance = transfer_desc.src_ * database->get_primitives_per_node() * data_provenance.get_size_in_bytes();
                size_t offset_in_render_VBO_provenance = transfer_desc.dst_ * database->get_primitives_per_node() * data_provenance.get_size_in_bytes();
                size_t provenance_slot_size = database->get_primitives_per_node() * data_provenance.get_size_in_bytes();
                device->opengl_api().glBufferSubData(GL_ARRAY_BUFFER, offset_in_render_VBO_provenance, provenance_slot_size, fix_b_.fix_buffer_provenance_ + offset_in_temp_VBO_provenance);
            }
        }
        break;
    }
    default:
        break;
    }

    return uploaded_nodes != 0;
}
}
}
