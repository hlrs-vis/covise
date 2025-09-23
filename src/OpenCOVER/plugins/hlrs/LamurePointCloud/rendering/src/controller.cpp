// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/controller.h>
#include <chrono>

namespace lamure
{
namespace ren
{
std::mutex controller::mutex_;
bool controller::is_instanced_ = false;
controller *controller::single_ = nullptr;

controller::controller() : num_contexts_registered_(0), num_models_registered_(0), ms_since_last_node_upload_(0), latest_timestamp_(std::chrono::system_clock::now()) {}

controller::~controller()
{
    std::lock_guard<std::mutex> lock(mutex_);

    is_instanced_ = false;

    for(auto &cut_update_pool_it : cut_update_pools_)
    {
        cut_update_pool *pool = cut_update_pool_it.second;
        if(pool != nullptr)
        {
            delete pool;
            pool = nullptr;
        }
    }
    cut_update_pools_.clear();

    for(auto &gpu_context_it : gpu_contexts_)
    {
        gpu_context *ctx = gpu_context_it.second;
        if(ctx != nullptr)
        {
            delete ctx;
            ctx = nullptr;
        }
    }
    gpu_contexts_.clear();
}

controller *controller::get_instance()
{
    if(!is_instanced_)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if(!is_instanced_)
        {
            single_ = new controller();
            is_instanced_ = true;
        }

        return single_;
    }
    else
    {
        return single_;
    }
}

const context_t controller::num_contexts_registered()
{
    std::lock_guard<std::mutex> lock(mutex_);

    return num_contexts_registered_;
}

void controller::signal_system_reset()
{
    std::lock_guard<std::mutex> lock(mutex_);

    policy *policy = policy::get_instance();
    policy->set_reset_system(true);

    for(const auto &context : context_map_)
    {
        context_t context_id = context.second;
        reset_flags_history_[context_id].push(true);
    }
}

const bool controller::is_system_reset_signaled(const context_t context_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if(!reset_flags_history_[context_id].empty())
    {
        bool reset = reset_flags_history_[context_id].front();
        reset_flags_history_[context_id].pop();
        if(reset)
        {
            return true;
        }
    }

    return false;
}

void controller::reset_system()
{
    policy *policy = policy::get_instance();

    if(policy->reset_system())
    {
        model_database *database = model_database::get_instance();
        cut_database *cuts = cut_database::get_instance();
        controller *controller = controller::get_instance();

        context_t num_contexts_registered = controller->num_contexts_registered();
        for(context_t ctx_id = 0; ctx_id < num_contexts_registered; ++ctx_id)
        {
            // while(controller->is_cut_update_in_progress(ctx_id))
            // {};
        }

        std::lock_guard<std::mutex> lock(mutex_);

        if(policy->reset_system())
        {
            database->apply();
            cuts->reset();

            for(auto &cut_update_pool_it : cut_update_pools_)
            {
                cut_update_pool *pool = cut_update_pool_it.second;
                if(pool != nullptr)
                {
                    while(pool->is_running())
                    {
                    };
                    delete pool;
                    pool = nullptr;
                }
            }

            cut_update_pools_.clear();

            for(auto &gpu_context_it : gpu_contexts_)
            {
                gpu_context *context = gpu_context_it.second;
                if(context != nullptr)
                {
                    delete context;
                    context = nullptr;
                }
            }

            gpu_contexts_.clear();

            // disregard:
            // num_contexts_registered_ = 0;
            // num_views_registered_.clear();
            // context_map_.clear();

            // keep the model map!

            policy->set_reset_system(false);
        }
    }
}

void controller::reset_system(Data_Provenance const &data_provenance)
{
    policy *policy = policy::get_instance();

    if(policy->reset_system())
    {
        model_database *database = model_database::get_instance();
        cut_database *cuts = cut_database::get_instance();
        controller *controller = controller::get_instance();

        context_t num_contexts_registered = controller->num_contexts_registered();
        for(context_t ctx_id = 0; ctx_id < num_contexts_registered; ++ctx_id)
        {
            while(controller->is_cut_update_in_progress(ctx_id, data_provenance))
            {
            };
        }

        std::lock_guard<std::mutex> lock(mutex_);

        if(policy->reset_system())
        {
            database->apply();
            cuts->reset();

            for(auto &cut_update_pool_it : cut_update_pools_)
            {
                cut_update_pool *pool = cut_update_pool_it.second;
                if(pool != nullptr)
                {
                    while(pool->is_running())
                    {
                    };
                    delete pool;
                    pool = nullptr;
                }
            }

            cut_update_pools_.clear();

            for(auto &gpu_context_it : gpu_contexts_)
            {
                gpu_context *context = gpu_context_it.second;
                if(context != nullptr)
                {
                    delete context;
                    context = nullptr;
                }
            }

            gpu_contexts_.clear();

            // disregard:
            // num_contexts_registered_ = 0;
            // num_views_registered_.clear();
            // context_map_.clear();

            // keep the model map!

            policy->set_reset_system(false);
        }
    }
}

context_t controller::deduce_context_id(const gua_context_desc_t context_desc)
{
    auto context_it = context_map_.find(context_desc);

    if(context_it != context_map_.end())
    {
        return context_map_[context_desc];
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            context_map_[context_desc] = num_contexts_registered_;
            view_map_[num_contexts_registered_] = std::unordered_map<context_t, view_t>();
            num_views_registered_.push_back(0);

            gpu_contexts_.insert(std::make_pair(num_contexts_registered_, new gpu_context(num_contexts_registered_)));
#ifdef LAMURE_ENABLE_INFO
            std::cout << "lamure: registered context id " << num_contexts_registered_ << std::endl;
#endif
            ++num_contexts_registered_;
        }
        return deduce_context_id(context_desc);
    }
}

view_t controller::deduce_view_id(const gua_context_desc_t context_desc, const gua_view_desc_t view_desc)
{
    context_t context_id = deduce_context_id(context_desc);

    auto view_it = view_map_[context_id].find(view_desc);

    if(view_it != view_map_[context_id].end())
    {
        return view_map_[context_id][view_desc];
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            view_map_[context_id][view_desc] = num_views_registered_[context_id];
#ifdef LAMURE_ENABLE_INFO
            std::cout << "lamure: registered view id " << num_views_registered_[context_id] << " on context id " << context_id << std::endl;
#endif
            ++num_views_registered_[context_id];
        }
        return deduce_view_id(context_desc, view_desc);
    }
}

model_t controller::deduce_model_id(const gua_model_desc_t &model_desc)
{
    auto model_it = model_map_.find(model_desc);

    if(model_it != model_map_.end())
    {
        return model_map_[model_desc];
    }
    else
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            model_map_[model_desc] = num_models_registered_;
#ifdef LAMURE_ENABLE_INFO
            std::cout << "lamure: registered model id " << num_models_registered_ << std::endl;
#endif
            ++num_models_registered_;
        }

        return deduce_model_id(model_desc);
    }
}

const bool controller::is_cut_update_in_progress(const context_t context_id, Data_Provenance const &data_provenance)
{
    auto gpu_context_it = gpu_contexts_.find(context_id);

    if(gpu_context_it == gpu_contexts_.end())
    {
        throw std::runtime_error("lamure: controller::Gpu Context not found for context: " + context_id);
    }

    auto cut_update_it = cut_update_pools_.find(context_id);

    if(cut_update_it != cut_update_pools_.end())
    {
        return cut_update_it->second->is_running();
    }
    else
    {
        gpu_context *ctx = gpu_context_it->second;
        if(!ctx->is_created())
        {
            throw std::runtime_error("lamure: controller::Gpu Context not created for context: " + context_id);
        }
        cut_update_pools_[context_id] = new cut_update_pool(context_id, ctx->upload_budget_in_nodes(), ctx->render_budget_in_nodes(), data_provenance);
        return is_cut_update_in_progress(context_id, data_provenance);
    }

    return true;
}

const bool controller::is_cut_update_in_progress(const context_t context_id)
{
    auto gpu_context_it = gpu_contexts_.find(context_id);

    if(gpu_context_it == gpu_contexts_.end())
    {
        throw std::runtime_error("lamure: controller::Gpu Context not found for context: " + context_id);
    }

    auto cut_update_it = cut_update_pools_.find(context_id);

    if(cut_update_it != cut_update_pools_.end())
    {
        return cut_update_it->second->is_running();
    }
    else
    {
        gpu_context *ctx = gpu_context_it->second;
        if(!ctx->is_created())
        {
            throw std::runtime_error("lamure: controller::Gpu Context not created for context: " + context_id);
        }

        cut_update_pools_[context_id] = new cut_update_pool(context_id, ctx->upload_budget_in_nodes(), ctx->render_budget_in_nodes());
        return is_cut_update_in_progress(context_id);
    }

    return true;
}

void controller::dispatch(const context_t context_id, scm::gl::render_device_ptr device, Data_Provenance const& data_provenance)
{
    auto gpu_context_it = gpu_contexts_.find(context_id);

    if (gpu_context_it == gpu_contexts_.end())
    {
        throw std::runtime_error("lamure: controller::Gpu Context not found for context: " + context_id);
    }

    auto cut_update_it = cut_update_pools_.find(context_id);

    if (cut_update_it != cut_update_pools_.end())
    {
        lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
        cuts->swap(context_id);

        // cut_update_it->second->dispatch_cut_update(gpu_context_it->second->get_temporary_storages().storage_a_, gpu_context_it->second->get_temporary_storages().storage_b_,
        // gpu_context_it->second->get_temporary_storages_provenance().storage_a_, gpu_context_it->second->get_temporary_storages_provenance().storage_b_);

        cut_update_it->second->dispatch_cut_update(gpu_context_it->second->get_fix_a().fix_buffer_, gpu_context_it->second->get_fix_b().fix_buffer_,
            gpu_context_it->second->get_fix_a().fix_buffer_provenance_, gpu_context_it->second->get_fix_b().fix_buffer_provenance_);

        //GLenum first_error = device->opengl_api().glGetError();

        if (cuts->is_front_modified(context_id))
        {
            cut_database_record::temporary_buffer current = cuts->get_buffer(context_id);

            gpu_context* ctx = gpu_context_it->second;

            // ctx->unmap_temporary_storage(current, device, data_provenance);

            if (ctx->update_primary_buffer_fix(current, device, data_provenance))
            {
                ms_since_last_node_upload_ = 0;
            }
            cuts->signal_upload_complete(context_id);
            // ctx->map_temporary_storage(current, device, data_provenance);
        }
        //first_error = device->opengl_api().glGetError();

    }
    else
    {
        gpu_context* ctx = gpu_context_it->second;
        if (!ctx->is_created())
        {
            // throw std::runtime_error(
            //    "lamure: controller::Gpu Context not created for context: " + context_id);

            // fix for gua:
            ctx->create(device, data_provenance);
            //int first_error = device->opengl_api().glGetError();

        }

        cut_update_pools_[context_id] = new cut_update_pool(context_id, ctx->upload_budget_in_nodes(), ctx->render_budget_in_nodes(), data_provenance);
        dispatch(context_id, device, data_provenance);
    }

    {
        auto const& current_time_stamp = std::chrono::system_clock::now();
        ms_since_last_node_upload_ += (std::chrono::duration_cast<std::chrono::duration<int, std::milli>>(current_time_stamp - latest_timestamp_).count());
        latest_timestamp_ = current_time_stamp;
    }
}

void controller::dispatch(const context_t context_id, scm::gl::render_device_ptr device)
{
    auto gpu_context_it = gpu_contexts_.find(context_id);

    if (gpu_context_it == gpu_contexts_.end())
    {
        throw std::runtime_error("lamure: controller::Gpu Context not found for context: " + context_id);
    }

    auto cut_update_it = cut_update_pools_.find(context_id);

    if (cut_update_it != cut_update_pools_.end())
    {
        lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
        cuts->swap(context_id);

        cut_update_it->second->dispatch_cut_update(gpu_context_it->second->get_temporary_storages().storage_a_, gpu_context_it->second->get_temporary_storages().storage_b_,
            gpu_context_it->second->get_temporary_storages_provenance().storage_a_, gpu_context_it->second->get_temporary_storages_provenance().storage_b_);

        if (cuts->is_front_modified(context_id))
        {
            cut_database_record::temporary_buffer current = cuts->get_buffer(context_id);

            gpu_context* ctx = gpu_context_it->second;
            ctx->unmap_temporary_storage(current, device);

            if (ctx->update_primary_buffer(current, device))
            {
                ms_since_last_node_upload_ = 0;
            }

            cuts->signal_upload_complete(context_id);
            ctx->map_temporary_storage(current, device);
        }
    }
    else
    {
        gpu_context* ctx = gpu_context_it->second;
        if (!ctx->is_created())
        {
            // throw std::runtime_error(
            //    "lamure: controller::Gpu Context not created for context: " + context_id);

            // fix for gua:
            ctx->create(device);
        }

        cut_update_pools_[context_id] = new cut_update_pool(context_id, ctx->upload_budget_in_nodes(), ctx->render_budget_in_nodes());
        dispatch(context_id, device);
    }

    {
        auto const& current_time_stamp = std::chrono::system_clock::now();
        ms_since_last_node_upload_ += (std::chrono::duration_cast<std::chrono::duration<int, std::milli>>(current_time_stamp - latest_timestamp_).count());
        latest_timestamp_ = current_time_stamp;
    }
}

const bool controller::is_model_present(const gua_model_desc_t model_desc) { return model_map_.find(model_desc) != model_map_.end(); }

scm::gl::buffer_ptr controller::get_context_buffer(const context_t context_id, scm::gl::render_device_ptr device)
{
    auto gpu_context_it = gpu_contexts_.find(context_id);

    if (gpu_context_it == gpu_contexts_.end())
    {
        throw std::runtime_error("lamure: controller::Gpu Context not found for context: " + context_id);
    }

    return gpu_context_it->second->get_context_buffer(device);
}

scm::gl::buffer_ptr controller::get_context_buffer(const context_t context_id, scm::gl::render_device_ptr device, Data_Provenance const& data_provenance)
{
    auto gpu_context_it = gpu_contexts_.find(context_id);

    if (gpu_context_it == gpu_contexts_.end())
    {
        throw std::runtime_error("lamure: controller::Gpu Context not found for context: " + context_id);
    }

    return gpu_context_it->second->get_context_buffer(device, data_provenance);
}

scm::gl::vertex_array_ptr controller::get_context_memory(const context_t context_id, bvh::primitive_type type, scm::gl::render_device_ptr device)
{
    auto gpu_context_it = gpu_contexts_.find(context_id);

    if (gpu_context_it == gpu_contexts_.end())
    {
        throw std::runtime_error("lamure: controller::Gpu Context not found for context: " + context_id);
    }

    return gpu_context_it->second->get_context_memory(type, device);
}

scm::gl::vertex_array_ptr controller::get_context_memory(const context_t context_id, bvh::primitive_type type, scm::gl::render_device_ptr device, Data_Provenance const &data_provenance)
{
    auto gpu_context_it = gpu_contexts_.find(context_id);

    if(gpu_context_it == gpu_contexts_.end())
    {
        throw std::runtime_error("lamure: controller::Gpu Context not found for context: " + context_id);
    }

    return gpu_context_it->second->get_context_memory(type, device, data_provenance);
}

} // namespace ren

} // namespace lamure
