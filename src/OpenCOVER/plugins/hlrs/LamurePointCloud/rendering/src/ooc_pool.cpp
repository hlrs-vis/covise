// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/ooc_pool.h>

namespace lamure
{
namespace ren
{
ooc_pool::ooc_pool(const uint32_t num_threads,
    const size_t size_of_slot_in_bytes)
    : locked_(false)
    , size_of_slot_(size_of_slot_in_bytes)
    , num_threads_(num_threads)
    , shutdown_(false)
    , bytes_loaded_(0)
    , active_threads_(0)
{
    assert(num_threads_ > 0);

    semaphore_.set_min_signal_count(1);
    semaphore_.set_max_signal_count(std::numeric_limits<size_t>::max());

    model_database* database = model_database::get_instance();
    const model_t N = database->num_models();

    priority_queue_.initialize(LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE, N);

    node_stride_by_model_.resize(N);
    prims_per_node_by_model_.resize(N);
    lod_files_.resize(N);
    prov_files_.clear();

    for (model_t m = 0; m < N; ++m) {
        node_stride_by_model_[m]    = database->get_node_size(m);
        prims_per_node_by_model_[m] = database->get_primitives_per_node(m);

        const std::string bvh_filename = database->get_model(m)->get_bvh()->get_filename();
        const std::string base_name    = bvh_filename.substr(0, bvh_filename.find_last_of(".") + 1);
        const std::string file_ext     = bvh_filename.substr(base_name.size());
        const std::string bvh_suffix   = file_ext.substr(3);
        lod_files_[m] = base_name + "lod" + bvh_suffix;
    }

    threads_.reserve(num_threads_);
    for (uint32_t i = 0; i < num_threads_; ++i) {
        threads_.push_back(std::thread(&ooc_pool::run, this));
    }
}

ooc_pool::ooc_pool(const uint32_t num_threads,
    const size_t size_of_slot_in_bytes,
    const size_t size_of_slot_provenance,
    Data_Provenance const& data_provenance)
    : locked_(false)
    , size_of_slot_(size_of_slot_in_bytes)
    , size_of_slot_provenance_(size_of_slot_provenance)
    , num_threads_(num_threads)
    , shutdown_(false)
    , bytes_loaded_(0)
    , active_threads_(0)
{
    assert(num_threads_ > 0);

    _data_provenance = data_provenance;

    semaphore_.set_min_signal_count(1);
    semaphore_.set_max_signal_count(std::numeric_limits<size_t>::max());

    model_database* database = model_database::get_instance();
    const model_t N = database->num_models();

    priority_queue_.initialize(LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE, N);

    node_stride_by_model_.resize(N);
    prims_per_node_by_model_.resize(N);
    lod_files_.resize(N);
    prov_files_.resize(N);

    for (model_t m = 0; m < N; ++m) {
        node_stride_by_model_[m]    = database->get_node_size(m);
        prims_per_node_by_model_[m] = database->get_primitives_per_node(m);

        const std::string bvh_filename = database->get_model(m)->get_bvh()->get_filename();
        const std::string base_name    = bvh_filename.substr(0, bvh_filename.find_last_of(".") + 1);
        const std::string file_ext     = bvh_filename.substr(base_name.size());
        const std::string bvh_suffix   = file_ext.substr(3);
        lod_files_[m]  = base_name + "lod" + bvh_suffix;
        prov_files_[m] = bvh_filename.substr(0, bvh_filename.size() - 3) + "prov";
    }

    threads_.reserve(num_threads_);
    for (uint32_t i = 0; i < num_threads_; ++i) {
        threads_.push_back(std::thread(&ooc_pool::run, this));
    }
}

ooc_pool::~ooc_pool()
{
    shutdown();
}

void ooc_pool::shutdown()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
            return;
        }
        shutdown_ = true;
        semaphore_.shutdown();
    }

    for (size_t i = 0; i < threads_.size(); ++i) {
        semaphore_.signal(1);
    }

    for(size_t i = 0; i < threads_.size(); ++i)
    {
        if(threads_[i].joinable())
        {
            threads_[i].join();
        }
    }
    threads_.clear();
}

bool ooc_pool::is_shutdown()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_;
}

void ooc_pool::lock()
{
    mutex_.lock();
    locked_ = true;
}

void ooc_pool::unlock()
{
    locked_ = false;
    mutex_.unlock();
}

void ooc_pool::begin_measure()
{
    std::lock_guard<std::mutex> lock(mutex_);
    bytes_loaded_ = 0;
}

void ooc_pool::end_measure()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "megabytes loaded: " << bytes_loaded_ / 1024 / 1024 << std::endl;
}

void ooc_pool::run()
{
    if (lod_files_.empty() || node_stride_by_model_.empty() ||
        (_data_provenance.get_size_in_bytes() > 0 &&
            (prov_files_.empty() || prims_per_node_by_model_.empty())))
    {
        model_database* database = model_database::get_instance();
        const model_t N = database->num_models();

        if (node_stride_by_model_.size() != N)         node_stride_by_model_.assign(N, 0);
        if (prims_per_node_by_model_.size() != N)      prims_per_node_by_model_.assign(N, 0);
        if (lod_files_.size() != N)                    lod_files_.assign(N, std::string());
        if (_data_provenance.get_size_in_bytes() > 0 && prov_files_.size() != N)
            prov_files_.assign(N, std::string());

        for (model_t m = 0; m < N; ++m) {
            if (node_stride_by_model_[m] == 0)
                node_stride_by_model_[m] = database->get_node_size(m);
            if (prims_per_node_by_model_[m] == 0)
                prims_per_node_by_model_[m] = database->get_primitives_per_node(m);

            if (lod_files_[m].empty() ||
                (_data_provenance.get_size_in_bytes() > 0 && prov_files_[m].empty()))
            {
                const std::string bvh = database->get_model(m)->get_bvh()->get_filename();
                const std::string base = bvh.substr(0, bvh.find_last_of(".") + 1);
                const std::string ext  = bvh.substr(base.size());
                const std::string bvh_suffix = ext.substr(3); // nach ".bvh"

                if (lod_files_[m].empty())
                    lod_files_[m] = base + "lod" + bvh_suffix;

                if (_data_provenance.get_size_in_bytes() > 0 && prov_files_[m].empty())
                    prov_files_[m] = bvh.substr(0, bvh.size() - 3) + "prov";
            }
        }
    }

    std::unique_ptr<char[]> local_cache(new char[size_of_slot_]);
    std::unique_ptr<char[]> local_cache_provenance;
    if (_data_provenance.get_size_in_bytes() > 0) {
        local_cache_provenance.reset(new char[size_of_slot_provenance_]);
    }

    while (true)
    {
        semaphore_.wait();
        if (is_shutdown())
            break;

        cache_queue::job job = priority_queue_.top_job();
        if (job.node_id_ == invalid_node_t)
            continue;

        active_threads_++;

        if (job.model_id_ >= node_stride_by_model_.size() ||
            job.model_id_ >= lod_files_.size())
        {
            active_threads_--;
            continue;
        }

        const size_t stride_in_bytes = node_stride_by_model_[job.model_id_];
        const size_t offset_in_bytes = job.node_id_ * stride_in_bytes;

        lod_stream access;
        access.open(lod_files_[job.model_id_]);
        access.read(local_cache.get(), offset_in_bytes, stride_in_bytes);
        access.close();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            bytes_loaded_ += stride_in_bytes;
        }

        assert(job.slot_mem_ != nullptr);
        memcpy(job.slot_mem_, local_cache.get(), stride_in_bytes);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            history_.push_back(job);
        }

        if (_data_provenance.get_size_in_bytes() > 0 &&
            job.model_id_ < prims_per_node_by_model_.size() &&
            job.model_id_ < prov_files_.size() &&
            job.slot_mem_provenance_ != nullptr)
        {
            const size_t stride_in_bytes_prov =
                prims_per_node_by_model_[job.model_id_] * _data_provenance.get_size_in_bytes();
            const size_t offset_in_bytes_prov = job.node_id_ * stride_in_bytes_prov;

            provenance_stream access_prov;
            access_prov.open(prov_files_[job.model_id_]);
            access_prov.read(local_cache_provenance.get(), offset_in_bytes_prov, stride_in_bytes_prov);
            access_prov.close();

            {
                std::lock_guard<std::mutex> lock(mutex_);
                bytes_loaded_ += stride_in_bytes_prov;
            }
            memcpy(job.slot_mem_provenance_, local_cache_provenance.get(), stride_in_bytes_prov);
        }

        active_threads_--;
    }
}

void ooc_pool::resolve_cache_history(cache_index *index)
{
    assert(locked_);

    for(auto &entry : history_)
    {
        index->apply_slot(entry.slot_id_, entry.model_id_, entry.node_id_);
        priority_queue_.pop_job(entry);
    }

    history_.clear();
}

void ooc_pool::perform_queue_maintenance(cache_index *index)
{
    assert(locked_);

    while (priority_queue_.num_jobs() > 0)
    {
        cache_queue::job job = priority_queue_.top_job();

        assert(job.slot_id_ != invalid_slot_t);

        priority_queue_.pop_job(job);
        index->unreserve_slot(job.slot_id_);
    }
}

bool ooc_pool::acknowledge_request(cache_queue::job job)
{
    if (is_shutdown()) return false;
    const bool success = priority_queue_.push_job(job);
    if (success) semaphore_.signal(1);
    return success;
}

cache_queue::query_result ooc_pool::acknowledge_query(const model_t model_id, const node_t node_id)
{
    return priority_queue_.is_node_indexed(model_id, node_id);
}

void ooc_pool::acknowledge_update(const model_t model_id, const node_t node_id, int32_t priority)
{
    if (is_shutdown()) return;
    priority_queue_.update_job(model_id, node_id, priority);
}

void ooc_pool::wait_for_idle()
{
    while (priority_queue_.num_jobs() > 0 || active_threads_ > 0) {
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
} // namespace ren

} // namespace lamure