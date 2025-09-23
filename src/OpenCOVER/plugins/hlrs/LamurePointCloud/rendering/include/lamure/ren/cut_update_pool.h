// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_CUT_UPDATE_POOL_H_
#define REN_CUT_UPDATE_POOL_H_


#include <lamure/ren/config.h>
#include <lamure/semaphore.h>

#include <lamure/utils.h>
#include <vector>

#include <lamure/ren/cut_database.h>
#include <lamure/ren/model_database.h>
#include <lamure/ren/policy.h>

#include <lamure/memory_status.h>
#include <lamure/ren/camera.h>
#include <lamure/ren/cut.h>
#include <lamure/ren/cut_update_index.h>
#include <lamure/ren/cut_update_queue.h>
#include <lamure/ren/gpu_cache.h>
#include <lamure/ren/ooc_cache.h>

namespace lamure
{
namespace ren
{
class cut_update_pool
{
  public:
    cut_update_pool(const context_t context_id, const node_t upload_budget_in_nodes, const node_t render_budget_in_nodes, Data_Provenance const &data_provenance);
    cut_update_pool(const context_t context_id, const node_t upload_budget_in_nodes, const node_t render_budget_in_nodes);
    virtual ~cut_update_pool();

    const uint32_t num_threads() const { return num_threads_; };

    void dispatch_cut_update(char *current_gpu_storage_A, char *current_gpu_storage_B, char *current_gpu_storage_A_provenance, char *current_gpu_storage_B_provenance);
    // void                    dispatch_cut_update(char* current_gpu_storage_A, char* current_gpu_storage_B);
    const bool is_running();

  protected:
    void initialize(bool provenance = false);
    const bool prepare();

    void split_node(const cut_update_index::action &item);
    void collapse_node(const cut_update_index::action &item);
    void cut_update_split_again(const cut_update_index::action &split_action);

    const bool is_all_nodes_in_cut(const model_t model_id, const std::vector<node_t> &node_ids, const std::set<node_t> &cut);
    const bool is_node_in_frustum(const view_t view_id, const model_t model_id, const node_t node_id, const scm::gl::frustum &frustum);
    const bool is_no_node_in_frustum(const view_t view_id, const model_t model_id, const std::vector<node_t> &node_ids, const scm::gl::frustum &frustum);

    const float calculate_node_error(const view_t view_id, const model_t model_id, const node_t node_id);

    /*virtual*/ void run();
    void shutdown();

    void cut_master();
    void cut_analysis(view_t view_id, model_t model_id);
    void cut_update();
    void compile_transfer_list();
    void compile_render_list();
#ifdef LAMURE_CUT_UPDATE_ENABLE_PREFETCHING
    void prefetch_routine();
#endif

  private:
    bool is_shutdown();

    Data_Provenance _data_provenance;
    context_t context_id_;

    bool locked_;
    semaphore semaphore_;
    std::mutex mutex_;

    uint32_t num_threads_;
    std::vector<std::thread> threads_;

    bool shutdown_;

    cut_update_queue job_queue_;

    gpu_cache *gpu_cache_;
    cut_update_index *index_;

    std::vector<cut_database_record::slot_update_desc> transfer_list_;
    std::vector<std::vector<std::vector<cut::node_slot_aggregate>>> render_list_;

    char *current_gpu_storage_A_;
    char *current_gpu_storage_B_;
    char *current_gpu_storage_;

    char *current_gpu_storage_A_provenance_;
    char *current_gpu_storage_B_provenance_;
    char *current_gpu_storage_provenance_;

    cut_database_record::temporary_buffer current_gpu_buffer_;

    std::map<view_t, camera> user_cameras_;
    std::map<view_t, float> height_divided_by_top_minus_bottoms_;
    std::map<model_t, scm::math::mat4f> model_transforms_;
    std::map<model_t, float> model_thresholds_;

    scm::math::mat4f previous_camera_view_;

    size_t upload_budget_in_nodes_;
    size_t render_budget_in_nodes_;
// size_t                  out_of_core_budget_in_nodes_;

#ifdef LAMURE_CUT_UPDATE_ENABLE_MODEL_TIMEOUT
    size_t cut_update_counter_;
    std::map<model_t, size_t> model_freshness_;
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_PREFETCHING
    std::vector<cut_update_index::action> pending_prefetch_set_;
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_REPEAT_MODE
    boost::timer::cpu_timer master_timer_;

    boost::timer::nanosecond_type last_frame_elapsed_;
#endif

    semaphore master_semaphore_;
    bool master_dispatched_;
};
}
} // namespace lamure

#endif // REN_CUT_UPDATE_POOL_H_
