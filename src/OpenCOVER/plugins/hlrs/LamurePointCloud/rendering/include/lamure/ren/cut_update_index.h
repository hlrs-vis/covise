// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_CUT_UPDATE_INDEX_H_
#define REN_CUT_UPDATE_INDEX_H_

#include <lamure/types.h>
#include <lamure/utils.h>
#include <lamure/ren/config.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <set>
#include <map>
#include <mutex>
#include <assert.h>
#include <algorithm>
#include <stack>

#include <lamure/ren/model_database.h>



namespace lamure {
namespace ren {


class cut_update_index
{
public:

    enum queue_t
    {
        KEEP = 0,
        MUST_SPLIT = 1,
        MUST_COLLAPSE = 2,
        COLLAPSE_ON_NEED = 3,
        MAYBE_COLLAPSE = 4,
        NUM_QUEUES = 5
    };

    struct action
    {
        explicit action(
            const queue_t queue,
            const view_t view_id,
            const model_t model_id,
            const node_t node_id,
            const float error)
            : queue_(queue),
            view_id_(view_id),
            model_id_(model_id),
            node_id_(node_id),
            error_(error)
        {};

        explicit action()
            : queue_(queue_t::NUM_QUEUES),
            model_id_(invalid_model_t),
            node_id_(invalid_node_t),
            error_(0.f) {};

        queue_t         queue_;
        view_t          view_id_;
        model_t         model_id_;
        node_t          node_id_;
        float           error_;
    };

    struct actioncompare
    {
        bool operator() (const action& l, const action& r) const
        {
            return l.error_ < r.error_;
        }
    };

                        cut_update_index();
    virtual             ~cut_update_index();

    inline const view_t num_views() const { return num_views_; };
    inline const model_t num_models() const { return num_models_; };
    inline const std::set<view_t>& view_ids() const { return view_ids_; };

    void                update_policy(const view_t num_views);

    const node_t        num_nodes(const model_t model_id) const;
    const size_t        fan_factor(const model_t model_id) const;
    const size_t        num_actions(const queue_t queue);

    void                push_action(const action& action, bool sort);
    const action        front_action(const queue_t queue);
    const action        back_action(const queue_t queue);
    void                pop_front_action(const queue_t queue);
    void                Popback_action(const queue_t queue);

    const std::set<node_t>& get_current_cut(const view_t view_id, const model_t model_id);
    const std::set<node_t>& get_previous_cut(const view_t view_id, const model_t model_id);
    void                swap_cuts();
    void                reset_cut(const view_t view_id, const model_t model_id);

    void                cancel_action(const view_t view_id, const model_t model_id, const node_t node_id);
    void                approve_action(const action& action);
    void                reject_action(const action& action);

    const node_t        get_child_id(const model_t model_id, const node_t node_id, const node_t child_index) const;
    const node_t        get_parent_id(const model_t model_id, const node_t node_id) const;
    void                get_all_siblings(const model_t model_id, const node_t node_id, std::vector<node_t>& siblings) const;
    void                get_all_children(const model_t model_id, const node_t node_id, std::vector<node_t>& children) const;

    void                sort();

private:

    enum cut_front
    {
        FRONT_A = 0,
        FRONT_B = 1,
        INVALID_FRONT = 2
    };

    void                add_action(const action& action, bool sort);

    void                swap(const queue_t queue, const size_t slot_id_0, const size_t slot_id_1);
    void                shuffle_up(const queue_t queue, const size_t slot_id);
    void                shuffle_down(const queue_t queue, const size_t slot_id);

    size_t              num_slots_[queue_t::NUM_QUEUES];
    view_t              num_views_;
    model_t             num_models_;
    std::mutex          mutex_;

    std::vector<uint32_t> fan_factor_table_;
    std::vector<node_t> num_nodes_table_;
    std::set<view_t> view_ids_;

    std::vector<action> slots_[queue_t::NUM_QUEUES];
    std::stack<action> initial_queue_;

    //mapping [queue] (model, node) to slot
    std::vector<std::map<node_t, std::set<slot_t>>> slot_maps_[queue_t::NUM_QUEUES];

    cut_front            current_cut_front_;
    //[user][model][node]
    std::map<view_t, std::vector<std::set<node_t>>> front_a_cuts_;
    std::map<view_t, std::vector<std::set<node_t>>> front_b_cuts_;

};



} } // namespace lamure


#endif // REN_CUT_UPDATE_INDEX_H_
