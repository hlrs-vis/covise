// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/gpu_cache.h>

namespace lamure {
namespace ren {

gpu_cache::
gpu_cache(const slot_t num_slots)
    : cache(num_slots),
    transfer_budget_(0),
    transfer_slots_written_(0) {
    model_database* database = model_database::get_instance();
    transfer_list_.resize(database->num_models());
}

gpu_cache::
~gpu_cache() {
    transfer_list_.clear();
}

void gpu_cache::
reset_transfer_list() {
    model_database* database = model_database::get_instance();
    transfer_list_.clear();
    transfer_list_.resize(database->num_models());
}

const bool gpu_cache::
register_node(const model_t model_id, const node_t node_id) {
    if (is_node_resident(model_id, node_id)) {
        return false;
    }

    if (transfer_budget_ > 0) {
        --transfer_budget_;
    }

    node_t least_recently_used_slot = index_->reserve_slot();

    index_->apply_slot(least_recently_used_slot, model_id, node_id);

    transfer_list_[model_id].insert(node_id);

    return true;
}


void gpu_cache::
remove_from_transfer_list(const model_t model_id, const node_t node_id) {
    if (transfer_list_[model_id].find(node_id) != transfer_list_[model_id].end()) {
        transfer_list_[model_id].erase(node_id);
        ++transfer_budget_;
    }
}



} } // namespace lamure
