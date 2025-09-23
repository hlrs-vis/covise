// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/cache.h>


namespace lamure
{

namespace ren
{

cache::
cache(const slot_t num_slots)
    : num_slots_(num_slots), slot_size_(0) {
    model_database* database = model_database::get_instance();

    slot_size_ = database->get_slot_size();
    index_ = new cache_index(database->num_models(), num_slots_);
}

cache::
~cache() {
    if (index_ != nullptr) {
        delete index_;
        index_ = nullptr;
    }
}

const bool cache::
is_node_resident(const model_t model_id, const node_t node_id) {
    return index_->is_node_indexed(model_id, node_id);
}

const slot_t cache::
num_free_slots() {
    return index_->num_free_slots();
}

const slot_t cache::
slot_id(const model_t model_id, const node_t node_id) {
    return index_->get_slot(model_id, node_id);
}

void cache::
aquire_node(const context_t context_id, const view_t view_id, const model_t model_id, const node_t node_id) {
    if (index_->is_node_indexed(model_id, node_id)) {
        uint32_t hash_id = ((((uint32_t)context_id) & 0xFFFF) << 16) | (((uint32_t)view_id) & 0xFFFF);
        index_->aquire_slot(hash_id, model_id, node_id);
    }
}

void cache::
release_node(const context_t context_id, const view_t view_id, const model_t model_id, const node_t node_id) {
    if (index_->is_node_indexed(model_id, node_id)) {
        uint32_t hash_id = ((((uint32_t)context_id) & 0xFFFF) << 16) | (((uint32_t)view_id) & 0xFFFF);
        index_->release_slot(hash_id, model_id, node_id);
    }

}

const bool cache::
release_node_invalidate(const context_t context_id, const view_t view_id, const model_t model_id, const node_t node_id) {
    if (index_->is_node_indexed(model_id, node_id)) {
        uint32_t hash_id = ((((uint32_t)context_id) & 0xFFFF) << 16) | (((uint32_t)view_id) & 0xFFFF);
        return index_->release_slot_invalidate(hash_id, model_id, node_id);
    }

    return false;
}

void cache::
lock() {
    mutex_.lock();
}

void cache::
unlock() {
    mutex_.unlock();
}


} // namespace ren

} // namespace lamure


