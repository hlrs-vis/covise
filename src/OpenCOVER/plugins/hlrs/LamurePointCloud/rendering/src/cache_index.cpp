// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/cache_index.h>


namespace lamure
{

namespace ren
{

cache_index::
cache_index(const model_t num_models, const slot_t num_slots)
    : num_models_(num_models), num_slots_(num_slots), num_free_slots_(num_slots) {
    assert(num_slots > 0);

    try {
      for (slot_t i = 0; i < num_slots_ + 2; ++i) {
        slots_.push_back(cache_index_node(invalid_model_t, invalid_node_t, i - 1, i + 1));
      }

      slots_[0].prev_ = invalid_slot_t;
      slots_[num_slots_ + 1].next_ = invalid_slot_t;

      maps_.resize(num_models_);
    }
    catch (...) {
    }
}

cache_index::
~cache_index() {

}

const slot_t cache_index::
num_free_slots() {
    std::lock_guard<std::mutex> lock(mutex_);
    return num_free_slots_;
}

const slot_t cache_index::
reserve_slot() {
    std::lock_guard<std::mutex> lock(mutex_);

    assert(num_free_slots_ > 0);

    slot_t slot_id = slots_[0].next_;

    //we shouldn't reserve something if the cache is full
    assert(slot_id != invalid_slot_t);
    assert(slot_id < num_slots_+1);

    cache_index_node& node = slots_[slot_id];

    //assert slot was in linked list
    assert(node.prev_ != invalid_slot_t);
    assert(node.next_ != invalid_slot_t);

    //remove node from linked list
    slots_[node.prev_].next_ = node.next_;
    slots_[node.next_].prev_ = node.prev_;

    node.prev_ = invalid_slot_t;
    node.next_ = invalid_slot_t;

    assert(node.views_.empty());

    if (node.node_id_ != invalid_node_t) {
        maps_[node.model_id_].erase(node.node_id_);
    }

    node.node_id_ = invalid_node_t;
    node.model_id_ = invalid_model_t;

    if (num_free_slots_ > 0) {
        --num_free_slots_;
    }

    return slot_id-1;
}

void cache_index::
apply_slot(const slot_t slot_id, const model_t model_id, const node_t node_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    cache_index_node& node = slots_[slot_id+1];

    //these raise when slot was not reserved
    assert(node.prev_ == invalid_slot_t);
    assert(node.next_ == invalid_slot_t);
    assert(node.node_id_ == invalid_node_t);
    assert(node.model_id_ == invalid_model_t);
    assert(node.views_.empty());
    assert(maps_[model_id].find(node_id) == maps_[model_id].end());

    node.node_id_ = node_id;
    node.model_id_ = model_id;

    //insert node at tail
    node.prev_ = slots_[num_slots_+1].prev_;
    node.next_ = num_slots_+1;

    slots_[slots_[num_slots_+1].prev_].next_ = slot_id+1;
    slots_[num_slots_+1].prev_ = slot_id+1;

    maps_[model_id][node_id] = slot_id+1;

    if (num_free_slots_ < num_slots_) {
        ++num_free_slots_;
    }
}


void cache_index::
unreserve_slot(const slot_t slot_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    cache_index_node& node = slots_[slot_id+1];

    //assert slot was reserved and is not in linked list
    assert(node.prev_ == invalid_slot_t);
    assert(node.next_ == invalid_slot_t);

    //assert slot was not aquired by any views
    assert(node.views_.empty());

    //insert to head
    node.prev_ = 0;
    node.next_ = slots_[0].next_;

    slots_[slots_[0].next_].prev_ = slot_id+1;
    slots_[0].next_ = slot_id+1;

    //section below is not really necessary,
    //but let's keep it for sanity
    {
        if (node.node_id_ != invalid_node_t) {
            maps_[node.model_id_].erase(node.node_id_);
        }

        node.node_id_ = invalid_node_t;
        node.model_id_ = invalid_model_t;

        node.views_.clear();
    }

    if (num_free_slots_ < num_slots_) {
        ++num_free_slots_;
    }
}

const slot_t cache_index::
get_slot(const model_t model_id, const node_t node_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    const auto it = maps_[model_id].find(node_id);

    //this raises when slot was not applied
    assert(it != maps_[model_id].end());

    //assert that this slot was actually aquired
    slot_t slot_id = it->second;
    
    //this raises if attempting to access a slot that was not aquired
    //and, thus, is in danger of being overriden very soon
    assert(!slots_[slot_id].views_.empty());

    //assert slot was removed from linked list
    assert(slots_[slot_id].prev_ == invalid_slot_t);
    assert(slots_[slot_id].next_ == invalid_slot_t);

    assert(slot_id != invalid_slot_t);

    return slot_id-1;
}

const bool cache_index::
is_node_indexed(const model_t model_id, const node_t node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return maps_[model_id].find(node_id) != maps_[model_id].end();
}

const bool cache_index::
is_node_aquired(const model_t model_id, const node_t node_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    const auto it = maps_[model_id].find(node_id);
    if (it == maps_[model_id].end()) {
      return false;
    }

    slot_t slot_id = it->second;
    cache_index_node& node = slots_[slot_id];

    return !node.views_.empty();
}

void cache_index::
aquire_slot(const view_t view_id, const model_t model_id, const node_t node_id) {

    std::lock_guard<std::mutex> lock(mutex_);

    const auto it = maps_[model_id].find(node_id);

    //this raises when node was not applied
    assert(it != maps_[model_id].end());

    slot_t slot_id = it->second;
    cache_index_node& node = slots_[slot_id];

    if (node.views_.find(view_id) == node.views_.end()) {
        node.views_.insert(view_id);

        //if slot was not removed from linked list
        if (node.prev_ != invalid_slot_t || node.next_ != invalid_slot_t) {
            assert(node.prev_ != invalid_slot_t);
            assert(node.next_ != invalid_slot_t);

            //remove node from linked list
            slots_[node.prev_].next_ = node.next_;
            slots_[node.next_].prev_ = node.prev_;

            node.prev_ = invalid_slot_t;
            node.next_ = invalid_slot_t;

            if (num_free_slots_ > 0) {
                --num_free_slots_;
            }

        }
    }


}

void cache_index::
release_slot(const view_t view_id, const model_t model_id, const node_t node_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    const auto it = maps_[model_id].find(node_id);

    //this raises when node was not  applied
    assert(it != maps_[model_id].end());

    slot_t slot_id = it->second;
    cache_index_node& node = slots_[slot_id];

    if (node.views_.find(view_id) != node.views_.end()) {
        node.views_.erase(view_id);

        if (node.views_.empty()) {
            //if slot was removed from linked list
            if (node.prev_ == invalid_slot_t && node.next_ == invalid_slot_t) {
                //insert node at tail
                node.prev_ = slots_[num_slots_+1].prev_;
                node.next_ = num_slots_+1;

                slots_[slots_[num_slots_+1].prev_].next_ = slot_id;
                slots_[num_slots_+1].prev_ = slot_id;

                if (num_free_slots_ < num_slots_) {
                    ++num_free_slots_;
                }
            }
        }

    }

}

const bool cache_index::
release_slot_invalidate(const view_t view_id, const model_t model_id, const node_t node_id) {
    //purpose: unregister view from node,
    //if no views remain, invalidate node (remove from index)
    //and insert slot to head
    //return true if and only if the slot was invalidated
    //during current function call

    std::lock_guard<std::mutex> lock(mutex_);

    const auto it = maps_[model_id].find(node_id);

    //this raises when node was not  applied
    assert(it != maps_[model_id].end());

    slot_t slot_id = it->second;
    cache_index_node& node = slots_[slot_id];

    if (node.views_.find(view_id) != node.views_.end()) {
        node.views_.erase(view_id);

        if (node.views_.empty()) {
            //if slot was removed from linked list
            if (node.prev_ == invalid_slot_t && node.next_ == invalid_slot_t) {
                //insert to head
                node.prev_ = 0;
                node.next_ = slots_[0].next_;

                slots_[slots_[0].next_].prev_ = slot_id;
                slots_[0].next_ = slot_id;

                if (num_free_slots_ < num_slots_) {
                    ++num_free_slots_;
                }

                //invalidate slot
                if (node.node_id_ != invalid_node_t) {
                    maps_[node.model_id_].erase(node.node_id_);
                }

                node.node_id_ = invalid_node_t;
                node.model_id_ = invalid_model_t;


                return true;

            }

        }

    }

    return false;
}


} // namespace ren

} // namespace lamure
