// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/cache_queue.h>

namespace lamure
{

namespace ren
{

cache_queue::
cache_queue()
: num_slots_(0),
  num_models_(0),
  mode_(update_mode::UPDATE_NEVER),
  initialized_(false) {

}

cache_queue::
~cache_queue() {

}

const size_t cache_queue::
num_jobs() {
    std::lock_guard<std::mutex> lock(mutex_);
    return num_slots_;
}

const cache_queue::query_result cache_queue::
is_node_indexed(const model_t model_id, const node_t node_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    assert(initialized_);
    assert(model_id < num_models_);

    query_result result = query_result::NOT_INDEXED;

    if (requested_set_[model_id].find(node_id) != requested_set_[model_id].end()) {
        result = query_result::INDEXED_AS_LOADING;

        if (mode_ != update_mode::UPDATE_NEVER) {
            if (pending_set_[model_id].find(node_id) == pending_set_[model_id].end()) {
                result = query_result::INDEXED_AS_WAITING;
            }
        }
    }

    return result;
}

void cache_queue::
initialize(const update_mode mode, const model_t num_models) {
    std::lock_guard<std::mutex> lock(mutex_);

    assert(!initialized_);

    mode_ = mode;
    num_models_ = num_models;

    requested_set_.resize(num_models_);

    if (mode_ != update_mode::UPDATE_NEVER) {
        pending_set_.resize(num_models_);
    }

    initialized_ = true;
}

bool cache_queue::
push_job(const job& job) {
    std::lock_guard<std::mutex> lock(mutex_);

    assert(initialized_);
    assert(job.model_id_ < num_models_);

    if (requested_set_[job.model_id_].find(job.node_id_) == requested_set_[job.model_id_].end()) {
        slots_.push_back(job);
        requested_set_[job.model_id_][job.node_id_] = num_slots_;

        ++num_slots_;

        shuffle_up(num_slots_-1);

        return true;
    }

    return false;

}

const cache_queue::job cache_queue::
top_job() {
    std::lock_guard<std::mutex> lock(mutex_);

    job job;

    if (num_slots_ > 0) {
        job = slots_.front();

        if (mode_ != update_mode::UPDATE_NEVER) {
            pending_set_[job.model_id_].insert(job.node_id_);
        }

        swap(0, num_slots_-1);
        slots_.pop_back();

        --num_slots_;

        shuffle_down(0);
    }

    return job;
}

void cache_queue::
pop_job(const job& job) {
    std::lock_guard<std::mutex> lock(mutex_);

    assert(job.model_id_ < num_models_);

    requested_set_[job.model_id_].erase(job.node_id_);

    if (mode_ != update_mode::UPDATE_NEVER) {
        assert(pending_set_[job.model_id_].find(job.node_id_) != pending_set_[job.model_id_].end());

        if (pending_set_[job.model_id_].find(job.node_id_) != pending_set_[job.model_id_].end()) {
            pending_set_[job.model_id_].erase(job.node_id_);
        }
    }
}

void cache_queue::
update_job(const model_t model_id, const node_t node_id, int32_t priority) {
    if (mode_ == update_mode::UPDATE_NEVER) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    assert(model_id < num_models_);

    if (pending_set_[model_id].find(node_id) != pending_set_[model_id].end()) {
        return;
    }

    const auto it = requested_set_[model_id].find(node_id);

    //assert(it != requested_set_[model_id].end());

    if (it == requested_set_[model_id].end()) {
        return;
    }

    size_t slot_id = it->second;

    if (priority < slots_[slot_id].priority_) {
        if (mode_ == update_mode::UPDATE_ALWAYS || mode_ == update_mode::UPDATE_DECREMENT_ONLY) {
            slots_[slot_id].priority_ = priority;
            shuffle_down(slot_id);
        }
    }
    else if (priority > slots_[slot_id].priority_) {
        if (mode_ == update_mode::UPDATE_ALWAYS || mode_ == update_mode::UPDATE_INCREMENT_ONLY) {
            slots_[slot_id].priority_ = priority;
            shuffle_up(slot_id);
        }
    }

}

const cache_queue::abort_result cache_queue::
abort_job(const job& job) {
    abort_result result = abort_result::ABORT_FAILED;

    if (mode_ != update_mode::UPDATE_NEVER) {
        std::lock_guard<std::mutex> lock(mutex_);

        const auto it = requested_set_[job.model_id_].find(job.node_id_);

        if (it != requested_set_[job.model_id_].end()) {
            if (pending_set_[job.model_id_].find(job.node_id_) == pending_set_[job.model_id_].end()) {
                size_t slot_id = it->second;

                swap(slot_id, num_slots_-1);
                slots_.pop_back();
                --num_slots_;
                shuffle_down(slot_id);
                requested_set_[job.model_id_].erase(job.node_id_);

                result = abort_result::ABORT_SUCCESS;
            }
        }
    }

    return result;
}

void cache_queue::
swap(const size_t slot_id_0, const size_t slot_id_1) {
    job& job0 = slots_[slot_id_0];
    job& job1 = slots_[slot_id_1];

    requested_set_[job0.model_id_][job0.node_id_] = slot_id_1;
    requested_set_[job1.model_id_][job1.node_id_] = slot_id_0;
    std::swap(slots_[slot_id_0], slots_[slot_id_1]);
}

void cache_queue::
shuffle_up(const size_t slot_id) {
    if (slot_id == 0) {
        return;
    }

    size_t parent_slot_id = (size_t)std::floor((slot_id-1)/2);

    if (slots_[slot_id].priority_ < slots_[parent_slot_id].priority_) {
        return;
    }

    swap(slot_id, parent_slot_id);

    shuffle_up(parent_slot_id);
}

void cache_queue::
shuffle_down(const size_t slot_id) {
    size_t left_child_id = slot_id*2 + 1;
    size_t right_child_id = slot_id*2 + 2;

    size_t replace_id = slot_id;

    if (right_child_id < num_slots_) {
        bool left_greater = slots_[right_child_id].priority_ < slots_[left_child_id].priority_;

        if (left_greater && slots_[slot_id].priority_ < slots_[left_child_id].priority_) {
            replace_id = left_child_id;
        }
        else if (!left_greater && slots_[slot_id].priority_ < slots_[right_child_id].priority_) {
            replace_id = right_child_id;
        }
    }
    else if (left_child_id < num_slots_) {
        if (slots_[slot_id].priority_ < slots_[left_child_id].priority_) {
            replace_id = left_child_id;
        }
    }

    if (replace_id != slot_id) {
        swap(slot_id, replace_id);
        shuffle_down(replace_id);
    }
}

} // namespace ren

} // namespace lamure

