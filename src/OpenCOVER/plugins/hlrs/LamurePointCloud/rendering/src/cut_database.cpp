// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/cut_database.h>


namespace lamure
{

namespace ren
{

std::mutex cut_database::mutex_;
bool cut_database::is_instanced_ = false;
cut_database* cut_database::single_ = nullptr;

cut_database::
cut_database() {

}

cut_database::
~cut_database() {
    std::lock_guard<std::mutex> lock(mutex_);

    is_instanced_ = false;

    for (auto& record_it : records_) {
        cut_database_record* record = record_it.second;
        if (record != nullptr) {
            delete record;
            record = nullptr;
        }
    }

    records_.clear();
}

cut_database* cut_database::
get_instance() {
    if (!is_instanced_) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!is_instanced_) {
            single_ = new cut_database();
            is_instanced_ = true;
        }

        return single_;
    }
    else {
        return single_;
    }
}

void cut_database::
reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& record_it : records_) {
        cut_database_record* record = record_it.second;
        if (record != nullptr) {
            delete record;
            record = nullptr;
        }
    }

    records_.clear();
}

void cut_database::
expand(const context_t context_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    //critical section: check for missing entry again
    //to prevent double initialization

    auto it = records_.find(context_id);

    if (it == records_.end()) {
        records_[context_id] = new cut_database_record(context_id);
    }
}

void cut_database::
swap(const context_t context_id) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->swap_front();
    }
    else {
        expand(context_id);
        swap(context_id);
    }
}

void cut_database::
send_camera(context_t const context_id, view_t const view_id, const camera& camera) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->set_camera(view_id, camera);
    }
    else {
        expand(context_id);
        send_camera(context_id, view_id, camera);
    }
}

void cut_database::
send_height_divided_by_top_minus_bottom(context_t const context_id, view_t const view_id, const float& height_divided_by_top_minus_bottom_value) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
      it->second->set_height_divided_by_top_minus_bottom(view_id, height_divided_by_top_minus_bottom_value);
    }
    else {
        expand(context_id);
        send_height_divided_by_top_minus_bottom(context_id, view_id, height_divided_by_top_minus_bottom_value);
    }
}

void cut_database::
send_transform(context_t const context_id, model_t const model_id, const scm::math::mat4f& transform) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->set_transform(model_id, transform);
    }
    else {
        expand(context_id);
        send_transform(context_id, model_id, transform);
    }
}

void cut_database::
send_rendered(const context_t context_id, const model_t model_id) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->set_rendered(model_id);
    }
    else {
        expand(context_id);
        send_rendered(context_id, model_id);
    }
}

void cut_database::
send_threshold(context_t const context_id, model_t const model_id, const float threshold) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->set_threshold(model_id, threshold);
    }
    else {
        expand(context_id);
        send_threshold(context_id, model_id, threshold);
    }
}


void cut_database::
receive_cameras(const context_t context_id, std::map<view_t, camera>& cameras) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->receive_cameras(cameras);
    }
    else {
        expand(context_id);
        receive_cameras(context_id, cameras);
    }
}

void cut_database::
receive_height_divided_by_top_minus_bottoms(const context_t context_id, std::map<view_t, float>& height_divided_by_top_minus_bottom) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->receive_height_divided_by_top_minus_bottoms(height_divided_by_top_minus_bottom);
    }
    else {
        expand(context_id);
        receive_height_divided_by_top_minus_bottoms(context_id, height_divided_by_top_minus_bottom);
    }
}


void cut_database::
receive_transforms(const context_t context_id, std::map<model_t, scm::math::mat4f>& transforms) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->receive_transforms(transforms);
    }
    else {
        expand(context_id);
        receive_transforms(context_id, transforms);
    }

}

void cut_database::
receive_rendered(const context_t context_id, std::set<model_t>& rendered) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->receive_rendered(rendered);
    }
    else {
        expand(context_id);
        receive_rendered(context_id, rendered);
    }
}

void cut_database::
receive_thresholds(const context_t context_id, std::map<model_t, float>& thresholds) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->receive_thresholds(thresholds);
    }
    else {
        expand(context_id);
        receive_thresholds(context_id, thresholds);
    }

}

void cut_database::
set_cut(const context_t context_id, const view_t view_id, const model_t model_id, cut& cut) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->set_cut(view_id, model_id, cut);
    }
    else {
        expand(context_id);
        set_cut(context_id, view_id, model_id, cut);
    }
}

cut& cut_database::
get_cut(const context_t context_id, const view_t view_id, const model_t model_id) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        return it->second->get_cut(view_id, model_id);
    }
    else {
        expand(context_id);
        return get_cut(context_id, view_id, model_id);
    }
}

std::vector<cut_database_record::slot_update_desc>& cut_database::
get_updated_set(const context_t context_id) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        return it->second->get_updated_set();
    }
    else {
        expand(context_id);
        return get_updated_set(context_id);
    }
}

void cut_database::
set_updated_set(const context_t context_id, std::vector<cut_database_record::slot_update_desc>& updated_set) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->set_updated_set(updated_set);
    }
    else {
        expand(context_id);
        set_updated_set(context_id, updated_set);
    }
}

const bool cut_database::
is_front_modified(const context_t context_id) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        return it->second->is_front_modified();
    }
    else {
        expand(context_id);
        return is_front_modified(context_id);
    }
}

const bool cut_database::
is_swap_required(const context_t context_id) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        return it->second->is_swap_required();
    }
    else {
        expand(context_id);
        return is_swap_required(context_id);
    }
}


void cut_database::
set_is_front_modified(const context_t context_id, const bool front_modified) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->set_is_front_modified(front_modified);
    }
    else {
        expand(context_id);
        set_is_front_modified(context_id, front_modified);
    }
}

void cut_database::
set_is_swap_required(const context_t context_id, const bool swap_required) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->set_is_swap_required(swap_required);
    }
    else {
        expand(context_id);
        set_is_swap_required(context_id, swap_required);
    }
}

void cut_database::
signal_upload_complete(const context_t context_id) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->signal_upload_complete();
    }
    else {
        expand(context_id);
        signal_upload_complete(context_id);
    }
}

const cut_database_record::temporary_buffer cut_database::
get_buffer(const context_t context_id) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        return it->second->get_buffer();
    }
    else {
        expand(context_id);
        return get_buffer(context_id);
    }
}

void cut_database::
set_buffer(const context_t context_id, const cut_database_record::temporary_buffer buffer) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->set_buffer(buffer);
    }
    else {
        expand(context_id);
        set_buffer(context_id, buffer);
    }
}

void cut_database::
lock_record(const context_t context_id) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->lock_front();
    }
    else {
        expand(context_id);
        lock_record(context_id);
    }
}

void cut_database::
unlock_record(const context_t context_id) {
    auto it = records_.find(context_id);

    if (it != records_.end()) {
        it->second->unlock_front();
    }
    else {
        expand(context_id);
        unlock_record(context_id);
    }

}


} // namespace ren

} // namespace lamure




