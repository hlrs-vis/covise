// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/cut_database_record.h>


namespace lamure
{

namespace ren
{

cut_database_record::
cut_database_record(const context_t context_id)
    : context_id_(context_id),
    is_swap_required_(false),
    current_front_(record_front::FRONT_A),
    front_a_is_modified_(false),
    front_b_is_modified_(false),
    front_a_buffer_(temporary_buffer::BUFFER_A),
    front_b_buffer_(temporary_buffer::BUFFER_A) {

}

cut_database_record::
~cut_database_record() {

}

void cut_database_record::
swap_front() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (is_swap_required_) {
        is_swap_required_ = false;

        if (current_front_ == record_front::FRONT_A) {
            current_front_ = record_front::FRONT_B;
            front_b_rendered_.clear();
        }
        else {
            current_front_ = record_front::FRONT_A;
            front_a_rendered_.clear();
        }
    }
}

void cut_database_record::
set_camera(const view_t view_id, const camera& camera) {
#ifdef LAMURE_DATABASE_SAFE_MODE
    std::lock_guard<std::mutex> lock(mutex_);
#endif
    if (current_front_ == record_front::FRONT_A) {
        expand_front_a(view_id, 0);

        auto it = front_a_cameras_.find(view_id);

        if (it != front_a_cameras_.end()) {
            front_a_cameras_.erase(it);
        }

        front_a_cameras_.insert(std::make_pair(view_id, camera));

    }
    else {
        expand_front_b(view_id, 0);

        auto it = front_b_cameras_.find(view_id);

        if (it != front_b_cameras_.end()) {
            front_b_cameras_.erase(it);
        }

        front_b_cameras_.insert(std::make_pair(view_id, camera));

    }

}

void cut_database_record::
set_height_divided_by_top_minus_bottom(const view_t view_id, float const height_divided_by_top_minus_bottom) {
#ifdef LAMURE_DATABASE_SAFE_MODE
    std::lock_guard<std::mutex> lock(mutex_);
#endif
    if (current_front_ == record_front::FRONT_A) {
        expand_front_a(view_id, 0);

        auto it = front_a_height_divided_by_top_minus_bottom_.find(view_id);

        if (it != front_a_height_divided_by_top_minus_bottom_.end()) {
            front_a_height_divided_by_top_minus_bottom_.erase(it);
        }

        front_a_height_divided_by_top_minus_bottom_.insert(std::make_pair(view_id, height_divided_by_top_minus_bottom));

    }
    else {
        expand_front_b(view_id, 0);

        auto it = front_b_height_divided_by_top_minus_bottom_.find(view_id);

        if (it != front_b_height_divided_by_top_minus_bottom_.end()) {
            front_b_height_divided_by_top_minus_bottom_.erase(it);
        }

        front_b_height_divided_by_top_minus_bottom_.insert(std::make_pair(view_id, height_divided_by_top_minus_bottom));

    }

}

void cut_database_record::
set_transform(const view_t model_id, const scm::math::mat4f& transform) {
#ifdef LAMURE_DATABASE_SAFE_MODE
    std::lock_guard<std::mutex> lock(mutex_);
#endif
    if (current_front_ == record_front::FRONT_A) {
        expand_front_a(0, model_id);

        auto it = front_a_transforms_.find(model_id);

        if (it != front_a_transforms_.end()) {
            front_a_transforms_.erase(it);
        }

        front_a_transforms_.insert(std::make_pair(model_id, transform));

    }
    else {
        expand_front_b(0, model_id);

        auto it = front_b_transforms_.find(model_id);

        if (it != front_b_transforms_.end()) {
            front_b_transforms_.erase(it);
        }

        front_b_transforms_.insert(std::make_pair(model_id, transform));

    }


}

void cut_database_record::
set_rendered(const model_t model_id) {
#ifdef LAMURE_DATABASE_SAFE_MODE
    std::lock_guard<std::mutex> lock(mutex_);
#endif
    if (current_front_ == record_front::FRONT_A) {
        expand_front_a(0, model_id);

        auto it = front_a_rendered_.find(model_id);

        if (it != front_a_rendered_.end()) {
            front_a_rendered_.erase(it);
        }

        front_a_rendered_.insert(model_id);

    }
    else {
        expand_front_b(0, model_id);

        auto it = front_b_rendered_.find(model_id);

        if (it != front_b_rendered_.end()) {
            front_b_rendered_.erase(it);
        }

        front_b_rendered_.insert(model_id);

    }

}

void cut_database_record::
set_threshold(const model_t model_id, const float threshold) {
#ifdef LAMURE_DATABASE_SAFE_MODE
    std::lock_guard<std::mutex> lock(mutex_);
#endif
    if (current_front_ == record_front::FRONT_A) {
        expand_front_a(0, model_id);

        auto it = front_a_thresholds_.find(model_id);

        if (it != front_a_thresholds_.end()) {
            front_a_thresholds_.erase(it);
        }

        front_a_thresholds_.insert(std::make_pair(model_id, threshold));

    }
    else {
        expand_front_b(0, model_id);

        auto it = front_b_thresholds_.find(model_id);

        if (it != front_b_thresholds_.end()) {
            front_b_thresholds_.erase(it);
        }

        front_b_thresholds_.insert(std::make_pair(model_id, threshold));

    }

}



void cut_database_record::
receive_transforms(std::map<model_t, scm::math::mat4f>& transforms) {
    //transforms.clear();

    std::lock_guard<std::mutex> lock(mutex_);

    if (current_front_ == record_front::FRONT_A) {
        for (const auto& trans_it : front_b_transforms_) {
            scm::math::mat4f transform = trans_it.second;
            model_t model_id = trans_it.first;
            transforms[model_id] = transform;
        }
    }
    else {
        for (const auto& trans_it : front_a_transforms_) {
            scm::math::mat4f transform = trans_it.second;
            model_t model_id = trans_it.first;
            transforms[model_id] = transform;
        }
    }
}

void cut_database_record::
receive_cameras(std::map<view_t, camera>& cameras) {
    //cameras.clear();

    std::lock_guard<std::mutex> lock(mutex_);

    if (current_front_ == record_front::FRONT_A) {
        for (const auto& cam_it : front_b_cameras_) {
            camera camera = cam_it.second;
            view_t view_id = cam_it.first;
            cameras[view_id] = camera;
        }
    }
    else {
        for (const auto& cam_it : front_a_cameras_) {
            camera camera = cam_it.second;
            view_t view_id = cam_it.first;
            cameras[view_id] = camera;
        }
    }
}

void cut_database_record::
receive_height_divided_by_top_minus_bottoms(std::map<view_t, float>& height_divided_by_top_minus_bottoms) {
    //height_divided_by_top_minus_bottoms.clear();

    std::lock_guard<std::mutex> lock(mutex_);

    if (current_front_ == record_front::FRONT_A) {
        for (const auto& hdtmb_it : front_b_height_divided_by_top_minus_bottom_) {

            float hdtmb = hdtmb_it.second;
            view_t view_id = hdtmb_it.first;
            height_divided_by_top_minus_bottoms[view_id] = hdtmb;
        }
    }
    else {
        for (const auto& hdtmb_it : front_a_height_divided_by_top_minus_bottom_) {
            float hdtmb = hdtmb_it.second;
            view_t view_id = hdtmb_it.first;
            height_divided_by_top_minus_bottoms[view_id] = hdtmb;
        }
    }
}


void cut_database_record::
receive_rendered(std::set<model_t>& rendered) {
    //rendered.clear();

    std::lock_guard<std::mutex> lock(mutex_);

    if (current_front_ == record_front::FRONT_A) {
        rendered = front_b_rendered_;
    }
    else {
        rendered = front_a_rendered_;
    }

}

void cut_database_record::
receive_thresholds(std::map<model_t, float>& thresholds) {
    //thresholds.clear();

    std::lock_guard<std::mutex> lock(mutex_);

    if (current_front_ == record_front::FRONT_A) {
        for (const auto& threshold_it : front_b_thresholds_) {
            float threshold = threshold_it.second;
            view_t model_id = threshold_it.first;
            thresholds[model_id] = threshold;
        }

    }
    else {
        for (const auto& threshold_it : front_a_thresholds_) {
            float threshold = threshold_it.second;
            view_t model_id = threshold_it.first;
            thresholds[model_id] = threshold;
        }
    }

}


void cut_database_record::
expand_front_a(const view_t view_id, const model_t model_id) {
    while (model_id >= front_a_cuts_.size())
    {
        //expand cut front B
        front_a_cuts_.push_back(std::vector<cut>());
        front_a_transforms_.insert(std::make_pair(model_id, scm::math::mat4f::identity()));
        front_a_thresholds_.insert(std::make_pair(model_id, LAMURE_DEFAULT_THRESHOLD));
    }

    view_t new_view_id = front_a_cuts_[model_id].size();

    while (view_id >= front_a_cuts_[model_id].size())
    {
        front_a_cuts_[model_id].push_back(cut(context_id_, new_view_id, model_id));
        front_a_height_divided_by_top_minus_bottom_.insert(std::make_pair(new_view_id, 1000.0f));
        ++new_view_id;
    }

    assert(model_id < front_a_cuts_.size());
    assert(view_id < front_a_cuts_[model_id].size());

}


void cut_database_record::
expand_front_b(const view_t view_id, const model_t model_id) {
    while (model_id >= front_b_cuts_.size())
    {
        //expand cut front B
        front_b_cuts_.push_back(std::vector<cut>());
        front_b_transforms_.insert(std::make_pair(model_id, scm::math::mat4f::identity()));
        front_b_thresholds_.insert(std::make_pair(model_id, LAMURE_DEFAULT_THRESHOLD));
    }

    view_t new_view_id = front_b_cuts_[model_id].size();

    while (view_id >= front_b_cuts_[model_id].size())
    {
        front_b_cuts_[model_id].push_back(cut(context_id_, new_view_id, model_id));
        front_b_height_divided_by_top_minus_bottom_.insert(std::make_pair(new_view_id, 1000.0f));
        ++new_view_id;
    }

    assert(model_id < front_b_cuts_.size());
    assert(view_id < front_b_cuts_[model_id].size());


}

cut& cut_database_record::
get_cut(const view_t view_id, const model_t model_id) {
    if (current_front_ == record_front::FRONT_A) {
        expand_front_a(view_id, model_id);
        return front_a_cuts_[model_id][view_id];
    }
    else {
        expand_front_b(view_id, model_id);
        return front_b_cuts_[model_id][view_id];
    }

}

void cut_database_record::
set_cut(const view_t view_id, const model_t model_id, cut& cut) {
    is_swap_required_ = true;

    if (current_front_ == record_front::FRONT_A) {
        expand_front_b(view_id, model_id);
        front_b_cuts_[model_id][view_id] = cut;
    }
    else {
        expand_front_a(view_id, model_id);
        front_a_cuts_[model_id][view_id] = cut;
    }

}

std::vector<cut_database_record::slot_update_desc>& cut_database_record::
get_updated_set() {
    if (current_front_ == record_front::FRONT_A) {
        return front_a_updated_set_;
    }
    else {
        return front_b_updated_set_;
    }

}

void cut_database_record::
set_updated_set(std::vector<cut_database_record::slot_update_desc>& updated_set) {
    is_swap_required_ = true;

    if (current_front_ == record_front::FRONT_A) {
        front_b_updated_set_ = updated_set;
    }
    else {
        front_a_updated_set_ = updated_set;
    }

}

const bool cut_database_record::
is_front_modified() const {
    if (current_front_ == record_front::FRONT_A) {
        return front_a_is_modified_;
    }
    else {
        return front_b_is_modified_;
    }

};

void cut_database_record::
set_is_front_modified(const bool front_modified) {
    is_swap_required_ = true;

    if (current_front_ == record_front::FRONT_A) {
        front_b_is_modified_ = front_modified;
    }
    else {
        front_a_is_modified_ = front_modified;
    }
};

void cut_database_record::
signal_upload_complete() {
    if (current_front_ == record_front::FRONT_A) {
        front_a_is_modified_ = false;
    }
    else {
        front_b_is_modified_ = false;
    }
}

const bool cut_database_record::
is_swap_required() const {
    return is_swap_required_;
};

void cut_database_record::
set_is_swap_required(const bool swap_required) {
    is_swap_required_ = true;
};

const cut_database_record::temporary_buffer cut_database_record::
get_buffer() const {
    if (current_front_ == record_front::FRONT_A) {
        return front_a_buffer_;
    }
    else {
        return front_b_buffer_;
    }

};

void cut_database_record::
set_buffer(const cut_database_record::temporary_buffer buffer) {
    is_swap_required_ = true;

    if (current_front_ == record_front::FRONT_A) {
        front_b_buffer_ = buffer;
    }
    else {
        front_a_buffer_ = buffer;
    }
};

void cut_database_record::
lock_front() {
    mutex_.lock();
}

void cut_database_record::
unlock_front() {
    mutex_.unlock();
}


} // namespace ren

} // namespace lamure




