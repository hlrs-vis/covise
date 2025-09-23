// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/model_database.h>
#include <lamure/ren/controller.h>

namespace lamure
{

namespace ren
{

std::mutex model_database::mutex_;
bool model_database::is_instanced_ = false;
model_database* model_database::single_ = nullptr;
bool model_database::contains_only_compressed_data_ = true;
bool model_database::contains_trimesh_ = false;

model_database::
model_database()
: num_datasets_(0),
  num_datasets_pending_(0),
  primitives_per_node_(0),
  primitives_per_node_pending_(0) {

}

model_database::
~model_database() {
    std::lock_guard<std::mutex> lock(mutex_);

    is_instanced_ = false;

    for (const auto& model_it : datasets_) {
        dataset* model = model_it.second;
        if (model != nullptr) {
            delete model;
            model = nullptr;
        }
    }

    datasets_.clear();
}

model_database* model_database::
get_instance() {
    if (!is_instanced_) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!is_instanced_) { //double-checked locking
            single_ = new model_database();
            is_instanced_ = true;
        }

        return single_;
    }
    else {
        return single_;
    }
}

void model_database::
apply() {
    std::lock_guard<std::mutex> lock(mutex_);

    num_datasets_ = num_datasets_pending_;
    primitives_per_node_ = primitives_per_node_pending_;
}

const model_t model_database::
add_model(const std::string& filepath, const std::string& model_key) {

    if (controller::get_instance()->is_model_present(model_key)) {
        return controller::get_instance()->deduce_model_id(model_key);
    }

    dataset* model = new dataset(filepath);

    if (model->is_loaded()) {
        const bvh* bvh = model->get_bvh();

        if( lamure::ren::bvh::primitive_type::POINTCLOUD_QZ != bvh->get_primitive() ) {
            model_database::contains_only_compressed_data_ = false;
        }

        if( lamure::ren::bvh::primitive_type::TRIMESH == bvh->get_primitive() ) {
            model_database::contains_trimesh_ = true;
        }

        if (num_datasets_ == 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (num_datasets_ == 0) {
                primitives_per_node_pending_ = bvh->get_primitives_per_node();
                primitives_per_node_ = primitives_per_node_pending_;
            }
        }

        if (bvh->get_primitives_per_node() > primitives_per_node_pending_) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (bvh->get_primitives_per_node() > primitives_per_node_pending_) {
                primitives_per_node_pending_ = bvh->get_primitives_per_node();
            }
        }

        model_t model_id = 0;

        {
            std::lock_guard<std::mutex> lock(mutex_);

            model_id = controller::get_instance()->deduce_model_id(model_key);

            model->model_id_ = model_id;
            datasets_[model_id] = model;

            ++num_datasets_pending_;

            num_datasets_ = num_datasets_pending_;
            primitives_per_node_ = primitives_per_node_pending_;

            controller::get_instance()->signal_system_reset();
        }

        switch (bvh->get_primitive()) {

          case bvh::primitive_type::POINTCLOUD:
#ifdef LAMURE_ENABLE_INFO
            std::cout << "lamure: pointcloud " << model_id << ": " << filepath << std::endl;
#endif
            break;


          case bvh::primitive_type::POINTCLOUD_QZ:
#ifdef LAMURE_ENABLE_INFO
            std::cout << "lamure: pointcloud_qz " << model_id << ": " << filepath << std::endl;
#endif

          case bvh::primitive_type::TRIMESH:
#ifdef LAMURE_ENABLE_INFO
            std::cout << "lamure: trimesh " << model_id << ": " << filepath << std::endl;
#endif
            break;

          default:
            throw std::runtime_error(
                "lamure: unknown primitive type: " + std::to_string(bvh->get_primitive()));
            break; 

        }

        return model_id;

    }
    else {
        throw std::runtime_error(
            "lamure: model_database::Model was not loaded");
    }

    return invalid_model_t;

}

dataset* model_database::
get_model(const model_t model_id) {
    if (datasets_.find(model_id) != datasets_.end()) {
        return datasets_[model_id];
    }
    throw std::runtime_error(
        "lamure: model_database::Model was not found:" + std::to_string(model_id));
    return nullptr;
}

const size_t model_database::
get_primitive_size(const bvh::primitive_type type) const {
    switch (type) {
        case bvh::primitive_type::POINTCLOUD:
            return sizeof(dataset::serialized_surfel);
        case bvh::primitive_type::POINTCLOUD_QZ:
            return sizeof(dataset::serialized_surfel_qz);
        case bvh::primitive_type::TRIMESH:
            return sizeof(dataset::serialized_vertex);
        default: break;
    }
    throw std::runtime_error(
        "lamure: model_database::Invalid primitive type has size 0");
    return 0;
}

const size_t model_database::
get_node_size(const model_t model_id) const {
    auto model_it = datasets_.find(model_id);
    if (model_it != datasets_.end()) {
        const bvh* bvh = model_it->second->get_bvh();
        return get_primitive_size(bvh->get_primitive()) * bvh->get_primitives_per_node();
    }
    throw std::runtime_error(
        "lamure: model_database::Model was not found:" + std::to_string(model_id));
    return 0;

}

const size_t model_database::
get_primitives_per_node(const model_t model_id) const {
    auto model_it = datasets_.find(model_id);
    if (model_it != datasets_.end()) {
        const bvh* bvh = model_it->second->get_bvh();
        return bvh->get_primitives_per_node();
    }
    throw std::runtime_error(
        "lamure: model_database::Model was not found:" + std::to_string(model_id));
    return 0;

}

const size_t model_database::
get_slot_size() const {
    //return the combined slot size in bytes for both trimeshes and pointclouds
    if( model_database::contains_only_compressed_data_ ) {
      return primitives_per_node_ * sizeof(dataset::serialized_surfel_qz);
    }
    else {
      if (model_database::contains_trimesh_) {
        return primitives_per_node_ * sizeof(dataset::serialized_vertex);
      }
      else {
        return primitives_per_node_ * sizeof(dataset::serialized_surfel);    
      }
    }
    return 0;
    
}

const size_t model_database::
get_primitives_per_node() const {
    //return the combined primitives per node for both trimeshes and pointclouds
    return primitives_per_node_;
}

} // namespace ren

} // namespace lamure


