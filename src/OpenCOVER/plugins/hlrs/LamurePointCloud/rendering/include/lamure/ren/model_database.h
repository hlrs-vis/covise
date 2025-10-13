// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_MODEL_DATABASE_H_
#define REN_MODEL_DATABASE_H_

#include <unordered_map>
#include <mutex>

#include <lamure/utils.h>
#include <lamure/types.h>
#include <lamure/ren/dataset.h>
#include <lamure/ren/config.h>
#include <lamure/ren/platform.h>

#include <scm/gl_core/query_objects.h>

namespace lamure {
namespace ren {

class model_database
{
public:

                        model_database(const model_database&) = delete;
                        model_database& operator=(const model_database&) = delete;
    virtual             ~model_database() noexcept;

    static model_database* get_instance();
    static void destroy_instance();

    const model_t       add_model(const std::string& filepath, const std::string& model_key);
    dataset*            get_model(const model_t model_id);
    void                apply();
    void                reset();

    const model_t       num_models() const { return num_datasets_; };

    const size_t        get_primitive_size(const bvh::primitive_type type) const;
    const size_t        get_node_size(const model_t model_id) const;

    const size_t        get_slot_size() const;
    const size_t        get_primitives_per_node() const;
    const size_t        get_primitives_per_node(const model_t model_id) const;

    static bool         contains_only_compressed_data_;
    static bool         contains_trimesh_;
protected:

                        model_database();
    static bool         is_instanced_;
    static model_database* single_;

private:
    static std::mutex   mutex_;
    mutable std::mutex state_mutex_;

    std::unordered_map<model_t, dataset*> datasets_;

    model_t             num_datasets_;
    model_t             num_datasets_pending_;
    size_t              primitives_per_node_;
    size_t              primitives_per_node_pending_;


};


} } // namespace lamure


#endif // REN_MODEL_DATABASE_H_
