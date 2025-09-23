// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_CUT_DATABASE_RECORD_H_
#define REN_CUT_DATABASE_RECORD_H_

#include <unordered_map>
#include <lamure/utils.h>
#include <lamure/types.h>
#include <mutex>
#include <lamure/ren/cut.h>
#include <lamure/ren/camera.h>
#include <lamure/ren/config.h>

namespace lamure {
namespace ren {

class cut_database_record
{
public:

    enum temporary_buffer
    {
        BUFFER_A = 0,
        BUFFER_B = 1,
        BUFFER_COUNT = 2
    };

    enum record_front
    {
        FRONT_A = 0,
        FRONT_B = 1,
        FRONT_COUNT = 2
    };

    struct slot_update_desc
    {
        explicit slot_update_desc(
            const slot_t src,
            const slot_t dst)
            : src_(src), dst_(dst) {};

        slot_t src_;
        slot_t dst_;
    };

                        cut_database_record(const context_t context_id);
                        cut_database_record(const cut_database_record&) = delete;
                        cut_database_record& operator=(const cut_database_record&) = delete;
    virtual             ~cut_database_record();

    void                set_cut(const view_t view_id, const model_t model_id, cut& cut);
    cut&                get_cut(const view_t view_id, const model_t model_id);

    std::vector<slot_update_desc>& get_updated_set();
    void                set_updated_set(std::vector<slot_update_desc>& updated_set);

    const bool          is_front_modified() const;
    const bool          is_swap_required() const;
    void                set_is_front_modified(const bool front_modified);
    void                set_is_swap_required(const bool swap_required);
    void                signal_upload_complete();

    const temporary_buffer get_buffer() const;
    void                set_buffer(const temporary_buffer buffer);

    void                swap_front();

    void                lock_front();
    void                unlock_front();

    void                set_camera(const view_t view_id, const camera& cam);
    void                set_height_divided_by_top_minus_bottom(const view_t view_id, float const height_divided_by_top_minus_bottom);
    void                set_transform(const model_t model_id, const scm::math::mat4f& transform);
    void                set_rendered(const model_t model_id);
    void                set_threshold(const model_t model_id, const float threshold);

    void                receive_cameras(std::map<view_t, camera>& cameras);
    void                receive_height_divided_by_top_minus_bottoms(std::map<view_t, float>& height_divided_by_top_minus_bottoms);
    void                receive_transforms(std::map<model_t, scm::math::mat4f>& transforms);
    void                receive_rendered(std::set<model_t>& rendered);
    void                receive_thresholds(std::map<model_t, float>& thresholds);

protected:

    void                expand_front_a(const view_t view_id, const model_t model_id);
    void                expand_front_b(const view_t view_id, const model_t model_id);

private:
    /* data */
    std::mutex          mutex_;

    context_t           context_id_;

    bool                is_swap_required_;

    record_front         current_front_;
    
    //dim: [model_id][view_id]
    std::vector<std::vector<cut>> front_a_cuts_;
    std::vector<std::vector<cut>> front_b_cuts_;

    bool                front_a_is_modified_;
    bool                front_b_is_modified_;

    temporary_buffer     front_a_buffer_;
    temporary_buffer     front_b_buffer_;

    std::vector<slot_update_desc> front_a_updated_set_;
    std::vector<slot_update_desc> front_b_updated_set_;

    std::map<view_t, camera> front_a_cameras_;
    std::map<view_t, camera> front_b_cameras_;

    std::map<view_t, float> front_a_height_divided_by_top_minus_bottom_;
    std::map<view_t, float> front_b_height_divided_by_top_minus_bottom_;

    std::map<model_t, scm::math::mat4f> front_a_transforms_;
    std::map<model_t, scm::math::mat4f> front_b_transforms_;

    std::set<model_t> front_a_rendered_;
    std::set<model_t> front_b_rendered_;

    std::map<model_t, float> front_a_thresholds_;
    std::map<model_t, float> front_b_thresholds_;
};


} } // namespace lamure


#endif // REN_CUT_DATABASE_RECORD_H_
