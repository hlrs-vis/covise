// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_APP_MANAGEMENT_H_
#define REN_APP_MANAGEMENT_H_

#include <lamure/utils.h>
#include <lamure/types.h>

#include "renderer.h"
#include <lamure/ren/config.h>

#include <lamure/ren/model_database.h>
#include <lamure/ren/controller.h>
#include <lamure/ren/cut.h>
#include <lamure/ren/cut_update_pool.h>


#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <FreeImagePlus.h>

#include <lamure/ren/ray.h>

struct snapshot_session_descriptor {
    snapshot_session_descriptor() : num_taken_screenshots(0),
                                    session_filename_(""),
                                    recorded_view_vector_(),
                                    snapshot_resolution_(scm::math::vec2ui(0,0)),
                                    snapshot_session_enabled_(false)
                                    {}

    //set_session_filename();

    uint32_t get_num_taken_screenshots() {
      return num_taken_screenshots;
    }   

    void increment_screenshot_counter() {
        ++num_taken_screenshots;
    }

    std::string get_screenshot_name() { 

        return
          std::to_string(num_taken_screenshots) 
        + "__" + std::to_string(snapshot_resolution_[0]) 
        + "_" + std::to_string(snapshot_resolution_[1]);}

    unsigned num_taken_screenshots;
    std::string session_filename_;
    std::vector<scm::math::mat4d> recorded_view_vector_;
    scm::math::vec2ui snapshot_resolution_;
    bool snapshot_session_enabled_;
};

class management
{
public:
                        management(std::vector<std::string> const& model_filenames,
                            std::vector<scm::math::mat4f> const& model_transformations,
                            const std::set<lamure::model_t>& visible_set,
                            const std::set<lamure::model_t>& invisible_set,
                            snapshot_session_descriptor& snap_descriptor);
                        
    virtual             ~management();

                        management(const management&) = delete;
                        management& operator=(const management&) = delete;

    bool                MainLoop();
    void                update_trackball(int x, int y);
    void                RegisterMousePresses(int button, int state, int x, int y);
    void                dispatchKeyboardInput(unsigned char key);
    void                dispatchSpecialInput(int key);
    void                dispatchSpecialInputRelease(int key);
    void                dispatchResize(int w, int h);

    void                resolve_movement(double elapsed_time_ms);

    void                PrintInfo();
    void                SetSceneName();

    float               error_threshold_;
    void                IncreaseErrorThreshold();
    void                DecreaseErrorThreshold();

    void                start_update_viewer_position_thread();

    void                interpolate_between_measurement_transforms(const bool& allow_interpolation);
    void                set_interpolation_step_size(const float& interpolation_step_size);

    void                enable_culling(const bool& enable);

    void                forward_background_color(float bg_r, float bg_g, float bg_b);
protected:

    void                Toggledispatching();
    void                toggle_camera_session();
    void                record_next_camera_position();
    void                create_quality_measurement_resources();

private:
    
    size_t              num_taken_screenshots_;
    bool const          allow_user_input_;
    bool                screenshot_session_started_;
    bool                camera_recording_enabled_;
    std::string const   current_session_filename_;
    std::string         current_session_file_path_;
    unsigned            num_recorded_camera_positions_;

#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
    split_screen_renderer* renderer_;
    lamure::ren::camera*   active_camera_left_;
    lamure::ren::camera*   active_camera_right_;
    bool                control_left_;
#endif

#ifndef LAMURE_RENDERING_USE_SPLIT_SCREEN
    Renderer* renderer_;
#endif

    lamure::ren::camera*   active_camera_;

    int32_t             width_;
    int32_t             height_;

    float importance_;

    bool test_send_rendered_;

    lamure::view_t         num_cameras_;
    std::vector<lamure::ren::camera*> cameras_;

    lamure::ren::camera::mouse_state mouse_state_;

    bool                fast_travel_;
    unsigned            travel_speed_mode_;

    bool                dispatch_;
    bool                trigger_one_update_;

    scm::math::mat4f    reset_matrix_;
    float               reset_diameter_;

    uint32_t            current_session_number_;
    double              current_update_timeout_timer_;

    lamure::model_t     num_models_;

    scm::math::vec3f    detail_translation_;
    float               detail_angle_;
    float               near_plane_;
    float               far_plane_;

    std::vector<scm::math::mat4f> model_transformations_;
    std::vector<std::string> model_filenames_;

    snapshot_session_descriptor& measurement_session_descriptor_;
    //std::vector<scm::math::mat4d>& recorded_view_vector_;
    float interpolation_time_point_;
    bool use_interpolation_on_measurement_session_;
    float movement_on_interpolation_per_frame_;
    double snapshot_framerate_counter_;
    size_t snapshot_frame_counter_;

    bool                is_updating_pvs_position_;

    bool                use_wasd_camera_control_scheme_;
    int                 mouse_last_x_;
    int                 mouse_last_y_;

    bool                is_moving_forward_  = false;
    bool                is_moving_backward_ = false;
    bool                is_moving_left_  = false;
    bool                is_moving_right_ = false;

#ifdef LAMURE_CUT_UPDATE_ENABLE_MEASURE_SYSTEM_PERFORMANCE
    boost::timer::cpu_timer system_performance_timer_;
    boost::timer::cpu_timer system_result_timer_;
#endif
};


#endif // REN_MANAGEMENT_H_


