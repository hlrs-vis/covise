// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_CAMERA_H_
#define REN_CAMERA_H_

#include <scm/gl_core.h>
#include <scm/gl_core/primitives/frustum.h>
#include <scm/gl_util/viewer/camera.h>

#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <lamure/ren/platform.h>
#include <lamure/ren/trackball.h>
#include <lamure/types.h>

namespace lamure
{
namespace ren
{
class camera
{
  public:
    enum class control_type
    {
        mouse
    };

    struct mouse_state
    {
        bool lb_down_;
        bool mb_down_;
        bool rb_down_;
        mouse_state() : lb_down_(false), mb_down_(false), rb_down_(false) {}
    };

    camera() : cam_state_(CAM_STATE_GUA){};
    camera(const view_t view_id, double near, scm::math::mat4d const& view, scm::math::mat4d const& proj);
    //camera(const view_t view_id, float near, float far, scm::math::mat4f const& view, scm::math::mat4f const& proj, scm::math::mat4f const& init_tb_mat, float distance, bool fast_travel = false, bool touch_screen_mode = false);
    camera(const view_t view_id, scm::math::mat4f const& init_tb_mat, double distance);
    //camera(const view_t view_id, scm::math::vec3d models_center, double distance, double fov, double aspect_ratio, double near, double far);
    //camera(const view_t view_id, scm::math::vec3d models_center, double fov, double aspect_ratio, double near, double far);
    camera(const view_t view_id, double left, double right, double bottom, double top, double near, double far, scm::math::vec3d eye, scm::math::vec3d center, scm::math::vec3d up, double look_dist);
    camera(const view_t view_id, double near, double far, scm::math::mat4d view_matrix_, scm::math::mat4d projection_matrix_);
    virtual ~camera();

    void event_callback(uint16_t code, float value);
    const view_t view_id() const { return view_id_; };

    void calc_view_to_screen_space_matrix(scm::math::vec2f const &win_dimensions);
    scm::gl::frustum::classification_result const cull_against_frustum(scm::gl::frustum const &frustum, scm::gl::box const &b) const;
    void set_dolly_sens_(double ds) { dolly_sens_ = ds; }

    void update_trackball_mouse_pos(double x, double y);
    void update_trackball(double x, double y, int window_width, int window_height, mouse_state const &mouse_state);
    void update_camera(double x, double y, int window_width, int window_height, mouse_state const &mouse_state, bool keys[]);
    
    //void set_frustum_orientation(scm::math::vec3 newCameraPosition, scm::math::vec3 newTargetPosition, scm::math::vec3 newUpVector);
    //void set_frustum_dimensions(float fov, float aspect_ratio, float near, float far_plane_value);
    
    
    scm::gl::frustum const get_predicted_frustum(scm::math::mat4f const &in_cam_or_mat);

    inline const double near_plane_value() const { return near_; }
    inline const double far_plane_value() const { return far_; }

    void set_trackball_matrix(scm::math::mat4d const& tb_matrix) { trackball_.set_transform(tb_matrix); }

    void set_view_matrix(const scm::math::mat4d &in_view);
    void set_view_matrix(scm::math::mat4d& in_view);
    void set_projection_matrix(const scm::math::mat4d &in_proj);
    void set_projection_matrix(scm::math::mat4d& in_proj);

    void set_projection_matrix(float opening_angle, float aspect_ratio, float near_, float far_);

    void set_cam_pos(scm::math::vec3d const& cam_pos);

    const scm::math::mat4d& trackball_matrix() const { return trackball_.transform(); };

    scm::math::mat4f const get_view_matrix() const;
    scm::math::mat4f const get_projection_matrix() const;

    scm::math::mat4f get_view_matrix();
    scm::math::mat4f get_projection_matrix();

    //scm::math::mat4f const get_mvp_matrix() const;

    scm::math::mat4d const get_high_precision_view_matrix() const;
    //scm::math::mat4d const get_high_precision_projection_matrix() const;
    //scm::math::mat4d const get_high_mvp_matrix() const;

    scm::math::vec3d get_cam_pos();

    scm::math::mat4f get_cam_matrix();
    //scm::math::mat4d get_hp_cam_matrix();

    //void const update_frustum();    
    scm::gl::frustum const get_frustum_by_model(scm::math::mat4d const& model) const;
    scm::gl::frustum const get_frustum_by_model(scm::math::mat4f const& model) const;

    scm::gl::frustum const get_frustum() const;
    std::vector<scm::math::vec3d> get_frustum_corners() const;
    std::vector<scm::math::vec3d> get_frustum_corners_by_model(scm::math::mat4d const& model) const;

    void    write_view_matrix(std::ofstream& matrix_stream);
    void    set_trackball_center_of_rotation(const scm::math::vec3f& cor);

    void    translate(const float& x, const float& y, const float& z);
    void    rotate(const float& x, const float& y, const float& z);


  protected:
    enum camera_state
    {
        CAM_STATE_GUA,
        CAM_STATE_LAMURE,
        INVALID_CAM_STATE
    };

    float const remap_value(float value, float oldMin, float oldMax, float newMin, float newMax) const;
    float const transfer_values(float currentValue, float maxValue) const;

  private:
    view_t view_id_;

    double left_;
    double right_;
    double bottom_;
    double top_;
    double near_;
    double far_;
    scm::math::vec3d center_;
    scm::math::vec3d up_;
    double look_dist_;
    scm::math::vec3d eye_;
    scm::math::mat4d eye_matrix_;
    scm::math::mat4d model_matrix_;
    scm::math::mat4d view_matrix_;
    scm::math::mat4d projection_matrix_;
    //scm::math::mat4f mvp_matrix_;

    scm::gl::frustum frustum_;

    double fov_;
    double aspect_ratio_;

    lamure::ren::trackball trackball_;

    double trackball_init_x_;
    double trackball_init_y_;

    double dolly_sens_;

    control_type controlType_;

    static std::mutex transform_update_mutex_;

    bool is_in_touch_screen_mode_;

    double sum_trans_x_;
    double sum_trans_y_;
    double sum_trans_z_;
    double sum_rot_x_;
    double sum_rot_y_;
    double sum_rot_z_;

    camera_state cam_state_;
};
}
} // namespace lamure

#endif // REN_CAMERA_H_
