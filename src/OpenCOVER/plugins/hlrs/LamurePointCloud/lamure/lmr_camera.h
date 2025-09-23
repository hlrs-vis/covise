// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef CO_REN_CAMERA_H_
#define CO_REN_CAMERA_H_

#include <lamure/ren/camera.h>

class lmr_camera : public lamure::ren::camera
{

protected:
    enum camera_state
    {
        CAM_STATE_GUA,
        CAM_STATE_LAMURE,
        INVALID_CAM_STATE
    };

    float const remap_value(float value, float oldMin, float oldMax, float newMin, float newMax) const;
    float const transfer_values(float currentValue, float maxValue) const;


public:
    enum class control_type
    {
        mouse
    };



   lmr_camera(
        lamure::view_t view_id,
        scm::math::mat4d view_matrix,
        scm::math::mat4d projection_matrix,
        scm::gl::frustum frustum,
        float near_plane_value,
        float far_plane_value,
        control_type controlType,
        double sum_trans_x,
        double sum_trans_y,
        double sum_trans_z,
        double sum_rot_x,
        double sum_rot_y,
        double sum_rot_z,
        camera_state cam_state);

    ~lmr_camera();

    
    static lmr_camera* instance(lamure::view_t view_id);

    struct mouse_state
    {
        bool lb_down_;
        bool mb_down_;
        bool rb_down_;

        mouse_state() : lb_down_(false), mb_down_(false), rb_down_(false) {}
    };

    const lamure::view_t view_id() const { return view_id_; };

    scm::math::mat4 calc_get_projection_matrix(float opening_angle, float aspect_ratio, float nearplane, float farplane);

    void set_trackball_matrix(scm::math::mat4d const& tb_matrix);

    void set_projection_matrix(float opening_angle, float aspect_ratio, float near, float far);

    void calc_view_to_screen_space_matrix(scm::math::vec2f const &win_dimensions);

    scm::math::vec3d get_cam_pos();
    scm::math::mat4f get_cam_matrix();

    scm::gl::frustum::classification_result const cull_against_frustum(scm::gl::frustum const &frustum, scm::gl::box const &b) const;

    scm::gl::frustum const get_frustum_by_model(scm::math::mat4 const &model) const;

    scm::gl::frustum const get_predicted_frustum(scm::math::mat4f const &in_cam_or_mat);

    inline const float near_plane_value() const { return near_plane_value_; }
    inline const float far_plane_value() const { return far_plane_value_; }

    inline const void set_near_plane_value(float np) {  near_plane_value_ = np; }
    inline const void set_far_plane_value(float fp) {  far_plane_value_ = fp; }

    std::vector<scm::math::vec3d> get_frustum_corners() const;

    //set projection and view matrix previously
    void calc_set_frustum();
    scm::gl::frustum calc_get_frustum();

    void set_frustum(scm::gl::frustum frustum);

    scm::math::mat4 get_view_matrix();
    scm::math::mat4 get_projection_matrix();

    scm::math::mat4d get_hp_view_matrix();
    scm::math::mat4d get_hp_projection_matrix();

    void set_hp_view_matrix(scm::math::mat4d m);
    void set_hp_projection_matrix(scm::math::mat4d m);

    void set_lookat_matrix(scm::math::mat4d m);

    void write_view_matrix(std::ofstream& matrix_stream);

    void translate(const float& x, const float& y, const float& z);
    void rotate(const float& x, const float& y, const float& z);



  private:
    lamure::view_t view_id_;

    scm::math::mat4d view_matrix_;
    scm::math::mat4d projection_matrix_;
    scm::gl::frustum frustum_;

    float near_plane_value_;
    float far_plane_value_;

    control_type controlType_;

    static std::mutex transform_update_mutex_;

    double sum_trans_x_;
    double sum_trans_y_;
    double sum_trans_z_;
    double sum_rot_x_;
    double sum_rot_y_;
    double sum_rot_z_;

    camera_state cam_state_;
};

#endif // CO_REN_CAMERA_H_
