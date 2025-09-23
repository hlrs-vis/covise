// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/camera.h>

namespace lamure
{
namespace ren
{
std::mutex camera::transform_update_mutex_;
/*
camera::camera(const view_t view_id, double near, scm::math::mat4d const &view, scm::math::mat4d const &proj)
    : view_id_(view_id), view_matrix_(view), projection_matrix_(proj), near_(near), far_(1000.0f), trackball_init_x_(0.0), trackball_init_y_(0.0), dolly_sens_(0.5f),
      is_in_touch_screen_mode_(0), sum_trans_x_(0), sum_trans_y_(0), sum_trans_z_(0), sum_rot_x_(0), sum_rot_y_(0), sum_rot_z_(0), cam_state_(CAM_STATE_GUA)
{
    frustum_ = scm::gl::frustum(scm::math::mat4f(projection_matrix_ * view_matrix_));
}

camera::camera(const view_t view_id, scm::math::vec3d models_center, double distance, double fov, double aspect_ratio, double near, double far)
    : view_id_(view_id), near_(near), far_(far), fov_(fov), aspect_ratio_(aspect_ratio),
    trackball_init_x_(0.0), trackball_init_y_(0.0), dolly_sens_(0.5f), is_in_touch_screen_mode_(0),
    sum_trans_x_(0), sum_trans_y_(0), sum_trans_z_(0), sum_rot_x_(0), sum_rot_y_(0), sum_rot_z_(0), 
    cam_state_(CAM_STATE_LAMURE), controlType_(control_type(0))
{
    scm::math::mat4d init_tb_mat =
        scm::math::make_look_at_matrix(
            models_center + scm::math::vec3d(0., 0.1, -0.01),
            models_center,
            scm::math::vec3d(0.0, 0.0, 1.0));

    scm::math::mat4d projection_matrix_;
    scm::math::perspective_matrix(projection_matrix_, fov_, aspect_ratio_, near_, far_);
    trackball_.set_transform(scm::math::mat4d(init_tb_mat));
    trackball_.dolly(distance);
    frustum_ = scm::gl::frustum(scm::math::mat4f(projection_matrix_ * trackball_.transform()));
};


camera::camera(const view_t view_id, scm::math::vec3d models_center, double fov, double aspect_ratio, double near, double far)
    : view_id_(view_id), near_(near), far_(far), fov_(fov), aspect_ratio_(aspect_ratio),
    cam_state_(CAM_STATE_GUA), controlType_(control_type(0))
{
    eye_ = scm::math::vec3d(0., (near_ - far_) / 2, 0.0);
    eye_matrix_ = scm::math::make_translation(eye_);

    view_matrix_ = scm::math::make_look_at_matrix(
        eye_,
        scm::math::vec3d::zero(),
        scm::math::vec3d(0.0, 0.0, 1.0)
    );
    //model_matrix_ = scm::math::make_translation(models_center);

    scm::math::perspective_matrix(projection_matrix_, fov_, aspect_ratio_, near_, far_);
    frustum_ = scm::gl::frustum(scm::math::mat4f(projection_matrix_ * view_matrix_));
};*/

camera::camera(const view_t view_id, scm::math::mat4f const& init_tb_mat, double distance)
    : view_id_(view_id), near_(0), far_(0), trackball_init_x_(0.0), trackball_init_y_(0.0), dolly_sens_(0.5f), is_in_touch_screen_mode_(0),
    sum_trans_x_(0), sum_trans_y_(0), sum_trans_z_(0), sum_rot_x_(0), sum_rot_y_(0), sum_rot_z_(0), cam_state_(CAM_STATE_LAMURE), controlType_(control_type(0))
{
    // set_projection_matrix(30.0f, float(800)/float(600), 0.01f, 100.0f);
    // scm::math::perspective_matrix(projection_matrix_, 60.f, float(800)/float(600), 0.1f, 100.0f);
    // frustum_ = scm::gl::frustum(projection_matrix_);
    //  scm::math::perspective_matrix(projection_matrix_, 60.f, float(800)/float(600), 0.1f, 100.0f);

    trackball_.set_transform(scm::math::mat4d(init_tb_mat));
    trackball_.dolly(distance);
}


camera::camera(const view_t view_id, double left, double right, double bottom, double top, double near, double far, 
    scm::math::vec3d eye, scm::math::vec3d center, scm::math::vec3d up, double look_dist) 
    : view_id_(view_id), left_(left), right_(right), bottom_(bottom), top_(top), near_(near), far_(far), 
    eye_(eye), center_(center), up_(up), look_dist_(look_dist),
    cam_state_(CAM_STATE_GUA), controlType_(control_type(0))
{
    double fovy = 2.0 * std::atan((top_ - bottom_) / (2.0 * near_));
    double aspect = (right_ - left_) / (top_ - bottom_);
    eye_matrix_ = scm::math::make_translation(eye_);
    view_matrix_ = scm::math::make_look_at_matrix(eye_, center_, up_);
    model_matrix_ = scm::math::mat4f::identity();
    scm::math::perspective_matrix(projection_matrix_, fov_, aspect_ratio_, near_, far_);
    frustum_ = scm::gl::frustum(scm::math::mat4f(projection_matrix_ * view_matrix_));
};


camera::camera(const view_t view_id, double near, double far, scm::math::mat4d view_matrix, scm::math::mat4d projection_matrix) 
    : view_id_(view_id), near_(near), far_(far), view_matrix_(view_matrix), projection_matrix_(projection_matrix), cam_state_(CAM_STATE_GUA), controlType_(control_type(0))
{
    eye_ = scm::math::vec3d(0.0, (near_ - far_) / 2, 0.0);
    eye_matrix_ = scm::math::make_translation(eye_);
    frustum_ = scm::gl::frustum(scm::math::mat4f(projection_matrix_ * view_matrix_));
};


camera::~camera() {}


void camera::set_trackball_center_of_rotation(const scm::math::vec3f &cor)
{
    trackball_.set_dolly(0.f);
    scm::math::mat4f cm = scm::math::inverse(scm::math::mat4f(trackball_.transform()));
    scm::math::vec3f pos = scm::math::vec3f(cm[12], cm[13], cm[14]);
    if(scm::math::length(pos - cor) < 0.001f) { return; }
    scm::math::vec3f up = scm::math::vec3f(cm[4], cm[5], cm[6]);
    scm::math::mat4 look_at = scm::math::make_look_at_matrix(cor + scm::math::vec3f(0.f, 0.f, 0.001f), cor, up);
    trackball_.set_transform(scm::math::mat4d::identity());
    trackball_.set_transform(scm::math::mat4d(look_at));
    trackball_.dolly(scm::math::length(pos - (cor + scm::math::vec3f(0.f, 0.f, 0.001f))));
}


void camera::event_callback(uint16_t code, float value)
{
    std::lock_guard<std::mutex> lock(transform_update_mutex_);

    float const transV = 3.0f;
    float const rotV = 5.0f;
    float const rotVz = 15.0f;

    if(std::abs(value) < 0.0)
        value = 0;

    if(is_in_touch_screen_mode_ == true)
    {
        switch(code)
        {
        case 1:
            code = 2;
            break;
        case 2:
            value = -value;
            code = 1;
            break;
        case 4:

            code = 5;
            break;
        case 5:
            value = -value;
            code = 4;
            break;
        }
    }

    if(code == 0)
    {
        sum_trans_x_ = remap_value(-value, -500, 500, -transV, transV);
    }
    if(code == 2)
    {
        sum_trans_y_ = remap_value(value, -500, 500, -transV, transV);
    }
    if(code == 1)
    {
        sum_trans_z_ = remap_value(-value, -500, 500, -transV, transV);
    }
    if(code == 3)
    {
        sum_rot_x_ = remap_value(-value, -500, 500, -rotV, rotV); // 0
    }
    if(code == 5)
    {
        sum_rot_y_ = remap_value(value, -500, 500, -rotV, rotV); // 0
    }
    if(code == 4)
    {
        sum_rot_z_ = remap_value(-value, -500, 500, -rotVz, rotVz);
    }
}


scm::gl::frustum::classification_result const camera::cull_against_frustum(scm::gl::frustum const &frustum, scm::gl::box const &b) const { return frustum.classify(b); }


scm::gl::frustum const camera::get_frustum_by_model(scm::math::mat4d const &model) const
{
    switch(cam_state_)
    {
    case CAM_STATE_LAMURE:
        return scm::gl::frustum(scm::math::mat4f(projection_matrix_ * trackball_.transform() * model));
        break;

    case CAM_STATE_GUA:
        return scm::gl::frustum(scm::math::mat4f(projection_matrix_ * view_matrix_ * model));
        break;

    default:
        break;
    }
    return scm::gl::frustum();
}



scm::gl::frustum const camera::get_frustum_by_model(scm::math::mat4f const& model) const
{
    switch (cam_state_)
    {
    case CAM_STATE_LAMURE:
        {
        scm::math::mat4d tr = trackball_.transform();
        return scm::gl::frustum(scm::math::mat4f(projection_matrix_ * tr) * model);
        }
        break;

    case CAM_STATE_GUA:
        return scm::gl::frustum(scm::math::mat4f(projection_matrix_ * view_matrix_) * model);
        break;

    default:
        break;
    }
    return scm::gl::frustum();
}


scm::gl::frustum const camera::get_frustum() const {
    return frustum_;
};


void camera::set_projection_matrix(float opening_angle, float aspect_ratio, float near, float far)
{
    //scm::math::perspective_matrix(projection_matrix_, opening_angle, aspect_ratio, near, far);

    //near_plane_value_ = near;
    //far_plane_value_ = far;

    //frustum_ = scm::gl::frustum(projection_matrix_ * scm::math::mat4f(trackball_.transform()));
}


void camera::set_projection_matrix(scm::math::mat4d& in_proj) {

    projection_matrix_ = in_proj;
    frustum_.update(scm::math::mat4f(projection_matrix_ * view_matrix_));
}


void camera::set_projection_matrix(const scm::math::mat4d& in_proj) {

    projection_matrix_ = in_proj;
    frustum_.update(scm::math::mat4f(projection_matrix_ * view_matrix_));
}

void camera::set_view_matrix(scm::math::mat4d& in_view) {
    switch (cam_state_)
    {
    case CAM_STATE_LAMURE:
        trackball_.set_transform(in_view);
        frustum_.update(scm::math::mat4f(projection_matrix_ * view_matrix_));
        break;
    case CAM_STATE_GUA:
        view_matrix_ = in_view;
        frustum_.update(scm::math::mat4f(projection_matrix_ * view_matrix_));
        break;
    default:
        break;
    }
}

void camera::set_view_matrix(const scm::math::mat4d &in_view) {
    switch(cam_state_)
    {
    case CAM_STATE_LAMURE:
        trackball_.set_transform(in_view);
        frustum_.update(scm::math::mat4f(projection_matrix_ * view_matrix_));
        break;
    case CAM_STATE_GUA:
        view_matrix_ = in_view;
        frustum_.update(scm::math::mat4f(projection_matrix_ * view_matrix_));
        break;
    default:
        break;
    }
}


void camera::update_trackball_mouse_pos(double x, double y) {
    trackball_init_x_ = x;
    trackball_init_y_ = y;
}


void camera::update_trackball(double x, double y, int window_width, int window_height, mouse_state const &mouse_state) {
    double nx = 2.0 * double(x - (window_width / 2)) / double(window_width);
    double ny = 2.0 * double(window_height - y - (window_height / 2)) / double(window_height);

    if(mouse_state.lb_down_)
    {
        trackball_.rotate(trackball_init_x_, trackball_init_y_, nx, ny);
    }
    if(mouse_state.rb_down_)
    {
        trackball_.dolly(dolly_sens_ * 0.5 * (ny - trackball_init_y_));
    }
    if(mouse_state.mb_down_)
    {
        double f = dolly_sens_ < 1.0 ? 0.02 : 0.3;
        trackball_.translate(f * (nx - trackball_init_x_), f * (ny - trackball_init_y_));
    }

    trackball_init_y_ = ny;
    trackball_init_x_ = nx;
}


void camera::write_view_matrix(std::ofstream &matrix_stream) {
    scm::math::mat4d t_mat = trackball_.transform();
    matrix_stream << t_mat[0] << " " << t_mat[1] << " " << t_mat[2] << " " << t_mat[3] << " " << t_mat[4] << " " << t_mat[5] << " " << t_mat[6] << " " << t_mat[7] << " " << t_mat[8] << " " << t_mat[9]
                  << " " << t_mat[10] << " " << t_mat[11] << " " << t_mat[12] << " " << t_mat[13] << " " << t_mat[14] << " " << t_mat[15] << "\n";
}


float const camera::transfer_values(float currentValue, float maxValue) const { return std::pow((std::abs(currentValue) / std::abs(maxValue)), 4); }


float const camera::remap_value(float value, float oldMin, float oldMax, float newMin, float newMax) const {
    float intermediateValue = (((value - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin;
    return transfer_values(intermediateValue, newMax) * intermediateValue;
}


scm::math::mat4d const camera::get_high_precision_view_matrix() const {
    if(cam_state_ == CAM_STATE_LAMURE)
    {
        return trackball_.transform();
    }
    return scm::math::mat4d(view_matrix_);
}


scm::math::mat4f const camera::get_projection_matrix() const { 
    return scm::math::mat4f(projection_matrix_);
}


scm::math::mat4f camera::get_projection_matrix() {
    return scm::math::mat4f(projection_matrix_);
}


std::vector<scm::math::vec3d> camera::get_frustum_corners_by_model(scm::math::mat4d const& model) const {
    std::vector<scm::math::vec4d> tmp(8);
    std::vector<scm::math::vec3d> result(8);

    scm::math::mat4d inverse_transform;

    if (CAM_STATE_LAMURE == cam_state_) {
        scm::math::mat4d tr = trackball_.transform();
        inverse_transform = scm::math::mat4f(scm::math::inverse(projection_matrix_ * tr * model));
    }
    else if (CAM_STATE_GUA == cam_state_) {
        inverse_transform = scm::math::mat4d(scm::math::inverse(projection_matrix_ * view_matrix_ * model));
    }
    tmp[0] = inverse_transform * scm::math::vec4d(-1, -1, -1, 1);
    tmp[1] = inverse_transform * scm::math::vec4d(-1, -1, 1, 1);
    tmp[2] = inverse_transform * scm::math::vec4d(-1, 1, -1, 1);
    tmp[3] = inverse_transform * scm::math::vec4d(-1, 1, 1, 1);
    tmp[4] = inverse_transform * scm::math::vec4d(1, -1, -1, 1);
    tmp[5] = inverse_transform * scm::math::vec4d(1, -1, 1, 1);
    tmp[6] = inverse_transform * scm::math::vec4d(1, 1, -1, 1);
    tmp[7] = inverse_transform * scm::math::vec4d(1, 1, 1, 1);

    for (int i(0); i < 8; ++i) {
        result[i] = tmp[i] / tmp[i][3];
    }
    return result;
}


std::vector<scm::math::vec3d> camera::get_frustum_corners() const {
    std::vector<scm::math::vec4d> tmp(8);
    std::vector<scm::math::vec3d> result(8);

    scm::math::mat4d inverse_transform;

    if(CAM_STATE_LAMURE == cam_state_) {
        scm::math::mat4d tr = trackball_.transform();
        inverse_transform = scm::math::mat4f(scm::math::inverse(projection_matrix_ * tr));
    }
    else if(CAM_STATE_GUA == cam_state_) {
        inverse_transform = scm::math::mat4d(scm::math::inverse(projection_matrix_ * view_matrix_));
    }
    tmp[0] = inverse_transform * scm::math::vec4d(-1, -1, -1, 1);
    tmp[1] = inverse_transform * scm::math::vec4d(-1, -1, 1, 1);
    tmp[2] = inverse_transform * scm::math::vec4d(-1, 1, -1, 1);
    tmp[3] = inverse_transform * scm::math::vec4d(-1, 1, 1, 1);
    tmp[4] = inverse_transform * scm::math::vec4d(1, -1, -1, 1);
    tmp[5] = inverse_transform * scm::math::vec4d(1, -1, 1, 1);
    tmp[6] = inverse_transform * scm::math::vec4d(1, 1, -1, 1);
    tmp[7] = inverse_transform * scm::math::vec4d(1, 1, 1, 1);

    for(int i(0); i < 8; ++i) {
        result[i] = tmp[i] / tmp[i][3];
    }
    return result;
}


void camera::update_camera(double x, double y, int window_width, int window_height, mouse_state const &mouse_state, bool keys[])
{
    double nx = 2.0 * double(x - (window_width / 2)) / double(window_width);
    double ny = 2.0 * double(window_height - y - (window_height / 2)) / double(window_height);
    double f = dolly_sens_ < 1.0 ? 0.02 : 0.3;
    // translations
    if(mouse_state.rb_down_ && keys[0])
    {
        translate(dolly_sens_ * 0.5 * (nx - trackball_init_x_), 0, 0);
    }
    if(mouse_state.rb_down_ && keys[1])
    {
        translate(0, dolly_sens_ * 0.5 * (ny - trackball_init_y_), 0);
    }
    if(mouse_state.rb_down_ && keys[2])
    {
        translate(0, 0, dolly_sens_ * 1 * (ny - trackball_init_y_));
    }
    // rotations
    if(mouse_state.lb_down_ && keys[0])
    {
        rotate(dolly_sens_ * 0.5 * (ny - trackball_init_y_), 0, 0);
    }
    if(mouse_state.lb_down_ && keys[1])
    {
        rotate(0, dolly_sens_ * 0.5 * (ny - trackball_init_y_), 0);
    }
    if(mouse_state.lb_down_ && keys[2])
    {
        rotate(0, 0, dolly_sens_ * 0.5 * (nx - trackball_init_x_));
    }
    trackball_init_y_ = ny;
    trackball_init_x_ = nx;
}


void camera::translate(const float& x, const float& y, const float& z) {
    if(cam_state_ == CAM_STATE_GUA) {
        view_matrix_ = scm::math::make_translation(x, y, z) * scm::math::mat4f(view_matrix_);
    }
    else if(cam_state_ == CAM_STATE_LAMURE) {
        scm::math::mat4d trackball_trans = trackball_.transform();
        trackball_.set_transform(scm::math::make_translation((double)x, (double)y, (double)z) * trackball_trans);
    }
}


void camera::rotate(const float& x, const float& y, const float& z) {
    if(cam_state_ == CAM_STATE_GUA) {
        view_matrix_ = scm::math::make_rotation(x, scm::math::vec3f(1.0f, 0.0f, 0.0f)) * 
                      scm::math::make_rotation(y, scm::math::vec3f(0.0f, 1.0f, 0.0f)) * 
                      scm::math::make_rotation(z, scm::math::vec3f(0.0f, 0.0f, 1.0f)) * 
                      scm::math::mat4f(view_matrix_);
    }
    else if(cam_state_ == CAM_STATE_LAMURE) {
        scm::math::mat4d trackball_trans = trackball_.transform();
        trackball_.set_transform(scm::math::make_rotation((double)x, scm::math::vec3d(1.0, 0.0, 0.0)) * 
                                scm::math::make_rotation((double)y, scm::math::vec3d(0.0, 1.0, 0.0)) * 
                                scm::math::make_rotation((double)z, scm::math::vec3d(0.0, 0.0, 1.0)) * 
                                trackball_trans);
    }
}


scm::math::mat4f const camera::get_view_matrix() const
{
    switch(cam_state_)
    {
    case CAM_STATE_LAMURE:
        return scm::math::mat4f(trackball_.transform());
        break;
    case CAM_STATE_GUA:
        return scm::math::mat4f(view_matrix_);
        break;
    default:
        break;
    }
    return scm::math::mat4f();
}


scm::math::mat4f camera::get_view_matrix()
{
    switch (cam_state_)
    {
    case CAM_STATE_LAMURE:
        return scm::math::mat4f(trackball_.transform());
        break;
    case CAM_STATE_GUA:
        return scm::math::mat4f(view_matrix_);
        break;
    default:
        break;
    }
    return scm::math::mat4f();
}


scm::math::vec3d camera::get_cam_pos() {
    switch (cam_state_)
    {
    case CAM_STATE_LAMURE: {
        scm::math::mat4f tm = scm::math::mat4f(trackball_.transform());
        return scm::math::vec3d(tm[12], tm[13], tm[14]);
        break;
    }
    case CAM_STATE_GUA: {
        scm::math::mat4d vm = scm::math::inverse(view_matrix_);
        return scm::math::vec3d(vm[12], vm[13], vm[14]);
        break;
    }
    }
}


void camera::set_cam_pos(scm::math::vec3d const& cam_pos) {
    switch (cam_state_)
    {
    case CAM_STATE_LAMURE: {
        scm::math::mat4f tm = scm::math::mat4f(trackball_.transform());
        tm[12] = static_cast<float>(cam_pos[0]);
        tm[13] = static_cast<float>(cam_pos[1]);
        tm[14] = static_cast<float>(cam_pos[2]);
        frustum_.update(scm::math::mat4f(projection_matrix_) * tm);
        break;
    }
    case CAM_STATE_GUA: {
        scm::math::mat4d vm = scm::math::inverse(view_matrix_);
        vm[12] = static_cast<float>(cam_pos[0]);
        vm[13] = static_cast<float>(cam_pos[1]);
        vm[14] = static_cast<float>(cam_pos[2]);
        view_matrix_ = scm::math::inverse(vm);
        frustum_.update(scm::math::mat4f(projection_matrix_ * view_matrix_));
        break;
    }
    default: {

        break;
    }
    }
}


scm::math::mat4f camera::get_cam_matrix() {
    scm::math::mat4f cm = scm::math::inverse(scm::math::mat4f(trackball_.transform()));
    return cm;
}


}
}
