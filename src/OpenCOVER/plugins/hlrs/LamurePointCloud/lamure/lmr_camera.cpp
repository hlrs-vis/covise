// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/lmr_camera.h>


static lmr_camera* lmr_camera_ = NULL;
std::mutex lmr_camera::transform_update_mutex_;

lmr_camera::~lmr_camera() {
    return;
}


lmr_camera::lmr_camera( lamure::view_t view_id,
                        scm::math::mat4d view_matrix = scm::math::mat4d::identity(),
                        scm::math::mat4d projection_matrix = scm::math::mat4d::identity(),
                        scm::gl::frustum frustum = scm::math::mat4::identity(),
                        float near_plane_value = 0.001,
                        float far_plane_value = 1000,
                        control_type controlType = control_type::mouse,
                        double sum_trans_x = 0.0,
                        double sum_trans_y = 0.0,
                        double sum_trans_z = 0.0,
                        double sum_rot_x = 0.0,
                        double sum_rot_y = 0.0,
                        double sum_rot_z = 0.0,
                        camera_state cam_state = camera_state::CAM_STATE_GUA) :  
    view_id_(view_id),
    view_matrix_(view_matrix),
    projection_matrix_(projection_matrix),
    frustum_(frustum),
    near_plane_value_(near_plane_value),
    far_plane_value_(far_plane_value),
    controlType_(controlType),
    sum_trans_x_(sum_trans_x),
    sum_trans_y_(sum_trans_y),
    sum_trans_z_(sum_trans_z),
    sum_rot_x_(sum_rot_x),
    sum_rot_y_(sum_rot_y),
    sum_rot_z_(sum_rot_z),
    cam_state_(cam_state) {
};


lmr_camera* lmr_camera::instance(lamure::view_t view_id) {
    if (!lmr_camera_ || lmr_camera_->view_id() != view_id) {
        std::cout << "New lmr_camera with view_id " << view_id << " was created." << std::endl;
        return new lmr_camera(view_id);
    }
    else if (lmr_camera_->view_id() == view_id) {
        return lmr_camera_;
    }
    else {
        std::cout << "error in creating lmr_camera.";
        return NULL;
    }
}


void lmr_camera::set_hp_view_matrix(scm::math::mat4d m) {
    view_matrix_ = m;
};

void lmr_camera::set_hp_projection_matrix(scm::math::mat4d m) {
    projection_matrix_ = m;
};


scm::math::mat4 lmr_camera::get_view_matrix() {
    return scm::math::mat4(view_matrix_);
}

scm::math::mat4 lmr_camera::get_projection_matrix() {
    return scm::math::mat4(projection_matrix_);
}

scm::math::mat4d lmr_camera::get_hp_view_matrix() {
    return scm::math::mat4d(view_matrix_);
}

scm::math::mat4d lmr_camera::get_hp_projection_matrix() {
    return scm::math::mat4d(projection_matrix_);
}

scm::math::mat4 lmr_camera::calc_get_projection_matrix(float opening_angle, float aspect_ratio, float nearplane, float farplane) {
    scm::math::mat4 temp_matrix = scm::math::mat4::identity();
    scm::math::perspective_matrix(temp_matrix, opening_angle, aspect_ratio, nearplane, farplane);
    return temp_matrix;
}


void lmr_camera::set_projection_matrix(float opening_angle, float aspect_ratio, float nearplane, float farplane) {
    scm::math::mat4f temp_matrix;
    for(int i=0;i<16;i++)
        temp_matrix[i] = (float)projection_matrix_[i];
    scm::math::perspective_matrix(temp_matrix, opening_angle, aspect_ratio, nearplane, farplane);

    near_plane_value_ = nearplane;
    far_plane_value_ = farplane;
    frustum_ = scm::gl::frustum(scm::math::mat4f(this->projection_matrix_ * view_matrix_));
}

void lmr_camera::set_lookat_matrix(scm::math::mat4d tb_matrix) {
    view_matrix_ = tb_matrix;
}


void lmr_camera::calc_set_frustum() {
    frustum_ = scm::gl::frustum(scm::math::mat4f(projection_matrix_ * view_matrix_));
}


scm::gl::frustum lmr_camera::calc_get_frustum() {
    return scm::gl::frustum(scm::math::mat4f(projection_matrix_ * view_matrix_));
}


void lmr_camera::set_frustum(scm::gl::frustum frustum) {
    frustum_ = frustum;
}


scm::gl::frustum::classification_result const lmr_camera::cull_against_frustum(scm::gl::frustum const& frustum, scm::gl::box const& b) const { 
    return frustum.classify(b); 
}


scm::gl::frustum const lmr_camera::get_frustum_by_model(scm::math::mat4 const& model) const {
    return scm::gl::frustum(scm::math::mat4(projection_matrix_ * view_matrix_) * model);
}

float const lmr_camera::transfer_values(float currentValue, float maxValue) const { 
    return std::pow((std::abs(currentValue) / std::abs(maxValue)), 4); 
}

float const lmr_camera::remap_value(float value, float oldMin, float oldMax, float newMin, float newMax) const {
    float intermediateValue = (((value - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin;
    return transfer_values(intermediateValue, newMax) * intermediateValue;
}

std::vector<scm::math::vec3d> lmr_camera::get_frustum_corners() const {
    std::vector<scm::math::vec4d> tmp(8);
    std::vector<scm::math::vec3d> result(8);

    scm::math::mat4d inverse_transform;
    inverse_transform = scm::math::mat4d(scm::math::inverse(projection_matrix_ * view_matrix_));
    
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


void lmr_camera::
translate(const float& x, const float& y, const float& z) {
    view_matrix_ = scm::math::make_translation(x, y, z) * scm::math::mat4f(view_matrix_);
}


void lmr_camera::
rotate(const float& x, const float& y, const float& z) {
    view_matrix_ = scm::math::make_rotation(x, scm::math::vec3f(1.0f, 0.0f, 0.0f)) *
        scm::math::make_rotation(y, scm::math::vec3f(0.0f, 1.0f, 0.0f)) *
        scm::math::make_rotation(z, scm::math::vec3f(0.0f, 0.0f, 1.0f)) *
        scm::math::mat4(view_matrix_);
}


scm::math::mat4 lmr_camera::get_cam_matrix() {
    scm::math::mat4 cm = scm::math::inverse(scm::math::mat4(view_matrix_));
    return cm;
}
