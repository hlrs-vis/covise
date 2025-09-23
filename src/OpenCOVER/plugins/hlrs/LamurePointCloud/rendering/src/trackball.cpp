// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/trackball.h>


namespace lamure {
namespace ren {

trackball::trackball()
 : transform_(scm::math::mat4d::identity()),
   radius_(1.0),
   dolly_(0.0) {

}

trackball::~trackball() {

}

double trackball::
project_to_sphere(double x, double y) const {
    double len_sqr = x*x + y*y;
    double len = scm::math::sqrt(len_sqr);

    if (len < radius_ / scm::math::sqrt(2.0)) {
        return (scm::math::sqrt(radius_ * radius_ - len_sqr));
    } else {
        return ((radius_ * radius_) / (2.0 * len));
    }
}

void trackball::
rotate(double fx, double fy, double tx, double ty) {
    scm::math::vec3d start(fx, fy, project_to_sphere(fx, fy));
    scm::math::vec3d end(tx, ty, project_to_sphere(tx, ty));

    scm::math::vec3d diff(end - start);
    double diff_len = scm::math::length(diff);

    scm::math::vec3d rot_axis(cross(start, end));

    double rot_angle = 2.0 * asin(scm::math::clamp(diff_len/(2.0 * radius_), -1.0, 1.0));

    scm::math::mat4d tmp(scm::math::mat4d::identity());

    scm::math::mat4d tmp_dolly(scm::math::mat4d::identity());
    scm::math::mat4d tmp_dolly_inv(scm::math::mat4d::identity());
    scm::math::translate(tmp_dolly, 0.0, 0.0, dolly_);
    scm::math::translate(tmp_dolly_inv, 0.0, 0.0, -dolly_);

    scm::math::rotate(tmp, scm::math::rad2deg(rot_angle), rot_axis);

    transform_ = tmp_dolly * tmp * tmp_dolly_inv * transform_;
}


void trackball::
translate(double x, double y) {
    double dolly_abs = abs(dolly_);
    double near_dist = 1.0;

    scm::math::mat4d tmp(scm::math::mat4d::identity());

    scm::math::translate(tmp,
              x * (near_dist + dolly_abs),
              y * (near_dist + dolly_abs),
              0.0);

    scm::math::mat4d tmp_dolly(scm::math::mat4d::identity());
    scm::math::mat4d tmp_dolly_inv(scm::math::mat4d::identity());
    scm::math::translate(tmp_dolly, 0.0, 0.0, dolly_);
    scm::math::translate(tmp_dolly_inv, 0.0, 0.0, -dolly_);

    transform_ = tmp_dolly * tmp * tmp_dolly_inv * transform_;
}


void trackball::
dolly(double y) {
    scm::math::mat4d tmp_dolly(scm::math::mat4d::identity());
    scm::math::mat4d tmp_dolly_inv(scm::math::mat4d::identity());
    scm::math::translate(tmp_dolly, 0.0, 0.0, dolly_);
    scm::math::translate(tmp_dolly_inv, 0.0, 0.0, -dolly_);

    dolly_ -= y;
    scm::math::mat4d tmp(scm::math::mat4d::identity());
    scm::math::translate(tmp, 0.0, 0.0, dolly_);

    transform_ = tmp * tmp_dolly_inv * transform_;
}



}
}
