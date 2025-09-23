// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_REN_TRACKBALL_H_INCLUDED
#define LAMURE_REN_TRACKBALL_H_INCLUDED

#include <scm/core/math.h>
#include <scm/gl_core/math/math.h>

namespace lamure {
namespace ren {

class trackball
{
public:
    trackball();
    ~trackball();

    void rotate(double fx, double fy, double tx, double ty);
    void translate(double x, double y);
    void dolly(double y);

    const scm::math::mat4d& transform() const { return transform_; };
    void set_transform(const scm::math::mat4d& transform) { transform_ = transform; };
    double dolly() const { return dolly_; };
    void set_dolly(const double dolly) { dolly_ = dolly; };

private:
    double project_to_sphere(double x, double y) const;

    scm::math::mat4d transform_;
    double radius_;
    double dolly_;

};

}
}

#endif
