// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef COMMON_sphere_H_
#define COMMON_sphere_H_

#include <lamure/platform.h>
#include <lamure/types.h>

namespace lamure
{

class bounding_box;

class sphere
{
public:
                        sphere(const vec3r center = vec3r(0.0),
                               const real radius = 0.0)
                               : center_(center), radius_(radius)
                            {

                            }

    virtual             ~sphere() {}

    inline const vec3r  center() const { return center_; }
    inline const real  radius() const { return radius_; }

    inline bool         operator==(const sphere& rhs) const
                            { return center_ == rhs.center_ && radius_ == rhs.radius_; }
    inline bool         operator!=(const sphere& rhs) const
                            { return !(operator==(rhs)); }

    const bounding_box   get_bounding_box() const;

    bool		        contains(const vec3r& point) const;

    real                clamp_to_AABB_face(real actual_value, 
                                           real min_BB_value, real max_BB_value) const;

    vec3r               get_closest_point_on_AABB(const bounding_box& bounding_box) const;

    bool		        intersects_or_contains(const bounding_box& bounding_box) const;

private:

    vec3r               center_;
    real                radius_;

};

} // namespace lamure

#endif // COMMON_sphere_H_
