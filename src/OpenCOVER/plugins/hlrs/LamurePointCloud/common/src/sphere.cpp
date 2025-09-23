// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/sphere.h>

#include <lamure/bounding_box.h>

namespace lamure
{

const bounding_box sphere::
get_bounding_box() const
{
    return bounding_box(
        vec3r(center_.x - radius_, center_.y - radius_, center_.z - radius_),
        vec3r(center_.x + radius_, center_.y + radius_, center_.z + radius_)
    );
}

bool sphere::
contains(const vec3r& point) const
{
    const real distance_to_center = scm::math::length_sqr(point - center_);
    return distance_to_center <= sqrt(radius_);
}

real sphere::
clamp_to_AABB_face(real actual_value, real min_BB_value, real max_BB_value) const {
    return std::max(std::min(actual_value, max_BB_value), min_BB_value);
}

vec3r sphere::
get_closest_point_on_AABB(const bounding_box& bounding_box) const {
    const vec3r min = bounding_box.min();
    const vec3r max = bounding_box.max();

    vec3r closest_point_on_AABB(0.0, 0.0, 0.0);

    for(int dim_idx = 0; dim_idx < 3; ++dim_idx) {
        closest_point_on_AABB[dim_idx] 
            = clamp_to_AABB_face(center_[dim_idx], min[dim_idx], max[dim_idx]);
    }

    return closest_point_on_AABB;
}

bool sphere::
intersects_or_contains(const bounding_box& bounding_box) const
{
    vec3r closest_point_on_AABB = get_closest_point_on_AABB(bounding_box);

    return ( scm::math::length_sqr( center_ - closest_point_on_AABB ) <= radius_*radius_);
}

} // namespace lamure
