// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/bounding_box.h>

#include <algorithm>

namespace lamure
{

const uint8_t bounding_box::
get_longest_axis() const
{
    const vec3r d = get_dimensions();
    return d.x > d.y ?
        (d.x > d.z ? 0 : (d.y > d.z ? 1 : 2)) :
        (d.y > d.z ? 1 : 2);
}

const uint8_t bounding_box::
get_shortest_axis() const
{
    const vec3r d = get_dimensions();
    return d.x < d.y ?
        (d.x < d.z ? 0 : (d.y < d.z ? 1 : 2)) :
        (d.y < d.z ? 1 : 2);
}

void bounding_box::
expand(const vec3r& point)
{
    if (is_valid()) {
        min_.x = std::min(min_.x, point.x);
        min_.y = std::min(min_.y, point.y);
        min_.z = std::min(min_.z, point.z);

        max_.x = std::max(max_.x, point.x);
        max_.y = std::max(max_.y, point.y);
        max_.z = std::max(max_.z, point.z);
    }
    else {
        min_ = max_ = point;
    }
}

void bounding_box::
expand(const vec3r& point, const real radius)
{
    if (is_valid()) {
      min_.x = std::min(min_.x, point.x - radius);
      min_.y = std::min(min_.y, point.y - radius);
      min_.z = std::min(min_.z, point.z - radius);
                                
      max_.x = std::max(max_.x, point.x + radius);
      max_.y = std::max(max_.y, point.y + radius);
      max_.z = std::max(max_.z, point.z + radius);
    }
    else {
        min_ = point - radius;
        max_ = point + radius;
    }
}

void bounding_box::
expand(const bounding_box& bounding_box)
{
    if (bounding_box.is_valid()) {
        if (is_valid()) {
            min_.x = std::min(min_.x, bounding_box.min().x);
            min_.y = std::min(min_.y, bounding_box.min().y);
            min_.z = std::min(min_.z, bounding_box.min().z);

            max_.x = std::max(max_.x, bounding_box.max().x);
            max_.y = std::max(max_.y, bounding_box.max().y);
            max_.z = std::max(max_.z, bounding_box.max().z);
        }
        else {
            *this = bounding_box;
        }
    }
}

void bounding_box::
expand_by_disk(const vec3r& surfel_center, const scm::math::vec3f& surfel_normal, const real surfel_radius) {

    const uint8_t X_AXIS = 0;
    const uint8_t Y_AXIS = 1;
    const uint8_t Z_AXIS = 2;

    auto calculate_half_offset = [](const vec3r& surf_center,
                                    const vec3r surf_normal,
                                    const real surf_radius, uint8_t axis) {
        vec3r point_along_axis(surf_center);

        point_along_axis[axis] += surf_radius;

        vec3r projected_point = point_along_axis - scm::math::dot(point_along_axis - surf_center, surf_normal) * surf_normal;

        return std::fabs(surf_center[axis] - projected_point[axis]);
    };

    vec3r normal = vec3r(surfel_normal);

    real half_offset_x = calculate_half_offset(surfel_center, normal, surfel_radius, X_AXIS);
    real half_offset_y = calculate_half_offset(surfel_center, normal, surfel_radius, Y_AXIS);
    real half_offset_z = calculate_half_offset(surfel_center, normal, surfel_radius, Z_AXIS);

    if (is_valid()) {
      min_.x = std::min(min_.x, surfel_center.x - half_offset_x);
      min_.y = std::min(min_.y, surfel_center.y - half_offset_y);
      min_.z = std::min(min_.z, surfel_center.z - half_offset_z);

      max_.x = std::max(max_.x, surfel_center.x + half_offset_x);
      max_.y = std::max(max_.y, surfel_center.y + half_offset_y);
      max_.z = std::max(max_.z, surfel_center.z + half_offset_z);
    }
    else {
      min_.x = surfel_center.x - half_offset_x;
      min_.y = surfel_center.y - half_offset_y;
      min_.z = surfel_center.z - half_offset_z;

      max_.x = surfel_center.x + half_offset_x;
      max_.y = surfel_center.y + half_offset_y;
      max_.z = surfel_center.z + half_offset_z;
    }
}

void bounding_box::
shrink(const bounding_box& bounding_box)
{
    assert(contains(bounding_box));


}

} // namespace lamure
