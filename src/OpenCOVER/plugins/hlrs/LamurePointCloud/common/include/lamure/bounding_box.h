// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef COMMON_BOUNDING_BOX_H_
#define COMMON_BOUNDING_BOX_H_

#include <lamure/platform.h>
#include <lamure/types.h>
#include <lamure/sphere.h>

namespace lamure
{

class bounding_box
{
public:

    explicit            bounding_box() : min_(vec3r(1.0)), max_(vec3r(-1.0)) {}

    explicit            bounding_box(const vec3r& min,
                                    const vec3r& max)
                            : min_(min), max_(max) {

                                assert((min_[0] <= max_[0]) && 
                                       (min_[1] <= max_[1]) && 
                                       (min_[2] <= max_[2]));
                            }

    const vec3r         min() const { return min_; }
    vec3r&              min() { return min_; }

    const vec3r         max() const { return max_; }
    vec3r&              max() { return max_; }

    const bool          is_invalid() const { 
                            return min_.x > max_.x || 
                                   min_.y > max_.y || 
                                   min_.z > max_.z; 
                        }

    const bool          is_valid() const { return !is_invalid(); }

    const vec3r         get_dimensions() const { 
                            assert(is_valid());
                            return max_ - min_;
                        }

    const vec3r         get_center() const { 
                            assert(is_valid());
                            return (max_ + min_) / 2.0;
                        }

    const uint8_t       get_longest_axis() const;
    const uint8_t       get_shortest_axis() const;

    const bool          contains(const vec3r& point) const {
                            assert(is_valid());
                            return min_.x <= point.x && point.x <= max_.x &&
                                   min_.y <= point.y && point.y <= max_.y &&
                                   min_.z <= point.z && point.z <= max_.z;
                        }

    const bool          contains(const bounding_box& bounding_box) const {
                            assert(is_valid());
                            assert(bounding_box.is_valid());
                            return contains(bounding_box.min()) &&
                                   contains(bounding_box.max());
                        }

    const bool          contains(const sphere& sphere) const {
                            assert(is_valid());
                            return contains(sphere.get_bounding_box());
                        }

    const bool          intersects(const bounding_box& bounding_box) const {
                            assert(is_valid());
                            assert(bounding_box.is_valid());
                            return !(max_.x < bounding_box.min().x || 
                                     max_.y < bounding_box.min().y || 
                                     max_.z < bounding_box.min().z || 
                                     min_.x > bounding_box.max().x || 
                                     min_.y > bounding_box.max().y ||
                                     min_.z > bounding_box.max().z);
                        }

    void                expand(const vec3r& point);

    void                expand(const vec3r& point, const real radius);

    void                expand(const bounding_box& bounding_box);

    void                expand_by_disk(const vec3r& surfel_center, const scm::math::vec3f& surfel_normal, const real surfel_radius);

    void                shrink(const bounding_box& bounding_box);

    inline bool         operator==(const bounding_box& rhs) const
                            { return min_ == rhs.min_ && max_ == rhs.max_; }
    inline bool         operator!=(const bounding_box& rhs) const
                            { return !(operator==(rhs)); }

private:

    vec3r               min_;
    vec3r               max_;

};

}

#endif // COMMON_BOUNDING_BOX_H_

