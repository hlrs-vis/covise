// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_RAY_H_
#define REN_RAY_H_

#include <bitset>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>

#include <lamure/ren/bvh.h>
#include <lamure/ren/dataset.h>
#include <lamure/ren/model_database.h>
#include <lamure/ren/ooc_cache.h>

#include <lamure/ren/platform.h>
#include <lamure/types.h>

#include <lamure/ren/dataset.h>
#include <lamure/semaphore.h>

#include <scm/core/math.h>
#include <scm/gl_core/primitives/box.h>

namespace lamure
{
namespace ren
{
class ray
{
  public:
    // this is for splat-based picking
    struct intersection
    {
        scm::math::vec3f position_;
        scm::math::vec3f normal_;
        float distance_;
        float error_;
        float error_raw_;

        intersection()
            : position_(scm::math::vec3::zero()), normal_(scm::math::vec3f::one()), distance_(0.f), error_(std::numeric_limits<float>::max()){

                                                                                                    };

        intersection(const scm::math::vec3f &position, const scm::math::vec3f &normal)
            : position_(position), normal_(normal), distance_(0.f), error_(std::numeric_limits<float>::max()){

                                                                    };
    };

    // this is for bvh-based picking
    struct intersection_bvh
    {
        scm::math::vec3f position_;
        float tmin_;
        float tmax_;
        float representative_radius_;
        std::string bvh_filename_;

        intersection_bvh()
            : position_(scm::math::vec3::zero()), tmin_(std::numeric_limits<float>::max()), tmax_(std::numeric_limits<float>::lowest()), representative_radius_(std::numeric_limits<float>::max()),
              bvh_filename_(""){

              };
    };

    ray();
    ray(const scm::math::vec3f &origin, const scm::math::vec3f &direction, const float max_distance);
    ~ray();

    const scm::math::vec3f &origin() const { return origin_; };
    const scm::math::vec3f &direction() const { return direction_; };
    const float max_distance() const { return max_distance_; };

    void set_origin(const scm::math::vec3f &origin) { origin_ = origin; };
    void set_direction(const scm::math::vec3f &direction) { direction_ = direction; };
    void set_max_distance(const float max_distance) { max_distance_ = max_distance; };

    // this is a interpolation picking interface,
    //(all models, splat-based, fits a plane)
    const bool intersect(const float aabb_scale, scm::math::vec3f &ray_up_vector, const float cone_diameter, const unsigned int max_depth, const unsigned int surfel_skip, intersection &intersect);

    // this is a BVH-only picking interface,
    //(all models, BVH-based, disambiguation)
    const bool intersect_bvh(const std::set<std::string> &model_filenames, const float aabb_scale, intersection_bvh &intersection);

    // this is a splat-based pick of a single model,
    //(single model, splat-based)
    const bool intersect_model(const model_t model_id, const scm::math::mat4f &model_transform, const float aabb_scale, const unsigned int max_depth, const unsigned int surfel_skip,
                               const bool is_wysiwyg, intersection &intersect);

    // this is a BVH-based pick of a single model,
    //(single model, BVH-based)
    const bool intersect_model_bvh(const model_t model_id, const scm::math::mat4f &model_transform, const float aabb_scale, intersection_bvh &intersection);

  protected:
    const bool intersect_model_unsafe(const model_t model_id, const scm::math::mat4f &model_transform, const float aabb_scale, const unsigned int max_depth, const unsigned int surfel_skip,
                                      const bool is_wysiwyg, intersection &intersection);
    static const bool intersect_aabb(const scm::gl::boxf &bb, const scm::math::vec3f &ray_origin, const scm::math::vec3f &ray_direction, scm::math::vec2f &t);
    static const bool intersect_surfel(const dataset::serialized_surfel &surfel, const scm::math::vec3f &ray_origin, const scm::math::vec3f &ray_direction, float &t);

  private:
    scm::math::vec3f origin_;
    scm::math::vec3f direction_;
    float max_distance_;
};

class ray_queue
{
  public:
    struct ray_job
    {
        ray ray_;
        int id_;

        ray_job() : ray_(scm::math::vec3f::zero(), scm::math::vec3f::zero(), -1.f), id_(-1) {}

        ray_job(unsigned int id, const ray &ray) : ray_(ray), id_(id) {}
    };

    ray_queue();
    ~ray_queue();

    void push_job(const ray_job &job);
    const ray_job pop_job();

    void wait();
    void relaunch();
    const bool is_shutdown();
    const unsigned int num_jobs();

  private:
    std::queue<ray_job> queue_;
    std::mutex mutex_;
    semaphore semaphore_;
    bool is_shutdown_;
};
}
}

#endif
