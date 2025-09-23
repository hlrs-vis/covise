// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/ray.h>

namespace lamure
{
namespace ren
{
ray::ray() : origin_(scm::math::vec3f::zero()), direction_(scm::math::vec3f::one()), max_distance_(-1.f) {}

ray::ray(const scm::math::vec3f &origin, const scm::math::vec3f &direction, const float max_distance) : origin_(origin), direction_(direction), max_distance_(max_distance) {}

ray::~ray() {}

const bool ray::intersect(const float aabb_scale, scm::math::vec3f &ray_up_vector, const float bundle_radius, const unsigned int max_depth, const unsigned int surfel_skip,
                          ray::intersection &intersection)
{
    scm::math::vec3f up_vector = scm::math::normalize(ray_up_vector);
    scm::math::vec3f right_vector = scm::math::normalize(scm::math::cross(up_vector, direction_));

    std::srand(255);

    std::vector<ray> rays;

    rays.push_back(ray(origin_, direction_, max_distance_));
    rays.push_back(ray(origin_ + up_vector * bundle_radius, direction_, max_distance_));
    rays.push_back(ray(origin_ - up_vector * bundle_radius, direction_, max_distance_));
    rays.push_back(ray(origin_ - right_vector * bundle_radius, direction_, max_distance_));
    rays.push_back(ray(origin_ + right_vector * bundle_radius, direction_, max_distance_));

    unsigned int num_rays = 5;

    for(unsigned int i = 0; i < 8; ++i)
    {
        if(num_rays < 255)
        {
            float angle = 2 * 3.1415926535f * (std::rand() / (float)RAND_MAX);

            scm::math::mat2f rot;
            rot.m00 = std::cos(angle);
            rot.m01 = std::sin(angle);
            rot.m02 = -std::sin(angle);
            rot.m03 = rot.m00;

            float r = (std::rand() / (float)RAND_MAX) * bundle_radius;

            scm::math::vec2f p0 = rot * scm::math::vec2f(r, 0.f);
            scm::math::vec2f p1 = rot * scm::math::vec2f(-r, 0.f);
            scm::math::vec2f p2 = rot * scm::math::vec2f(0.f, r);
            scm::math::vec2f p3 = rot * scm::math::vec2f(0.f, -r);

            rays.push_back(ray(origin_ + right_vector * p0.x + up_vector * p0.y, direction_, max_distance_));
            rays.push_back(ray(origin_ + right_vector * p1.x + up_vector * p1.y, direction_, max_distance_));
            rays.push_back(ray(origin_ + right_vector * p2.x + up_vector * p2.y, direction_, max_distance_));
            rays.push_back(ray(origin_ + right_vector * p3.x + up_vector * p3.y, direction_, max_distance_));
            num_rays += 4;
        }
    }

    std::vector<ray::intersection> intersections;
    std::vector<float> best_errors;
    for(unsigned int i = 0; i < num_rays; ++i)
    {
        intersections.push_back(ray::intersection());
        best_errors.push_back(std::numeric_limits<float>::max());
    }

    model_database *database = model_database::get_instance();
    ooc_cache *ooc_cache = ooc_cache::get_instance();

    ooc_cache->lock();
    ooc_cache->refresh();

    ray_queue job_queue;
    for(unsigned int i = 0; i < num_rays; ++i)
    {
        job_queue.push_job(ray_queue::ray_job(i, rays[i]));
    }

    unsigned int num_threads = 16;

    std::vector<std::thread> threads;
    for(unsigned int i = 0; i < num_threads; ++i)
    {
        threads.push_back(std::thread([&] {

            while(true)
            {
                job_queue.wait();

                if(job_queue.is_shutdown())
                {
                    break;
                }

                ray_queue::ray_job job = job_queue.pop_job();

                if(job.id_ >= 0)
                {
                    ray &ray = rays[job.id_];
                    ray::intersection temp;
                    for(model_t model_id = 0; model_id < database->num_models(); ++model_id)
                    {
                        const scm::math::mat4f &model_transform = database->get_model(model_id)->transform();
                        if(ray.intersect_model_unsafe(model_id, model_transform, aabb_scale, max_depth, surfel_skip, false, temp))
                        {
                            if(temp.error_ < best_errors[job.id_])
                            {
                                best_errors[job.id_] = temp.error_;
                                intersections[job.id_] = temp;
                            }
                        }
                    }
                }
            }

        }));
    }

    for(auto &thread : threads)
    {
        thread.join();
    }

    unsigned int num_rays_hit = 0;
    for(const auto &error : best_errors)
    {
        if(error < std::numeric_limits<float>::max())
        {
            ++num_rays_hit;
        }
    }

    ooc_cache->unlock();

    if(num_rays_hit > num_rays / 4)
    {
        // fit the plane
        scm::math::vec3f plane_center = scm::math::vec3f::zero();
        float avg_distance = 0.f;
        for(unsigned int i = 0; i < num_rays; ++i)
        {
            if(best_errors[i] < std::numeric_limits<float>::max())
            {
                plane_center += intersections[i].position_;
                avg_distance += intersections[i].distance_;
            }
        }

        float denom = 1.f / (float)num_rays_hit;
        plane_center *= denom;
        avg_distance *= denom;

        scm::math::mat3f covariance_mat = scm::math::mat3f::zero();

        for(unsigned int i = 0; i < num_rays; ++i)
        {
            if(best_errors[i] < std::numeric_limits<float>::max())
            {
                scm::math::vec3f &c = intersections[i].position_;
                covariance_mat.m00 += std::pow(c.x - plane_center.x, 2);
                covariance_mat.m01 += (c.x - plane_center.x) * (c.y - plane_center.y);
                covariance_mat.m02 += (c.x - plane_center.x) * (c.z - plane_center.z);

                covariance_mat.m03 += (c.y - plane_center.y) * (c.x - plane_center.x);
                covariance_mat.m04 += std::pow(c.y - plane_center.y, 2);
                covariance_mat.m05 += (c.y - plane_center.y) * (c.z - plane_center.z);

                covariance_mat.m06 += (c.z - plane_center.z) * (c.x - plane_center.x);
                covariance_mat.m07 += (c.z - plane_center.z) * (c.y - plane_center.y);
                covariance_mat.m08 += std::pow(c.z - plane_center.z, 2);
            }
        }

        scm::math::mat3f inv_covariance_mat = scm::math::inverse(covariance_mat);
        scm::math::vec3f v = scm::math::vec3f(1.f, 1.f, 1.f);
        scm::math::vec3f plane_normal = scm::math::normalize(v * inv_covariance_mat);
        unsigned int iteration = 0;
        while(iteration++ < 255 && v != plane_normal)
        {
            v = plane_normal;
            plane_normal = scm::math::normalize(v * inv_covariance_mat);
        }

        if(scm::math::dot(plane_normal, direction_) > 0.f)
        {
            plane_normal *= -1.f;
        }

        intersection.normal_ = plane_normal;
        intersection.position_ = plane_center;
        intersection.distance_ = avg_distance;

        // construct hessian normal form
        float d = -scm::math::dot(plane_center, plane_normal);

        // obtain maximum absolute distance
        float max_plane_distance = 0.f;
        for(unsigned int i = 0; i < num_rays; ++i)
        {
            if(best_errors[i] < std::numeric_limits<float>::max())
            {
                scm::math::vec3f &c = intersections[i].position_;
                float plane_distance = scm::math::abs(plane_normal.x * c.x + plane_normal.y * c.y + plane_normal.z * c.z + d);
                max_plane_distance = std::max(max_plane_distance, plane_distance);
            }
        }

        // hack max plane distance into some result
        // intersection.distance_ = max_plane_distance;
        intersection.error_ = max_plane_distance;

        return true;
    }

    return false;
}

const bool ray::intersect_model(const model_t model_id, const scm::math::mat4f &model_transform, const float aabb_scale, const unsigned int max_depth, const unsigned int surfel_skip, bool is_wysiwyg,
                                ray::intersection &intersection)
{
    ooc_cache *ooc_cache = ooc_cache::get_instance();
    ooc_cache->lock();
    ooc_cache->refresh();

    bool result = intersect_model_unsafe(model_id, model_transform, aabb_scale, max_depth, surfel_skip, is_wysiwyg, intersection);

    ooc_cache->unlock();

    return result;
}

const bool ray::intersect_model_unsafe(const model_t model_id, const scm::math::mat4f &model_transform, const float aabb_scale, const unsigned int max_depth, const unsigned int surfel_skip,
                                       bool is_wysiwyg, ray::intersection &intersection)
{
    model_database *database = model_database::get_instance();
    if(model_id >= database->num_models())
    {
        return false;
    }

    ooc_cache *ooc_cache = ooc_cache::get_instance();

    const bvh *tree = database->get_model(model_id)->get_bvh();
    if(tree->get_primitive() != bvh::primitive_type::POINTCLOUD)
    {
        return false;
    }

    unsigned int fan_factor = tree->get_fan_factor();
    node_t num_nodes = tree->get_num_nodes();
    uint32_t num_surfels_per_node = database->get_primitives_per_node();

    scm::math::mat4f inverse_model_transform = scm::math::inverse(model_transform);
    scm::math::vec3f object_ray_origin = inverse_model_transform * origin_;
    scm::math::vec3f object_ray_aux = inverse_model_transform * (origin_ + direction_ * max_distance_);
    scm::math::vec3f object_ray_direction = object_ray_aux - object_ray_origin;
    float object_ray_max_distance = scm::math::length(object_ray_direction);
    object_ray_direction = scm::math::normalize(object_ray_direction);

    bool has_hit = false;
    std::stack<node_t> candidates;
    candidates.push(0);

    // check if model has started loading, otherwise we cant do nothin
    if(!ooc_cache->is_node_resident_and_aquired(model_id, 0))
    {
        return false;
    }

    unsigned int valid_max_depth = max_depth == 0 ? 255 : max_depth;
    unsigned int valid_surfel_skip = surfel_skip == 0 ? 1 : surfel_skip;

    float max_intersection_error = 6.f;

    while(!candidates.empty())
    {
        node_t current_parent_id = candidates.top();
        candidates.pop();

        bool no_child_available = true;

        for(node_t i = 0; i < (node_t)fan_factor; ++i)
        {
            node_t node_id = tree->get_child_id(current_parent_id, i);

            if(node_id == invalid_node_t)
            {
                continue;
            }

            if(node_id >= num_nodes)
            {
                continue;
            }

            if(!ooc_cache->is_node_resident_and_aquired(model_id, node_id))
            {
                continue;
            }

            no_child_available = false;

            scm::math::vec2f t = scm::math::vec2f::zero();
            if(!intersect_aabb(tree->get_bounding_boxes()[node_id], object_ray_origin, object_ray_direction, t))
            {
                continue;
            }

            // check if node too far away
            if(t.x > object_ray_max_distance)
            {
                continue;
            }

            bool all_children_in_memory = true;
            for(node_t k = 0; k < fan_factor; ++k)
            {
                node_t child_id = tree->get_child_id(node_id, k);
                if(child_id == invalid_node_t)
                {
                    all_children_in_memory = false;
                    break;
                }

                if(child_id >= num_nodes)
                {
                    all_children_in_memory = false;
                    break;
                }

                if(!ooc_cache->is_node_resident_and_aquired(model_id, child_id))
                {
                    all_children_in_memory = false;
                    break;
                }
            }

            bool intersect_splats = false;

            if(all_children_in_memory)
            {
                bool we_do_not_intersect_either_child = true;
                for(node_t k = 0; k < fan_factor; ++k)
                {
                    node_t child_id = tree->get_child_id(node_id, k);

                    if(child_id == invalid_node_t)
                    {
                        we_do_not_intersect_either_child = true;
                        break;
                    }

                    if(child_id >= num_nodes)
                    {
                        we_do_not_intersect_either_child = true;
                        break;
                    }

                    scm::math::vec2f t1 = scm::math::vec2f::zero();
                    if(intersect_aabb(tree->get_bounding_boxes()[child_id], object_ray_origin, object_ray_direction, t1))
                    {
                        we_do_not_intersect_either_child = false;
                        break;
                    }
                }

                if(we_do_not_intersect_either_child)
                {
                    intersect_splats = true;
                }
                else
                {
                    if(tree->get_depth_of_node(node_id) + 1 < valid_max_depth)
                    {
                        candidates.push(node_id);
                    }
                    else
                    {
                        intersect_splats = true;
                    }
                }
            }
            else
            {
                intersect_splats = true;
            }

            if(intersect_splats)
            {
                if(tree->get_visibility(node_id) == bvh::node_visibility::NODE_INVISIBLE)
                {
                    continue;
                }

                float object_to_world_scale = max_distance_ / object_ray_max_distance;

                dataset::serialized_surfel *surfels = (dataset::serialized_surfel *)ooc_cache->node_data(model_id, node_id);
                for(unsigned int k = 0; k < num_surfels_per_node; k += valid_surfel_skip)
                {
                    dataset::serialized_surfel &surfel = surfels[k];

                    if(surfel.size >= std::numeric_limits<float>::min())
                    {
                        float ts = -1.f;
                        if(intersect_surfel(surfel, object_ray_origin, object_ray_direction, ts))
                        {
                            if(ts != ts || ts <= 0.f)
                            {
                                continue;
                            }

                            scm::math::vec3f splat_plane_intersection = origin_ + direction_ * ts * object_to_world_scale;
                            scm::math::vec3f splat_position = model_transform * scm::math::vec3f(surfel.x, surfel.y, surfel.z);
                            float splat_plane_distance = scm::math::length(splat_position - splat_plane_intersection);

                            if(scm::math::length(splat_position - origin_) < max_distance_)
                            {
                                if(surfel.size <= std::numeric_limits<float>::min())
                                {
                                    continue;
                                }

                                if(is_wysiwyg)
                                {
                                    if(splat_plane_distance > object_to_world_scale * surfel.size * LAMURE_WYSIWYG_SPLAT_SCALE)
                                    {
                                        continue;
                                    }
                                }

                                float intersection_distance = scm::math::length(splat_plane_intersection - origin_);
                                float error = 0.01f * intersection_distance + splat_plane_distance;
                                // float error = splat_plane_distance;

                                if(error < intersection.error_ && error < max_intersection_error)
                                {
                                    intersection.error_ = error;
                                    intersection.error_raw_ = splat_plane_distance;

                                    has_hit = true;
                                    intersection.distance_ = intersection_distance;
                                    intersection.position_ = splat_plane_intersection;

                                    scm::math::mat4f normal_transform = scm::math::transpose(inverse_model_transform);
                                    scm::math::vec3f plane_normal = normal_transform * scm::math::vec3f(surfel.nx, surfel.ny, surfel.nz);
                                    intersection.normal_ = scm::math::normalize(plane_normal);
                                    if(scm::math::dot(intersection.normal_, direction_) > 0.f)
                                    {
                                        intersection.normal_ *= -1.f;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // fix: no node other than root in ram
        if(no_child_available && current_parent_id == 0 && !has_hit)
        {
            node_t node_id = current_parent_id;

            float object_to_world_scale = max_distance_ / object_ray_max_distance;

            dataset::serialized_surfel *surfels = (dataset::serialized_surfel *)ooc_cache->node_data(model_id, node_id);
            for(unsigned int k = 0; k < num_surfels_per_node; k += valid_surfel_skip)
            {
                dataset::serialized_surfel &surfel = surfels[k];

                if(surfel.size >= std::numeric_limits<float>::min())
                {
                    float ts = -1.f;
                    if(intersect_surfel(surfel, object_ray_origin, object_ray_direction, ts))
                    {
                        if(ts != ts || ts <= 0.f)
                        {
                            continue;
                        }

                        scm::math::vec3f splat_plane_intersection = origin_ + direction_ * ts * object_to_world_scale;
                        scm::math::vec3f splat_position = model_transform * scm::math::vec3f(surfel.x, surfel.y, surfel.z);
                        float splat_plane_distance = scm::math::length(splat_position - splat_plane_intersection);

                        if(scm::math::length(splat_position - origin_) < max_distance_)
                        {
                            if(surfel.size <= std::numeric_limits<float>::min())
                            {
                                continue;
                            }

                            if(is_wysiwyg)
                            {
                                if(splat_plane_distance > object_to_world_scale * surfel.size * LAMURE_WYSIWYG_SPLAT_SCALE)
                                {
                                    continue;
                                }
                            }

                            float intersection_distance = scm::math::length(splat_plane_intersection - origin_);
                            float error = 0.01f * intersection_distance + splat_plane_distance;
                            // float error = splat_plane_distance;

                            if(error < intersection.error_ && error < max_intersection_error)
                            {
                                intersection.error_ = error;
                                intersection.error_raw_ = splat_plane_distance;

                                has_hit = true;
                                intersection.distance_ = intersection_distance;
                                intersection.position_ = splat_plane_intersection;

                                scm::math::mat4f normal_transform = scm::math::transpose(inverse_model_transform);
                                scm::math::vec3f plane_normal = normal_transform * scm::math::vec3f(surfel.nx, surfel.ny, surfel.nz);
                                intersection.normal_ = scm::math::normalize(plane_normal);
                                if(scm::math::dot(intersection.normal_, direction_) > 0.f)
                                {
                                    intersection.normal_ *= -1.f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return has_hit;
}

const bool ray::intersect_bvh(const std::set<std::string> &model_filenames, const float aabb_scale, ray::intersection_bvh &intersection)
{
    model_database *database = model_database::get_instance();

    std::vector<ray::intersection_bvh> intersections;

    // iterate the models, intersect the onces that the user wants
    for(model_t model_id = 0; model_id < database->num_models(); ++model_id)
    {
        std::string bvh_filename = database->get_model(model_id)->get_bvh()->get_filename();
        ray::intersection_bvh temp;

        if(model_filenames.find(bvh_filename) != model_filenames.end())
        {
            const scm::math::mat4f &model_transform = database->get_model(model_id)->transform();

            ray ray(origin_, direction_, max_distance_);
            if(ray.intersect_model_bvh(model_id, model_transform, aabb_scale, temp))
            {
                intersections.push_back(temp);
            }
        }
    }

    // std::cout << "num intersections: " << intersections.size() << std::endl;

    if(intersections.empty())
    {
        return false;
    }
    if(intersections.size() == 1)
    {
        intersection = intersections[0];
        return true;
    }

    // now we have to run the disambiguation code
    // heres the rules:
    // if there is a intersection in the list such that its tmax is smaller than the
    // tmins of all others, then this is our picked model
    // if no such intersection exists, then, we search for the first out and then
    // return the smallest representative radius that has its in smaller than the first out.

    // obtain the smallest max
    float smallest_tmax = std::numeric_limits<float>::max();
    unsigned int id_of_smallest_tmax = 0;
    for(unsigned int i = 0; i < intersections.size(); ++i)
    {
        const auto &candidate = intersections[i];
        if(smallest_tmax > candidate.tmax_)
        {
            id_of_smallest_tmax = i;
            smallest_tmax = candidate.tmax_;
        }
    }

    // now test the smallest tmax against all other ins
    bool accept_smallest_tmax = true;
    for(unsigned int i = 0; i < intersections.size(); ++i)
    {
        const auto &candidate = intersections[i];
        if(i == id_of_smallest_tmax)
        {
            continue;
        }
        // if there exists one other intersection such that tmin < smallest_tmax,
        // then we cannot use this approach
        if(candidate.tmin_ < smallest_tmax)
        {
            accept_smallest_tmax = false;
            break;
        }
    }

    // if we accept the smallest tmax, we return it
    if(accept_smallest_tmax)
    {
        intersection = intersections[id_of_smallest_tmax];
        return true;
    }

    // else, we need to figure out a better candidate
    // figure out the smallest representative candidate that has in smaller than out
    unsigned int id_of_smallest_representative = 0;
    float smallest_representative = std::numeric_limits<float>::max();
    for(unsigned int i = 0; i < intersections.size(); ++i)
    {
        const auto &candidate = intersections[i];
        // only consider candidate if its tmin is smaller than the smallest tmax
        if(i != id_of_smallest_tmax && candidate.tmin_ < smallest_tmax)
        {
            if(smallest_representative > candidate.representative_radius_)
            {
                id_of_smallest_representative = i;
                smallest_representative = candidate.representative_radius_;
            }
        }
    }

    intersection = intersections[id_of_smallest_representative];
    return true;
}

const bool ray::intersect_model_bvh(const model_t model_id, const scm::math::mat4f &model_transform, const float aabb_scale, ray::intersection_bvh &intersection)
{
    model_database *database = model_database::get_instance();
    if(model_id >= database->num_models())
    {
        return false;
    }

    const bvh *tree = database->get_model(model_id)->get_bvh();
    if(tree->get_primitive() != bvh::primitive_type::POINTCLOUD)
    {
        return false;
    }

    unsigned int fan_factor = tree->get_fan_factor();
    node_t num_nodes = tree->get_num_nodes();

    scm::math::mat4f inverse_model_transform = scm::math::inverse(model_transform);
    scm::math::vec3f object_ray_origin = inverse_model_transform * origin_;
    scm::math::vec3f object_ray_aux = inverse_model_transform * (origin_ + direction_ * max_distance_);
    scm::math::vec3f object_ray_direction = object_ray_aux - object_ray_origin;
    float object_ray_max_distance = scm::math::length(object_ray_direction);
    object_ray_direction = scm::math::normalize(object_ray_direction);

    bool has_hit = false;
    std::stack<node_t> candidates;
    candidates.push(0);

    while(!candidates.empty())
    {
        node_t current_parent_id = candidates.top();
        candidates.pop();

        for(node_t i = 0; i < (node_t)fan_factor; ++i)
        {
            node_t node_id = tree->get_child_id(current_parent_id, i);

            if(node_id == invalid_node_t)
            {
                continue;
            }

            if(node_id >= num_nodes)
            {
                continue;
            }

            if(tree->get_visibility(node_id) == bvh::node_visibility::NODE_INVISIBLE)
            {
                if(tree->get_depth_of_node(node_id) == tree->get_depth())
                {
                    continue;
                }
            }

            auto bb = tree->get_bounding_boxes()[node_id];

            scm::math::vec2f t = scm::math::vec2f::zero();
            if(!intersect_aabb(tree->get_bounding_boxes()[node_id], object_ray_origin, object_ray_direction, t))
            {
                continue;
            }

            // check if node too far away
            if(t.x > object_ray_max_distance)
            {
                continue;
            }

            bool hit_now = false;
            if(tree->get_depth_of_node(node_id) == tree->get_depth())
            {
                hit_now = true;
            }
            else
            {
                candidates.push(node_id);
            }

            if(hit_now)
            {
                float object_to_world_scale = max_distance_ / object_ray_max_distance;
                float world_tmin = std::max(0.f, t.x) * object_to_world_scale;
                float world_tmax = std::min(t.y, object_ray_max_distance) * object_to_world_scale;

                if(world_tmin < intersection.tmin_)
                { // not good enough
                    // if (tree->get_average_surfel_radius(node_id) < intersection.representative_radius_) {
                    intersection.tmin_ = world_tmin;
                    intersection.tmax_ = world_tmax;
                    intersection.position_ = origin_ + direction_ * world_tmin;
                    intersection.representative_radius_ = tree->get_avg_primitive_extent(node_id);
                    intersection.bvh_filename_ = tree->get_filename();
                    has_hit = true;
                }
            }
        }
    }

    return has_hit;
}

const bool ray::intersect_aabb(const scm::gl::boxf &bb, const scm::math::vec3f &ray_origin, const scm::math::vec3f &ray_direction, scm::math::vec2f &t)
{
    scm::math::vec3f t1 = ((bb.min_vertex() - ray_origin) / ray_direction);
    scm::math::vec3f t2 = ((bb.max_vertex() - ray_origin) / ray_direction);

    scm::math::vec3f tmin1 = scm::math::vec3f(std::min(t1.x, t2.x), std::min(t1.y, t2.y), std::min(t1.z, t2.z));
    scm::math::vec3f tmax1 = scm::math::vec3f(std::max(t1.x, t2.x), std::max(t1.y, t2.y), std::max(t1.z, t2.z));

    float tmin = std::max(std::max(tmin1.x, tmin1.y), tmin1.z);
    float tmax = std::min(std::min(tmax1.x, tmax1.y), tmax1.z);

    if(tmax >= 0.f && tmax >= tmin)
    {
        t = scm::math::vec2f(tmin, tmax);
        return true;
    }

    return false;
}

const bool ray::intersect_surfel(const dataset::serialized_surfel &surfel, const scm::math::vec3f &ray_origin, const scm::math::vec3f &ray_direction, float &t)
{
    scm::math::vec3f plane_normal = scm::math::vec3f(surfel.nx, surfel.ny, surfel.nz);
    scm::math::vec3f plane_origin = scm::math::vec3f(surfel.x, surfel.y, surfel.z);

    if(scm::math::dot(plane_normal, ray_direction) < 0.f)
    {
        plane_normal *= -1.f;
    }

    float t0 = -1.f;
    float t1 = -1.f;

    // intersect plane
    float denom = scm::math::dot(plane_normal, ray_direction);
#if 1
    if(denom > std::numeric_limits<float>::min())
    {
        scm::math::vec3f pd = plane_origin - ray_origin;
        t0 = scm::math::dot(pd, plane_normal) / denom;
    }
#endif
#if 1
    denom = scm::math::dot(-plane_normal, ray_direction);
    if(denom > std::numeric_limits<float>::min())
    {
        scm::math::vec3f pd = plane_origin - ray_origin;
        t1 = scm::math::dot(pd, -plane_normal) / denom;
    }
#endif
    t = std::max(t0, t1);

    if(t >= 0.f)
    {
        return true;
    }

    return false;
}

ray_queue::ray_queue() : is_shutdown_(false)
{
    semaphore_.set_min_signal_count(1);
    semaphore_.set_max_signal_count(std::numeric_limits<float>::max());
}

ray_queue::~ray_queue() {}

void ray_queue::wait() { semaphore_.wait(); }

const bool ray_queue::is_shutdown() { return is_shutdown_; }

void ray_queue::relaunch() { is_shutdown_ = false; }

const unsigned int ray_queue::num_jobs() { return (unsigned int)queue_.size(); }

void ray_queue::push_job(const ray_queue::ray_job &job)
{
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(job);
    semaphore_.signal(1);
}

const ray_queue::ray_job ray_queue::pop_job()
{
    std::lock_guard<std::mutex> lock(mutex_);

    ray_job job;

    if(!queue_.empty())
    {
        job = queue_.front();
        queue_.pop();
    }

    if(queue_.empty())
    {
        is_shutdown_ = true;
        semaphore_.shutdown();
    }

    return job;
}
}
}
