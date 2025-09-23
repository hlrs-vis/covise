// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/management_id_histogram_renderer_corners.h"

#include <set>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <lamure/ren/bvh.h>
#include <sstream>
#include <chrono>
#include <thread>

#include "lamure/pvs/pvs_database.h"

#include <iostream>

namespace lamure
{
namespace pvs
{

management_id_histogram_renderer_corners::
management_id_histogram_renderer_corners(std::vector<std::string> const& model_filenames,
    std::vector<scm::math::mat4f> const& model_transformations,
    std::set<lamure::model_t> const& visible_set,
    std::set<lamure::model_t> const& invisible_set) : management_base(model_filenames, model_transformations, visible_set, invisible_set)
{
    first_phase_ = true;
    smallest_cell_size_ = scm::math::vec3d(1.0, 1.0, 1.0);

    current_corner_index_x_ = 0;
    current_corner_index_y_ = 0;
    current_corner_index_z_ = 0;
}

management_id_histogram_renderer_corners::
~management_id_histogram_renderer_corners()
{
}

bool management_id_histogram_renderer_corners::
MainLoop()
{
    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    lamure::ren::controller* controller = lamure::ren::controller::get_instance();
    lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();

    bool signal_shutdown = false;

    controller->reset_system();

    lamure::context_t context_id = controller->deduce_context_id(0);
    lamure::view_t view_id = controller->deduce_view_id(context_id, active_camera_->view_id());
    
#ifndef LAMURE_PVS_USE_AS_RENDERER
    scm::math::vec3d look_dir;
    scm::math::vec3d up_dir(0.0, 1.0, 0.0);
    
    float opening_angle = 90.0f;        // TODO: these two should also be computed per cell (these constants only work in the regular box case)
    float aspect_ratio = 1.0f;
    float near_plane = 0.01f;

    // Only used in second phase.
    scm::math::vec3d current_corner_pos;

    if(first_phase_)
    {
        // First phase: render at centers of view cells.
        const view_cell* current_cell = visibility_grid_->get_cell_at_index(current_grid_index_);

        switch(direction_counter_)
        {
            case 0:
                look_dir = scm::math::vec3d(1.0, 0.0, 0.0);

                near_plane = current_cell->get_size().x * 0.5f;
                break;

            case 1:
                look_dir = scm::math::vec3d(-1.0, 0.0, 0.0);

                near_plane = current_cell->get_size().x * 0.5f;
                break;

            case 2:
                look_dir = scm::math::vec3d(0.0, 1.0, 0.0);
                up_dir = scm::math::vec3d(0.0, 0.0, 1.0);

                near_plane = current_cell->get_size().y * 0.5f;
                break;

            case 3:
                look_dir = scm::math::vec3d(0.0, -1.0, 0.0);
                up_dir = scm::math::vec3d(0.0, 0.0, 1.0);

                near_plane = current_cell->get_size().y * 0.5f;
                break;

            case 4:
                look_dir = scm::math::vec3d(0.0, 0.0, 1.0);
                
                near_plane = current_cell->get_size().z * 0.5f;
                break;

            case 5:
                look_dir = scm::math::vec3d(0.0, 0.0, -1.0);

                near_plane = current_cell->get_size().z * 0.5f;
                break;
                
            default:
                break;
        }

        active_camera_->set_projection_matrix(opening_angle, aspect_ratio, near_plane, far_plane_);
        active_camera_->set_view_matrix(scm::math::make_look_at_matrix(current_cell->get_position_center(), current_cell->get_position_center() + look_dir, up_dir));  // look_at(eye, center, up)
    }
    else
    {
        // Second phase: go along view cell corners.
        scm::math::vec3d start_corner = visibility_grid_->get_position_center() - visibility_grid_->get_size() * 0.5;
        scm::math::vec3d current_corner_offset(smallest_cell_size_.x * (double)current_corner_index_x_, smallest_cell_size_.y * (double)current_corner_index_y_, smallest_cell_size_.z * (double)current_corner_index_z_);

        current_corner_pos = start_corner + current_corner_offset;

        switch(direction_counter_)
        {
            case 0:
                look_dir = scm::math::vec3d(1.0, 0.0, 0.0);
                near_plane = smallest_cell_size_.x * 0.1f;
                break;

            case 1:
                look_dir = scm::math::vec3d(-1.0, 0.0, 0.0);
                near_plane = smallest_cell_size_.x * 0.1f;
                break;

            case 2:
                look_dir = scm::math::vec3d(0.0, 1.0, 0.0);
                up_dir = scm::math::vec3d(0.0, 0.0, 1.0);
                near_plane = smallest_cell_size_.y * 0.1f;
                break;

            case 3:
                look_dir = scm::math::vec3d(0.0, -1.0, 0.0);
                up_dir = scm::math::vec3d(0.0, 0.0, 1.0);
                near_plane = smallest_cell_size_.y * 0.1f;
                break;

            case 4:
                look_dir = scm::math::vec3d(0.0, 0.0, 1.0);
                near_plane = smallest_cell_size_.z * 0.1f;
                break;

            case 5:
                look_dir = scm::math::vec3d(0.0, 0.0, -1.0);
                near_plane = smallest_cell_size_.z * 0.1f;
                break;
                
            default:
                break;
        }

        active_camera_->set_projection_matrix(opening_angle, aspect_ratio, near_plane, far_plane_);
        active_camera_->set_view_matrix(scm::math::make_look_at_matrix(current_corner_pos, current_corner_pos + look_dir, up_dir));  // look_at(eye, center, up)
    }

#ifdef LAMURE_PVS_MEASURE_PERFORMANCE
    // Performance measurement of cut update.
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    start_time = std::chrono::system_clock::now();
#endif

    if(first_frame_)
    {
        smallest_cell_size_ = visibility_grid_->get_cell_at_index(0)->get_size();

        // Find the smallest cell size in each dimension in the grid.
        for(size_t cell_index = 1; cell_index < visibility_grid_->get_cell_count(); ++cell_index)
        {
            scm::math::vec3d current_cell_size = visibility_grid_->get_cell_at_index(cell_index)->get_size();
            
            if(current_cell_size.x < smallest_cell_size_.x)
            {
                smallest_cell_size_.x = current_cell_size.x;
            }
            if(current_cell_size.y < smallest_cell_size_.y)
            {
                smallest_cell_size_.y = current_cell_size.y;
            }
            if(current_cell_size.z < smallest_cell_size_.z)
            {
                smallest_cell_size_.z = current_cell_size.z;
            }
        }

        controller->dispatch(context_id, renderer_->device());
        first_frame_ = false;
    }
    else
    {
        std::vector<unsigned int> old_cut_lengths(num_models_, 0);
        bool done = false;
        int repetition_counter = 0;

        while (!done)
        {
            // Cut update runs asynchronous, so wait until it is done.
            if (!controller->is_cut_update_in_progress(context_id))
            {
                bool length_changed = false;
#endif

                for (lamure::model_t model_index = 0; model_index < num_models_; ++model_index)
                {
                    lamure::model_t model_id = controller->deduce_model_id(std::to_string(model_index));

                #ifndef LAMURE_PVS_USE_AS_RENDERER
                    // Check if the cut length changed in comparison to previous frame.
                    lamure::ren::cut& cut = cuts->get_cut(context_id, view_id, model_id);
                    if(cut.complete_set().size() > old_cut_lengths[model_index])
                    {
                        length_changed = true;
                    }
                    old_cut_lengths[model_index] = cut.complete_set().size();
                #endif

                    cuts->send_transform(context_id, model_id, model_transformations_[model_id]);
                    cuts->send_threshold(context_id, model_id, error_threshold_ / importance_);

                    // Send rendered, threshold, camera, ... 
                    cuts->send_rendered(context_id, model_id);
                    database->get_model(model_id)->set_transform(model_transformations_[model_id]);

                    lamure::view_t cam_id = controller->deduce_view_id(context_id, active_camera_->view_id());
                    cuts->send_camera(context_id, cam_id, *active_camera_);

                    std::vector<scm::math::vec3d> corner_values = active_camera_->get_frustum_corners();
                    double top_minus_bottom = scm::math::length((corner_values[2]) - (corner_values[0]));
                    float height_divided_by_top_minus_bottom = lamure::ren::policy::get_instance()->window_height() / top_minus_bottom;

                    cuts->send_height_divided_by_top_minus_bottom(context_id, cam_id, height_divided_by_top_minus_bottom);
                }

                controller->dispatch(context_id, renderer_->device());

#ifndef LAMURE_PVS_USE_AS_RENDERER
                // Stop if no length change was detected.
                if(!length_changed)
                {
                    ++repetition_counter;

                    if(repetition_counter >= 10)
                    {
                        done = true;
                    }
                }
                else
                {
                    repetition_counter = 0;
                }
            }

            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

#ifdef LAMURE_PVS_MEASURE_PERFORMANCE
    end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    total_cut_update_time_ += elapsed_seconds.count();
#endif
#endif

#ifdef LAMURE_PVS_USE_AS_RENDERER
    // Update PVS database with current camera position before rendering.
    pvs_database* pvs = pvs_database::get_instance();
    if(update_position_for_pvs_)
    {
        scm::math::mat4f cm = scm::math::inverse(scm::math::mat4f(active_camera_->trackball_matrix()));
        scm::math::vec3d cam_pos = scm::math::vec3d(cm[12], cm[13], cm[14]);
        pvs->set_viewer_position(cam_pos);
    }
#endif

#ifdef LAMURE_PVS_MEASURE_PERFORMANCE
    // Measure rendering performance.
    start_time = std::chrono::system_clock::now();
#endif

    renderer_->set_radius_scale(importance_);
    renderer_->render(context_id, *active_camera_, view_id, controller->get_context_memory(context_id, lamure::ren::bvh::primitive_type::POINTCLOUD, renderer_->device()), 0);

#ifdef LAMURE_PVS_MEASURE_PERFORMANCE
    end_time = std::chrono::system_clock::now();
    elapsed_seconds = end_time - start_time;
    total_render_time_ += elapsed_seconds.count();
#endif

#ifdef LAMURE_PVS_USE_AS_RENDERER
    // Output current view matrix for debug purpose.
    std::stringstream add_info_string;
    add_info_string << "visibility threshold: " << visibility_threshold_ << std::endl;

    add_info_string << "camera position:\n";
    scm::math::mat4f cm = scm::math::inverse(scm::math::mat4f(active_camera_->trackball_matrix()));
    for(int index = 11; index < 15; ++index)
    {
        add_info_string << cm[index] << "  ";
    }
    add_info_string << std::endl;

    add_info_string << "use PVS: " << pvs->is_activated() << std::endl;
    add_info_string << "update pos: " << update_position_for_pvs_ << std::endl;

    renderer_->display_status(add_info_string.str());
    //renderer_->display_status("");
#endif

#ifndef LAMURE_PVS_USE_AS_RENDERER
    if(!first_frame_ && first_phase_)
    {
        // Analyze histogram data of current rendered image.
        if(renderer_->get_rendered_node_count() > 0)
        {
        #ifdef LAMURE_PVS_MEASURE_PERFORMANCE
            // Measure histogram creation performance.
            start_time = std::chrono::system_clock::now();
        #endif

            id_histogram hist = renderer_->create_node_id_histogram(false, (direction_counter_ * visibility_grid_->get_cell_count()) + current_grid_index_);
            std::map<model_t, std::vector<node_t>> visible_ids = hist.get_visible_nodes(width_ * height_, visibility_threshold_);

            for(std::map<model_t, std::vector<node_t>>::iterator iter = visible_ids.begin(); iter != visible_ids.end(); ++iter)
            {
                model_t model_id = iter->first;

                for(node_t node_id : iter->second)
                {
                    visibility_grid_->set_cell_visibility(current_grid_index_, model_id, node_id, true);
                }
            }

        #ifdef LAMURE_PVS_MEASURE_PERFORMANCE
            end_time = std::chrono::system_clock::now();
            elapsed_seconds = end_time - start_time;
            total_histogram_evaluation_time_ += elapsed_seconds.count();
        #endif 
        }

        // Collect data to calculate average depth of nodes per model.
        for (lamure::model_t model_index = 0; model_index < num_models_; ++model_index)
        {
            lamure::model_t model_id = controller->deduce_model_id(std::to_string(model_index));
            lamure::ren::cut& cut = cuts->get_cut(context_id, view_id, model_id);
            std::vector<lamure::ren::cut::node_slot_aggregate> renderable = cut.complete_set();

            // Count nodes in the current cut.
            for(auto const& node_slot_aggregate : renderable)
            {
                total_depth_rendered_nodes_[model_index][current_grid_index_] = total_depth_rendered_nodes_[model_index][current_grid_index_] + database->get_model(model_id)->get_bvh()->get_depth_of_node(node_slot_aggregate.node_id_);
                total_num_rendered_nodes_[model_index][current_grid_index_] = total_num_rendered_nodes_[model_index][current_grid_index_] + 1;
            }
        }

        // Update current cell index and direction.
        current_grid_index_++;
        if(current_grid_index_ == visibility_grid_->get_cell_count())
        {
            current_grid_index_ = 0;
            direction_counter_++;

            if(direction_counter_ == 6)
            {
                direction_counter_ = 0;
                first_phase_ = false;
            }
        }

        if(current_grid_index_ % 8 == 0)
        {
            // Calculate current rendering state so user gets visual feedback on the preprocessing progress.
            size_t num_cells = visibility_grid_->get_cell_count();
            float total_rendering_steps = num_cells * 6;
            float current_rendering_step = (num_cells * direction_counter_) + current_grid_index_;
            float current_percentage_done = (current_rendering_step / total_rendering_steps) * 100.0f;
            std::cout << "\rfirst phase rendering in progress [" << current_percentage_done << "]       " << std::flush;

            if(current_percentage_done == 100.0f)
            {
                std::cout << std::endl;
            }
        }
    }
    else if(!first_frame_ && !first_phase_)
    {
        if(renderer_->get_rendered_node_count() > 0)
        {
        #ifdef LAMURE_PVS_MEASURE_PERFORMANCE
            // Measure histogram creation performance.
            start_time = std::chrono::system_clock::now();
        #endif

            id_histogram hist = renderer_->create_node_id_histogram(false, (direction_counter_ * visibility_grid_->get_cell_count()) + current_grid_index_);
            std::map<model_t, std::vector<node_t>> visible_ids = hist.get_visible_nodes(width_ * height_, visibility_threshold_);

            for(std::map<model_t, std::vector<node_t>>::iterator iter = visible_ids.begin(); iter != visible_ids.end(); ++iter)
            {
                model_t model_id = iter->first;

                for(node_t node_id : iter->second)
                {
                    // Visibility is to be set in up to 8 view cells surrounding the corner that was rendered.
                    for(double z_dir = -1.0; z_dir < 2.0; z_dir += 2.0)
                    {
                        for(double y_dir = -1.0; y_dir < 2.0; y_dir += 2.0)
                        {
                            for(double x_dir = -1.0; x_dir < 2.0; x_dir += 2.0)
                            {
                                scm::math::vec3d potentially_cell_pos = current_corner_pos + (smallest_cell_size_ * scm::math::vec3d(x_dir, y_dir, z_dir) * 0.1);
                                visibility_grid_->set_cell_visibility(potentially_cell_pos, model_id, node_id, true);
                            }
                        }
                    }
                }
            }

        #ifdef LAMURE_PVS_MEASURE_PERFORMANCE
            end_time = std::chrono::system_clock::now();
            elapsed_seconds = end_time - start_time;
            total_histogram_evaluation_time_ += elapsed_seconds.count();
        #endif 
        }

        // Update current render position and direction.
        size_t max_index_x = std::round(visibility_grid_->get_size().x / smallest_cell_size_.x) + 1;
        size_t max_index_y = std::round(visibility_grid_->get_size().y / smallest_cell_size_.y) + 1;
        size_t max_index_z = std::round(visibility_grid_->get_size().z / smallest_cell_size_.z) + 1;

        ++current_corner_index_x_;

        if(current_corner_index_x_ == max_index_x)
        {
            current_corner_index_x_ = 0;
            ++current_corner_index_y_;

            if(current_corner_index_y_ == max_index_y)
            {
                current_corner_index_y_ = 0;
                ++current_corner_index_z_;

                if(current_corner_index_z_ == max_index_z)
                {
                    current_corner_index_z_ = 0;
                    ++direction_counter_;

                    if(direction_counter_ == 6)
                    {
                        //direction_counter_ = 0;
                        signal_shutdown = true;
                    }
                }
            }
        }

        if(current_corner_index_x_ % 8 == 0)
        {
            // Calculate current rendering state so user gets visual feedback on the preprocessing progress.
            size_t num_corners = max_index_x * max_index_y * max_index_z;
            float total_rendering_steps = num_corners * 6;
            float current_rendering_step = (num_corners * direction_counter_) + (current_corner_index_z_ * max_index_y * max_index_x + current_corner_index_y_ * max_index_x + current_corner_index_x_);
            float current_percentage_done = (current_rendering_step / total_rendering_steps) * 100.0f;
            std::cout << "\rsecond phase rendering in progress [" << current_percentage_done << "]       " << std::flush;

            if(current_percentage_done == 100.0f)
            {
                std::cout << std::endl;
            }
        }
    }
#endif

    // Once the visibility test is complete ...
    if(signal_shutdown)
    {
    #ifdef LAMURE_PVS_MEASURE_PERFORMANCE
        start_time = std::chrono::system_clock::now();
    #endif

        // ... calculate which nodes are inside the view cells based on the average depth of the nodes inside the cuts during rendering.
        std::cout << "start check for nodes inside grid cells..." << std::endl;
        check_for_nodes_within_cells(total_depth_rendered_nodes_, total_num_rendered_nodes_);
        std::cout << "node check finished" << std::endl;

    #ifdef LAMURE_PVS_MEASURE_PERFORMANCE
        end_time = std::chrono::system_clock::now();
        elapsed_seconds = end_time - start_time;
        double node_within_cell_check_time = elapsed_seconds.count();
        start_time = std::chrono::system_clock::now();
    #endif

        // Hardcoded heresy. This grid type applies visibility propagation at runtime.
        if(visibility_grid_->get_grid_type() != "octree_hierarchical_v3")
        {
            // ... set visibility of LOD-trees based on rendered nodes.
            std::cout << "start visibility propagation..." << std::endl;
            emit_node_visibility(visibility_grid_);
            std::cout << "visibility propagation finished" << std::endl;
        }

    #ifdef LAMURE_PVS_MEASURE_PERFORMANCE
        end_time = std::chrono::system_clock::now();
        elapsed_seconds = end_time - start_time;
        double visibility_propagation_time = elapsed_seconds.count();

        std::string performance_file_path = pvs_file_path_;
        performance_file_path.resize(performance_file_path.size() - 4);
        performance_file_path += "_performance.txt";

        std::ofstream file_out;
        file_out.open(performance_file_path);

        file_out << "---------- average performance in seconds ----------" << std::endl;
        file_out << "cut update: " << total_cut_update_time_ / (6 * visibility_grid_->get_cell_count()) << std::endl;
        file_out << "rendering: " << total_render_time_ / (6 * visibility_grid_->get_cell_count()) << std::endl;
        file_out << "histogram evaluation: " << total_histogram_evaluation_time_ / (6 * visibility_grid_->get_cell_count()) << std::endl;

        file_out << "\n---------- total performance in seconds ----------" << std::endl;
        file_out << "cut update: " << total_cut_update_time_ << std::endl;
        file_out << "rendering: " << total_render_time_ << std::endl;
        file_out << "histogram evaluation: " << total_histogram_evaluation_time_ << std::endl;
        file_out << "node in cell check: " << node_within_cell_check_time << std::endl;
        file_out << "visibility propagation: " << visibility_propagation_time << std::endl;
        file_out << std::endl;

        file_out.close();
    #endif
    }

    return signal_shutdown;
}

}
}
