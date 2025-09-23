// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/management_base.h"

#include <set>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <lamure/ren/bvh.h>
#include <sstream>
#include <chrono>
#include <thread>

#include "lamure/pvs/pvs_database.h"
#include "lamure/pvs/grid_regular.h"

namespace lamure
{
namespace pvs
{

management_base::
management_base(std::vector<std::string> const& model_filenames,
    std::vector<scm::math::mat4f> const& model_transformations,
    std::set<lamure::model_t> const& visible_set,
    std::set<lamure::model_t> const& invisible_set)
    :   renderer_(nullptr),
        model_filenames_(model_filenames),
        model_transformations_(model_transformations),

        test_send_rendered_(true),
        active_camera_(nullptr),
        num_models_(0),
        dispatch_(true),
        error_threshold_(LAMURE_DEFAULT_THRESHOLD),
        near_plane_(0.001f),
        far_plane_(1000.f),
        importance_(1.f)

{
    visibility_threshold_ = 0.0001f;
    first_frame_ = true;
    visibility_grid_ = nullptr;
    current_grid_index_ = 0;
    direction_counter_ = 0;
    update_position_for_pvs_ = true;

#ifdef LAMURE_PVS_MEASURE_PERFORMANCE
    total_cut_update_time_ = 0.0;
    total_render_time_ = 0.0;
    total_histogram_evaluation_time_ = 0.0;
#endif

#ifndef LAMURE_PVS_USE_AS_RENDERER
    lamure::pvs::pvs_database::get_instance()->activate(false);
#endif

    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();

    for (const auto& filename : model_filenames_)
    {
        database->add_model(filename, std::to_string(num_models_));
        ++num_models_;
    }

    total_depth_rendered_nodes_.resize(num_models_);
    total_num_rendered_nodes_.resize(num_models_);

    float scene_diameter = far_plane_;
    for (lamure::model_t model_id = 0; model_id < database->num_models(); ++model_id)
    {
        const auto& bb = database->get_model(model_id)->get_bvh()->get_bounding_boxes()[0];
        scene_diameter = std::max(scm::math::length(bb.max_vertex()-bb.min_vertex()), scene_diameter);
        model_transformations_[model_id] = model_transformations_[model_id] * scm::math::make_translation(database->get_model(model_id)->get_bvh()->get_translation());
    }
    far_plane_ = 2.0f * scene_diameter;

    auto root_bb = database->get_model(0)->get_bvh()->get_bounding_boxes()[0];
    scm::math::vec3 center = model_transformations_[0] * root_bb.center();
    scm::math::mat4f reset_matrix = scm::math::make_look_at_matrix(center + scm::math::vec3f(0.0f, 0.0f, 0.1f), center, scm::math::vec3f(0.0f, 1.0f,0.0f));
    float reset_diameter = scm::math::length(root_bb.max_vertex()-root_bb.min_vertex());

    std::cout << "model center : " << center << std::endl;
    std::cout << "model size : " << reset_diameter << std::endl;

    active_camera_ = new lamure::ren::camera(0, reset_matrix, reset_diameter);

    // Increase camera movement speed for debugging purpose.
    active_camera_->set_dolly_sens_(20.5f);

    renderer_ = new Renderer(model_transformations_, visible_set, invisible_set);
}

management_base::
~management_base()
{
    if (active_camera_ != nullptr)
    {
        delete active_camera_;
        active_camera_ = nullptr;
    }
    if (renderer_ != nullptr)
    {
        delete renderer_;
        renderer_ = nullptr;
    }
}

void management_base::
check_for_nodes_within_cells(const std::vector<std::vector<size_t>>& total_depths, const std::vector<std::vector<size_t>>& total_nums)
{
    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();

    for(model_t model_index = 0; model_index < database->num_models(); ++model_index)
    {
        for(size_t cell_index = 0; cell_index < visibility_grid_->get_cell_count(); ++cell_index)
        {
            // Create bounding box of view cell.
            const view_cell* current_cell = visibility_grid_->get_cell_at_index(cell_index);
                
            vec3r min_vertex(current_cell->get_position_center() - (current_cell->get_size() * 0.5f));
            vec3r max_vertex(current_cell->get_position_center() + (current_cell->get_size() * 0.5f));
            bounding_box cell_bounds(min_vertex, max_vertex);

            // We can get the first and last index of the nodes on a certain depth inside the bvh.
            unsigned int average_depth = total_depth_rendered_nodes_[model_index][cell_index] / total_num_rendered_nodes_[model_index][cell_index];

            node_t start_index = database->get_model(model_index)->get_bvh()->get_first_node_id_of_depth(average_depth);
            node_t end_index = start_index + database->get_model(model_index)->get_bvh()->get_length_of_depth(average_depth);

            for(node_t node_index = start_index; node_index < end_index; ++node_index)
            {
                // Create bounding box of node.
                scm::gl::boxf node_bounding_box = database->get_model(model_index)->get_bvh()->get_bounding_boxes()[node_index];
                vec3r min_vertex = vec3r(node_bounding_box.min_vertex()) + database->get_model(model_index)->get_bvh()->get_translation();
                vec3r max_vertex = vec3r(node_bounding_box.max_vertex()) + database->get_model(model_index)->get_bvh()->get_translation();
                bounding_box node_bounds(min_vertex, max_vertex);

                // check if the bounding boxes collide.
                if(cell_bounds.intersects(node_bounds))
                {
                    visibility_grid_->set_cell_visibility(cell_index, model_index, node_index, true);
                }
            }
        }
    }
}

void management_base::
emit_node_visibility(grid* visibility_grid)
{
    float steps_finished = 0.0f;
    float total_steps = visibility_grid_->get_cell_count();

    // Advance node visibility downwards and upwards in the LOD-hierarchy.
    // Since only a single LOD-level was rendered in the visibility test, this is necessary to produce a complete PVS.
    #pragma omp parallel for
    for(size_t cell_index = 0; cell_index < visibility_grid->get_cell_count(); ++cell_index)
    {
        const view_cell* current_cell = visibility_grid->get_cell_at_index(cell_index);
        std::map<model_t, std::vector<node_t>> visible_indices = current_cell->get_visible_indices();

        for(std::map<model_t, std::vector<node_t>>::const_iterator map_iter = visible_indices.begin(); map_iter != visible_indices.end(); ++map_iter)
        {
            for(node_t node_index = 0; node_index < map_iter->second.size(); ++node_index)
            {
                node_t visible_node_id = map_iter->second.at(node_index);

                // Communicate visibility to children and parents nodes of visible nodes.
                set_node_children_visible(cell_index, current_cell, map_iter->first, visible_node_id);
                set_node_parents_visible(cell_index, current_cell, map_iter->first, visible_node_id);
            }       
        }

        #pragma omp critical
        {
            // Calculate current node propagation state so user gets visual feedback on the preprocessing progress.
            steps_finished++;
            float current_percentage_done = (steps_finished / total_steps) * 100.0f;
            std::cout << "\rvisibility propagation in progress [" << current_percentage_done << "]       " << std::flush;
        }
    }

    std::cout << std::endl;
}

void management_base::
set_node_parents_visible(const size_t& cell_id, const view_cell* cell, const model_t& model_id, const node_t& node_id)
{
    // Set parents of a visible node visible, too.
    // Necessary since only a single LOD-level is rendered during the visibility test.
    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    node_t parent_id = database->get_model(model_id)->get_bvh()->get_parent_id(node_id);
    
    if(parent_id != lamure::invalid_node_t && !cell->get_visibility(model_id, parent_id))
    {
        visibility_grid_->set_cell_visibility(cell_id, model_id, parent_id, true);
        set_node_parents_visible(cell_id, cell, model_id, parent_id);
    }
}

void management_base::
set_node_children_visible(const size_t& cell_id, const view_cell* cell, const model_t& model_id, const node_t& node_id)
{
    // Set children of a visible node visible, too.
    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    uint32_t fan_factor = database->get_model(model_id)->get_bvh()->get_fan_factor();

    for(uint32_t child_index = 0; child_index < fan_factor; ++child_index)
    {
        node_t child_id = database->get_model(model_id)->get_bvh()->get_child_id(node_id, child_index);
        if(child_id < database->get_model(model_id)->get_bvh()->get_num_nodes() && !cell->get_visibility(model_id, child_id))
        {
            visibility_grid_->set_cell_visibility(cell_id, model_id, child_id, true);
            set_node_children_visible(cell_id, cell, model_id, child_id);
        }
    }
}

void management_base::
update_trackball(int x, int y)
{
#ifdef ALLOW_INPUT
    active_camera_->update_trackball(x,y, width_, height_, mouse_state_);
#endif
}

void management_base::
RegisterMousePresses(int button, int state, int x, int y)
{
#ifdef ALLOW_INPUT
    switch (button)
    {
        case GLUT_LEFT_BUTTON:
            {
                mouse_state_.lb_down_ = (state == GLUT_DOWN) ? true : false;
            }
            break;
        case GLUT_MIDDLE_BUTTON:
            {
                mouse_state_.mb_down_ = (state == GLUT_DOWN) ? true : false;
            }
            break;
        case GLUT_RIGHT_BUTTON:
            {
                mouse_state_.rb_down_ = (state == GLUT_DOWN) ? true : false;
            }
            break;
    }

    float trackball_init_x = 2.f * float(x - (width_/2))/float(width_) ;
    float trackball_init_y = 2.f * float(height_ - y - (height_/2))/float(height_);

    active_camera_->update_trackball_mouse_pos(trackball_init_x, trackball_init_y);
#endif
}

void management_base::
dispatchKeyboardInput(unsigned char key)
{
#ifdef ALLOW_INPUT
    switch(key)
    {
        case 's':
        {
            id_histogram hist = renderer_->create_node_id_histogram(false, 0);
            //renderer_->compare_histogram_to_cut(hist, visibility_threshold_);
            apply_temporal_pvs(hist);
            break;
        }

        case 'x':
        {
            pvs_database::get_instance()->clear_visibility_grid();
            break;
        }

        case 'p':
        {
            pvs_database::get_instance()->activate(!pvs_database::get_instance()->is_activated());
            break;
        }

        case 'o':
        {
            update_position_for_pvs_ = !update_position_for_pvs_;
            break;
        }

        case 'e':
            visibility_threshold_ *= 1.1f;
            break;

        case 'd':
            visibility_threshold_ /= 1.1f;
            break;

        case 'q':
            Toggledispatching();
            break;

        case 'w':
            renderer_->toggle_bounding_box_rendering();
            break;
    }
#endif
}

void management_base::
dispatchResize(int w, int h)
{
    width_ = w;
    height_ = h;

    renderer_->reset_viewport(w,h);

    lamure::ren::policy* policy = lamure::ren::policy::get_instance();
    policy->set_window_width(w);
    policy->set_window_height(h);

    active_camera_->set_projection_matrix(30.0f, float(w)/float(h),  near_plane_, far_plane_);
}

void management_base::
Toggledispatching()
{
    dispatch_ = ! dispatch_;
}

void management_base::
DecreaseErrorThreshold()
{
    error_threshold_ -= 0.1f;
    if (error_threshold_ < LAMURE_MIN_THRESHOLD)
    {
        error_threshold_ = LAMURE_MIN_THRESHOLD;
    }
}

void management_base::
IncreaseErrorThreshold()
{
    error_threshold_ += 0.1f;
    if (error_threshold_ > LAMURE_MAX_THRESHOLD)
    {
        error_threshold_ = LAMURE_MAX_THRESHOLD;
    }
}

void management_base::
set_grid(grid* visibility_grid)
{
    visibility_grid_ = visibility_grid;

    // Set size of containers used to collect data on rendered node depth.
    if(visibility_grid_ != nullptr)
    {
        total_depth_rendered_nodes_.clear();
        total_num_rendered_nodes_.clear();

        for(model_t model_index = 0; model_index < num_models_; ++model_index)
        {
            total_depth_rendered_nodes_[model_index].resize(visibility_grid_->get_cell_count());
            total_num_rendered_nodes_[model_index].resize(visibility_grid_->get_cell_count());
        }
    }
}

void management_base::
set_pvs_file_path(const std::string& file_path)
{
    pvs_file_path_ = file_path;
}

void management_base::
apply_temporal_pvs(const id_histogram& hist)
{
    int numPixels = width_ * height_;
    std::map<model_t, std::vector<node_t>> visible_nodes = hist.get_visible_nodes(numPixels, visibility_threshold_);

    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    std::vector<lamure::node_t> ids;
    ids.resize(database->num_models());
    
    for(model_t model_index = 0; model_index < database->num_models(); ++model_index)
    {
        ids[model_index] = database->get_model(model_index)->get_bvh()->get_num_nodes();
    }

    grid_regular tmp_grid(1, 1000.0, scm::math::vec3d(0.0, 0.0, 0.0), ids);

    for(std::map<model_t, std::vector<node_t>>::iterator modelIter = visible_nodes.begin(); modelIter != visible_nodes.end(); ++modelIter)
    {
        for(lamure::node_t node_index = 0; node_index < modelIter->second.size(); ++node_index)
        {
            tmp_grid.set_cell_visibility(0, modelIter->first, modelIter->second[node_index], true);
        }
    }

    emit_node_visibility(&tmp_grid);

    tmp_grid.save_grid_to_file("/home/tiwo9285/tmp.grid");
    tmp_grid.save_visibility_to_file("/home/tiwo9285/tmp.pvs");

    pvs_database* pvs = pvs_database::get_instance();
    pvs->load_pvs_from_file("/home/tiwo9285/tmp.grid", "/home/tiwo9285/tmp.pvs", true);

    for (size_t cell_index = 0; cell_index < pvs->get_visibility_grid()->get_cell_count(); ++cell_index)
    {
        std::map<model_t, std::vector<node_t>> visible_nodes = pvs->get_visibility_grid()->get_cell_at_index(cell_index)->get_visible_indices();

        for(model_t model_index = 0; model_index < num_models_; ++model_index)
        {
            std::cout << "cell: " << cell_index << " model: " << model_index << std::endl;

            for(node_t node_index = 0; node_index < visible_nodes[model_index].size(); ++node_index)
            {
                std::cout << visible_nodes[model_index][node_index] << " ";
            }

            std::cout << std::endl;
        }
    }
}

}
}
