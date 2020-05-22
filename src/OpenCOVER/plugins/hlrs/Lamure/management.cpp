// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "management.h"

#include "utils.h"
#include <set>
#include <ctime>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <chrono>

#include <lamure/ren/bvh.h>
#include <set>
#include <lamure/pvs/pvs_database.h>

//#include <glm/glm.hpp>
//#include "MatrixInterpolation.hpp"

management::management(std::vector<std::string> const &model_filenames, std::vector<scm::math::mat4f> const &model_transformations, std::set<lamure::model_t> const &visible_set,
                       std::set<lamure::model_t> const &invisible_set, snapshot_session_descriptor &snap_descriptor)
    : num_taken_screenshots_(0), allow_user_input_(snap_descriptor.recorded_view_vector_.size() == 0), screenshot_session_started_(false), camera_recording_enabled_(false),
      // current_session_filename_(session_filename),
      current_session_file_path_(""), num_recorded_camera_positions_(0), renderer_(nullptr), model_filenames_(model_filenames), model_transformations_(model_transformations),
      // recorded_view_vector_(recorded_view_vector),
      measurement_session_descriptor_(snap_descriptor),
#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
      active_camera_left_(nullptr), active_camera_right_(nullptr), control_left_(true),
#endif
        test_send_rendered_(true),
        active_camera_(nullptr),
        mouse_state_(),
        num_models_(0),
        num_cameras_(1),
        fast_travel_(false),
	    travel_speed_mode_(0),
        dispatch_(true),
        trigger_one_update_(false),
        reset_matrix_(scm::math::mat4f::identity()),
        reset_diameter_(90.f),
        detail_translation_(scm::math::vec3f::zero()),
        detail_angle_(0.f),
        error_threshold_(LAMURE_DEFAULT_THRESHOLD),
        near_plane_(0.001f),
        far_plane_(1000.f),
        importance_(1.f)

{

    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    is_updating_pvs_position_ = true;

#ifdef LAMURE_RENDERING_ENABLE_LAZY_MODELS_TEST
    assert(model_filenames_.size() > 0);
    lamure::model_t model_id = database->add_model(model_filenames_[0], std::to_string(num_models_));
    ++num_models_;
#else
    for(const auto &filename : model_filenames_)
    {
        lamure::model_t model_id = database->add_model(filename, std::to_string(num_models_));
        ++num_models_;
    }
#endif

    //std::cout << database->get_model(0)->get_bvh()->get_depth() << std::endl;

    {
        float scene_diameter = far_plane_;
        for(lamure::model_t model_id = 0; model_id < database->num_models(); ++model_id)
        {
            const auto &bb = database->get_model(model_id)->get_bvh()->get_bounding_boxes()[0];
            scene_diameter = std::max(scm::math::length(bb.max_vertex() - bb.min_vertex()), scene_diameter);
            model_transformations_[model_id] = model_transformations_[model_id] * scm::math::make_translation(database->get_model(model_id)->get_bvh()->get_translation());
        }
        far_plane_ = 2.0f * scene_diameter;
        if (database->num_models() > 0)
        {
            auto root_bb = database->get_model(0)->get_bvh()->get_bounding_boxes()[0];
            scm::math::vec3 center = model_transformations_[0] * root_bb.center();
            reset_matrix_ = scm::math::make_look_at_matrix(center + scm::math::vec3f(0.f, 0.1f, -0.01f), center, scm::math::vec3f(0.f, 1.f, 0.f));
            reset_diameter_ = scm::math::length(root_bb.max_vertex() - root_bb.min_vertex());
            std::cout << "model center : " << center << std::endl;
            std::cout << "model size : " << reset_diameter_ << std::endl;
        }

        for(lamure::view_t cam_id = 0; cam_id < num_cameras_; ++cam_id)
        {
            cameras_.push_back(new lamure::ren::camera(cam_id, reset_matrix_, reset_diameter_, false, false));
        }
    }

#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
    active_camera_left_ = cameras_[0];
    active_camera_right_ = cameras_[0];
#endif
    active_camera_ = cameras_[0];

#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
    renderer_ = new SplitScreenRenderer(model_transformations_);
#endif

#ifndef LAMURE_RENDERING_USE_SPLIT_SCREEN
    renderer_ = new Renderer(model_transformations_, visible_set, invisible_set);
#endif
    
    PrintInfo();

    current_update_timeout_timer_ = 0.0;
    interpolation_time_point_ = 0.0f;
    use_interpolation_on_measurement_session_ = false;
    movement_on_interpolation_per_frame_ = 10.0f;
    snapshot_framerate_counter_ = 0.0;
    snapshot_frame_counter_ = 0;

    use_wasd_camera_control_scheme_ = false;
    renderer_->toggle_use_user_defined_background_color();
}

management::~management()
{
    for(auto &cam : cameras_)
    {
        if(cam != nullptr)
        {
            delete cam;
            cam = nullptr;
        }
    }
    if(renderer_ != nullptr)
    {
        delete renderer_;
        renderer_ = nullptr;
    }
}


void management::resolve_movement(double elapsed_time_ms) {

  double travel_speed_multiplicator = (travel_speed_mode_ == 0 ? 0.5f
                                     : travel_speed_mode_ == 1 ? 20.5f
                                     : travel_speed_mode_ == 2 ? 100.5f
                                     : 300.5f);

  double movement_factor = travel_speed_multiplicator * elapsed_time_ms;
  if(is_moving_forward_) {
    active_camera_->translate(0.0f, 0.0f, movement_factor*10.0f);
  }
  if(is_moving_backward_) {
    active_camera_->translate(0.0f, 0.0f, movement_factor*-10.0f);
  }
  if(is_moving_left_) {
    active_camera_->translate(movement_factor*10.0f, 0.0f, 0.0f);
  }
  if(is_moving_right_) {
    active_camera_->translate(movement_factor*-10.0f, 0.0f, 0.0f);
  }
}

bool management::MainLoop()
{

    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
    start_time = std::chrono::system_clock::now();

    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    lamure::ren::controller* controller = lamure::ren::controller::get_instance();
    lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
    lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();

    bool signal_shutdown = false;

    // PVS position update.
    if(is_updating_pvs_position_)
    {
        scm::math::mat4f cm = scm::math::inverse(scm::math::mat4f(active_camera_->trackball_matrix()));
        scm::math::vec3d cam_pos = scm::math::vec3d(cm[12], cm[13], cm[14]);
        pvs->set_viewer_position(cam_pos);
    }

#if 0
    for (unsigned int model_id = 0; model_id < database->num_models(); ++model_id) {
       model_transformations_[model_id] = model_transformations_[model_id] * scm::math::make_translation(28.f, -389.f, -58.f);
       renderer_->send_model_transform(model_id, model_transformations_[model_id]);
    }

#endif

    controller->reset_system();

    lamure::context_t context_id = controller->deduce_context_id(0);

    controller->dispatch(context_id, renderer_->device());

#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
    lamure::view_t view_id_left = controller->deduce_view_id(context_id, active_camera_left_->view_id());
    lamure::view_t view_id_right = controller->deduce_view_id(context_id, active_camera_right_->view_id());
#else

    lamure::view_t view_id = controller->deduce_view_id(context_id, active_camera_->view_id());

#endif

#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
    renderer_->render(context_id, *active_camera_left_, view_id_left, 0, controller->get_context_memory(context_id, lamure::ren::bvh::primitive_type::POINTCLOUD, renderer_->device()),
                      num_recorded_camera_positions_);
    renderer_->render(context_id, *active_camera_right_, view_id_right, 1, controller->get_context_memory(context_id, lamure::ren::bvh::primitive_type::POINTCLOUD, renderer_->device()),
                      num_recorded_camera_positions_);
#else
    renderer_->set_radius_scale(importance_);
    renderer_->render(context_id, *active_camera_, view_id, controller->get_context_memory(context_id, lamure::ren::bvh::primitive_type::POINTCLOUD, renderer_->device()),
                      num_recorded_camera_positions_);
#endif

    std::string status_string("");

    if(camera_recording_enabled_)
    {
        //status_string += "Session recording (#"+std::to_string(current_session_number_) +") : ON\n";
    }
    else
    {
        //status_string += "Session recording: OFF\n";
    }

    // Session file handling. Used to capture screenshots for test or error measurement purpose.
    if(!allow_user_input_)
    {
        status_string += std::to_string(measurement_session_descriptor_.recorded_view_vector_.size()+1) + " views left to write.\n";
        const double max_timeout_timer_time = 60.0;

        if(use_interpolation_on_measurement_session_)
        {
            // Interpolation between session files saved transformations. Allows to follow a track through the scene.
            if( measurement_session_descriptor_.recorded_view_vector_.size() > 1)
            {
                scm::math::mat4d& from_transform = measurement_session_descriptor_.recorded_view_vector_[measurement_session_descriptor_.recorded_view_vector_.size() - 1];
                scm::math::mat4d& to_transform = measurement_session_descriptor_.recorded_view_vector_[measurement_session_descriptor_.recorded_view_vector_.size() - 2];

            /*    glm::mat4 glm_from_transform(from_transform[0], from_transform[1], from_transform[2], from_transform[3],
                                            from_transform[4], from_transform[5], from_transform[6], from_transform[7],
                                            from_transform[8], from_transform[9], from_transform[10], from_transform[11],
                                            from_transform[12], from_transform[13], from_transform[14], from_transform[15]);
                glm::mat4 glm_to_transform(to_transform[0], to_transform[1], to_transform[2], to_transform[3],
                                            to_transform[4], to_transform[5], to_transform[6], to_transform[7],
                                            to_transform[8], to_transform[9], to_transform[10], to_transform[11],
                                            to_transform[12], to_transform[13], to_transform[14], to_transform[15]);

                glm::mat4 glm_interpolated_transform = interpolate(glm_from_transform, glm_to_transform, interpolation_time_point_);

                // Cast back to schism matrix to set camera view matrix.
                scm::math::mat4d interpolated_transform(glm_interpolated_transform[0][0], glm_interpolated_transform[0][1], glm_interpolated_transform[0][2], glm_interpolated_transform[0][3],
                                                        glm_interpolated_transform[1][0], glm_interpolated_transform[1][1], glm_interpolated_transform[1][2], glm_interpolated_transform[1][3],
                                                        glm_interpolated_transform[2][0], glm_interpolated_transform[2][1], glm_interpolated_transform[2][2], glm_interpolated_transform[2][3],
                                                        glm_interpolated_transform[3][0], glm_interpolated_transform[3][1], glm_interpolated_transform[3][2], glm_interpolated_transform[3][3]);*/
                //active_camera_->set_view_matrix(interpolated_transform);

                // Take screenshots.
                size_t ms_since_update = controller->ms_since_last_node_upload();

                // Used to determine average frame rate over 1 second before taking the screenshot.
                if(ms_since_update > 2000 || current_update_timeout_timer_ > (max_timeout_timer_time - 1.0))
                {
                    snapshot_framerate_counter_ += renderer_->get_fps();
                    ++snapshot_frame_counter_;
                }
                else
                {
                    snapshot_framerate_counter_ = 0.0f;
                    snapshot_frame_counter_ = 0;
                }
                
                if(ms_since_update > 3000 || current_update_timeout_timer_ > max_timeout_timer_time)
                {
                    double average_framerate = snapshot_framerate_counter_ / (double)snapshot_frame_counter_;

                    auto const& resolution = measurement_session_descriptor_.snapshot_resolution_;
                    renderer_->take_screenshot("../quality_measurement/session_screenshots/" + measurement_session_descriptor_.session_filename_, 
                                                measurement_session_descriptor_.get_screenshot_name() + "_fps_" + std::to_string(average_framerate));

                    measurement_session_descriptor_.increment_screenshot_counter();
                    controller->reset_ms_since_last_node_upload();

                    current_update_timeout_timer_ = 0.0;

                    scm::math::vec3d start_pos(from_transform[12], from_transform[13], from_transform[14]);
                    scm::math::vec3d end_pos(to_transform[12], to_transform[13], to_transform[14]);
                    double total_distance = scm::math::length(start_pos - end_pos);

                    interpolation_time_point_ += movement_on_interpolation_per_frame_ / total_distance;
                }
                else
                {
                    status_string += std::to_string(((3000 - ms_since_update) / 100) * 100 ) + " ms until next buffer snapshot.\n";
                    status_string += std::to_string((int)(max_timeout_timer_time - current_update_timeout_timer_)) + " s until forced snapshot.\n";
                }

                // Interpolate between next two points if finished goal.
                if(interpolation_time_point_ >= 1.0f)
                {
                    interpolation_time_point_ = 0.0f;
                    measurement_session_descriptor_.recorded_view_vector_.pop_back();
                }

            }
            else
            {
                // leave the main loop
                signal_shutdown = true;
            }
        }
        else
        {

            // Classic way. Take screenshots only at transformations saved in session file.
            size_t ms_since_update = controller->ms_since_last_node_upload();

            if ( ms_since_update > 3000 || current_update_timeout_timer_ > max_timeout_timer_time)
            {
                if ( screenshot_session_started_ )
                    
                    if(measurement_session_descriptor_.get_num_taken_screenshots() )
                    {
                        auto const& resolution = measurement_session_descriptor_.snapshot_resolution_;
                        renderer_->take_screenshot("../quality_measurement/session_screenshots/" + measurement_session_descriptor_.session_filename_, 
                                                    measurement_session_descriptor_.get_screenshot_name() );
                    }

                    measurement_session_descriptor_.increment_screenshot_counter();

                if(!measurement_session_descriptor_.recorded_view_vector_.empty() )
                {
                    if (! screenshot_session_started_ )
                    {
                        screenshot_session_started_ = true;
                    }

                    active_camera_->set_view_matrix(measurement_session_descriptor_.recorded_view_vector_.back());
                    controller->reset_ms_since_last_node_upload();
                    measurement_session_descriptor_.recorded_view_vector_.pop_back();
                }
                else
                {
                    // leave the main loop
                    signal_shutdown = true;
                }

                current_update_timeout_timer_ = 0.0;
            }
            else
            {
                status_string += std::to_string(((3000 - ms_since_update) / 100) * 100 ) + " ms until next buffer snapshot.\n";
                status_string += std::to_string((int)(max_timeout_timer_time - current_update_timeout_timer_)) + " s until forced snapshot.\n";
            }

        }
    }

    if(pvs->is_activated())
    {
        status_string += "PVS: ON\n";
    }
    else
    {
        status_string += "PVS: OFF\n";
    }



    renderer_->display_status(status_string);

    if(dispatch_ || trigger_one_update_)
    {
        if(trigger_one_update_)
        {
            trigger_one_update_ = false;
        }

        for(lamure::model_t model_id = 0; model_id < num_models_; ++model_id)
        {
            lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));

            cuts->send_transform(context_id, m_id, model_transformations_[m_id]);
            cuts->send_threshold(context_id, m_id, error_threshold_ / importance_);

            // if (visible_set_.find(model_id) != visible_set_.end())
            if(!test_send_rendered_)
            {
                if(model_id > num_models_ / 2)
                {
                    cuts->send_rendered(context_id, m_id);
                }
            }
            else
            {
                cuts->send_rendered(context_id, m_id);
            }

            database->get_model(m_id)->set_transform(model_transformations_[m_id]);
        }

        for(auto &cam : cameras_)
        {
            lamure::view_t cam_id = controller->deduce_view_id(context_id, cam->view_id());
            cuts->send_camera(context_id, cam_id, *cam);

            std::vector<scm::math::vec3d> corner_values = cam->get_frustum_corners();
            double top_minus_bottom = scm::math::length((corner_values[2]) - (corner_values[0]));
            float height_divided_by_top_minus_bottom = lamure::ren::policy::get_instance()->window_height() / top_minus_bottom;

            cuts->send_height_divided_by_top_minus_bottom(context_id, cam_id, height_divided_by_top_minus_bottom);
        }

        // controller->dispatch(context_id, renderer_->device());
    }

#ifdef LAMURE_CUT_UPDATE_ENABLE_MEASURE_SYSTEM_PERFORMANCE
    system_performance_timer_.stop();
    boost::timer::cpu_times const elapsed_times(system_performance_timer_.elapsed());
    boost::timer::nanosecond_type const elapsed(elapsed_times.system + elapsed_times.user);

    if(elapsed >= boost::timer::nanosecond_type(1.0f * 1000 * 1000 * 1000)) // 1 second
    {
        boost::timer::cpu_times const result_elapsed_times(system_result_timer_.elapsed());
        boost::timer::nanosecond_type const result_elapsed(result_elapsed_times.system + result_elapsed_times.user);

        std::cout << "no cut update after " << result_elapsed / (1000 * 1000 * 1000) << " seconds" << std::endl;
        system_performance_timer_.start();
    }
    else
    {
        system_performance_timer_.resume();
    }

#endif

    end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    double frame_time = elapsed_seconds.count();


    resolve_movement(frame_time);

    current_update_timeout_timer_ += frame_time;

    return signal_shutdown;
}

void management::update_trackball(int x, int y) { active_camera_->update_trackball(x, y, width_, height_, mouse_state_); }

void management::RegisterMousePresses(int button, int state, int x, int y)
{
    if(use_wasd_camera_control_scheme_)
    {
        if(mouse_state_.lb_down_)
        {
            double delta_x = 100.0 * double(x - mouse_last_x_) / double(width_);
            double delta_y = 100.0 * double(y - mouse_last_y_) / float(height_);
            active_camera_->rotate((double)delta_y, (double)delta_x, 0.0);
        }
        else if(mouse_state_.rb_down_)
        {
            double delta_x = 100.0 * double(x - mouse_last_x_) / double(width_);
            active_camera_->rotate(0.0, 0.0, (double)delta_x);
        }

        mouse_last_x_ = x;
        mouse_last_y_ = y;
    }
    else
    {
        active_camera_->update_trackball(x,y, width_, height_, mouse_state_);
    }


    if(! allow_user_input_)
    {
        return;
    }

    switch (button)
    {
     /*   case GLUT_LEFT_BUTTON:
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
            break;*/
    }

    if(use_wasd_camera_control_scheme_)
    {
        mouse_last_x_ = x;
        mouse_last_y_ = y;
    }
    else
    {

        float trackball_init_x = 2.f * float(x - (width_/2))/float(width_) ;
        float trackball_init_y = 2.f * float(height_ - y - (height_/2))/float(height_);

        active_camera_->update_trackball_mouse_pos(trackball_init_x, trackball_init_y);

        renderer_->mouse(button, state, x, y, *active_camera_);
    }
}

void management::dispatchKeyboardInput(unsigned char key)
{
    if(! allow_user_input_)
    {
        return;
    }

    bool override_center_of_rotation = false;

    switch(key)
    {
    case 'p':
        renderer_->toggle_provenance_rendering();
        break;
    case 'm':
      renderer_->toggle_do_measurement();
      break;
      
    case 'c':
      renderer_->toggle_culling();
      break;

    case '+':
        importance_ += 0.1f;
        importance_ = std::min(importance_, 1.0f);
        std::cout << "importance: " << importance_ << std::endl;
        break;

    case '-':
        importance_ -= 0.1f;
        importance_ = std::max(0.1f, importance_);
        std::cout << "importance: " << importance_ << std::endl;
        break;

    case 'y':
        test_send_rendered_ = !test_send_rendered_;
        std::cout << "send rendered: " << test_send_rendered_ << std::endl;
        break;
    case 'w':
        renderer_->toggle_bounding_box_rendering();
        break;
    case 'U':
        renderer_->change_point_size(1.0f);
        break;
    case 'u':
        renderer_->change_point_size(0.1f);
        break;
    case 'J':
        renderer_->change_point_size(-1.0f);
        break;
    case 'j':
        renderer_->change_point_size(-0.1f);
        break;
    case 'P':
    {
        // Toggle PVS activation.
        lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();
        pvs->activate(!pvs->is_activated());
        break;
    }
    case 'o':
    {
        // Toggle the position update to the pvs.
        is_updating_pvs_position_ = !is_updating_pvs_position_;

        std::cout << "PVS viewer position update: " << ( (true == is_updating_pvs_position_) ? "ON" : "OFF") << "\n";

        break;
    }
    case 'l':
    {
        // Toggle the rendering of the view cells of the loaded pvs.
        renderer_->toggle_pvs_grid_cell_rendering();
        break;
    }
    case '5':
    {
        renderer_->toggle_culling();
        break;
    }
    case 't':
#ifndef LAMURE_RENDERING_USE_SPLIT_SCREEN
        renderer_->toggle_visible_set();
#endif
        break;

#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
    case '1':

        control_left_ = !control_left_;
        if(control_left_)
            active_camera_ = active_camera_left_;
        else
            active_camera_ = active_camera_right_;
        break;

    case '2':
        control_left_ = !control_left_;
        if(control_left_)
            active_camera_ = active_camera_left_;
        else
            active_camera_ = active_camera_right_;
#else
    case '1':
        renderer_->switch_render_mode(RenderMode::HQ_ONE_PASS);
        break;

    case '2':
        renderer_->switch_render_mode(RenderMode::HQ_TWO_PASS);
        break;

    case '3':
        renderer_->switch_render_mode(RenderMode::LQ_ONE_PASS);
        break;
#endif

    case ' ':
#ifdef LAMURE_RENDERING_ENABLE_MULTI_VIEW_TEST
#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
    {
        if(control_left_)
        {
            lamure::view_t current_camera_id = active_camera_left_->view_id();
            active_camera_left_ = cameras_[(++current_camera_id) % num_cameras_];
            active_camera_ = active_camera_left_;
        }
        else
        {
            lamure::view_t current_camera_id = active_camera_right_->view_id();
            active_camera_right_ = cameras_[(++current_camera_id) % num_cameras_];
            active_camera_ = active_camera_right_;
        }

        renderer_->toggle_camera_info(active_camera_left_->view_id(), active_camera_right_->view_id());
    }
#else
    {
        lamure::view_t current_camera_id = active_camera_->view_id();
        active_camera_ = cameras_[(++current_camera_id) % num_cameras_];
        renderer_->toggle_camera_info(active_camera_->view_id());
    }
#endif
#endif
    break;

    case 'd':
        this->Toggledispatching();
        renderer_->toggle_cut_update_info();
        break;

    case 'z':
    {
        lamure::ren::ooc_cache *ooc_cache = lamure::ren::ooc_cache::get_instance();
        ooc_cache->begin_measure();
    }
    break;

    case 'Z':
    {
        lamure::ren::ooc_cache *ooc_cache = lamure::ren::ooc_cache::get_instance();
        ooc_cache->end_measure();
    }
    break;

    case 'e':
        trigger_one_update_ = true;
        break;

    case 'x':
    {
#ifdef LAMURE_RENDERING_ENABLE_MULTI_VIEW_TEST
        cameras_.push_back(new lamure::ren::camera(num_cameras_, reset_matrix_, reset_diameter_, fast_travel_));
        int w = width_;
        int h = height_;
#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
        w /= 2;
#endif

        cameras_.back()->set_projection_matrix(30.0f, float(w) / float(h), near_plane_, far_plane_);

        ++num_cameras_;
#endif
    }
    break;

    case 'f':
      {
	      ++travel_speed_mode_;
	      if(travel_speed_mode_ > 4){
	        travel_speed_mode_ = 0;
	      }
	      const float travel_speed = (travel_speed_mode_ == 0 ? 0.5f
				          : travel_speed_mode_ == 1 ? 5.5f
				          : travel_speed_mode_ == 2 ? 20.5f
				          : travel_speed_mode_ == 3 ? 100.5f
				          : 300.5f);
	      std::cout << "setting travel speed to " << travel_speed << std::endl;
	      for (auto& cam : cameras_){ 
	        cam->set_dolly_sens_(travel_speed);
	      }
      }
      break;
      
      case 'F':
      {
	      if(travel_speed_mode_ > 0){
	        --travel_speed_mode_;
	      }
	      const float travel_speed = (travel_speed_mode_ == 0 ? 0.5f
				          : travel_speed_mode_ == 1 ? 5.5f
				          : travel_speed_mode_ == 2 ? 20.5f
				          : travel_speed_mode_ == 3 ? 100.5f
				          : 300.5f);
	      std::cout << "setting travel speed to " << travel_speed << std::endl;
	      for (auto& cam : cameras_){ 
	        cam->set_dolly_sens_(travel_speed);
	      }
      }
      break;
#ifndef LAMURE_RENDERING_USE_SPLIT_SCREEN
    
 
    case 'Q':
    {
        {
            travel_speed_mode_ = 0;
        }
        const float travel_speed = (travel_speed_mode_ == 0 ? 0.5f : travel_speed_mode_ == 1 ? 20.5f : travel_speed_mode_ == 2 ? 100.5f : 300.5f);
        std::cout << "setting travel speed to " << travel_speed << std::endl;
        for(auto &cam : cameras_)
        {
            cam->set_dolly_sens_(travel_speed);
        }
    }
    break;
#endif
#ifndef LAMURE_RENDERING_USE_SPLIT_SCREEN

    case 'V':
    {
        override_center_of_rotation = true;
    }

#if 1
    case 'v':
    {
        scm::math::mat4f cm = scm::math::inverse(scm::math::mat4f(active_camera_->trackball_matrix()));
        scm::math::vec3f cam_pos = scm::math::vec3f(cm[12], cm[13], cm[14]);
        scm::math::vec3f cam_fwd = -scm::math::normalize(scm::math::vec3f(cm[8], cm[9], cm[10]));
        scm::math::vec3f cam_right = scm::math::normalize(scm::math::vec3f(cm[4], cm[5], cm[6]));
        scm::math::vec3f cam_up = scm::math::normalize(scm::math::vec3f(cm[0], cm[1], cm[2]));

        float max_distance = 100000.0f;

        lamure::ren::ray::intersection intersectn;
        std::vector<lamure::ren::ray::intersection> dbg_intersections;
        lamure::ren::ray ray_brush(cam_pos, cam_fwd, max_distance);

        // sample params for single pick (wysiwg)
        unsigned int max_depth = 255;
        unsigned int surfel_skip = 1;
        float plane_dim = 0.11f; // e.g. 5.0 for valley, 0.05 for seradina rock

#if 0 /*INTERPOLATION PICK*/
                if (intersection_ray.intersect(1.0f, cam_up, plane_dim, max_depth, surfel_skip, intersectn)) {
#ifdef LAMURE_ENABLE_INFO
                    std::cout << "intersection distance: " << intersectn.distance_ << std::endl;
                    std::cout << "intersection position: " << intersectn.position_ << std::endl;
#endif
#ifndef LAMURE_RENDERING_USE_SPLIT_SCREEN
                    renderer_->clear_line_begin();
                    renderer_->clear_line_end();
                    scm::math::vec3f intersection_position = cam_pos + cam_fwd * intersectn.distance_;
                    renderer_->add_line_begin(intersection_position);
                    renderer_->add_line_end(intersection_position + intersectn.normal_ * 5.f);
                    //std::cout << "num debug intersections " << dbg_intersections.size() << std::endl;
                    for (const auto& dbg : dbg_intersections) {
                       renderer_->add_line_begin(dbg.position_);
                       renderer_->add_line_end(dbg.position_ - (cam_fwd) * 5.f);
                       //renderer_->add_line_end(dbg.position_ - dbg.normal_ * 5.f);

                    }
#endif
                }

#elif 1 /*SINGLE PICK SPLAT-BASED*/
 lamure::ren::model_database *database = lamure::ren::model_database::get_instance();
 for(lamure::model_t model_id = 0; model_id < database->num_models(); ++model_id)
 {
     scm::math::mat4f model_transform = database->get_model(model_id)->transform();
     lamure::ren::ray::intersection temp;
     if(ray_brush.intersect_model(model_id, model_transform, 1.0f, max_depth, surfel_skip, true, temp))
     {
         intersectn = temp;
     }
 }

#elif 0 /*SINGLE PICK BVH-BASED*/

        lamure::ren::model_database *database = lamure::ren::model_database::get_instance();
        for(lamure::model_t model_id = 0; model_id < database->num_models(); ++model_id)
        {
            scm::math::mat4f model_transform = database->get_model(model_id)->transform();
            lamure::ren::ray::Intersectionbvh temp;
            if(intersection_ray.intersect_model_bvh(model_id, model_transform, 1.0f, temp))
            {
                // std::cout << "hit i model id " << model_id << " distance: " << temp.tmin_ << std::endl;
                intersectn.position_ = temp.position_;
                intersectn.normal_ = scm::math::vec3f(0.0f, 1.0f, 0.f);
                intersectn.error_ = 0.f;
                intersectn.distance_ = temp.tmin_;
            }
        }

#else /*DISAMBIGUATION SINGLE PICK BVH-BASED*/

        // compile list of model kdn filenames
        lamure::ren::model_database *database = lamure::ren::model_database::get_instance();
        std::set<std::string> bvh_filenames;
        for(lamure::model_t model_id = 0; model_id < database->num_models(); ++model_id)
        {
            const std::string &bvh_filename = database->get_model(model_id)->get_bvh()->get_filename();
            bvh_filenames.insert(bvh_filename);
        }

        // now test the list of files
        lamure::ren::ray::intersection_bvh temp;
        if(intersection_ray.intersect_bvh(bvh_filenames, 1.0f, temp))
        {
            intersectn.position_ = temp.position_;
            intersectn.normal_ = scm::math::vec3f(0.f, 1.0f, 1.0f);
            intersectn.error_ = 0.f;
            intersectn.distance_ = temp.tmin_;
            std::cout << temp.bvh_filename_ << std::endl;
        }

#endif

#ifdef LAMURE_ENABLE_INFO
        std::cout << "intersection distance: " << intersectn.distance_ << std::endl;
        std::cout << "intersection position: " << intersectn.position_ << std::endl;
#endif

        if(intersectn.error_ < std::numeric_limits<float>::max())
        {
            renderer_->clear_line_begin();
            renderer_->clear_line_end();
            renderer_->add_line_begin(intersectn.position_);
            renderer_->add_line_end(intersectn.position_ + intersectn.normal_ * 5.f);
            // std::cout << "num debug intersections " << dbg_intersections.size() << std::endl;
            for(const auto &dbg : dbg_intersections)
            {
                renderer_->add_line_begin(dbg.position_);
                // renderer_->add_line_end(dbg.position_ - (cam_fwd) * 5.f);
                renderer_->add_line_end(dbg.position_ - dbg.normal_ * 5.f);
            }
        }

#if 1
        if(override_center_of_rotation)
        {
            // move center of rotation to intersection
            if(intersectn.error_ < std::numeric_limits<float>::max())
            {
                active_camera_->set_trackball_center_of_rotation(intersectn.position_);
            }
        }
#endif
    }

    break;

#endif

#endif

    case 'r':
    case 'R':
        toggle_camera_session();
        break;

    case 'a':
        record_next_camera_position();
        break;

    case '0':
        active_camera_->set_trackball_matrix(scm::math::mat4d(reset_matrix_));
        break;

    case '8':
        renderer_->toggle_use_user_defined_background_color();
        break;

    case '9':
        renderer_->toggle_display_info();
        break;

    case 'k':
        DecreaseErrorThreshold();
        std::cout << "error threshold: " << error_threshold_ << std::endl;
        break;
    case 'i':
        IncreaseErrorThreshold();
        std::cout << "error threshold: " << error_threshold_ << std::endl;
        break;


    }
}


void management::
dispatchSpecialInput(int key) {
  if(! allow_user_input_) {
    return;
  }
  /*
  switch (key) {

    // Change camera movement mode.
    case GLUT_KEY_F1:
        use_wasd_camera_control_scheme_ = !use_wasd_camera_control_scheme_;
        break;
    case GLUT_KEY_UP:
        is_moving_forward_ = true;
      break;
    case GLUT_KEY_DOWN:
        is_moving_backward_ = true;
      break;
    case GLUT_KEY_LEFT:
        is_moving_left_ = true;
      break;
    case GLUT_KEY_RIGHT:
        is_moving_right_ = true;
      break;

  }*/
}

void management::
dispatchSpecialInputRelease(int key) {
  if(! allow_user_input_) {
    return;
  }
  /*
  switch (key) {

    case GLUT_KEY_UP:
        is_moving_forward_ = false;
      break;
    case GLUT_KEY_DOWN:
        is_moving_backward_ = false;
      break;
    case GLUT_KEY_LEFT:
        is_moving_left_ = false;
      break;
    case GLUT_KEY_RIGHT:
        is_moving_right_ = false;
      break;

  }*/
}

void management::
dispatchResize(int w, int h)
{
    width_ = w;
    height_ = h;

#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
    w /= 2;
#endif

    // if snapshots are taken, use the user specified resolution
    if(measurement_session_descriptor_.snapshot_session_enabled_)
    {
        renderer_->reset_viewport(measurement_session_descriptor_.snapshot_resolution_[0], measurement_session_descriptor_.snapshot_resolution_[1]);
    }
    else
    { // otherwise react on window resizing
        renderer_->reset_viewport(w, h);
    }
    lamure::ren::policy *policy = lamure::ren::policy::get_instance();
    policy->set_window_width(w);
    policy->set_window_height(h);

    for(auto &cam : cameras_)
    {
        if(measurement_session_descriptor_.snapshot_session_enabled_)
        {
            cam->set_projection_matrix(30.0f, float(measurement_session_descriptor_.snapshot_resolution_[0]) / float(measurement_session_descriptor_.snapshot_resolution_[1]), near_plane_, far_plane_);
        }
        else
        {
            cam->set_projection_matrix(30.0f, float(w) / float(h), near_plane_, far_plane_);
        }
    }
}


void management::
forward_background_color(float bg_r, float bg_g, float bg_b) {
    renderer_->set_user_defined_background_color(bg_r, bg_g, bg_b);
}

void management::
Toggledispatching()
{
    dispatch_ = ! dispatch_;
};

void management::DecreaseErrorThreshold()
{
    error_threshold_ -= 0.1f;
    if(error_threshold_ < LAMURE_MIN_THRESHOLD)
        error_threshold_ = LAMURE_MIN_THRESHOLD;
}

void management::IncreaseErrorThreshold()
{
    error_threshold_ += 0.1f;
    if(error_threshold_ > LAMURE_MAX_THRESHOLD)
        error_threshold_ = LAMURE_MAX_THRESHOLD;
}

void management::PrintInfo()
{
    std::cout << "\n"
              << "Controls: w - enable/disable bounding box rendering\n"
              << "\n"
              << "          U/u - increase point size by 1.0/0.1\n"
              << "          J/j - decrease point size by 1.0/0.1\n"
              << "\n"
              << "          o - switch to circle/ellipse rendering\n"
              << "          c - toggle normal clamping\n"
              << "\n"
              << "          A/a - increase clamping ratio by 0.1/0.01f\n"
              << "          S/s - decrease clamping ratio by 0.1/0.01f\n"
              << "\n"
              << "          d - toggle dispatching\n"
              << "          e - trigger 1 dispatch, if dispatch is frozen\n"
              << "          f - toggle fast travel\n"
              << "          . - toggle fullscreen\n"
              << "\n"
              << "          i - increase error threshold\n"
              << "          k - decrease error threshold\n"
              << "\n"
              << "          + (NUMPAD) - increase importance\n"
              << "          - (NUMPAD) - decrease importance\n"
              <<

#ifdef LAMURE_RENDERING_ENABLE_MULTI_VIEW_TEST
        "\n"
              << "          x - add camera\n"
              << "          Space - switch to next camera\n"
              <<
#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
        "\n"
              << "          1 - control left screen"
              << "\n"
              << "          2 - control right screen"
              <<
#endif
#endif

        "\n";
}

void management::toggle_camera_session() { camera_recording_enabled_ = !camera_recording_enabled_; }

void management::record_next_camera_position()
{
    if(camera_recording_enabled_)
    {
        create_quality_measurement_resources();
    }
}

void management::create_quality_measurement_resources()
{
    std::string base_quality_measurement_path = "../quality_measurement/";
    std::string session_file_prefix = "session_";

    if(!boost::filesystem::exists(base_quality_measurement_path))
    {
        std::cout << "Creating Folder.\n\n";
        boost::filesystem::create_directories(base_quality_measurement_path);
    }

    if(current_session_file_path_.empty())
    {
        boost::filesystem::directory_iterator begin(base_quality_measurement_path), end;

        int num_existing_sessions = std::count_if(begin, end, [](const boost::filesystem::directory_entry &d) { return !boost::filesystem::is_directory(d.path()); });

        current_session_number_ = num_existing_sessions + 1;

        current_session_file_path_ = base_quality_measurement_path + session_file_prefix + std::to_string(current_session_number_) + ".csn";
    }

    std::ofstream camera_session_file(current_session_file_path_, std::ios_base::out | std::ios_base::app);
    active_camera_->write_view_matrix(camera_session_file);
    camera_session_file.close();
}

void management::
interpolate_between_measurement_transforms(const bool& allow_interpolation)
{
    use_interpolation_on_measurement_session_ = allow_interpolation;
}

void management::
set_interpolation_step_size(const float& interpolation_step_size)
{
    movement_on_interpolation_per_frame_ = interpolation_step_size;
}

void management::
enable_culling(const bool& enable)
{
    renderer_->enable_culling(enable);
}
