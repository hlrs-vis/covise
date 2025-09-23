// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/renderer.h"

#include <ctime>
#include <chrono>

#include <lamure/config.h>


#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <FreeImagePlus.h>

#include <scm/gl_core/render_device/opengl/gl_core.h>

#include "lamure/pvs/pvs_database.h"

#define NUM_BLENDED_FRAGS 18
#define RENDER_TO_SCREEN

Renderer::
Renderer(std::vector<scm::math::mat4f> const& model_transformations,
         const std::set<lamure::model_t>& visible_set,
         const std::set<lamure::model_t>& invisible_set)
    : near_plane_(0.f),
      far_plane_(1000.0f),
      point_size_factor_(1.0f),
      blending_threshold_(0.01f),
      render_bounding_boxes_(false),
      elapsed_ms_since_cut_update_(0),
      render_mode_(RenderMode::VISIBLE_NODE_PASS),
      visible_set_(visible_set),
      invisible_set_(invisible_set),
      render_visible_set_(true),
      fps_(0.0),
      rendered_splats_(0),
      is_cut_update_active_(true),
      current_cam_id_(0),
      display_info_(true),
      model_transformations_(model_transformations),
      radius_scale_(1.f)
{

    lamure::ren::policy* policy = lamure::ren::policy::get_instance();
    win_x_ = policy->window_width();
    win_y_ = policy->window_height();

    initialize_schism_device_and_shaders(win_x_, win_y_);
    initialize_VBOs();
    reset_viewport(win_x_, win_y_);

    calculate_radius_scale_per_model();
}

Renderer::
~Renderer()
{
    filter_nearest_.reset();
    color_blending_state_.reset();

    depth_state_disable_.reset();


    visible_node_shader_program_.reset();
    node_texture_shader_program_.reset();

    visible_node_id_fbo_.reset();
    visible_node_depth_buffer_.reset();
    visible_node_id_texture_.reset();

    screen_quad_.reset();

    context_.reset();
    device_.reset();
}

void Renderer::
upload_uniforms(lamure::ren::camera const& camera) const
{
    using namespace lamure::ren;
    using namespace scm::gl;
    using namespace scm::math;

    //model_database* database = model_database::get_instance();
    //uint32_t number_of_surfels_per_node = database->get_primitives_per_node();
    //unsigned num_blend_f = NUM_BLENDED_FRAGS;

    // visible node pass
    visible_node_shader_program_->uniform("near_plane", near_plane_);
    visible_node_shader_program_->uniform("far_plane", far_plane_);
    visible_node_shader_program_->uniform("point_size_factor", point_size_factor_);

    // render node id pass
    node_texture_shader_program_->uniform_sampler("in_color_texture", 0);

    context_->clear_default_color_buffer(FRAMEBUFFER_BACK, vec4f(0.0f, 0.0f, .0f, 1.0f)); // how the image looks like, if nothing is drawn
    context_->clear_default_depth_stencil_buffer();

    context_->apply();
}

void Renderer::
upload_transformation_matrices(lamure::ren::camera const& camera, lamure::model_t const model_id, RenderPass const pass) const {
    using namespace lamure::ren;

    scm::math::mat4f    model_matrix        = model_transformations_[model_id];
    scm::math::mat4f    projection_matrix   = camera.get_projection_matrix();

#if 1
    scm::math::mat4d    vm = camera.get_high_precision_view_matrix();
    scm::math::mat4d    mm = scm::math::mat4d(model_matrix);
    scm::math::mat4d    vmd = vm * mm;
    
    scm::math::mat4f    model_view_matrix = scm::math::mat4f(vmd);

    scm::math::mat4d    mvpd = scm::math::mat4d(projection_matrix) * vmd;
    
#define DEFAULT_PRECISION 31
#else
    scm::math::mat4f    model_view_matrix   = view_matrix * model_matrix;
#endif

    float total_radius_scale = radius_scale_;// * radius_scale_per_model_[model_id];

    switch(pass)
    {         
        case RenderPass::VISIBLE_NODE:
            visible_node_shader_program_->uniform("mvp_matrix", scm::math::mat4f(mvpd) );
            visible_node_shader_program_->uniform("model_view_matrix", model_view_matrix);
            visible_node_shader_program_->uniform("inv_mv_matrix", scm::math::mat4f(scm::math::transpose(scm::math::inverse(vmd))));
            visible_node_shader_program_->uniform("model_radius_scale", total_radius_scale);
        break;

        default:
            //LOGGER_ERROR("Unknown Pass ID used in function 'upload_transformation_matrices'");
            std::cout << "Unknown Pass ID used in function 'upload_transformation_matrices'\n";
            break;

    }

    context_->apply();
}

void Renderer::
render_depth(lamure::context_t context_id, 
            lamure::ren::camera const& camera, 
            const lamure::view_t view_id, 
            scm::gl::vertex_array_ptr const& render_VAO, 
            std::set<lamure::model_t> const& current_set, std::vector<uint32_t>& frustum_culling_results)
{
    using namespace lamure;
    using namespace lamure::ren;

    using namespace scm::gl;
    using namespace scm::math;

    cut_database* cuts = cut_database::get_instance();
    model_database* database = model_database::get_instance();

    size_t number_of_surfels_per_node = database->get_primitives_per_node();

    /***************************************************************************************
    *******************************BEGIN DEPTH PASS*****************************************
    ****************************************************************************************/

    {
        context_->clear_depth_stencil_buffer(visible_node_id_fbo_);
        context_->clear_color_buffer(visible_node_id_fbo_, 0, vec4f(0.0f, 0.0f, 0.0f, 0.0f));
        
        context_->set_frame_buffer(visible_node_id_fbo_);

        context_->set_rasterizer_state(no_backface_culling_rasterizer_state_);
        context_->set_viewport(viewport(vec2ui(0, 0), 1 * vec2ui(win_x_, win_y_)));

        context_->bind_program(visible_node_shader_program_);


        context_->bind_vertex_array(render_VAO);
        //context_->bind_texture(visible_node_id_texture_, filter_nearest_, 0);
        context_->apply();

        node_t actually_rendered_nodes = 0;

        for (auto& model_id : current_set)
        {
            cut& cut = cuts->get_cut(context_id, view_id, model_id);

            std::vector<cut::node_slot_aggregate> renderable = cut.complete_set();

            const bvh* bvh = database->get_model(model_id)->get_bvh();

            if (bvh->get_primitive() != bvh::primitive_type::POINTCLOUD)
            {
                continue;
            }

            size_t surfels_per_node_of_model = bvh->get_primitives_per_node();
            std::vector<scm::gl::boxf>const & bounding_box_vector = bvh->get_bounding_boxes();


            upload_transformation_matrices(camera, model_id, RenderPass::VISIBLE_NODE);

            scm::gl::frustum frustum_by_model = camera.get_frustum_by_model(model_transformations_[model_id]);

            // Set model ID so it may be rendered to the resulting image in the Alpha-channel.
            visible_node_shader_program_->uniform("model_id", (GLint)model_id );

            for(auto const& node_slot_aggregate : renderable)
            {
                uint32_t node_culling_result = camera.cull_against_frustum( frustum_by_model ,bounding_box_vector[ node_slot_aggregate.node_id_ ] );

                // Set node ID so it may be rendered to the resulting image in the RGB-channel.
                visible_node_shader_program_->uniform("node_id", (GLint)node_slot_aggregate.node_id_ );

                if( (node_culling_result != 1) )
                {
                    // If the app is not used as preprocessor, but as renderer, the current visibility may already be applied.
                    lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();
                    if(!pvs->get_viewer_visibility(model_id, node_slot_aggregate.node_id_))
                    {
                        continue;
                    }

                    context_->apply();
#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
                    scm::gl::timer_query_ptr depth_pass_timer_query = device_->create_timer_query();
                    context_->begin_query(depth_pass_timer_query);
#endif

                    context_->draw_arrays(PRIMITIVE_POINT_LIST, (node_slot_aggregate.slot_id_) * number_of_surfels_per_node, surfels_per_node_of_model);

#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
                    context_->collect_query_results(depth_pass_timer_query);
                    depth_pass_time += depth_pass_timer_query->result();
#endif
                    ++actually_rendered_nodes;
                }
            }
       }

       rendered_splats_ = actually_rendered_nodes * database->get_primitives_per_node();
    }

#ifdef RENDER_TO_SCREEN
    /***************************************************************************************
    *******************************BEGIN TEXTURE DRAW PASS**********************************
    ****************************************************************************************/
    {
        context_->clear_default_color_buffer(FRAMEBUFFER_BACK, vec4f(0.0f, 0.0f, .0f, 1.0f));
        context_->clear_default_depth_stencil_buffer();

        context_->set_default_frame_buffer();
        context_->bind_program(node_texture_shader_program_);
        context_->bind_texture(visible_node_id_texture_, filter_nearest_, 0);
        context_->apply();

    #ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
        scm::gl::timer_query_ptr hole_filling_pass_timer_query = device_->create_timer_query();
        context_->begin_query(hole_filling_pass_timer_query);
    #endif

        screen_quad_->draw(context_);

    #ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
        context_->end_query(hole_filling_pass_timer_query);
        context_->collect_query_results(hole_filling_pass_timer_query);
        hole_filling_pass_time += hole_filling_pass_timer_query->result();
    #endif
    }
#endif
}


std::string const Renderer::
strip_whitespace(std::string const& in_string) {
  return boost::regex_replace(in_string, boost::regex("^ +| +$|( ) +"), "$1");

}

//checks for prefix AND removes it (+ whitespace) if it is found; 
//returns true, if prefix was found; else false
bool const Renderer::
parse_prefix(std::string& in_string, std::string const& prefix) {

 uint32_t num_prefix_characters = prefix.size();

 bool prefix_found 
  = (!(in_string.size() < num_prefix_characters ) 
     && strncmp(in_string.c_str(), prefix.c_str(), num_prefix_characters ) == 0); 

  if( prefix_found ) {
    in_string = in_string.substr(num_prefix_characters);
    in_string = strip_whitespace(in_string);
  }

  return prefix_found;
}

bool Renderer::
read_shader(std::string const& path_string, 
                 std::string& shader_string) {


  if ( !boost::filesystem::exists( path_string ) ) {
    std::cout << "WARNING: File " << path_string << "does not exist." <<  std::endl;
    return false;
  }

  std::ifstream shader_source(path_string, std::ios::in);
  std::string line_buffer;

  std::string include_prefix("INCLUDE");

  std::size_t slash_position = path_string.find_last_of("/\\");
  std::string const base_path =  path_string.substr(0,slash_position+1);

  while( std::getline(shader_source, line_buffer) ) {
    line_buffer = strip_whitespace(line_buffer);
    //std::cout << line_buffer << "\n";

    if( parse_prefix(line_buffer, include_prefix) ) {
      std::string filename_string = line_buffer;
      read_shader(base_path+filename_string, shader_string);
    } else {
      shader_string += line_buffer+"\n";
    }
  }

  return true;
}

void Renderer::
render(lamure::context_t context_id, lamure::ren::camera const& camera, const lamure::view_t view_id, scm::gl::vertex_array_ptr render_VAO, const unsigned current_camera_session)
{
    using namespace lamure;
    using namespace lamure::ren;

    update_frustum_dependent_parameters(camera);
    upload_uniforms(camera);

    using namespace scm::gl;
    using namespace scm::math;

    model_database* database = model_database::get_instance();
    cut_database* cuts = cut_database::get_instance();

    model_t num_models = database->num_models();

    //determine set of models to render
    std::set<lamure::model_t> current_set;
    for (lamure::model_t model_id = 0; model_id < num_models; ++model_id)
    {
        auto vs_it = visible_set_.find(model_id);
        auto is_it = invisible_set_.find(model_id);

        if (vs_it == visible_set_.end() && is_it == invisible_set_.end())
        {
            current_set.insert(model_id);
        }
        else if (vs_it != visible_set_.end())
        {
            if (render_visible_set_)
            {
                current_set.insert(model_id);
            }
        }
        else if (is_it != invisible_set_.end())
        {
            if (!render_visible_set_)
            {
                current_set.insert(model_id);
            }
        }
    }


    rendered_splats_ = 0;

    std::vector<uint32_t>                       frustum_culling_results;

    uint32_t size_of_culling_result_vector = 0;

    for (auto& model_id : current_set) {
        cut& cut = cuts->get_cut(context_id, view_id, model_id);

        std::vector<cut::node_slot_aggregate> renderable = cut.complete_set();

        size_of_culling_result_vector += renderable.size();
    }

     frustum_culling_results.clear();
     frustum_culling_results.resize(size_of_culling_result_vector);

#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
     size_t depth_pass_time = 0;
     size_t accumulation_pass_time = 0;
     size_t normalization_pass_time = 0;
     size_t hole_filling_pass_time = 0;
#endif

    {
        switch(render_mode_)
        {
            case (RenderMode::VISIBLE_NODE_PASS):
                render_depth(context_id, 
                            camera, 
                            view_id, 
                            render_VAO, 
                            current_set, 
                            frustum_culling_results);
                break;
        }


        context_->reset();
        frame_time_.stop();
        frame_time_.start();

        if (true)
        {
            //schism bug ? time::to_seconds yields milliseconds
            if (scm::time::to_seconds(frame_time_.accumulated_duration()) > 100.0)
            {
                fps_ = 1000.0f / scm::time::to_seconds(frame_time_.average_duration());
                frame_time_.reset();
            }
        }
        //if(display_info_)
        //display_status(current_camera_session);

        context_->reset();
    }

#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
uint64_t total_time = depth_pass_time + accumulation_pass_time + normalization_pass_time + hole_filling_pass_time;

std::cout << "depth pass        : " << depth_pass_time / ((float)(1000000)) << "ms (" << depth_pass_time        /( (float)(total_time) ) << ")\n"
          << "accumulation pass : " << accumulation_pass_time / ((float)(1000000)) << "ms (" << accumulation_pass_time /( (float)(total_time) )<< ")\n"
          << "normalization pass: " << normalization_pass_time / ((float)(1000000)) << "ms (" << normalization_pass_time/( (float)(total_time) )<< ")\n"
          << "hole filling  pass: " << hole_filling_pass_time / ((float)(1000000)) << "ms (" << hole_filling_pass_time /( (float)(total_time) )<< ")\n\n";

#endif
}


void Renderer::send_model_transform(const lamure::model_t model_id, const scm::math::mat4f& transform) {
    model_transformations_[model_id] = transform;
}

void Renderer::display_status(std::string const& information_to_display)
{
    std::stringstream os;
   // os.setprecision(5);
    os
      <<"FPS:   "<<std::setprecision(4)<<fps_<<"\n"
      //<<"# Points:   "<< (rendered_splats_ / 100000) / 10.0f<< " Mio. \n" 
      <<"# Nodes:   "<< rendered_splats_ / lamure::ren::model_database::get_instance()->get_primitives_per_node() << "\n"
      <<"Render Mode: " ;
      switch(render_mode_) {
        case(RenderMode::VISIBLE_NODE_PASS):
            os << "Visible Node\n";
            break;

        default:
            os << "RenderMode not implemented\n";
      }
      
    os << information_to_display;
    os << "\n";
    
    renderable_text_->text_string(os.str());
    text_renderer_->draw_shadowed(context_, scm::math::vec2i(20, win_y_- 40), renderable_text_);
}

void Renderer::
initialize_VBOs()
{
    // init the GL context
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;


    filter_nearest_ = device_->create_sampler_state(FILTER_MIN_MAG_LINEAR, WRAP_CLAMP_TO_EDGE);

    no_backface_culling_rasterizer_state_ = device_->create_rasterizer_state(FILL_SOLID, CULL_NONE, ORIENT_CCW, false, false, 0.0, false, false);

    visible_node_id_fbo_ = device_->create_frame_buffer();
    visible_node_depth_buffer_ = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_D24, 1, 1, 1);
    visible_node_id_fbo_->attach_depth_stencil_buffer(visible_node_depth_buffer_);

    visible_node_id_texture_ = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_RGBA_8UI , 1, 1, 1);
    visible_node_id_fbo_->attach_color_buffer(0, visible_node_id_texture_);

    screen_quad_.reset(new quad_geometry(device_, vec2f(-1.0f, -1.0f), vec2f(1.0f, 1.0f)));


    color_blending_state_ = device_->create_blend_state(true, FUNC_ONE, FUNC_ONE, FUNC_ONE, FUNC_ONE, EQ_FUNC_ADD, EQ_FUNC_ADD);


    depth_state_disable_ = device_->create_depth_stencil_state(false, true, scm::gl::COMPARISON_NEVER);

    depth_state_test_without_writing_ = device_->create_depth_stencil_state(true, false, scm::gl::COMPARISON_LESS_EQUAL);
}

bool Renderer::
initialize_schism_device_and_shaders(int resX, int resY)
{
    std::string root_path = LAMURE_SHADERS_DIR;
    char * val;
        val = getenv( "COVISEDIR" );
        root_path=val;
        root_path=root_path+"/share/covise/shaders";


    std::string node_visibility_vs_source;
    std::string node_visibility_gs_source;
    std::string node_visibility_fs_source;

    std::string node_texture_vs_source;
    std::string node_texture_fs_source;

    try
    {
        using scm::io::read_text_file;

        if (!read_shader(root_path + "/pvs/node_visibility.glslv", node_visibility_vs_source)
            || !read_shader(root_path + "/pvs/node_visibility.glslg", node_visibility_gs_source)
            || !read_shader(root_path + "/pvs/node_visibility.glslf", node_visibility_fs_source)
            || !read_shader(root_path + "/pvs/node_render_texture.glslv", node_texture_vs_source)
            || !read_shader(root_path + "/pvs/node_render_texture.glslf", node_texture_fs_source)
           )
           {
               scm::err() << "error reading shader files" << scm::log::end;
               return false;
           }
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    device_.reset(new scm::gl::render_device());
    context_ = device_->main_context();
    scm::out() << *device_ << scm::log::end;

    visible_node_shader_program_ = device_->create_program(
        boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER,   node_visibility_vs_source ))
                              (device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, node_visibility_gs_source ))
                              (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, node_visibility_fs_source ))
    );

    node_texture_shader_program_ = device_->create_program(
        boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER,   node_texture_vs_source ))
                              (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, node_texture_fs_source ))
    );

    if (!visible_node_shader_program_
         || !node_texture_shader_program_
       )
    {
        scm::err() << "error creating shader programs" << scm::log::end;
        return false;
    }


    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    try {
        font_face_ptr output_font(new font_face(device_, std::string(LAMURE_FONTS_DIR) + "/Ubuntu.ttf", 30, 0, font_face::smooth_lcd));
        text_renderer_  =     scm::make_shared<text_renderer>(device_);
        renderable_text_    = scm::make_shared<scm::gl::text>(device_, output_font, font_face::style_regular, "sick, sad world...");

        mat4f   fs_projection = make_ortho_matrix(0.0f, static_cast<float>(win_x_),
                                                  0.0f, static_cast<float>(win_y_), -1.0f, 1.0f);
        text_renderer_->projection_matrix(fs_projection);

        renderable_text_->text_color(math::vec4f(1.0f, 1.0f, 1.0f, 1.0f));
        renderable_text_->text_kerning(true);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(std::string("vtexture_system::vtexture_system(): ") + e.what());
    }

    return true;
}

void Renderer::reset_viewport(int w, int h)
{
    //reset viewport
    win_x_ = w;
    win_y_ = h;
    context_->set_viewport(scm::gl::viewport(scm::math::vec2ui(0, 0), scm::math::vec2ui(w, h)));

    //reset frame buffers and textures
    visible_node_id_fbo_ = device_->create_frame_buffer();
    visible_node_depth_buffer_ = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_D24, 1, 1, 1);
    visible_node_id_fbo_->attach_depth_stencil_buffer(visible_node_depth_buffer_);

    visible_node_id_texture_ = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_RGBA_8UI , 1, 1, 1);
    visible_node_id_fbo_->attach_color_buffer(0, visible_node_id_texture_);

    //reset orthogonal projection matrix for text rendering
    scm::math::mat4f   fs_projection = scm::math::make_ortho_matrix(0.0f, static_cast<float>(win_x_),
                                                                    0.0f, static_cast<float>(win_y_), -1.0f, 1.0f);
    text_renderer_->projection_matrix(fs_projection);
}

void Renderer::
update_frustum_dependent_parameters(lamure::ren::camera const& camera)
{
    near_plane_ = camera.near_plane_value();
    far_plane_  = camera.far_plane_value();

    std::vector<scm::math::vec3d> corner_values = camera.get_frustum_corners();
    double top_minus_bottom = scm::math::length((corner_values[2]) - (corner_values[0]));

    height_divided_by_top_minus_bottom_ = win_y_ / top_minus_bottom;
}

void Renderer::
calculate_radius_scale_per_model()
{
    using namespace lamure::ren;
    uint32_t num_models = (model_database::get_instance())->num_models();

    if(radius_scale_per_model_.size() < num_models)
      radius_scale_per_model_.resize(num_models);

    scm::math::vec4f x_unit_vec = scm::math::vec4f(1.0,0.0,0.0,0.0);
    for(unsigned int model_id = 0; model_id < num_models; ++model_id)
    {
     radius_scale_per_model_[model_id] = scm::math::length(model_transformations_[model_id] * x_unit_vec);
    }
}

//dynamic rendering adjustment functions
void Renderer::
toggle_bounding_box_rendering()
{
    render_bounding_boxes_ = ! render_bounding_boxes_;

    std::cout<<"bounding box visualisation: ";
    if(render_bounding_boxes_)
        std::cout<<"ON\n\n";
    else
        std::cout<<"OFF\n\n";
};

void Renderer::
change_point_size(float amount)
{
    point_size_factor_ += amount;
    if(point_size_factor_ < 0.0001f)
    {
        point_size_factor_ = 0.0001;
    }

    std::cout<<"set point size factor to: "<<point_size_factor_<<"\n\n";
}

void Renderer::
toggle_cut_update_info()
{
    is_cut_update_active_ = ! is_cut_update_active_;
}

void Renderer::
toggle_camera_info(const lamure::view_t current_cam_id)
{
    current_cam_id_ = current_cam_id;
}

void Renderer::
toggle_display_info()
{
    display_info_ = ! display_info_;
}

void Renderer::
toggle_visible_set()
{
    render_visible_set_ = !render_visible_set_;
}

void Renderer::
switch_render_mode(RenderMode const& render_mode)
{
    render_mode_ = render_mode;
}

lamure::pvs::id_histogram Renderer::
create_node_id_histogram(const bool& save_screenshot, const int& image_index) const
{
    // Make the BYTE array, factor of 4 because it's RGBA.
    GLubyte* pixels = new GLubyte[4 * win_x_ * win_y_];

    device_->opengl_api().glBindTexture(GL_TEXTURE_2D, visible_node_id_texture_->object_id());
    device_->opengl_api().glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, pixels);

    lamure::pvs::id_histogram hist;
    hist.create(pixels, win_x_ * win_y_);

    if(save_screenshot)
    {
        std::string screenshot_path = "/home/tiwo9285/";
        std::string screenshot_name = "test" + std::to_string(image_index);
        std::string file_extension = ".png";

        std::string full_path = screenshot_path + "/";
        {
            if(! boost::filesystem::exists(full_path)) {
               std::cout<<"Screenshot Folder did not exist. Creating Folder: " << full_path << "\n\n";
               boost::filesystem::create_directories(full_path);
            }
        }

        std::string filename = full_path + "color__" + screenshot_name + "__surfels_" + file_extension;

        FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, win_x_, win_y_, 4 * win_x_, 32, 0x0000FF, 0x00FF00, 0xFF0000, false);
        FreeImage_Save(FIF_PNG, image, filename.c_str(), 0);

        device_->opengl_api().glBindTexture(GL_TEXTURE_2D, 0);

        std::cout<<"Saved Screenshot: "<<filename.c_str()<<"\n\n";

        // Free resources
        FreeImage_Unload(image);
    }

    delete [] pixels;

    return hist;
}

// Debug stuff to output rendered nodes as histogram and check if histogram is valid within current cut.
void Renderer::
compare_histogram_to_cut(const lamure::pvs::id_histogram& hist, const float& visibility_threshold)
{
    int numPixels = win_x_ * win_y_;
    std::map<lamure::model_t, std::vector<lamure::node_t>> visible_nodes = hist.get_visible_nodes(numPixels, visibility_threshold);

    std::fstream f;
    f.open("/home/tiwo9285/test.txt", std::ios::out);
    for(std::map<lamure::model_t, std::map<lamure::node_t, size_t>>::iterator modelIter = hist.get_histogram().begin(); modelIter != hist.get_histogram().end(); ++modelIter)
    {
        for(std::map<lamure::node_t, size_t>::iterator nodeIter = modelIter->second.begin(); nodeIter != modelIter->second.end(); ++nodeIter)
        {
            f << "model: " << modelIter->first << "  node: " << nodeIter->first << "  amount: " << nodeIter->second << "  percent: " << ((float)nodeIter->second / (float)numPixels) * 100.0f << std::endl;
        }
    }

    f << "\n";

    lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    lamure::context_t context_id = 0;
    lamure::view_t view_id = 0;

    lamure::node_t node_counter = 0;
    lamure::node_t regular_node_counter = 0;
    lamure::node_t irregular_node_counter = 0;

    for (lamure::model_t model_id = 0; model_id < database->num_models(); ++model_id)
    {
        std::vector<lamure::node_t> rendered_nodes;

        lamure::ren::cut& cut = cuts->get_cut(context_id, view_id, model_id);
        std::vector<lamure::ren::cut::node_slot_aggregate> renderable = cut.complete_set();

        // Count nodes in the current cut.
        for(auto const& node_slot_aggregate : renderable)
        {
            rendered_nodes.push_back(node_slot_aggregate.node_id_);
            ++node_counter;
        }

        // Check rendered nodes against current cut. Count rendered nodes inside cut (regular) and not inside cut (irregular).
        for(size_t index = 0; index < visible_nodes[model_id].size(); ++index)
        {
            lamure::node_t irregular_node_id = visible_nodes[model_id][index];

            if(std::find(rendered_nodes.begin(), rendered_nodes.end(), irregular_node_id) == rendered_nodes.end())
            {
                f << irregular_node_id << "\n";
                irregular_node_counter++;
            }
            else
            {
                regular_node_counter++;
            }
        }
    }

    f << "cut nodes total: " << node_counter << std::endl;
    f << "regular rendered nodes: " << regular_node_counter << std::endl;
    f << "irregular rendered nodes: " << irregular_node_counter << std::endl;

    f.close();   
}

int Renderer::
get_rendered_node_count() const
{
    return rendered_splats_ / lamure::ren::model_database::get_instance()->get_primitives_per_node();
}
