// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "renderer.h"

#include <ctime>
#include <chrono>
#include <lamure/config.h>

#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <FreeImagePlus.h>

#include <scm/gl_core/render_device/opengl/gl_core.h>

#include <lamure/pvs/pvs_database.h>
#include <lamure/pvs/view_cell.h>

#define NUM_BLENDED_FRAGS 18

Renderer::
Renderer(std::vector<scm::math::mat4f> const& model_transformations,
         const std::set<lamure::model_t>& visible_set,
         const std::set<lamure::model_t>& invisible_set)
    : near_plane_(0.f),
      far_plane_(1000.0f),
      point_size_factor_(1.0f),
      blending_threshold_(0.01f),
      render_provenance_(0),
      do_measurement_(false),
      use_user_defined_background_color_(true),
      render_bounding_boxes_(false),
      elapsed_ms_since_cut_update_(0),
      render_mode_(RenderMode::HQ_TWO_PASS),
      visible_set_(visible_set),
      invisible_set_(invisible_set),
      render_visible_set_(true),
      fps_(0.0),
      rendered_splats_(0),
      is_cut_update_active_(true),
      current_cam_id_(0),
      display_info_(true),
#ifdef LAMURE_ENABLE_LINE_VISUALIZATION
      max_lines_(256),
#endif
      model_transformations_(model_transformations),
      radius_scale_(1.f),
      measurement_()
{
    render_pvs_grid_cells_ = false;
    render_occluded_geometry_ = false;

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

    pass1_visibility_shader_program_.reset();
    pass1_compressed_visibility_shader_program_.reset();
    pass2_accumulation_shader_program_.reset();
    pass2_compressed_accumulation_shader_program_.reset();
    pass3_pass_through_shader_program_.reset();

    trimesh_shader_program_.reset();

    bounding_box_vis_shader_program_.reset();
    pvs_grid_cell_vis_shader_program_.reset();

    pass1_depth_buffer_.reset();
    pass1_visibility_fbo_.reset();

    pass2_accumulated_color_buffer_.reset();

    pass2_accumulation_fbo_.reset();

    pass3_normalization_color_texture_.reset();
    pass3_normalization_normal_texture_.reset();

    pass1_linked_list_accumulate_program_.reset();
    pass2_linked_list_resolve_program_.reset();
    pass3_repair_program_.reset();

    LQ_one_pass_program_.reset();
    compressed_LQ_one_pass_program_.reset();

    screen_quad_.reset();

    context_.reset();
    device_.reset();

    min_es_distance_image_.reset();
    linked_list_buffer_texture_.reset();
}

void Renderer::
bind_storage_buffer(scm::gl::buffer_ptr buffer) {
    
    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    lamure::ren::policy*         policy   = lamure::ren::policy::get_instance();

    size_t size_of_nodes_in_bytes = database->get_primitive_size(lamure::ren::bvh::primitive_type::POINTCLOUD) * database->get_primitives_per_node();
    size_t render_budget_in_mb    = policy->render_budget_in_mb();

    size_t num_slots              = (render_budget_in_mb * 1024u * 1024u) / size_of_nodes_in_bytes;
    
    pass2_linked_list_resolve_program_->storage_buffer("point_attrib_ssbo", 0);
    context_->bind_storage_buffer(buffer, 0, 0, num_slots * size_of_nodes_in_bytes);
}

void Renderer::
bind_bvh_attributes_for_compression_ssbo_buffer(scm::gl::buffer_ptr& buffer, lamure::context_t context_id, std::set<lamure::model_t> const& current_set, lamure::view_t const view_id) {
    
    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
    lamure::ren::policy*         policy   = lamure::ren::policy::get_instance();

    size_t size_of_nodes_in_bytes = database->get_primitive_size(lamure::ren::bvh::primitive_type::POINTCLOUD_QZ) * database->get_primitives_per_node();
    size_t render_budget_in_mb    = policy->render_budget_in_mb();

    size_t num_slots              = (render_budget_in_mb * 1024u * 1024u) / size_of_nodes_in_bytes;
    size_t size_of_node_compression_slot = sizeof(float) * 8;
    if(nullptr == buffer) {
        std::cout << "INITIALIZED SSBO! \n";
        buffer = device_->create_buffer(scm::gl::BIND_STORAGE_BUFFER,
                                         scm::gl::USAGE_DYNAMIC_DRAW,
                                         num_slots * size_of_node_compression_slot,
                                         0);

        bvh_ssbo_cpu_data[int(context_id)].resize(num_slots*8);
    }


    //float* mapped_ssbo = (float*)device_->main_context()->map_buffer(buffer, scm::gl::access_mode::ACCESS_READ_WRITE);
    float* mapped_ssbo = (float*)device_->main_context()->map_buffer(buffer, scm::gl::access_mode::ACCESS_WRITE_ONLY);
    std::map<lamure::slot_t, std::pair<lamure::model_t, lamure::node_t> > fast_update_index_map;
    for (auto& model_id : current_set) {
        lamure::ren::cut& cut = cuts->get_cut(context_id, view_id, model_id);
        //only nodes that are in the cut could have been updated. Therefore, we write the complete cut into the map.


        const lamure::ren::bvh*  bvh = database->get_model(model_id)->get_bvh();

        for(auto const& node_slot_index_pair : cut.complete_set() ) {
            //std::cout << node_slot_index_pair.node_id_ << "\n";
            auto const& current_bounding_box = bvh->get_bounding_boxes()[node_slot_index_pair.node_id_];
            float avg_surfel_radius = bvh->get_avg_primitive_extent(node_slot_index_pair.node_id_);
            float max_radius_deviation = bvh->get_max_surfel_radius_deviation(node_slot_index_pair.node_id_);

            float min_radius = avg_surfel_radius - max_radius_deviation;
            float max_radius = avg_surfel_radius + max_radius_deviation;

            float bvh_data_to_write[8];
            bvh_data_to_write[0] = current_bounding_box.min_vertex()[0];
            bvh_data_to_write[1] = current_bounding_box.min_vertex()[1];
            bvh_data_to_write[2] = current_bounding_box.min_vertex()[2];
            bvh_data_to_write[3] = min_radius;
            bvh_data_to_write[4] = current_bounding_box.max_vertex()[0];
            bvh_data_to_write[5] = current_bounding_box.max_vertex()[1];
            bvh_data_to_write[6] = current_bounding_box.max_vertex()[2];
            bvh_data_to_write[7] = max_radius;


                for(int i = 0; i < 8; ++i) {
                    int64_t write_idx = 8*node_slot_index_pair.slot_id_ + i;
                    if( write_idx < 0 || write_idx >= num_slots * 8)
                        std::cout << "CURRENT WRITE IDX: " << write_idx << "; MAX_WRITE IDX: " << num_slots * 8 - 1 << "\n";
                    mapped_ssbo[8*node_slot_index_pair.slot_id_ + i] = bvh_data_to_write[i];
                }
            //fast_update_index_map[node_slot_index_pair.slot_id_] = std::make_pair(model_id, node_slot_index_pair.node_id_);
        }


    device_->main_context()->unmap_buffer(buffer);

    }

    pass1_compressed_visibility_shader_program_->uniform("num_primitives_per_node", int(database->get_primitives_per_node()) );
    pass1_compressed_visibility_shader_program_->storage_buffer("bvh_auxiliary_struct", 1);
    pass2_compressed_accumulation_shader_program_->uniform("num_primitives_per_node", int(database->get_primitives_per_node()) );
    pass2_compressed_accumulation_shader_program_->storage_buffer("bvh_auxiliary_struct", 1);
    compressed_LQ_one_pass_program_->uniform("num_primitives_per_node", int(database->get_primitives_per_node()) );
    compressed_LQ_one_pass_program_->storage_buffer("bvh_auxiliary_struct", 1);
    context_->bind_storage_buffer(buffer, 1, 0, int(num_slots * size_of_node_compression_slot) );
}

void Renderer::
upload_uniforms(lamure::ren::camera const& camera) const
{
    using namespace lamure::ren;
    using namespace scm::gl;
    using namespace scm::math;

    model_database* database = model_database::get_instance();
    uint32_t number_of_surfels_per_node = database->get_primitives_per_node();
    unsigned num_blend_f = NUM_BLENDED_FRAGS;

    pass1_visibility_shader_program_->uniform("near_plane", near_plane_);
    pass1_visibility_shader_program_->uniform("far_plane", far_plane_);
    pass1_visibility_shader_program_->uniform("point_size_factor", point_size_factor_);

    pass1_compressed_visibility_shader_program_->uniform("near_plane", near_plane_);
    pass1_compressed_visibility_shader_program_->uniform("far_plane", far_plane_);
    pass1_compressed_visibility_shader_program_->uniform("point_size_factor", point_size_factor_);

    pass2_accumulation_shader_program_->uniform("near_plane", near_plane_);
    pass2_accumulation_shader_program_->uniform("far_plane", far_plane_ );
    pass2_accumulation_shader_program_->uniform("point_size_factor", point_size_factor_);

    pass2_compressed_accumulation_shader_program_->uniform("near_plane", near_plane_);
    pass2_compressed_accumulation_shader_program_->uniform("far_plane", far_plane_);
    pass2_compressed_accumulation_shader_program_->uniform("point_size_factor", point_size_factor_);

    pass3_pass_through_shader_program_->uniform_sampler("in_color_texture", 0);
    pass3_pass_through_shader_program_->uniform_sampler("in_normal_texture", 1);

    pass_filling_program_->uniform_sampler("in_color_texture", 0);
    pass_filling_program_->uniform_sampler("depth_texture", 1);
    pass_filling_program_->uniform("win_size", scm::math::vec2f(win_x_, win_y_) );

    pass_filling_program_->uniform("background_color", use_user_defined_background_color_ ? user_defined_background_color_ : scm::math::vec3f(LAMURE_DEFAULT_COLOR_R, LAMURE_DEFAULT_COLOR_G, LAMURE_DEFAULT_COLOR_B) );


    pass1_linked_list_accumulate_program_->uniform_image("linked_list_buffer", 0);
    pass1_linked_list_accumulate_program_->uniform_image("fragment_count_img", 1);
    pass1_linked_list_accumulate_program_->uniform_image("min_es_distance_image", 2);

    pass1_linked_list_accumulate_program_->uniform("near_plane", near_plane_);
    pass1_linked_list_accumulate_program_->uniform("far_plane", far_plane_);
    pass1_linked_list_accumulate_program_->uniform("point_size_factor", point_size_factor_);
    pass1_linked_list_accumulate_program_->uniform("EPSILON", blending_threshold_);
    pass1_linked_list_accumulate_program_->uniform("surfels_per_node", number_of_surfels_per_node);
    pass1_linked_list_accumulate_program_->uniform("res_x", uint32_t(win_x_) );
    pass1_linked_list_accumulate_program_->uniform("num_blended_frags", num_blend_f);

    //pass2 resolve
    pass2_linked_list_resolve_program_->uniform("res_x", uint32_t(win_x_) );
    pass2_linked_list_resolve_program_->uniform("near_plane", near_plane_);
    pass2_linked_list_resolve_program_->uniform("far_plane", far_plane_);
    pass2_linked_list_resolve_program_->uniform("EPSILON", blending_threshold_);
    pass2_linked_list_resolve_program_->uniform("num_blended_frags", num_blend_f);
    pass2_linked_list_resolve_program_->uniform_image("linked_list_buffer", 0);
    pass2_linked_list_resolve_program_->uniform_image("fragment_count_img", 1);
    pass2_linked_list_resolve_program_->uniform_image("min_es_distance_image", 2);

    //pass3 repair
    pass3_repair_program_->uniform_sampler("in_color_texture", 0);
    pass3_repair_program_->uniform_sampler("depth_texture", 1);
    pass3_repair_program_->uniform("win_size", scm::math::vec2f(win_x_, win_y_) );

    pass3_repair_program_->uniform("background_color", use_user_defined_background_color_ ? user_defined_background_color_ : scm::math::vec3f(LAMURE_DEFAULT_COLOR_R, LAMURE_DEFAULT_COLOR_G, LAMURE_DEFAULT_COLOR_B) );

    LQ_one_pass_program_->uniform("near_plane", near_plane_);
    LQ_one_pass_program_->uniform("far_plane", far_plane_);
    LQ_one_pass_program_->uniform("point_size_factor", point_size_factor_);
    LQ_one_pass_program_->uniform("render_provenance", render_provenance_);

    compressed_LQ_one_pass_program_->uniform("near_plane", near_plane_);
    compressed_LQ_one_pass_program_->uniform("far_plane", far_plane_);
    compressed_LQ_one_pass_program_->uniform("point_size_factor", point_size_factor_);
    compressed_LQ_one_pass_program_->uniform("render_provenance", render_provenance_);

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

    switch(pass) {
        case RenderPass::DEPTH:
            pass1_visibility_shader_program_->uniform("mvp_matrix", scm::math::mat4f(mvpd) );
            pass1_visibility_shader_program_->uniform("model_view_matrix", model_view_matrix);
            pass1_visibility_shader_program_->uniform("inv_mv_matrix", scm::math::mat4f(scm::math::transpose(scm::math::inverse(vmd))));
            pass1_visibility_shader_program_->uniform("model_radius_scale", total_radius_scale);

            pass1_compressed_visibility_shader_program_->uniform("mvp_matrix", scm::math::mat4f(mvpd) );
            pass1_compressed_visibility_shader_program_->uniform("model_view_matrix", model_view_matrix);
            pass1_compressed_visibility_shader_program_->uniform("inv_mv_matrix", scm::math::mat4f(scm::math::transpose(scm::math::inverse(vmd))));
            pass1_compressed_visibility_shader_program_->uniform("model_radius_scale", total_radius_scale);
            break;

        case RenderPass::ACCUMULATION:
            pass2_accumulation_shader_program_->uniform("mvp_matrix", scm::math::mat4f(mvpd));
            pass2_accumulation_shader_program_->uniform("model_view_matrix", model_view_matrix);
            pass2_accumulation_shader_program_->uniform("inv_mv_matrix", scm::math::mat4f(scm::math::transpose(scm::math::inverse(vmd))));
            pass2_accumulation_shader_program_->uniform("model_radius_scale", total_radius_scale);

            pass2_compressed_accumulation_shader_program_->uniform("mvp_matrix", scm::math::mat4f(mvpd));
            pass2_compressed_accumulation_shader_program_->uniform("model_view_matrix", model_view_matrix);
            pass2_compressed_accumulation_shader_program_->uniform("inv_mv_matrix", scm::math::mat4f(scm::math::transpose(scm::math::inverse(vmd))));
            pass2_compressed_accumulation_shader_program_->uniform("model_radius_scale", total_radius_scale);
            break;            

        case RenderPass::LINKED_LIST_ACCUMULATION:
            pass1_linked_list_accumulate_program_->uniform("mvp_matrix", scm::math::mat4f(mvpd));
            pass1_linked_list_accumulate_program_->uniform("model_view_matrix", model_view_matrix);
            pass1_linked_list_accumulate_program_->uniform("inv_mv_matrix", scm::math::mat4f(scm::math::transpose(scm::math::inverse(vmd))));
            pass1_linked_list_accumulate_program_->uniform("model_radius_scale", total_radius_scale);
	    break;

        case RenderPass::ONE_PASS_LQ:
	  {
	    const scm::math::mat4f viewport_scale = scm::math::make_scale(win_x_ * 0.5f, win_y_ * 0.5f, 0.5f);;
	    const scm::math::mat4f viewport_translate = scm::math::make_translation(1.0f,1.0f,1.0f);
	    const scm::math::mat4d model_to_screen =  scm::math::mat4d(viewport_scale) * scm::math::mat4d(viewport_translate) * mvpd;
	    LQ_one_pass_program_->uniform("model_to_screen_matrix", scm::math::mat4f(model_to_screen));
        compressed_LQ_one_pass_program_->uniform("model_to_screen_matrix", scm::math::mat4f(model_to_screen));
	  }
            LQ_one_pass_program_->uniform("mvp_matrix", scm::math::mat4f(mvpd));
            LQ_one_pass_program_->uniform("model_view_matrix", model_view_matrix);
            LQ_one_pass_program_->uniform("inv_mv_matrix", scm::math::mat4f(scm::math::transpose(scm::math::inverse(vmd))));
            LQ_one_pass_program_->uniform("model_radius_scale", total_radius_scale);

            compressed_LQ_one_pass_program_->uniform("mvp_matrix", scm::math::mat4f(mvpd));
            compressed_LQ_one_pass_program_->uniform("model_view_matrix", model_view_matrix);
            compressed_LQ_one_pass_program_->uniform("inv_mv_matrix", scm::math::mat4f(scm::math::transpose(scm::math::inverse(vmd))));
            compressed_LQ_one_pass_program_->uniform("model_radius_scale", total_radius_scale);
        break;

        case RenderPass::BOUNDING_BOX:
            bounding_box_vis_shader_program_->uniform("projection_matrix", projection_matrix);
            bounding_box_vis_shader_program_->uniform("model_view_matrix", model_view_matrix );
            break;

#ifdef LAMURE_ENABLE_LINE_VISUALIZATION
        case RenderPass::LINE:
            line_shader_program_->uniform("projection_matrix", projection_matrix);
            line_shader_program_->uniform("view_matrix", view_matrix );
            break;
#endif
        case RenderPass::TRIMESH:
            trimesh_shader_program_->uniform("mvp_matrix", scm::math::mat4f(mvpd));
            break;

        default:
            //LOGGER_ERROR("Unknown Pass ID used in function 'upload_transformation_matrices'");
            std::cout << "Unknown Pass ID used in function 'upload_transformation_matrices'\n";
            break;

    }

    context_->apply();
}

void Renderer::
render_one_pass_LQ(lamure::context_t context_id, 
                   lamure::ren::camera const& camera, 
                   const lamure::view_t view_id, 
                   scm::gl::vertex_array_ptr const& render_VAO, 
                   std::set<lamure::model_t> const& current_set, 
                   std::vector<uint32_t>& frustum_culling_results) {

    using namespace lamure;
    using namespace lamure::ren;

    using namespace scm::gl;
    using namespace scm::math;

    cut_database* cuts = cut_database::get_instance();
    model_database* database = model_database::get_instance();

    size_t number_of_surfels_per_node = database->get_primitives_per_node();;



    bool is_any_model_compressed = false;
    for (auto& model_id : current_set) {
        const bvh* bvh = database->get_model(model_id)->get_bvh();

        if( bvh::primitive_type::POINTCLOUD_QZ == bvh->get_primitive() ) {
            is_any_model_compressed = true;
        }

    }

    if(is_any_model_compressed) {
        bind_bvh_attributes_for_compression_ssbo_buffer(bvh_ssbos_per_context[context_id], context_id, current_set, view_id);
    }

    /***************************************************************************************
    *******************************BEGIN LOW QUALIY PASS*****************************************
    ****************************************************************************************/

    {
      
      context_->clear_default_depth_stencil_buffer();
      context_->clear_default_color_buffer(FRAMEBUFFER_BACK, use_user_defined_background_color_ ? vec4f( user_defined_background_color_, 1.0f) : 
        vec4f(LAMURE_DEFAULT_COLOR_R, LAMURE_DEFAULT_COLOR_G, LAMURE_DEFAULT_COLOR_B, 1.0f));
      //context_->clear_default_color_buffer();


        context_->set_default_frame_buffer();

        context_->set_rasterizer_state(no_backface_culling_rasterizer_state_);
        context_->set_viewport(viewport(vec2ui(0, 0), 1 * vec2ui(win_x_, win_y_)));




        node_t node_counter = 0;
        node_t non_culled_node_idx = 0;
        for (auto& model_id : current_set)
        {
            cut& cut = cuts->get_cut(context_id, view_id, model_id);

            std::vector<cut::node_slot_aggregate>& renderable = cut.complete_set();

            const bvh* bvh = database->get_model(model_id)->get_bvh();

            auto primitive_type_to_use = bvh->get_primitive();
            if ( (primitive_type_to_use != bvh::primitive_type::POINTCLOUD) && (primitive_type_to_use != bvh::primitive_type::POINTCLOUD_QZ) ) {
                continue;
            }

            scm::gl::vertex_array_ptr const& render_VAO = lamure::ren::controller::get_instance()->get_context_memory(context_id, primitive_type_to_use, device_);
            context_->bind_vertex_array(render_VAO);

            if(  bvh::primitive_type::POINTCLOUD_QZ == primitive_type_to_use ) {
              context_->bind_program(compressed_LQ_one_pass_program_);
            } else {
              context_->bind_program(LQ_one_pass_program_); 
            }


            context_->apply();


            size_t surfels_per_node_of_model = bvh->get_primitives_per_node();
            //store culling result and push it back for second pass#

            std::vector<scm::gl::boxf>const & bounding_box_vector = bvh->get_bounding_boxes();

            upload_transformation_matrices(camera, model_id, RenderPass::ONE_PASS_LQ);

            scm::gl::frustum frustum_by_model = camera.get_frustum_by_model(model_transformations_[model_id]);

            for(auto const& node_slot_aggregate : renderable)
            {
                // Drop potentially occluded nodes.
                if(!render_occluded_geometry_ && !lamure::pvs::pvs_database::get_instance()->get_viewer_visibility(model_id, node_slot_aggregate.node_id_))
                {
                    continue;
                }

                uint32_t node_culling_result = camera.cull_against_frustum(frustum_by_model ,bounding_box_vector[node_slot_aggregate.node_id_]);
                frustum_culling_results[node_counter] = node_culling_result;

                if( (node_culling_result != 1) ) {
                  if(0 != render_provenance_) {
                    LQ_one_pass_program_->uniform("average_radius", bvh->get_avg_primitive_extent(node_slot_aggregate.node_id_));
                    compressed_LQ_one_pass_program_->uniform("average_radius", bvh->get_avg_primitive_extent(node_slot_aggregate.node_id_));
                    if(3 == render_provenance_){
                      const float accuracy = 1.0 - (bvh->get_depth_of_node(node_slot_aggregate.node_id_) * 1.0)/(bvh->get_depth() - 1);// 0...1
                      LQ_one_pass_program_->uniform("accuracy", accuracy);
                      compressed_LQ_one_pass_program_->uniform("accuracy", accuracy);
#if 0
                      const float min_accuracy = 0.1f;
                      const float max_accuracy = 0.0001f;
                      const float accuracy = std::max(max_accuracy, std::min(min_accuracy, bvh->get_avg_primitive_extent(bvh->get_num_nodes() - (bvh->get_num_nodes()/4))));
                      auto interpolate = [](float a, float b, float v){
                      const float t = (v - a)/(b - a);
                      return (1.0 - t) * 0.0 + t * 1.0;
                      };
                      LQ_one_pass_program_->uniform("accuracy", interpolate(min_accuracy, max_accuracy, accuracy));
#endif
                    }
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
                    ++non_culled_node_idx;
                }

                ++node_counter;
            }
        }

        rendered_splats_ = non_culled_node_idx * database->get_primitives_per_node();
    }

    
    scm::math::mat4f view_matrix = scm::math::mat4f(camera.get_high_precision_view_matrix());
    measurement_.drawInfo(device_, context_, text_renderer_, renderable_text_, win_x_, win_y_,
              camera.get_projection_matrix(), view_matrix, do_measurement_, display_info_);
    
}


void
Renderer::mouse(int button, int state, int x, int y, lamure::ren::camera const& camera){
  if(do_measurement_ && render_mode_ == RenderMode::LQ_ONE_PASS){
    scm::math::mat4f view_matrix = scm::math::mat4f(camera.get_high_precision_view_matrix());
    measurement_.mouse(device_, button, state, x, y, win_x_, win_y_,
		       camera.get_projection_matrix(), view_matrix);
  }
}



void Renderer::
render_one_pass_HQ(lamure::context_t context_id, 
                   lamure::ren::camera const& camera, 
                   const lamure::view_t view_id, 
                   scm::gl::vertex_array_ptr const& render_VAO, 
                   std::set<lamure::model_t> const& current_set, 
                   std::vector<uint32_t>& frustum_culling_results) {

    using namespace lamure;
    using namespace lamure::ren;

    using namespace scm::gl;
    using namespace scm::math;

    cut_database* cuts = cut_database::get_instance();
    model_database* database = model_database::get_instance();

    size_t number_of_surfels_per_node = database->get_primitives_per_node();

    /***************************************************************************************
    *******************************BEGIN LINKED_LIST_ACCUMULATION PASS**********************
    ****************************************************************************************/

    context_->clear_color_buffer(atomic_image_fbo_, 0, vec4f(0.0, 0.0, 0.0, 0.0) );
    context_->clear_color_buffer(atomic_image_fbo_, 1, vec4f(0.0, 0.0, 0.0, 0.0) );

    context_->clear_depth_stencil_buffer(pass1_visibility_fbo_);

     
    context_->set_frame_buffer(pass1_visibility_fbo_);


    context_->bind_image(linked_list_buffer_texture_, FORMAT_RGBA_16UI, scm::gl::access_mode::ACCESS_READ_WRITE, 0);
    context_->bind_image(atomic_fragment_count_image_, atomic_fragment_count_image_->format(), scm::gl::access_mode::ACCESS_READ_WRITE,1);
    context_->bind_image(min_es_distance_image_, min_es_distance_image_->format(), scm::gl::access_mode::ACCESS_READ_WRITE,2);


    context_->set_rasterizer_state(no_backface_culling_rasterizer_state_);
    context_->set_viewport(viewport(vec2ui(0, 0), 1 * vec2ui(win_x_, win_y_)));

    context_->bind_program(pass1_linked_list_accumulate_program_);

    context_->bind_vertex_array(render_VAO);
    context_->apply();

    auto compute_dist = [](lamure::vec3f const& v3, lamure::vec4r const& v4)
    {
        lamure::vec3r dist_vec(real(v3[2]) - v4[2]);
        return scm::math::length_sqr(dist_vec);
    };

    uint32_t swap_operations_performed = 0;
    uint32_t node_counter = 0;
    uint32_t non_culled_node_idx = 0;

    for (auto& model_id : current_set)
    {
        const bvh* bvh = database->get_model(model_id)->get_bvh();

        if (bvh->get_primitive() != bvh::primitive_type::POINTCLOUD)
        {
            continue;
        }

        cut& cut = cuts->get_cut(context_id, view_id, model_id);
        std::vector<cut::node_slot_aggregate> renderable = cut.complete_set();
        scm::math::mat4 inv_m_matrix = (  (( (camera.get_view_matrix()) ) * mat4(model_transformations_[model_id]) ) );

        std::vector<scm::gl::boxf>const & bounding_box_vector = bvh->get_bounding_boxes();
        std::sort(renderable.begin(), renderable.end(), [&](cut::node_slot_aggregate const & lhs,
                                                            cut::node_slot_aggregate const & rhs)
                                                            {  
                                                                bool result = compute_dist(scm::math::vec3f(0), vec4r(inv_m_matrix *  scm::math::vec4(bounding_box_vector[lhs.node_id_].center(),1)) ) <
                                                                             compute_dist(scm::math::vec3f(0), vec4r(inv_m_matrix *  scm::math::vec4(bounding_box_vector[rhs.node_id_].center(),1) ) );


                                                                if (result == false) {
                                                                    ++swap_operations_performed;
                                                                }
                                                                return  result;
                                                            } 
                  );
        
        upload_transformation_matrices(camera, model_id, RenderPass::LINKED_LIST_ACCUMULATION);
        
        scm::gl::frustum frustum_by_model = camera.get_frustum_by_model(model_transformations_[model_id]);

        size_t surfels_per_node_of_model = bvh->get_primitives_per_node();

        for(auto const& node_slot_aggregate : renderable)
        {
            // Drop potentially occluded nodes.
            if(!render_occluded_geometry_ && !lamure::pvs::pvs_database::get_instance()->get_viewer_visibility(model_id, node_slot_aggregate.node_id_))
            {
                continue;
            }

            uint32_t node_culling_result = camera.cull_against_frustum( frustum_by_model ,bounding_box_vector[ node_slot_aggregate.node_id_ ] );

            if( (node_culling_result != 1) )
            {
                context_->apply();
#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
                scm::gl::timer_query_ptr depth_pass_timer_query = device_->create_timer_query();
                context_->begin_query(depth_pass_timer_query);
#endif
                context_->draw_arrays(PRIMITIVE_POINT_LIST, (node_slot_aggregate.slot_id_) * number_of_surfels_per_node, surfels_per_node_of_model);
                ++non_culled_node_idx;
            }

            ++node_counter;
        }

        rendered_splats_ = non_culled_node_idx * database->get_primitives_per_node();
    }

    /***************************************************************************************
    *******************************BEGIN LINKED_LIST_RESOLVE PASS***************************
    ****************************************************************************************/
    {
        //context_->set_default_frame_buffer();
        context_->clear_color_buffer(pass3_normalization_fbo_, 0, vec4( 0.0, 0.0, 0.0, 0.0) );
        context_->clear_color_buffer(pass3_normalization_fbo_, 1, vec3( 0.0, 0.0, 0.0) );

        context_->set_frame_buffer(pass3_normalization_fbo_);
        context_->set_depth_stencil_state(depth_state_disable_);
        context_->bind_program(pass2_linked_list_resolve_program_);

        context_->bind_image(linked_list_buffer_texture_, FORMAT_RGBA_16UI, scm::gl::access_mode::ACCESS_READ_ONLY, 0);
        context_->bind_image(atomic_fragment_count_image_, atomic_fragment_count_image_->format(), scm::gl::access_mode::ACCESS_READ_ONLY,1);
        context_->bind_image(min_es_distance_image_, min_es_distance_image_->format(), scm::gl::access_mode::ACCESS_READ_ONLY, 2);

        bind_storage_buffer(controller::get_instance()->get_context_buffer(0, device_));
        context_->apply();

#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
        scm::gl::timer_query_ptr normalization_pass_timer_query = device_->create_timer_query();
        context_->begin_query(normalization_pass_timer_query);
#endif
        screen_quad_->draw(context_);
#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
        context_->end_query(normalization_pass_timer_query);
        context_->collect_query_results(normalization_pass_timer_query);
        normalization_pass_time += normalization_pass_timer_query->result();
#endif
    }

    /***************************************************************************************
    *******************************BEGIN HOLE_FILLING PASS**********************************
    ****************************************************************************************/
    {
        context_->set_default_frame_buffer();
        context_->bind_program(pass3_repair_program_);

        context_->bind_texture(pass3_normalization_color_texture_, filter_nearest_,   0);
        context_->bind_texture(min_es_distance_image_, filter_nearest_,   1);
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
}



void Renderer::
render_two_pass_HQ(lamure::context_t context_id, 
                   lamure::ren::camera const& camera, 
                   const lamure::view_t view_id, 
                   scm::gl::vertex_array_ptr const& dummy_VAO, 
                   std::set<lamure::model_t> const& current_set, 
                   std::vector<uint32_t>& frustum_culling_results)
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
        context_->clear_depth_stencil_buffer(pass1_visibility_fbo_);

        context_->set_frame_buffer(pass1_visibility_fbo_);

        context_->set_rasterizer_state(no_backface_culling_rasterizer_state_);

        context_->set_viewport(viewport(vec2ui(0, 0), vec2ui(win_x_, win_y_)));


        node_t node_counter = 0;
        bool is_any_model_compressed = false;
        for (auto& model_id : current_set) {
            const bvh* bvh = database->get_model(model_id)->get_bvh();

            if( bvh::primitive_type::POINTCLOUD_QZ == bvh->get_primitive() ) {
                is_any_model_compressed = true;
            }

        }

        if(is_any_model_compressed) {

            bind_bvh_attributes_for_compression_ssbo_buffer(bvh_ssbos_per_context[context_id], context_id, current_set, view_id);
        }

        for (auto& model_id : current_set)
        {
            cut& cut = cuts->get_cut(context_id, view_id, model_id);

            std::vector<cut::node_slot_aggregate>& renderable = cut.complete_set();

            const bvh* bvh = database->get_model(model_id)->get_bvh();


            bvh::primitive_type primitive_type_to_use = bvh->get_primitive();



            if(bvh::primitive_type::POINTCLOUD_QZ == primitive_type_to_use) {

              context_->bind_program(pass1_compressed_visibility_shader_program_);



            } else  {
              context_->bind_program(pass1_visibility_shader_program_);
            }

            scm::gl::vertex_array_ptr const& render_VAO = lamure::ren::controller::get_instance()->get_context_memory(context_id, primitive_type_to_use, device_);

            context_->bind_vertex_array(render_VAO);
            context_->apply();
            

/*
            if (bvh->get_primitive() == bvh::primitive_type::POINTCLOUD_QZ) {
                std::cout << "TRYING TO RENDER COMPRESSED DATA SET\n";
            }
*/
            if (bvh->get_primitive() != bvh::primitive_type::POINTCLOUD && bvh->get_primitive() != bvh::primitive_type::POINTCLOUD_QZ) {
                continue;
            }

            size_t surfels_per_node_of_model = bvh->get_primitives_per_node();
            //store culling result and push it back for second pass#

            std::vector<scm::gl::boxf>const & bounding_box_vector = bvh->get_bounding_boxes();

            upload_transformation_matrices(camera, model_id, RenderPass::DEPTH);

            scm::gl::frustum frustum_by_model = camera.get_frustum_by_model(model_transformations_[model_id]);

            for(auto const& node_slot_aggregate : renderable)
            {
                // Drop potentially occluded nodes.
                if(!render_occluded_geometry_ && !lamure::pvs::pvs_database::get_instance()->get_viewer_visibility(model_id, node_slot_aggregate.node_id_))
                {
                    continue;
                }

                uint32_t node_culling_result = camera.cull_against_frustum( frustum_by_model ,bounding_box_vector[ node_slot_aggregate.node_id_ ] );

                frustum_culling_results[node_counter] = node_culling_result;


                if( (node_culling_result != 1) ) {

#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
                    scm::gl::timer_query_ptr depth_pass_timer_query = device_->create_timer_query();
                    context_->begin_query(depth_pass_timer_query);
#endif

                    context_->draw_arrays(PRIMITIVE_POINT_LIST, (node_slot_aggregate.slot_id_) * number_of_surfels_per_node, surfels_per_node_of_model);

#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT

                    context_->collect_query_results(depth_pass_timer_query);
                    depth_pass_time += depth_pass_timer_query->result();
#endif
                }

                ++node_counter;
            }
       }
    }

    /***************************************************************************************
    *******************************BEGIN ACCUMULATION PASS**********************************
    ****************************************************************************************/
    {

        context_->clear_color_buffer(pass2_accumulation_fbo_ , 0, vec4f( .0f, .0f, .0f, 0.0f));
        context_->clear_color_buffer(pass2_accumulation_fbo_ , 1, vec4f( .0f, .0f, .0f, 0.0f));

        pass2_accumulation_fbo_->attach_depth_stencil_buffer(pass1_depth_buffer_);

        context_->set_frame_buffer(pass2_accumulation_fbo_);

        context_->set_rasterizer_state(no_backface_culling_rasterizer_state_);
        context_->set_blend_state(color_blending_state_);


              context_->apply();


        context_->set_depth_stencil_state(depth_state_test_without_writing_);
/*
        context_->bind_program(pass2_accumulation_shader_program_);
*/

        node_t node_counter = 0;
        node_t actually_rendered_nodes = 0;

        for (auto& model_id : current_set) {
            cut& cut = cuts->get_cut(context_id, view_id, model_id);

            std::vector<cut::node_slot_aggregate>& renderable = cut.complete_set();

            const bvh* bvh = database->get_model(model_id)->get_bvh();


            bvh::primitive_type primitive_type_to_use = bvh->get_primitive();

            if(bvh::primitive_type::POINTCLOUD_QZ == primitive_type_to_use) {
              context_->bind_program(pass2_compressed_accumulation_shader_program_);
            } else {
              context_->bind_program(pass2_accumulation_shader_program_);
            }

            if(bvh::primitive_type::POINTCLOUD_QZ == primitive_type_to_use) {
              scm::gl::vertex_array_ptr const& render_VAO = lamure::ren::controller::get_instance()->get_context_memory(context_id, primitive_type_to_use, device_);
              context_->bind_vertex_array(render_VAO);
              context_->apply();
            }



            if (bvh->get_primitive() != bvh::primitive_type::POINTCLOUD  && bvh->get_primitive() != bvh::primitive_type::POINTCLOUD_QZ) {
                continue;
            }

            size_t surfels_per_node_of_model = bvh->get_primitives_per_node();


            upload_transformation_matrices(camera, model_id, RenderPass::ACCUMULATION);

            for( auto const& node_slot_aggregate : renderable )
            {
                // Drop potentially occluded nodes.
                if(!render_occluded_geometry_ && !lamure::pvs::pvs_database::get_instance()->get_viewer_visibility(model_id, node_slot_aggregate.node_id_))
                {
                    continue;
                }


                if( frustum_culling_results[node_counter] != 1)  // 0 = inside, 1 = outside, 2 = intersectingS
                {

#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
                    scm::gl::timer_query_ptr accumulation_pass_timer_query = device_->create_timer_query();
                    context_->begin_query(accumulation_pass_timer_query);
#endif
                    context_->draw_arrays(PRIMITIVE_POINT_LIST, (node_slot_aggregate.slot_id_) * number_of_surfels_per_node, surfels_per_node_of_model);
#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
                    context_->end_query(accumulation_pass_timer_query);
                    context_->collect_query_results(accumulation_pass_timer_query);
                    accumulation_pass_time += accumulation_pass_timer_query->result();
#endif

                    ++actually_rendered_nodes;
                }

                ++node_counter;
            }
        }
        rendered_splats_ = actually_rendered_nodes * database->get_primitives_per_node();

    }

    /***************************************************************************************
    *******************************BEGIN NORMALIZATION PASS*********************************
    ****************************************************************************************/

    {
        //context_->set_default_frame_buffer();
        context_->clear_color_buffer(pass3_normalization_fbo_, 0, vec4( 0.0, 0.0, 0.0, 0.0) );
        context_->clear_color_buffer(pass3_normalization_fbo_, 1, vec3( 0.0, 0.0, 0.0) );
        
        
        context_->set_frame_buffer(pass3_normalization_fbo_);

        context_->set_depth_stencil_state(depth_state_disable_);

        context_->bind_program(pass3_pass_through_shader_program_);

        context_->bind_texture(pass2_accumulated_color_buffer_, filter_nearest_,   0);
        context_->bind_texture(pass2_accumulated_normal_buffer_, filter_nearest_, 1);
        context_->apply();

#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
        scm::gl::timer_query_ptr normalization_pass_timer_query = device_->create_timer_query();
        context_->begin_query(normalization_pass_timer_query);
#endif
        screen_quad_->draw(context_);
#ifdef LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT
        context_->end_query(normalization_pass_timer_query);
        context_->collect_query_results(normalization_pass_timer_query);
        normalization_pass_time += normalization_pass_timer_query->result();
#endif
    }

    /***************************************************************************************
    *******************************BEGIN RECONSTRUCTION PASS*********************************
    ****************************************************************************************/
    {
        context_->set_default_frame_buffer();

        context_->bind_program(pass_filling_program_);

        context_->bind_texture(pass3_normalization_color_texture_, filter_nearest_,   0);
        context_->bind_texture(pass1_depth_buffer_, filter_nearest_,   1);
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
    for (lamure::model_t model_id = 0; model_id < num_models; ++model_id) {
        auto vs_it = visible_set_.find(model_id);
        auto is_it = invisible_set_.find(model_id);

        if (vs_it == visible_set_.end() && is_it == invisible_set_.end()) {
            current_set.insert(model_id);
        }
        else if (vs_it != visible_set_.end()) {
            if (render_visible_set_) {
                current_set.insert(model_id);
            }
        }
        else if (is_it != invisible_set_.end()) {
            if (!render_visible_set_) {
                current_set.insert(model_id);
            }
        }

    }


    rendered_splats_ = 0;

    std::vector<uint32_t>                       frustum_culling_results;

    uint32_t size_of_culling_result_vector = 0;

    for (auto& model_id : current_set) {
        cut& cut = cuts->get_cut(context_id, view_id, model_id);

        std::vector<cut::node_slot_aggregate>& renderable = cut.complete_set();

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

        switch(render_mode_) {
            case (RenderMode::HQ_TWO_PASS):
                render_two_pass_HQ(context_id, 
                                   camera, 
                                   view_id, 
                                   render_VAO, 
                                   current_set, 
                                   frustum_culling_results);
                break;
            case (RenderMode::HQ_ONE_PASS):

                render_one_pass_HQ(context_id, 
                                   camera, 
                                   view_id, 
                                   render_VAO, 
                                   current_set, 
                                   frustum_culling_results);
                break;

            case (RenderMode::LQ_ONE_PASS):

                render_one_pass_LQ(context_id, 
                                   camera, 
                                   view_id, 
                                   render_VAO, 
                                   current_set, 
                                   frustum_culling_results);
                break;
        }



        //TRIMESH PASS
        context_->set_default_frame_buffer();
        context_->bind_program(trimesh_shader_program_);
               
        scm::gl::vertex_array_ptr memory = lamure::ren::controller::get_instance()->get_context_memory(context_id, bvh::primitive_type::TRIMESH, device_);
        context_->bind_vertex_array(memory);
        context_->apply();

        for (auto& model_id : current_set) {

            const bvh* bvh = database->get_model(model_id)->get_bvh();

            if (bvh->get_primitive() == bvh::primitive_type::TRIMESH) {
               cut& cut = cuts->get_cut(context_id, view_id, model_id);
               std::vector<cut::node_slot_aggregate>& renderable = cut.complete_set();

               upload_transformation_matrices(camera, model_id, RenderPass::TRIMESH);
               
               size_t surfels_per_node_of_model = bvh->get_primitives_per_node();

               std::vector<scm::gl::boxf>const & bounding_box_vector = bvh->get_bounding_boxes();

               scm::gl::frustum frustum_by_model = camera.get_frustum_by_model(model_transformations_[model_id]);

               for (auto const& node_slot_aggregate : renderable) {
                  uint32_t node_culling_result = camera.cull_against_frustum( frustum_by_model ,bounding_box_vector[ node_slot_aggregate.node_id_ ] );

                  //if( node_culling_result != 1)  // 0 = inside, 1 = outside, 2 = intersectingS
                  {
                      rendered_splats_ += database->get_primitives_per_node();
                      context_->apply();
                      context_->draw_arrays(PRIMITIVE_TRIANGLE_LIST, (node_slot_aggregate.slot_id_) * database->get_primitives_per_node(), surfels_per_node_of_model);
                  }
               }
            }
        }



        if(render_bounding_boxes_)
        {
            context_->set_default_frame_buffer();
            context_->bind_program(bounding_box_vis_shader_program_);
            context_->apply();

            node_t node_counter = 0;

            for (auto& model_id : current_set)
            {
                cut& c = cuts->get_cut(context_id, view_id, model_id);

                std::vector<cut::node_slot_aggregate>& renderable = c.complete_set();

                upload_transformation_matrices(camera, model_id, RenderPass::BOUNDING_BOX);

                for( auto const& node_slot_aggregate : renderable ) {

                    int culling_result = frustum_culling_results[node_counter];

                    if( culling_result  != 1 )  // 0 = inside, 1 = outside, 2 = intersectingS
                    {

                        scm::gl::boxf temp_box = database->get_model(model_id)->get_bvh()->get_bounding_boxes()[node_slot_aggregate.node_id_ ];
                        scm::gl::box_geometry box_to_render(device_,temp_box.min_vertex(), temp_box.max_vertex());

                        bounding_box_vis_shader_program_->uniform("culling_status", culling_result);

                        device_->opengl_api().glDisable(GL_DEPTH_TEST);
                        box_to_render.draw(context_, scm::gl::geometry::MODE_WIRE_FRAME);
                        device_->opengl_api().glEnable(GL_DEPTH_TEST);

                    }

                    ++node_counter;
                }
            }
        }

        // Render PVS grid cell outlines to visualize grid cell size and position.
        if(render_pvs_grid_cells_)
        {
            lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();

            if(pvs->get_visibility_grid() != nullptr)
            {
                context_->set_default_frame_buffer();
                context_->bind_program(pvs_grid_cell_vis_shader_program_);
                context_->apply();

                // Code copied from upload_transformation_matrices() because we need to manually set the model matrix.
                scm::math::mat4f model_matrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
                scm::math::mat4f projection_matrix = camera.get_projection_matrix();

                scm::math::mat4d    vm = camera.get_high_precision_view_matrix();
                scm::math::mat4d    mm = scm::math::mat4d(model_matrix);
                scm::math::mat4d    vmd = vm * mm;
    
                scm::math::mat4f model_view_matrix = scm::math::mat4f(vmd);

                pvs_grid_cell_vis_shader_program_->uniform("projection_matrix", projection_matrix);
                pvs_grid_cell_vis_shader_program_->uniform("model_view_matrix", model_view_matrix );

                for(size_t cell_index = 0; cell_index < pvs->get_visibility_grid()->get_cell_count(); ++cell_index)
                {
                    const lamure::pvs::view_cell* current_cell = pvs->get_visibility_grid()->get_cell_at_index(cell_index);

                    scm::math::vec3f min_corner(current_cell->get_position_center() - (current_cell->get_size() * 0.5f));
                    scm::math::vec3f max_corner(current_cell->get_position_center() + (current_cell->get_size() * 0.5f));
                    scm::gl::box_geometry box_to_render(device_, min_corner, max_corner);

                    //device_->opengl_api().glDisable(GL_DEPTH_TEST);
                    box_to_render.draw(context_, scm::gl::geometry::MODE_WIRE_FRAME);
                    //device_->opengl_api().glEnable(GL_DEPTH_TEST);
                }
            }
            else
            {
                std::cout << "no grid" << std::endl;
            }
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


#ifdef LAMURE_ENABLE_LINE_VISUALIZATION


    scm::math::vec3f* line_mem = (scm::math::vec3f*)device_->main_context()->map_buffer(line_buffer_, scm::gl::ACCESS_READ_WRITE);
    unsigned int num_valid_lines = 0;
    for (unsigned int i = 0; i < max_lines_; ++i) {
      if (i < line_begin_.size() && i < line_end_.size()) {
         line_mem[num_valid_lines*2+0] = line_begin_[i];
         line_mem[num_valid_lines*2+1] = line_end_[i];
         ++num_valid_lines; 
      }
    }

    device_->main_context()->unmap_buffer(line_buffer_);

    upload_transformation_matrices(camera, 0, LINE);
    device_->opengl_api().glDisable(GL_DEPTH_TEST);

    context_->set_default_frame_buffer();

    context_->bind_program(line_shader_program_);

    context_->bind_vertex_array(line_memory_);
    context_->apply();

    context_->draw_arrays(PRIMITIVE_LINE_LIST, 0, 2*num_valid_lines);


    device_->opengl_api().glEnable(GL_DEPTH_TEST);
#endif


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

    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();


  if(!display_info_){
    return;
  }

    std::stringstream os;
   // os.setprecision(5);
    os
      <<"FPS:   "<<std::setprecision(4)<<fps_<<"\n"
      <<"# Points:   "<< (rendered_splats_ / 100000) / 10.0f<< " Mio. \n"
      <<"# Nodes:   "<< (rendered_splats_ / database->get_primitives_per_node()) << "\n"
      <<"Render Mode: " ;
      switch(render_mode_) {
        case(RenderMode::HQ_ONE_PASS):
            os << "HQ One-Pass\n";
            break;

        case(RenderMode::HQ_TWO_PASS):
            os << "HQ Two-Pass\n";
            break;

        case(RenderMode::LQ_ONE_PASS):
            os << "LQ One-Pass:\nPress p for additional visualizations\n";
	      switch(render_provenance_){
	          case 0:
	            os << "Additional visualization: none\n";
	            break;
	          case 1:
	            os << "Output-sensitivity visualization (blue is good, red is bad)\n";
	            break;
	          case 2:
	            os << "View space normal mapped to color\n";
	            break;
	          case 3:
	            os << "Distance to highest data density (red means far away)\n";
	            break;
	          default:
	            break;
	      }
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

    pass1_visibility_fbo_ = device_->create_frame_buffer();

    pass1_depth_buffer_           = device_->create_texture_2d(vec2ui(win_x_, win_y_) * 1, FORMAT_D32F, 1, 1, 1);

    pass1_visibility_fbo_->attach_depth_stencil_buffer(pass1_depth_buffer_);



    pass2_accumulation_fbo_ = device_->create_frame_buffer();

    pass2_accumulated_color_buffer_   = device_->create_texture_2d(vec2ui(win_x_, win_y_) * 1, FORMAT_RGBA_32F , 1, 1, 1);

    pass2_accumulation_fbo_->attach_color_buffer(0, pass2_accumulated_color_buffer_);

    pass2_accumulated_normal_buffer_   = device_->create_texture_2d(vec2ui(win_x_, win_y_) * 1, FORMAT_RGB_32F , 1, 1, 1);

    pass2_accumulation_fbo_->attach_color_buffer(1, pass2_accumulated_normal_buffer_);

    pass2_accumulation_fbo_->attach_depth_stencil_buffer(pass1_depth_buffer_);


    pass3_normalization_fbo_ = device_->create_frame_buffer();

    pass3_normalization_color_texture_ = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_RGBA_8 , 1, 1, 1);
    pass3_normalization_normal_texture_ = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_RGB_8 , 1, 1, 1);

    pass3_normalization_fbo_->attach_color_buffer(0, pass3_normalization_color_texture_);
    pass3_normalization_fbo_->attach_color_buffer(1, pass3_normalization_normal_texture_);

   
    screen_quad_.reset(new quad_geometry(device_, vec2f(-1.0f, -1.0f), vec2f(1.0f, 1.0f)));


    color_blending_state_ = device_->create_blend_state(true, FUNC_ONE, FUNC_ONE, FUNC_ONE, FUNC_ONE, EQ_FUNC_ADD, EQ_FUNC_ADD);


    depth_state_disable_ = device_->create_depth_stencil_state(false, true, scm::gl::COMPARISON_NEVER);

    depth_state_test_without_writing_ = device_->create_depth_stencil_state(true, false, scm::gl::COMPARISON_LESS_EQUAL);

#ifdef LAMURE_ENABLE_LINE_VISUALIZATION
    std::size_t size_of_line_buffer = max_lines_ * sizeof(float) * 3 * 2;
    line_buffer_ = device_->create_buffer(scm::gl::BIND_VERTEX_BUFFER,
                                    scm::gl::USAGE_DYNAMIC_DRAW,
                                    size_of_line_buffer,
                                    0);
    line_memory_ = device_->create_vertex_array(scm::gl::vertex_format
            (0, 0, scm::gl::TYPE_VEC3F, sizeof(float)*3),
            boost::assign::list_of(line_buffer_));

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


bool Renderer::
initialize_schism_device_and_shaders(int resX, int resY)
{
    std::string root_path = LAMURE_SHADERS_DIR;

    std::string visibility_vs_source;
    std::string compressed_visibility_vs_source; //compressed_version
    std::string visibility_gs_source;
    std::string visibility_fs_source;

    std::string pass_trough_vs_source;
    std::string pass_trough_fs_source;

    std::string accumulation_vs_source;
    std::string compressed_accumulation_vs_source; //compressed_version
    std::string accumulation_gs_source;
    std::string accumulation_fs_source;

    std::string filling_vs_source;
    std::string filling_fs_source;

    std::string bounding_box_vs_source;
    std::string bounding_box_fs_source;

    std::string pvs_grid_cell_vs_source;
    std::string pvs_grid_cell_fs_source;

    std::string linked_list_accum_vs_source;
    std::string linked_list_accum_gs_source;
    std::string linked_list_accum_fs_source;

    std::string linked_list_resolve_vs_source;
    std::string linked_list_resolve_fs_source;

    std::string repair_program_vs_source;
    std::string repair_program_fs_source;

    std::string lq_one_pass_vs_source;
    std::string compressed_lq_one_pass_vs_source;
    std::string lq_one_pass_gs_source;
    std::string lq_one_pass_fs_source;

    std::string trimesh_vs_source;
    std::string trimesh_fs_source;

#ifdef LAMURE_ENABLE_LINE_VISUALIZATION
    std::string line_vs_source;
    std::string line_fs_source;
#endif
    try {

        using scm::io::read_text_file;


        if (!read_shader(root_path +  "/pass1_visibility_pass.glslv", visibility_vs_source)
            || !read_shader(root_path + "/pass1_compressed_visibility_pass.glslv", compressed_visibility_vs_source)
            || !read_shader(root_path + "/pass1_visibility_pass.glslg", visibility_gs_source)
            || !read_shader(root_path + "/pass1_visibility_pass.glslf", visibility_fs_source)
            || !read_shader(root_path + "/pass2_accumulation_pass.glslv", accumulation_vs_source)
            || !read_shader(root_path + "/pass2_compressed_accumulation_pass.glslv", compressed_accumulation_vs_source)
            || !read_shader(root_path + "/pass2_accumulation_pass.glslg", accumulation_gs_source)
            || !read_shader(root_path + "/pass2_accumulation_pass.glslf", accumulation_fs_source)
            || !read_shader(root_path + "/pass3_pass_through.glslv", pass_trough_vs_source)
            || !read_shader(root_path + "/pass3_pass_through.glslf", pass_trough_fs_source)
            || !read_shader(root_path + "/pass_reconstruction.glslv", filling_vs_source)
            || !read_shader(root_path + "/pass_reconstruction.glslf", filling_fs_source)
            || !read_shader(root_path + "/bounding_box_vis.glslv", bounding_box_vs_source)
            || !read_shader(root_path + "/bounding_box_vis.glslf", bounding_box_fs_source)
            || !read_shader(root_path + "/pvs_grid_cell_vis.glslv", pvs_grid_cell_vs_source)
            || !read_shader(root_path + "/pvs_grid_cell_vis.glslf", pvs_grid_cell_fs_source)
    	    || !read_shader(root_path + "/pass1_linked_list_accumulation.glslv", linked_list_accum_vs_source)
    	    || !read_shader(root_path + "/pass1_linked_list_accumulation.glslg", linked_list_accum_gs_source)
    	    || !read_shader(root_path + "/pass1_linked_list_accumulation.glslf", linked_list_accum_fs_source)
    	    || !read_shader(root_path + "/pass2_linked_list_resolve.glslv", linked_list_resolve_vs_source)
    	    || !read_shader(root_path + "/pass2_linked_list_resolve.glslf", linked_list_resolve_fs_source)         
    	    || !read_shader(root_path + "/pass3_repair.glslv", repair_program_vs_source)
    	    || !read_shader(root_path + "/pass3_repair.glslf", repair_program_fs_source)
            || !read_shader(root_path + "/lq_one_pass.glslv", lq_one_pass_vs_source)
            || !read_shader(root_path + "/compressed_lq_one_pass.glslv", compressed_lq_one_pass_vs_source)
            || !read_shader(root_path + "/lq_one_pass.glslg", lq_one_pass_gs_source)
            || !read_shader(root_path + "/lq_one_pass.glslf", lq_one_pass_fs_source)
            || !read_shader(root_path + "/trimesh.glslv", trimesh_vs_source)
            || !read_shader(root_path + "/trimesh.glslf", trimesh_fs_source)

#ifdef LAMURE_ENABLE_LINE_VISUALIZATION
            || !read_shader(root_path + "/lines_shader.glslv", line_vs_source)
            || !read_shader(root_path + "/lines_shader.glslf", line_fs_source)
#endif
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

    //using namespace boost::assign;
    std::cout << "shader: pass1_visibility_shader_program_" << std::endl;
    pass1_visibility_shader_program_ = device_->create_program(
                                                  boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, visibility_vs_source))
                                                                        (device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, visibility_gs_source))
                                                                        (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, visibility_fs_source))
                                                              );

    std::cout << "shader: pass1_compressed_visibility_shader_program_" << std::endl;
    pass1_compressed_visibility_shader_program_ = device_->create_program(
                                                  boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, compressed_visibility_vs_source))
                                                                        (device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, visibility_gs_source))
                                                                        (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, visibility_fs_source))
                                                              );

    std::cout << "shader: pass2_accumulation_shader_program_" << std::endl;
    pass2_accumulation_shader_program_ = device_->create_program(
                                                    boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, accumulation_vs_source))
                                                                          (device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, accumulation_gs_source))
                                                                          (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER,accumulation_fs_source))
                                                                );

    std::cout << "shader: pass2_compressed_accumulation_shader_program_" << std::endl;
    pass2_compressed_accumulation_shader_program_ = device_->create_program(
                                                    boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, compressed_accumulation_vs_source))
                                                                          (device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, accumulation_gs_source))
                                                                          (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER,accumulation_fs_source))
                                                              );
    
    std::cout << "shader: pass3_pass_through_shader_program_" << std::endl;
    pass3_pass_through_shader_program_ = device_->create_program(boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, pass_trough_vs_source))
                                                                (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, pass_trough_fs_source)));
    pass_filling_program_ = device_->create_program(boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, filling_vs_source))
                                                    (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, filling_fs_source)));

    std::cout << "shader: bounding_box_vis_shader_program_" << std::endl;
    bounding_box_vis_shader_program_ = device_->create_program(boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, bounding_box_vs_source))
                                                               (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, bounding_box_fs_source)));

    std::cout << "shader: pvs_grid_cell_vis_shader_program_" << std::endl;
    pvs_grid_cell_vis_shader_program_ = device_->create_program(boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, pvs_grid_cell_vs_source))
                                                               (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, pvs_grid_cell_fs_source)));

#ifdef LAMURE_ENABLE_LINE_VISUALIZATION

    std::cout << "shader: line_shader_program_" << std::endl;
    line_shader_program_ = device_->create_program(boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, line_vs_source))
                                                   (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, line_fs_source)));
#endif

    std::cout << "shader: pass1_linked_list_accumulate_program_" << std::endl;
    pass1_linked_list_accumulate_program_ = device_->create_program(
        boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, linked_list_accum_vs_source ))
                              (device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, linked_list_accum_gs_source ))
	                          (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, linked_list_accum_fs_source ))
    );

    std::cout << "shader: pass2_linked_list_resolve_program_" << std::endl;
    pass2_linked_list_resolve_program_ = device_->create_program(
        boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, linked_list_resolve_vs_source ))
	                          (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, linked_list_resolve_fs_source ))
    );

    std::cout << "shader: pass3_repair_program_" << std::endl;
    pass3_repair_program_ = device_->create_program(
        boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, repair_program_vs_source ))
	                          (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, repair_program_fs_source ))
    );

    std::cout << "shader: LQ_one_pass_program_" << std::endl;
    LQ_one_pass_program_ = device_->create_program(
        boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER,   lq_one_pass_vs_source ))
                              (device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, lq_one_pass_gs_source ))
                              (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, lq_one_pass_fs_source ))
    );

    std::cout << "shader: compressed_LQ_one_pass_program_" << std::endl;
    compressed_LQ_one_pass_program_ = device_->create_program(
        boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER,   compressed_lq_one_pass_vs_source ))
                              (device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, lq_one_pass_gs_source ))
                              (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, lq_one_pass_fs_source ))
    );

    std::cout << "shader: trimesh_shader_program_" << std::endl;
    trimesh_shader_program_ = device_->create_program(
       boost::assign::list_of(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, trimesh_vs_source))
                             (device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, trimesh_fs_source)) );

    if (    !pass1_visibility_shader_program_ || !pass1_compressed_visibility_shader_program_ 
         || !pass2_accumulation_shader_program_ || !pass2_compressed_accumulation_shader_program_ 
         || !pass3_pass_through_shader_program_ || !pass_filling_program_ 
         || !pass1_linked_list_accumulate_program_ || !pass2_linked_list_resolve_program_ || !pass3_repair_program_ || !trimesh_shader_program_
         || !LQ_one_pass_program_ || !compressed_LQ_one_pass_program_
         || !bounding_box_vis_shader_program_
         || !pvs_grid_cell_vis_shader_program_
#ifdef LAMURE_ENABLE_LINE_VISUALIZATION
        || !line_shader_program_
#endif
       ) {
        scm::err() << "error creating shader programs" << scm::log::end;
        return false;
    }


    scm::out() << *device_ << scm::log::end;


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
    pass1_visibility_fbo_ = device_->create_frame_buffer();

    pass1_depth_buffer_           =device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_D24, 1, 1, 1);

    pass1_visibility_fbo_->attach_depth_stencil_buffer(pass1_depth_buffer_);


    pass2_accumulation_fbo_ = device_->create_frame_buffer();

    pass2_accumulated_color_buffer_   = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_RGBA_32F , 1, 1, 1);
    pass2_accumulated_normal_buffer_   = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_RGB_32F , 1, 1, 1);
    
    pass2_accumulation_fbo_->attach_color_buffer(0, pass2_accumulated_color_buffer_);
    pass2_accumulation_fbo_->attach_color_buffer(1, pass2_accumulated_normal_buffer_);


    pass3_normalization_fbo_ = device_->create_frame_buffer();

    pass3_normalization_color_texture_ = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_RGBA_8 , 1, 1, 1);
    pass3_normalization_normal_texture_ = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_) * 1, scm::gl::FORMAT_RGB_8 , 1, 1, 1);

    pass3_normalization_fbo_->attach_color_buffer(0, pass3_normalization_color_texture_);
    pass3_normalization_fbo_->attach_color_buffer(1, pass3_normalization_normal_texture_);

    size_t total_num_pixels = win_x_ * win_y_ * NUM_BLENDED_FRAGS;

    linked_list_buffer_texture_ = device_->create_texture_buffer(scm::gl::FORMAT_RGBA_16UI, scm::gl::USAGE_DYNAMIC_COPY, sizeof(uint16_t) * 4 * total_num_pixels);


    atomic_image_fbo_ = device_->create_frame_buffer();
    atomic_fragment_count_image_ = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_), scm::gl::FORMAT_R_32UI , 1, 1, 1);;
    atomic_image_fbo_->attach_color_buffer(0, atomic_fragment_count_image_);

    min_es_distance_image_ = device_->create_texture_2d(scm::math::vec2ui(win_x_, win_y_), scm::gl::FORMAT_R_32UI, 1, 1, 1);
    atomic_image_fbo_->attach_color_buffer(1, min_es_distance_image_);

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


void Renderer::
toggle_provenance_rendering()
{
  ++render_provenance_;
  if(render_provenance_ == 4){
    render_provenance_ = 0;
  }

  std::cout<<"provenance rendering: ";
  if(render_provenance_ == 0)
    std::cout<<"OFF\n\n";
  else if(render_provenance_ == 1)
    std::cout<<"difference to average surfel radius\n\n";
  else if(render_provenance_ == 2)
    std::cout<<"normal\n\n";
  else// if(render_provenance_ == 3)
    std::cout<<"accuracy\n\n";

};


void Renderer::
toggle_do_measurement(){
  if(render_mode_ == RenderMode::LQ_ONE_PASS){
    do_measurement_ = !do_measurement_;
  }
};

void Renderer::
toggle_use_user_defined_background_color(){
  use_user_defined_background_color_ = !use_user_defined_background_color_;
};


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
toggle_pvs_grid_cell_rendering()
{
    render_pvs_grid_cells_ = ! render_pvs_grid_cells_;

    std::cout<<"pvs grid cell visualisation: ";
    if(render_pvs_grid_cells_)
        std::cout<<"ON\n\n";
    else
        std::cout<<"OFF\n\n";  
}


void Renderer::
change_point_size(float amount)
{
    point_size_factor_ += amount;
    if(point_size_factor_ < 0.0001f)
    {
        point_size_factor_ = 0.0001;
    }

    std::cout<<"set point size factor to: "<<point_size_factor_<<"\n\n";
};

void Renderer::
toggle_cut_update_info() {
    is_cut_update_active_ = ! is_cut_update_active_;
}

void Renderer::
toggle_camera_info(const lamure::view_t current_cam_id) {
    current_cam_id_ = current_cam_id;
}

void Renderer::
toggle_display_info() {
    display_info_ = ! display_info_;
}

void Renderer::
toggle_visible_set() {
    render_visible_set_ = !render_visible_set_;
}

void Renderer::
take_screenshot(std::string const& screenshot_path, std::string const& screenshot_name) {

    std::string file_extension = ".png";
    {

        std::string full_path = screenshot_path + "/";
        {
            if(! boost::filesystem::exists(full_path)) {
               std::cout<<"Screenshot Folder did not exist. Creating Folder: " << full_path << "\n\n";
               boost::filesystem::create_directories(full_path);
            }
        }

        

        // Make the BYTE array, factor of 3 because it's RBG.
        BYTE* pixels = new BYTE[ 3 * win_x_ * win_y_];

        
        device_->opengl_api().glBindTexture(GL_TEXTURE_2D, pass3_normalization_color_texture_->object_id());
        device_->opengl_api().glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, pixels);

        std::string ten_k_surfels = std::to_string((rendered_splats_ / 10000) );

        std::string filename = full_path + "color__" + screenshot_name + "__surfels_" + ten_k_surfels  + "_tk" + file_extension;

        FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, win_x_, win_y_, 3 * win_x_, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
        FreeImage_Save(FIF_PNG, image, filename.c_str(), 0);

        device_->opengl_api().glBindTexture(GL_TEXTURE_2D, pass3_normalization_normal_texture_->object_id());
        device_->opengl_api().glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, pixels);

        filename = full_path + "normal__" + screenshot_name + "__surfels_" + ten_k_surfels  + "_tk" + file_extension;

        image = FreeImage_ConvertFromRawBits(pixels, win_x_, win_y_, 3 * win_x_, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
        FreeImage_Save(FIF_PNG, image, filename.c_str(), 0);

        device_->opengl_api().glBindTexture(GL_TEXTURE_2D, 0);

        // Free resources
        FreeImage_Unload(image);
        delete [] pixels;

        std::cout<<"Saved Screenshot: "<<filename.c_str()<<"\n\n";
    }
}

void Renderer::
set_user_defined_background_color(float bg_r, float bg_g, float bg_b) {
  user_defined_background_color_ = scm::math::vec3f(bg_r, bg_g, bg_b);
}

void Renderer::
switch_render_mode(RenderMode const& render_mode) {
    render_mode_ = render_mode;
}

void Renderer::
toggle_culling()
{
    render_occluded_geometry_ = !render_occluded_geometry_;
}

void Renderer::
enable_culling(const bool& enable)
{
    render_occluded_geometry_ = !enable;
}

double Renderer::get_fps() const
{
    return fps_;
}

