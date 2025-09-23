// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef PVS_OLD_RENDERER_H_
#define PVS_OLD_RENDERER_H_

#include <lamure/pvs/pvs_preprocessing.h>
#include <lamure/ren/camera.h>
#include <lamure/ren/cut.h>

#include <lamure/ren/controller.h>
#include <lamure/ren/model_database.h>
#include <lamure/ren/cut_database.h>
#include <lamure/ren/controller.h>
#include <lamure/ren/policy.h>

#include <boost/assign/list_of.hpp>
#include <memory>

#include <fstream>

#include <scm/core.h>
#include <scm/log.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>
#include <scm/core/pointer_types.h>
#include <scm/core/io/tools.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/gl_util/data/imaging/texture_loader.h>
#include <scm/gl_util/viewer/camera.h>
#include <scm/gl_util/primitives/quad.h>
#include <scm/gl_util/primitives/box.h>

#include <scm/core/math.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/primitives/geometry.h>

#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/font/text.h>
#include <scm/gl_util/font/text_renderer.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>
#include <scm/gl_util/primitives/geometry.h>

#include <vector>
#include <map>
#include <lamure/types.h>
#include <lamure/utils.h>

#include "lamure/pvs/id_histogram.h"

#include <lamure/ren/ray.h>

//#define LAMURE_RENDERING_ENABLE_PERFORMANCE_MEASUREMENT


enum class RenderPass
{
    DEPTH                    = 0,
    ACCUMULATION             = 1,
    NORMALIZATION            = 2,
    LINKED_LIST_ACCUMULATION = 4,
    VISIBLE_NODE             = 6,
    BOUNDING_BOX             = 100,
    LINE                     = 101,
    TRIMESH                  = 300
};

enum class RenderMode
{
    VISIBLE_NODE_PASS = 1
};


class Renderer
{
public:
                        Renderer(std::vector<scm::math::mat4f> const& model_transformations,
                            const std::set<lamure::model_t>& visible_set,
                            const std::set<lamure::model_t>& invisible_set);
    virtual             ~Renderer();

    scm::gl::render_device_ptr          device_;
    scm::gl::render_context_ptr         context_;

    void                render(lamure::context_t context_id, lamure::ren::camera const& camera, const lamure::view_t view_id, scm::gl::vertex_array_ptr render_VAO, const unsigned current_camera_session);
    void                reset_viewport(int const x, int const y);

    void                send_model_transform(const lamure::model_t model_id, const scm::math::mat4f& transform);

    void                set_radius_scale(const float radius_scale) { radius_scale_ = radius_scale; };

    void                switch_render_mode(RenderMode const& render_mode);

    scm::gl::render_device_ptr device() const { return device_; }

    void                display_status(std::string const& information_to_display);

    

protected:
    bool                initialize_schism_device_and_shaders(int resX, int resY);
    void                initialize_VBOs();
    void                update_frustum_dependent_parameters(lamure::ren::camera const& camera);

    void                upload_uniforms(lamure::ren::camera const& camera) const;
    void                upload_transformation_matrices(lamure::ren::camera const& camera, lamure::model_t const model_id, RenderPass const pass_id) const;
    void                swap_temp_buffers();
    void                calculate_radius_scale_per_model();


    void                render_one_pass_LQ(lamure::context_t context_id, 
                                           lamure::ren::camera const& camera, 
                                           const lamure::view_t view_id, 
                                           scm::gl::vertex_array_ptr const& render_VAO,
                                           std::set<lamure::model_t> const& current_set, std::vector<uint32_t>& frustum_culling_results);

    void                render_depth(lamure::context_t context_id, 
                                    lamure::ren::camera const& camera, 
                                    const lamure::view_t view_id, 
                                    scm::gl::vertex_array_ptr const& render_VAO, 
                                    std::set<lamure::model_t> const& current_set, std::vector<uint32_t>& frustum_culling_results);

    void                render_two_pass_HQ(lamure::context_t context_id, 
                                           lamure::ren::camera const& camera, 
                                           const lamure::view_t view_id, 
                                           scm::gl::vertex_array_ptr const& render_VAO, 
                                           std::set<lamure::model_t> const& current_set, std::vector<uint32_t>& frustum_culling_results);


    bool                read_shader(std::string const& path_string, std::string& shader_string);
    bool const          parse_prefix(std::string& in_string, std::string const& prefix);
    std::string const   strip_whitespace(std::string const& in_string);
    
private:

        int                                         win_x_;
        int                                         win_y_;

        scm::gl::sampler_state_ptr                  filter_nearest_;
        scm::gl::blend_state_ptr                    color_blending_state_;

        scm::gl::depth_stencil_state_ptr            depth_state_disable_;
        scm::gl::depth_stencil_state_ptr            depth_state_test_without_writing_;

        scm::gl::rasterizer_state_ptr               no_backface_culling_rasterizer_state_;

        //shader programs
        scm::gl::program_ptr                        visible_node_shader_program_;
        scm::gl::program_ptr                        node_texture_shader_program_;

        //framebuffer and textures for different passes
        scm::gl::frame_buffer_ptr                   visible_node_id_fbo_;
        scm::gl::texture_2d_ptr                     visible_node_depth_buffer_;
        scm::gl::texture_2d_ptr                     visible_node_id_texture_;
	
        scm::shared_ptr<scm::gl::quad_geometry>     screen_quad_;

        float                                       height_divided_by_top_minus_bottom_;  //frustum dependent
        float                                       near_plane_;                          //frustum dependent
        float                                       far_plane_;   
        float                                       point_size_factor_;

	    float                                       blending_threshold_;

        bool                                        render_bounding_boxes_;

        //variables related to text rendering
        scm::gl::text_renderer_ptr                              text_renderer_;
        scm::gl::text_ptr                                       renderable_text_;
        scm::time::accum_timer<scm::time::high_res_timer>       frame_time_;
        double                                                  fps_;
        unsigned long long                                      rendered_splats_;
        bool                                                    is_cut_update_active_;
        lamure::view_t                                          current_cam_id_;


        bool                                                    display_info_;

        std::vector<scm::math::mat4f>                           model_transformations_;
        std::vector<float>                                      radius_scale_per_model_;
        float                                                   radius_scale_;

        size_t                                                  elapsed_ms_since_cut_update_;

        RenderMode                                              render_mode_;

        std::set<lamure::model_t> visible_set_;
        std::set<lamure::model_t> invisible_set_;
        bool render_visible_set_;


//methods for changing rendering settings dynamically
public:
    void toggle_bounding_box_rendering();
    void change_point_size(float amount);
    void toggle_cut_update_info();
    void toggle_camera_info(const lamure::view_t current_cam_id);
    void toggle_visible_set();
    void toggle_display_info();

    lamure::pvs::id_histogram create_node_id_histogram(const bool& save_screenshot, const int& image_index) const;
    void compare_histogram_to_cut(const lamure::pvs::id_histogram& hist, const float& visibility_threshold);

    int get_rendered_node_count() const;
};

#endif // PVS_OLD_RENDERER_H_
