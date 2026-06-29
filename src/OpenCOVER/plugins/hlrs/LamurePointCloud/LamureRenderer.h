#ifndef _LAMURE_RENDERER_H
#define _LAMURE_RENDERER_H

//gl
#ifndef __gl_h_
#include <GL/glew.h>
#endif

// Platform-specific headers
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <array>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <unordered_set>
#include <chrono>
#include <limits>
#include <atomic>

#include <scm/core/math.h>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Camera>
#include <osg/GraphicsContext>
#include <osg/State>
#include <osg/Matrix>
#include <osg/Group>
#include <osgViewer/Viewer>
#include <osgViewer/Renderer>
#include <osgText/Text>
#include <scm/core/pointer_types.h>
#include <lamure/ren/bvh.h>
#include <lamure/ren/camera.h>

#include <lamure/types.h>
#include <osg/NodeCallback>

#include "LamureEditTool.h"

class Lamure;
class LamureEditTool;
class LamureRenderer;
struct InitDrawCallback;

class DispatchDrawCallback : public osg::Drawable::DrawCallback {
public:
    explicit DispatchDrawCallback(Lamure* plugin);
    void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override;

private:
    Lamure* _plugin{ nullptr };
    LamureRenderer* _renderer{ nullptr };
};

class CutsDrawCallback : public osg::Drawable::DrawCallback {
public:
    CutsDrawCallback(LamureRenderer* renderer) : m_renderer(renderer) {}
    void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override;
private:
    LamureRenderer* m_renderer;
};

// Data attached to each model node (Geode) to identify it
class LamureModelData : public osg::Referenced {
public:
    lamure::model_t modelId;
    LamureModelData(lamure::model_t id) : modelId(id) {}
};

// Callback to handle drawing per model
class PointsDrawCallback : public osg::Drawable::DrawCallback {
public:
    PointsDrawCallback(LamureRenderer* renderer) : m_renderer(renderer) {}
    virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const;
private:
    LamureRenderer* m_renderer;
};

class InitDrawCallback : public osg::Drawable::DrawCallback {
public:
    explicit InitDrawCallback(Lamure* plugin);
    void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override;

private:
    Lamure* _plugin{nullptr};
    LamureRenderer* _renderer{nullptr};
};

class StatsDrawCallback : public osg::Drawable::DrawCallback
{
public:
    StatsDrawCallback(Lamure* plugin, osgText::Text* label, osgText::Text* values);
    void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const override;

private:
    Lamure *_plugin{nullptr};
    osg::observer_ptr<osgText::Text> _label;
    osg::observer_ptr<osgText::Text> _values;
    LamureRenderer *_renderer{nullptr};
};

class FrustumDrawCallback : public osg::Drawable::DrawCallback
{
public:
    explicit FrustumDrawCallback(Lamure* plugin);
    void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override;

private:
    Lamure* _plugin{nullptr};
    LamureRenderer* _renderer{nullptr};
};

// Callback to draw BVH bounding boxes (culling/drawing logic aligned with _ref_LamureRenderer.cpp).
class BoundingBoxDrawCallback : public osg::Drawable::DrawCallback {
public:
    BoundingBoxDrawCallback(LamureRenderer* renderer) : m_renderer(renderer) {}
    void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override;

private:
    LamureRenderer* m_renderer;
};

class LamureRenderer {
 
 
private:
    friend struct InitDrawCallback;
    friend class DispatchDrawCallback;
    friend class CutsDrawCallback;
    friend class PointsDrawCallback;
    Lamure* m_plugin{nullptr};

    //osg::ref_ptr<osg::Group> m_group;

    void syncEditBrushGeometry();

    mutable std::mutex m_renderMutex;
    std::condition_variable m_renderCondition;
    bool m_pauseRequested{false};
    mutable std::mutex m_sceneMutex;
    mutable std::mutex m_shader_mutex;
    bool m_shader_sources_loaded{false};

    // Private methods
    bool readShader(const std::string& pathString, std::string& shaderString, bool keepOptionalShaderCode);

    GLuint compileAndLinkShaders(std::string vsSource, std::string fsSource, uint8_t ctxId, std::string desc = "");
    GLuint compileAndLinkShaders(std::string vsSource, std::string gsSource, std::string fsSource, uint8_t ctxId, std::string desc = "");
    unsigned int compileShader(unsigned int type, const std::string &source, uint8_t ctxId, std::string desc = "");
    void uploadClipPlanes(GLint countLocation, GLint dataLocation) const;

    // --- OpenGL helper structs for context-local resources ---
    struct GLGeo {
        GLuint vao{0};
        GLuint vbo{0};
        GLuint ibo{0};
        void destroy() {
            if (ibo) { glDeleteBuffers(1, &ibo); ibo = 0; }
            if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
            if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
        }
    };

    struct GLShader {
        GLuint program{0};
        void destroy() {
            if (program) {
                glDeleteProgram(program);
                program = 0;
            }
        }
    };

    struct PointShader : GLShader {
        GLint  mvp_matrix_loc{-1};
        GLint  model_matrix_loc{-1};
        GLint  clip_plane_count_loc{-1};
        GLint  clip_plane_data_loc{-1};
        GLint  max_radius_loc{-1};
        GLint  min_radius_loc{-1};
        GLint  max_screen_size_loc{-1};
        GLint  min_screen_size_loc{-1};
        GLint  scale_radius_loc{-1};
        GLint  scale_radius_gamma_loc{-1};
        GLint  max_radius_cut_loc{-1};
        GLint  scale_projection_loc{-1};
        GLint  proj_col0_loc{-1};
        GLint  proj_col1_loc{-1};
        GLint  viewport_half_y_loc{-1};
        GLint  use_aniso_loc{-1};
        GLint  aniso_normalize_loc{-1};
    };

    struct PointColorShader : GLShader {
        GLint mvp_matrix_loc{-1};
        GLint model_matrix_loc{-1};
        GLint clip_plane_count_loc{-1};
        GLint clip_plane_data_loc{-1};
        GLint view_matrix_loc{-1};
        GLint normal_matrix_loc{-1};
        GLint max_radius_loc{-1};
        GLint min_radius_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint scale_radius_gamma_loc{-1};
        GLint max_radius_cut_loc{-1};
        GLint scale_radius_loc{-1};
        GLint scale_projection_loc{-1};
        GLint proj_col0_loc{-1};
        GLint proj_col1_loc{-1};
        GLint viewport_half_y_loc{-1};
        GLint use_aniso_loc{-1};
        GLint aniso_normalize_loc{-1};
        GLint show_normals_loc{-1};
        GLint show_accuracy_loc{-1};
        GLint show_radius_dev_loc{-1};
        GLint show_output_sens_loc{-1};
        GLint accuracy_loc{-1};
        GLint average_radius_loc{-1};
    };

    struct PointColorLightingShader : GLShader {
        GLint mvp_matrix_loc{-1};
        GLint model_matrix_loc{-1};
        GLint clip_plane_count_loc{-1};
        GLint clip_plane_data_loc{-1};
        GLint view_matrix_loc{-1};
        GLint normal_matrix_loc{-1};
        GLint max_radius_loc{-1};
        GLint min_radius_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint scale_radius_gamma_loc{-1};
        GLint max_radius_cut_loc{-1};
        GLint scale_radius_loc{-1};
        GLint scale_projection_loc{-1};
        GLint proj_col0_loc{-1};
        GLint proj_col1_loc{-1};
        GLint viewport_half_y_loc{-1};
        GLint use_aniso_loc{-1};
        GLint aniso_normalize_loc{-1};
        GLint use_tone_mapping_loc{-1};
        GLint ambient_intensity_loc{-1};
        GLint specular_intensity_loc{-1};
        GLint shininess_loc{-1};
        GLint point_light_intensity_loc{-1};
        GLint point_light_pos_vs_loc{-1};
        GLint gamma_loc{-1};
        GLint show_normals_loc{-1};
        GLint show_accuracy_loc{-1};
        GLint show_radius_dev_loc{-1};
        GLint show_output_sens_loc{-1};
        GLint accuracy_loc{-1};
        GLint average_radius_loc{-1};
    };

    struct PointProvShader : GLShader {
        GLint mvp_matrix_loc{-1};
        GLint model_matrix_loc{-1};
        GLint clip_plane_count_loc{-1};
        GLint clip_plane_data_loc{-1};
        GLint max_radius_loc{-1};
        GLint min_radius_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint scale_radius_gamma_loc{-1};
        GLint max_radius_cut_loc{-1};
        GLint scale_radius_loc{-1};
        GLint scale_projection_loc{-1};
        GLint show_normals_loc{-1};
        GLint show_accuracy_loc{-1};
        GLint show_radius_dev_loc{-1};
        GLint show_output_sens_loc{-1};
        GLint accuracy_loc{-1};
        GLint average_radius_loc{-1};
        GLint channel_loc{-1};
        GLint heatmap_loc{-1};
        GLint heatmap_min_loc{-1};
        GLint heatmap_max_loc{-1};
        GLint heatmap_min_color_loc{-1};
        GLint heatmap_max_color_loc{-1};
    };

    struct SurfelShader : GLShader {
        GLint mvp_matrix_loc{-1};
        GLint model_view_matrix_loc{-1};
        GLint model_matrix_loc{-1};
        GLint clip_plane_count_loc{-1};
        GLint clip_plane_data_loc{-1};
        GLint max_radius_loc{-1};
        GLint min_radius_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint scale_radius_loc{-1};
        GLint scale_projection_loc{-1};
        GLint scale_radius_gamma_loc{-1};
        GLint max_radius_cut_loc{-1};
        GLint viewport_loc{-1};
        GLint use_aniso_loc{-1};
    };

    struct SurfelColorShader : GLShader {
        GLint mvp_matrix_loc{-1};
        GLint view_matrix_loc{-1};
        GLint model_view_matrix_loc{-1};
        GLint model_matrix_loc{-1};
        GLint clip_plane_count_loc{-1};
        GLint clip_plane_data_loc{-1};
        GLint normal_matrix_loc{-1};
        GLint min_radius_loc{-1};
        GLint max_radius_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint scale_radius_gamma_loc{-1};
        GLint max_radius_cut_loc{-1};
        GLint scale_radius_loc{-1};
        GLint viewport_loc{-1};
        GLint show_normals_loc{-1};
        GLint show_accuracy_loc{-1};
        GLint show_radius_dev_loc{-1};
        GLint show_output_sens_loc{-1};
        GLint accuracy_loc{-1};
        GLint average_radius_loc{-1};
        GLint scale_projection_loc{-1};
        GLint use_aniso_loc{-1};
    };

    struct SurfelColorLightingShader : GLShader {
        GLint mvp_matrix_loc{-1};
        GLint view_matrix_loc{-1};
        GLint model_view_matrix_loc{-1};
        GLint model_matrix_loc{-1};
        GLint clip_plane_count_loc{-1};
        GLint clip_plane_data_loc{-1};
        GLint normal_matrix_loc{-1};
        GLint max_radius_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint min_radius_loc{-1};
        GLint scale_radius_loc{-1};
        GLint scale_radius_gamma_loc{-1};
        GLint max_radius_cut_loc{-1};
        GLint viewport_loc{-1};
        GLint scale_projection_loc{-1};
        GLint show_normals_loc{-1};
        GLint show_accuracy_loc{-1};
        GLint show_radius_dev_loc{-1};
        GLint show_output_sens_loc{-1};
        GLint accuracy_loc{-1};
        GLint average_radius_loc{-1};
        GLint use_tone_mapping_loc{-1};
        GLint ambient_intensity_loc{-1};
        GLint specular_intensity_loc{-1};
        GLint shininess_loc{-1};
        GLint point_light_intensity_loc{-1};
        GLint point_light_pos_vs_loc{-1};
        GLint gamma_loc{-1};
        GLint use_aniso_loc{-1};
    };

    struct SurfelProvShader : GLShader {
        GLint mvp_matrix_loc{-1};
        GLint model_matrix_loc{-1};
        GLint clip_plane_count_loc{-1};
        GLint clip_plane_data_loc{-1};
        GLint min_radius_loc{-1};
        GLint max_radius_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint scale_radius_loc{-1};
        GLint scale_radius_gamma_loc{-1};
        GLint max_radius_cut_loc{-1};
        GLint viewport_loc{-1};
        GLint scale_projection_loc{-1};
        GLint show_normals_loc{-1};
        GLint show_accuracy_loc{-1};
        GLint show_radius_dev_loc{-1};
        GLint show_output_sens_loc{-1};
        GLint accuracy_loc{-1};
        GLint average_radius_loc{-1};
        GLint channel_loc{-1};
        GLint heatmap_loc{-1};
        GLint heatmap_min_loc{-1};
        GLint heatmap_max_loc{-1};
        GLint heatmap_min_color_loc{-1};
        GLint heatmap_max_color_loc{-1};
    };

    struct SurfelPass1Shader : GLShader {
        GLint mvp_matrix_loc{-1};
        GLint projection_matrix_loc{-1};
        GLint model_view_matrix_loc{-1};
        GLint model_matrix_loc{-1};
        GLint viewport_loc{-1};
        GLint use_aniso_loc{-1};
        GLint scale_projection_loc{-1};
        GLint scale_radius_gamma_loc{-1};
        GLint max_radius_cut_loc{-1};
        GLint max_radius_loc{-1};
        GLint min_radius_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint scale_radius_loc{-1};
    };

    struct SurfelPass2Shader : GLShader {
        GLint model_view_matrix_loc{-1};
        GLint projection_matrix_loc{-1};
        GLint normal_matrix_loc{-1};
        GLint depth_texture_loc{-1};
        GLint viewport_loc{-1};
        GLint use_aniso_loc{-1};
        GLint scale_projection_loc{-1};
        GLint max_radius_loc{-1};
        GLint min_radius_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint scale_radius_loc{-1};
        GLint scale_radius_gamma_loc{-1};
        GLint max_radius_cut_loc{-1};
        GLint show_normals_loc{-1};
        GLint show_accuracy_loc{-1};
        GLint show_radius_dev_loc{-1};
        GLint show_output_sens_loc{-1};
        GLint accuracy_loc{-1};
        GLint average_radius_loc{-1};
        GLint depth_range_loc{-1};
        GLint flank_lift_loc{-1};
        GLint coloring_loc{-1};
    };

    struct SurfelPass3Shader : GLShader {
        GLint in_color_texture_loc{-1};
        GLint in_normal_texture_loc{-1};
        GLint in_vs_position_texture_loc{-1};
        GLint in_depth_texture_loc{-1};
        GLint point_light_pos_vs_loc{-1};
        GLint point_light_intensity_loc{-1};
        GLint ambient_intensity_loc{-1};
        GLint specular_intensity_loc{-1};
        GLint shininess_loc{-1};
        GLint use_tone_mapping_loc{-1};
        GLint gamma_loc{-1};
        GLint lighting_loc{-1};
    };

    struct LineShader : GLShader {
        GLint in_color_location{-1};
        GLint mvp_matrix_location{-1};
    };

    static constexpr int kMaxClipPlanes = 6;
    std::array<scm::math::vec4f, kMaxClipPlanes> m_clip_planes;
    int m_clip_plane_count{0};
    int m_enabled_clip_distances{0};

    struct MultipassTarget {
        GLuint fbo = 0;
        GLuint texture_color = 0;
        GLuint texture_normal = 0;
        GLuint texture_position = 0;
        GLuint depth_texture = 0;
        int width = 0;
        int height = 0;
    };

    struct ContextResources {
        uint8_t ctx = -1;
        bool rendering = false;
        bool rendering_allowed = true;
        uint32_t frames_pending_drain = 0;
        // Initialization flags
        bool resources_initialized{false}; // VBOs, VAOs (Volatile: reset on model change)
        bool shaders_initialized{false};   // Programs (Persistent: keep across model change)
        bool initialized{false};           // Base context (Schism, Camera)

        // Geometry
        GLGeo geo_box;
        GLGeo geo_frustum;
        GLGeo geo_screen_quad;
        GLGeo geo_text;
        std::array<unsigned short, 24> box_idx = {{
            0, 1, 2, 3, 4, 5, 6, 7,
            0, 2, 1, 3, 4, 6, 5, 7,
            0, 4, 1, 5, 2, 6, 3, 7,
        }};
        std::array<unsigned short, 24> frustum_idx = {{
            0, 1, 2, 3, 4, 5, 6, 7,
            0, 2, 1, 3, 4, 6, 5, 7,
            0, 4, 1, 5, 2, 6, 3, 7,
        }};
        std::array<float, 24> frustum_vertices{{}};
        std::array<float, 18> screen_quad_vertex = {{
            -1.0f,  1.0f, 0.0f,   -1.0f, -1.0f, 0.0f,    1.0f, -1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,    1.0f, -1.0f, 0.0f,    1.0f,  1.0f, 0.0f
        }};
        std::string text;
        size_t num_text_vertices{0};
        GLuint text_atlas_texture{0};

        // Shaders
        PointShader sh_point;
        PointColorShader sh_point_color;
        PointColorLightingShader sh_point_color_lighting;
        PointProvShader sh_point_prov;
        SurfelShader sh_surfel;
        SurfelColorShader sh_surfel_color;
        SurfelColorLightingShader sh_surfel_color_lighting;
        SurfelProvShader sh_surfel_prov;
        SurfelPass1Shader sh_surfel_pass1;
        SurfelPass2Shader sh_surfel_pass2;
        SurfelPass3Shader sh_surfel_pass3;
        LineShader sh_line;
        GLShader sh_coverage_query;

        struct PixelQuerySlot {
            GLuint query_id{0};
            uint64_t frame_number{std::numeric_limits<uint64_t>::max()};
            double viewport_pixels{-1.0};
            bool issued{false};
            bool is_coverage{false};
        };
        static constexpr size_t kPixelQuerySlotCount = 128;
        std::array<PixelQuerySlot, kPixelQuerySlotCount> pixel_query_slots{};
        uint32_t pixel_query_next_slot{0};
        bool pixel_metrics_checked{false};
        bool pixel_metrics_supported{false};
        bool pixel_queries_ready{false};
        GLuint pixel_stencil_bit{0};
        uint64_t pixel_current_frame{std::numeric_limits<uint64_t>::max()};
        bool pixel_stencil_needs_clear{false};

        bool pixel_capture_active{false};
        GLuint pixel_total_query_active{0};
        double pixel_capture_viewport_pixels{-1.0};
        uint64_t pixel_capture_frame{std::numeric_limits<uint64_t>::max()};
        int pixel_capture_view_id{-1};
        bool pixel_capture_used_stencil{false};
        bool pixel_warned_no_stencil{false};
        uint64_t pixel_aggregate_frame{std::numeric_limits<uint64_t>::max()};
        double pixel_viewport_sum{0.0};
        std::unordered_set<int> pixel_viewport_accounted;
        std::unordered_map<int, double> pixel_view_covered_samples;

        GLboolean pixel_prev_stencil_test{GL_FALSE};
        GLint pixel_prev_stencil_func{GL_ALWAYS};
        GLint pixel_prev_stencil_ref{0};
        GLint pixel_prev_stencil_value_mask{~0};
        GLint pixel_prev_stencil_writemask{~0};
        GLint pixel_prev_stencil_fail{GL_KEEP};
        GLint pixel_prev_stencil_zfail{GL_KEEP};
        GLint pixel_prev_stencil_zpass{GL_KEEP};

        // FBOs
        std::unordered_map<int, MultipassTarget> multipass_targets;
        scm::gl::render_device_ptr  scm_device;
        scm::gl::render_context_ptr scm_context;
        std::unordered_map<const osg::Camera*, std::shared_ptr<lamure::ren::camera>> scm_cameras;
        std::unordered_map<const osg::Camera*, int> view_ids;
        std::mutex callback_mutex;
        std::unordered_map<const osg::GraphicsContext*, GLuint> point_vaos;
        std::unordered_map<const osg::GraphicsContext*, GLuint> box_vaos;
        std::unordered_map<const osg::GraphicsContext*, GLuint> screen_quad_vaos;
        uint64_t dispatch_frame{std::numeric_limits<uint64_t>::max()};
        bool dispatch_done{false};
        std::unordered_map<const osg::Camera*, osg::ref_ptr<osg::Camera>> hud_cameras;
        bool gpu_info_logged{false};
        bool gpu_consistency_checked{false};
        std::string gpu_uuid;
        std::string driver_uuid;
        std::string gpu_vendor;
        std::string gpu_renderer;
        std::string gpu_version;
        std::string gpu_key;

        // Batch draw vectors (per-context to avoid thread-unsafe static locals)
        std::vector<GLint> batch_firsts;
        std::vector<GLsizei> batch_counts;
#ifdef _OPENMP
        std::vector<std::vector<GLint>> tls_firsts;
        std::vector<std::vector<GLsizei>> tls_counts;
#endif
    };


    std::map<int, ContextResources> m_ctx_res;
    std::mutex m_ctx_mutex;
    std::atomic<bool> m_gpu_org_ready{false};

    // Schism objects
    std::unordered_set<int> m_initialized_context_ids;
    std::unordered_set<const void*> m_initialized_context_ptrs;

    osg::ref_ptr<osg::Camera>   m_osg_camera;


    osg::ref_ptr<osg::Group> m_frustum_group;

    // Geodes
    osg::ref_ptr<osg::Geode> m_init_geode;
    osg::ref_ptr<osg::Geode> m_dispatch_geode;
    osg::ref_ptr<osg::Geode> m_stats_geode;
    osg::ref_ptr<osg::Geode> m_frustum_geode;
    osg::ref_ptr<osg::Geode> m_edit_brush_geode;
    osg::ref_ptr<osg::MatrixTransform> m_edit_brush_transform;

    // Stateset
    osg::ref_ptr<osg::StateSet> m_init_stateset;
    osg::ref_ptr<osg::StateSet> m_dispatch_stateset;
    osg::ref_ptr<osg::StateSet> m_stats_stateset;
    osg::ref_ptr<osg::StateSet> m_frustum_stateset;

    // Geometry
    osg::ref_ptr<osg::Geometry> m_init_geometry;
    osg::ref_ptr<osg::Geometry> m_dispatch_geometry;
    osg::ref_ptr<osg::Geometry> m_frustum_geometry;

    // Framebuffers
    scm::gl::frame_buffer_ptr fbo;
    scm::gl::texture_2d_ptr fbo_color_buffer;
    scm::gl::texture_2d_ptr fbo_depth_buffer;
    scm::gl::frame_buffer_ptr pass1_fbo;
    scm::gl::frame_buffer_ptr pass2_fbo;
    scm::gl::frame_buffer_ptr pass3_fbo;
    scm::gl::texture_2d_ptr pass1_depth_buffer;
    scm::gl::texture_2d_ptr pass2_color_buffer;
    scm::gl::texture_2d_ptr pass2_normal_buffer;
    scm::gl::texture_2d_ptr pass2_view_space_pos_buffer;
    scm::gl::texture_2d_ptr pass2_depth_buffer;

    // Render states
    scm::gl::depth_stencil_state_ptr depth_state_disable;
    scm::gl::depth_stencil_state_ptr depth_state_less;
    scm::gl::depth_stencil_state_ptr depth_state_without_writing;
    scm::gl::rasterizer_state_ptr no_backface_culling_rasterizer_state;
    scm::gl::blend_state_ptr color_blending_state;
    scm::gl::blend_state_ptr color_no_blending_state;
    scm::gl::sampler_state_ptr filter_linear;
    scm::gl::sampler_state_ptr filter_nearest;
    scm::gl::sampler_state_ptr vt_filter_linear;
    scm::gl::sampler_state_ptr vt_filter_nearest;
    scm::gl::texture_2d_ptr bg_texture;

    // Shader sources
    std::string vis_point_vs_source;
    std::string vis_point_fs_source;
    std::string vis_point_color_vs_source;
    std::string vis_point_color_fs_source;
    std::string vis_point_prov_vs_source;
    std::string vis_point_prov_fs_source;
    std::string vis_point_color_lighting_vs_source;
    std::string vis_point_color_lighting_fs_source;
    std::string vis_surfel_vs_source;
    std::string vis_surfel_gs_source;
    std::string vis_surfel_fs_source;
    std::string vis_surfel_color_lighting_vs_source;
    std::string vis_surfel_color_lighting_gs_source;
    std::string vis_surfel_color_lighting_fs_source;
    std::string vis_surfel_color_vs_source;
    std::string vis_surfel_color_gs_source;
    std::string vis_surfel_color_fs_source;
    std::string vis_surfel_prov_vs_source;
    std::string vis_surfel_prov_gs_source;
    std::string vis_surfel_prov_fs_source;
    std::string vis_quad_vs_source;
    std::string vis_quad_fs_source;
    std::string vis_line_vs_source;
    std::string vis_line_fs_source;
    std::string vis_triangle_vs_source;
    std::string vis_triangle_fs_source;
    std::string vis_plane_vs_source;
    std::string vis_plane_fs_source;
    std::string vis_text_vs_source;
    std::string vis_text_fs_source;
    std::string vis_vt_vs_source;
    std::string vis_vt_fs_source;
    std::string vis_xyz_vs_source;
    std::string vis_xyz_gs_source;
    std::string vis_xyz_fs_source;
    std::string vis_surfel_pass1_vs_source;
    std::string vis_surfel_pass1_gs_source;
    std::string vis_surfel_pass1_fs_source;
    std::string vis_surfel_pass2_vs_source;
    std::string vis_surfel_pass2_gs_source;
    std::string vis_surfel_pass2_fs_source;
    std::string vis_surfel_pass3_vs_source;
    std::string vis_surfel_pass3_fs_source;
    std::string vis_debug_vs_source;
    std::string vis_debug_fs_source;
    std::string vis_xyz_qz_vs_source;
    std::string vis_xyz_qz_pass1_vs_source;
    std::string vis_xyz_qz_pass2_vs_source;
    std::string vis_box_vs_source;
    std::string vis_box_gs_source;
    std::string vis_box_fs_source;
    std::string vis_xyz_vs_lighting_source;
    std::string vis_xyz_gs_lighting_source;
    std::string vis_xyz_fs_lighting_source;
    std::string vis_xyz_pass2_vs_lighting_source;
    std::string vis_xyz_pass2_gs_lighting_source;
    std::string vis_xyz_pass2_fs_lighting_source;
    std::string vis_xyz_pass3_vs_lighting_source;
    std::string vis_xyz_pass3_fs_lighting_source;

    const osg::Camera* m_active_sync_camera{nullptr};
    bool m_has_frozen_matrices{false};
    bool m_last_sync_state{false};

public:
    Lamure* getPlugin() const noexcept { return m_plugin; }
    LamureRenderer(Lamure* lamure_plugin);
    ~LamureRenderer();
    bool notifyOn() const;
    bool gpuOrganizationReady() const { return m_gpu_org_ready.load(std::memory_order_acquire); }

    void init();
    void shutdown();
    void detachCallbacks();
    void syncHudCameras();

    bool beginFrame(int ctxId);
    void endFrame(int ctxId);
    bool pauseAndDrainFrames(uint32_t extraDrainFrames);
    void resumeRendering();
    bool isRendering() const;
    bool getModelMatrix(const osg::Node* node, osg::Matrixd& out) const;
    bool getModelViewProjectionFromRenderInfo(osg::RenderInfo& renderInfo, const osg::Node* node, osg::Matrixd& outModel, osg::Matrixd& outView, osg::Matrixd& outProj) const;
    void updateScmCameraFromRenderInfo(osg::RenderInfo& renderInfo, int ctxId);

    bool initCamera(ContextResources& res, osg::Camera* contextCamera, int viewId);
    void initFrustumResources(ContextResources& res);
    bool initLamureShader(ContextResources& res);
    void initSchismObjects(ContextResources& res);
    bool initGpus(ContextResources& res);
    void initUniforms(ContextResources& res);
    void initBoxResources(ContextResources& res);
    void initPclResources(ContextResources& res);
    void releaseMultipassTargets();
    void initializeMultipassTarget(MultipassTarget& target, int width, int height);
    void destroyMultipassTarget(MultipassTarget& target);

    lamure::ren::camera* getScmCamera(int ctxId, const osg::Camera* camera) {
        if (!camera) return nullptr;
        auto& res = getResources(ctxId);
        auto it = res.scm_cameras.find(camera);
        return (it != res.scm_cameras.end() && it->second) ? it->second.get() : nullptr;
    }

    const lamure::ren::camera* getScmCamera(int ctxId, const osg::Camera* camera) const {
        if (!camera) return nullptr;
        const auto& res = getResources(ctxId);
        auto it = res.scm_cameras.find(camera);
        return (it != res.scm_cameras.end() && it->second) ? it->second.get() : nullptr;
    }

    MultipassTarget& acquireMultipassTarget(int ctxId, int viewId, int width, int height);
    osg::ref_ptr<osg::Geode> getStatsGeode() { return m_stats_geode; }
    osg::ref_ptr<osg::Geode> getFrustumGeode() { return m_frustum_geode; }
    osg::ref_ptr<osg::MatrixTransform> getEditBrushTransform() { return m_edit_brush_transform; }
    osg::MatrixTransform* ensureEditBrushNode(osg::Group* parent);
    osg::Matrixd composeEditBrushMatrix(const osg::Matrixd& interactorMat, const osg::Matrixd& parentWorld, const osg::Matrixd& invRoot, double hullScale) const;
    osg::Matrixd updateEditBrushFromInteractor(const osg::Matrixd& interactorMat, const osg::Matrixd& invRoot, double hullScale);
    void setEditBrushTransform(const osg::Matrixd& brushMat);
    void destroyEditBrushNode(osg::Group* parent);

    scm::gl::render_device_ptr getDevice(int ctxId) { return getResources(ctxId).scm_device; }
    scm::gl::render_context_ptr getSchismContext(int ctxId) { return getResources(ctxId).scm_context; }

    enum class ShaderType {
        Point,
        PointColor,
        PointColorLighting,
        PointProv,
        Surfel,
        SurfelColor,
        SurfelColorLighting,
        SurfelProv,
        SurfelMultipass
    };

    struct ShaderInfo {
        ShaderType type;
        std::string name;
    };

    const std::vector<ShaderInfo> pcl_shader = {
        {ShaderType::Point,                 "Point"},
        {ShaderType::PointColor,            "Point Color"},
        {ShaderType::PointColorLighting,    "Point Color Lighting"},
        {ShaderType::PointProv,             "Point Prov"},
        {ShaderType::Surfel,                "Surfel"},
        {ShaderType::SurfelColor,           "Surfel Color"},
        {ShaderType::SurfelColorLighting,   "Surfel Color Lighting"},
        {ShaderType::SurfelProv,            "Surfel Prov"},
        {ShaderType::SurfelMultipass,       "Surfel Multipass"}
    };

    ShaderType m_active_shader_type = ShaderType::Point;

    const std::vector<ShaderInfo>& getPclShader() const { return pcl_shader; }
    
    // Shared CPU data (prepared once, read by all contexts)
    std::map<uint32_t, std::vector<uint32_t>> m_bvh_node_vertex_offsets;
    std::vector<float> m_shared_box_vertices;

    // Call this from main thread when models change, before rendering resumes
    void updateSharedBoxData();

    void setActiveShaderType(ShaderType t) { m_active_shader_type = t; }
    ShaderType getActiveShaderType() const { return m_active_shader_type; }
    void setFrameUniforms(const scm::math::mat4& projection_matrix, const scm::math::mat4& view_matrix, const scm::math::vec2& viewport, ContextResources& ctx);
    void setModelUniforms(const scm::math::mat4& mvp_matrix, const scm::math::mat4& model_matrix, ContextResources& ctx);
    void setNodeUniforms(const lamure::ren::bvh* bvh, uint32_t node_id, ContextResources& ctx);
    bool isModelVisible(std::size_t modelIndex) const;
    inline osg::GraphicsContext* getGC(osg::RenderInfo& ri) const {
        return ri.getState() ? ri.getState()->getGraphicsContext() : nullptr;
    }
    
    void getMatricesFromRenderInfo(osg::RenderInfo& renderInfo, osg::Matrixd& outView, osg::Matrixd& outProj);
    int resolveViewId(osg::RenderInfo& renderInfo) const;

    void updateActiveClipPlanes();
    void enableClipDistances();
    void disableClipDistances();
    int clipPlaneCount() const { return m_clip_plane_count; }
    bool supportsClipPlanes(ShaderType type) const;

    struct ContextTimingSample {
        uint64_t frame_number{std::numeric_limits<uint64_t>::max()};
        uint64_t rendered_primitives{0};
        uint64_t rendered_nodes{0};
        uint64_t rendered_bounding_boxes{0};
        double dispatch_ms{-1.0};
        double context_update_ms{-1.0};
        double cpu_cull_ms{-1.0};
        double cpu_draw_ms{-1.0};
        double gpu_ms{-1.0};
        double render_cpu_ms{-1.0};
        double samples_passed{-1.0};
        double covered_samples{-1.0};
        double viewport_pixels{-1.0};
        double coverage{-1.0};
        double overdraw{-1.0};
    };

    struct TimingSnapshot {
        uint64_t frame_number{0};
        uint64_t rendered_primitives{0};
        uint64_t rendered_nodes{0};
        uint64_t rendered_bounding_boxes{0};
        double cpu_update_ms{-1.0};
        double wait_ms{-1.0};
        double dispatch_ms{-1.0};
        double context_update_ms{-1.0};
        double cpu_cull_ms{-1.0};
        double cpu_draw_ms{-1.0};
        double gpu_ms{-1.0};
        double render_cpu_ms{-1.0};
        double samples_passed{-1.0};
        double covered_samples{-1.0};
        double viewport_pixels{-1.0};
        double coverage{-1.0};
        double overdraw{-1.0};
        std::vector<std::pair<int, ContextTimingSample>> per_context;
    };

    ContextResources& getResources(int ctxId);
    const ContextResources& getResources(int id) const {
        return const_cast<LamureRenderer*>(this)->getResources(id);
    }

    bool isTimingModeActive() const;
    void noteContextUpdateMs(int ctxId, uint64_t frameNo, double ms);
    void noteDispatchMs(int ctxId, uint64_t frameNo, double ms);
    void noteContextStats(int ctxId, uint64_t frameNo, double cullMs, double drawMs, double gpuMs);
    void noteContextPixelStats(int ctxId, uint64_t frameNo, double samplesPassed, double coveredSamples, double viewportPixels);
    void noteContextRenderCounts(int ctxId, uint64_t frameNo, uint64_t renderedPrimitives, uint64_t renderedNodes, uint64_t renderedBoundingBoxes);
    void noteGlobalStats(uint64_t frameNo, double cpuUpdateMs, double waitMs);
    void commitFrameTiming(int ctxId, uint64_t frameNo, const ContextTimingSample& sample);
    bool getDisplayTimingData(int ctxId, uint64_t currentFrameNo, ContextTimingSample& outContext, TimingSnapshot& outSummed) const;
    TimingSnapshot getTimingSnapshot(uint64_t preferredFrame = std::numeric_limits<uint64_t>::max()) const;
    std::string getTimingCompactString(uint64_t preferredFrame = std::numeric_limits<uint64_t>::max()) const;
    void updateLiveTimingFromRenderInfo(osg::RenderInfo& renderInfo, int ctxId);
    bool beginPixelMetricsCapture(int ctxId, uint64_t frameNo, int viewId, double viewportPixels);
    void endPixelMetricsCapture(int ctxId);
    void releasePixelMetricsQueries(ContextResources& res);

    const PointShader&                  getPointShader(int ctxId)                const { return getResources(ctxId).sh_point; }
    const PointColorShader&             getPointColorShader(int ctxId)           const { return getResources(ctxId).sh_point_color; }
    const PointColorLightingShader&     getPointColorLightingShader(int ctxId)   const { return getResources(ctxId).sh_point_color_lighting; }
    const PointProvShader&              getPointProvShader(int ctxId)            const { return getResources(ctxId).sh_point_prov; }

    const SurfelShader&                 getSurfelShader(int ctxId)               const { return getResources(ctxId).sh_surfel; }
    const SurfelColorShader&            getSurfelColorShader(int ctxId)          const { return getResources(ctxId).sh_surfel_color; }
    const SurfelColorLightingShader&    getSurfelColorLightingShader(int ctxId)  const { return getResources(ctxId).sh_surfel_color_lighting; }
    const SurfelProvShader&             getSurfelProvShader(int ctxId)           const { return getResources(ctxId).sh_surfel_prov; }

    const SurfelPass1Shader&            getSurfelPass1Shader(int ctxId)          const { return getResources(ctxId).sh_surfel_pass1; }
    const SurfelPass2Shader&            getSurfelPass2Shader(int ctxId)          const { return getResources(ctxId).sh_surfel_pass2; }
    const SurfelPass3Shader&            getSurfelPass3Shader(int ctxId)          const { return getResources(ctxId).sh_surfel_pass3; }

    const LineShader&                   getLineShader(int ctxId)                 const { return getResources(ctxId).sh_line; }

private:

    struct GlobalTimingSample {
        double cpu_update_ms{-1.0};
        double wait_ms{-1.0};
    };

    ContextTimingSample& upsertTimingSampleLocked(int ctxId, uint64_t frameNo);
    void trimTimingHistoryLocked();

    std::map<uint64_t, std::unordered_map<int, ContextTimingSample>> m_timing_frames;
    std::map<uint64_t, GlobalTimingSample> m_timing_global_by_frame;
    std::unordered_map<int, uint64_t> m_timing_latest_frame_by_ctx;
    mutable std::mutex m_timing_mutex;
    mutable uint64_t m_last_complete_timing_frame{std::numeric_limits<uint64_t>::max()};
    uint64_t m_live_timing_last_scanned_frame{std::numeric_limits<uint64_t>::max()};
    static constexpr size_t kTimingHistoryLimit = 256;
    bool m_stats_initialized{false};

};

#endif // _LAMURE_RENDERER_H
