#ifndef _LAMURE_RENDERER_H
#define _LAMURE_RENDERER_H

//gl
#ifndef __gl_h_
#include <GL/glew.h>
#endif

// Platform-specific headers
#ifdef _WIN32
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

#include <scm/core/math.h>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Camera>
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

class TextDrawCallback : public osg::Drawable::DrawCallback
{
public:
    TextDrawCallback(Lamure* plugin, osgText::Text* values);
    void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const override;

private:
    Lamure *_plugin{nullptr};
    osg::ref_ptr<osgText::Text> _values;
    LamureRenderer *_renderer{nullptr};
    mutable std::chrono::steady_clock::time_point _lastUpdateTime;
    std::chrono::milliseconds _minInterval;
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
    friend class PointsDrawCallback;
    Lamure* m_plugin{nullptr};
    LamureRenderer* m_renderer{nullptr};

    //osg::ref_ptr<osg::Group> m_group;

    void syncEditBrushGeometry();

    mutable std::mutex m_renderMutex;
    std::condition_variable m_renderCondition;
    bool m_renderingAllowed{true};
    bool m_pauseRequested{false};
    uint32_t m_framesPendingDrain{0};
    mutable std::mutex m_sceneMutex;

    // Private methods
    bool readShader(const std::string& pathString, std::string& shaderString, bool keepOptionalShaderCode);

    GLuint compileAndLinkShaders(std::string vsSource, std::string fsSource, uint8_t ctxId);
    GLuint compileAndLinkShaders(std::string vsSource, std::string gsSource, std::string fsSource, uint8_t ctxId);
    unsigned int compileShader(unsigned int type, const std::string& source, uint8_t ctxId);
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

    struct MultipassTargetKey {
        lamure::context_t context = 0;
        const osg::Camera* camera = nullptr;
        bool operator==(const MultipassTargetKey& other) const noexcept {
            return context == other.context && camera == other.camera;
        }
    };

    struct MultipassTargetKeyHash {
        std::size_t operator()(const MultipassTargetKey& key) const noexcept {
            std::size_t h1 = static_cast<std::size_t>(key.context);
            std::size_t h2 = std::hash<const void*>{}(static_cast<const void*>(key.camera));
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };

    struct ContextResources {
        uint8_t ctx = -1;
        int view_id = -1;
        bool rendering = false;
        // Geometry
        GLGeo geo_box;
        GLGeo geo_frustum;
        GLGeo geo_screen_quad;
        GLGeo geo_text;
        GLuint vao_pointcloud{0};
        bool vao_initialized{false};
        bool initialized{false};
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

        // FBOs
        std::unordered_map<MultipassTargetKey, MultipassTarget, MultipassTargetKeyHash> multipass_targets;
        scm::gl::render_device_ptr  scm_device;
        scm::gl::render_context_ptr scm_context;

        std::unique_ptr<lamure::ren::camera> scm_camera;
        bool dump_done = false;
        uint64_t last_camera_frame{std::numeric_limits<uint64_t>::max()};
    };

    std::map<int, ContextResources> m_ctx_res;
    std::mutex m_ctx_mutex;

    // Schism objects
    std::unordered_set<int> m_initialized_context_ids;
    std::unordered_set<const void*> m_initialized_context_ptrs;

    osg::ref_ptr<osg::Camera>   m_osg_camera;
    osg::ref_ptr<osg::Camera>   m_hud_camera;


    osg::ref_ptr<osg::Group> m_frustum_group;

    // Geodes
    osg::ref_ptr<osg::Geode> m_init_geode;
    osg::ref_ptr<osg::Geode> m_text_geode;
    osg::ref_ptr<osg::Geode> m_frustum_geode;
    osg::ref_ptr<osg::Geode> m_edit_brush_geode;
    osg::ref_ptr<osg::MatrixTransform> m_edit_brush_transform;

    // Stateset
    osg::ref_ptr<osg::StateSet> m_init_stateset;
    osg::ref_ptr<osg::StateSet> m_text_stateset;
    osg::ref_ptr<osg::StateSet> m_frustum_stateset;

    // Geometry
    osg::ref_ptr<osg::Geometry> m_init_geometry;
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

    void init();
    void shutdown();
    void detachCallbacks();

    bool beginFrame(int ctxId);
    void endFrame(int ctxId);
    bool pauseAndWaitForIdle(uint32_t extraDrainFrames);
    void resumeRendering();
    bool isRendering() const;
    const osg::MatrixTransform* getModelTransform(const osg::Node* node) const;
    void updateScmCameraFromRenderInfo(osg::RenderInfo& renderInfo, int ctxId);

    void initCamera(ContextResources& res);
    void initFrustumResources(ContextResources& res);
    void initLamureShader(ContextResources& res);
    void initSchismObjects(ContextResources& res);
    void initUniforms(ContextResources& res);
    void initBoxResources(ContextResources& res);
    void initPclResources(ContextResources& res);
    void releaseMultipassTargets();
    void initializeMultipassTarget(MultipassTarget& target, int width, int height);
    void destroyMultipassTarget(MultipassTarget& target);

    lamure::ren::camera* getScmCamera(int ctxId) {
        return getResources(ctxId).scm_camera.get();
    }

    const lamure::ren::camera* getScmCamera(int ctxId) const { return getResources(ctxId).scm_camera.get(); }

    MultipassTarget& acquireMultipassTarget(lamure::context_t contextID, const osg::Camera* camera, int width, int height);
    osg::ref_ptr<osg::Geode> getTextGeode() { return m_text_geode; }
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
    std::map<uint32_t, std::vector<uint32_t>> m_bvh_node_vertex_offsets;

    void setActiveShaderType(ShaderType t) { m_active_shader_type = t; }
    ShaderType getActiveShaderType() const { return m_active_shader_type; }
    void setFrameUniforms(const scm::math::mat4& projection_matrix, const scm::math::mat4& view_matrix, const scm::math::vec2& viewport, ContextResources& ctx);
    void setModelUniforms(const scm::math::mat4& mvp_matrix, const scm::math::mat4& model_matrix, ContextResources& ctx);
    void setNodeUniforms(const lamure::ren::bvh* bvh, uint32_t node_id, ContextResources& ctx);
    bool isModelVisible(std::size_t modelIndex) const;
    
    void getMatricesFromRenderInfo(osg::RenderInfo& renderInfo, osg::Matrixd& outView, osg::Matrixd& outProj);

    void updateActiveClipPlanes();
    void enableClipDistances();
    void disableClipDistances();
    int clipPlaneCount() const { return m_clip_plane_count; }
    bool supportsClipPlanes(ShaderType type) const;

    ContextResources& getResources(int ctxId);
    const ContextResources& getResources(int id) const {
        return const_cast<LamureRenderer*>(this)->getResources(id);
    }

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

};

#endif // _LAMURE_RENDERER_H
