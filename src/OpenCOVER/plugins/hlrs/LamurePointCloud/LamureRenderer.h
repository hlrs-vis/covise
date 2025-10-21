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

#include <scm/core/math.h>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Geode>
#include <osg/Camera>
#include <osg/State>
#include <osgViewer/Viewer>
#include <osgViewer/Renderer>
#include <osgText/Text>
#include <scm/core/pointer_types.h>
#include <lamure/ren/bvh.h>
#include <lamure/ren/camera.h>

class Lamure;
class LamureRenderer {


private:
    Lamure* m_plugin{nullptr};
    LamureRenderer* m_renderer{nullptr};

    //osg::ref_ptr<osg::Group> m_group;

    void flushGlCommands();
    void releaseSceneGraph();

    struct CameraBinding;
    CameraBinding& ensureCameraBinding(const osg::Camera* osgCamera);
    lamure::ren::camera* bindActiveCamera(const osg::Camera* osgCamera,
                                          const scm::math::mat4d& modelview,
                                          const scm::math::mat4d& projection,
                                          bool updateViewMatrix);
    void releaseCameraBindings();
    bool isHudCamera(const osg::Camera* camera) const;

    bool m_rendering{false};
    mutable std::mutex m_renderMutex;
    std::condition_variable m_renderCondition;
    bool m_renderingAllowed{true};
    bool m_pauseRequested{false};
    uint32_t m_framesPendingDrain{0};
    mutable std::mutex m_sceneMutex;

    // Private methods
    bool readShader(const std::string& pathString, std::string& shaderString, bool keepOptionalShaderCode);
    void initCamera();
    void printSettings() const;

    GLuint compileAndLinkShaders(std::string vsSource, std::string fsSource);
    GLuint compileAndLinkShaders(std::string vsSource, std::string gsSource, std::string fsSource);
    unsigned int createShader(const std::string& vertexShader, const std::string& fragmentShader, uint8_t ctxId);
    unsigned int compileShader(unsigned int type, const std::string& source, uint8_t ctxId);

    // Resource structs
    struct pcl_resource;
    struct box_resource;
    struct plane_resource;
    struct sphere_resource;
    struct coord_recourse;
    struct frustum_resource;
    struct text_resource;

    struct PointShader {
        GLuint program{0};
        GLint  mvp_matrix_loc{-1};
        GLint  max_radius_loc{-1};
        GLint  min_radius_loc{-1};
        GLint  max_screen_size_loc{-1};
        GLint  min_screen_size_loc{-1};
        GLint  scale_radius_loc{-1};
        GLint scale_radius_gamma_loc    {-1};
        GLint max_radius_cut_loc        {-1};
        GLint  scale_projection_loc{-1};
    };
    PointShader m_point_shader;

    struct PointColorShader {
        GLuint program{0};
        GLint mvp_matrix_loc            {-1}; // mat4  mvp_matrix
        GLint view_matrix_loc           {-1}; // mat4  view_matrix
        GLint normal_matrix_loc         {-1}; // mat3  normal_matrix
        GLint max_radius_loc            {-1}; // float max_radius
        GLint min_radius_loc            {-1}; // float min_radius
        GLint max_screen_size_loc       {-1};
        GLint min_screen_size_loc       {-1};
        GLint scale_radius_gamma_loc    {-1};
        GLint max_radius_cut_loc        {-1};
        GLint scale_radius_loc          {-1}; // float scale_radius
        GLint scale_projection_loc      {-1}; // float scale_projection
        GLint show_normals_loc          {-1}; // bool  show_normals
        GLint show_accuracy_loc         {-1}; // bool  show_accuracy
        GLint show_radius_dev_loc       {-1}; // bool  show_radius_deviation
        GLint show_output_sens_loc      {-1}; // bool  show_output_sensitivity
        GLint accuracy_loc              {-1}; // float accuracy
        GLint average_radius_loc        {-1}; // float average_radius
    };
    PointColorShader m_point_color_shader;

    struct PointColorLightingShader {
        GLuint program{0};
        GLint mvp_matrix_loc            {-1};
        GLint view_matrix_loc           {-1};
        GLint normal_matrix_loc         {-1};
        GLint max_radius_loc            {-1};
        GLint min_radius_loc            {-1};
        GLint max_screen_size_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint scale_radius_gamma_loc    {-1};
        GLint max_radius_cut_loc        {-1};
        GLint scale_radius_loc          {-1};
        GLint scale_projection_loc      {-1};
        // --- Unified float-based lighting uniforms ---
        GLint use_tone_mapping_loc      {-1};
        GLint ambient_intensity_loc     {-1};
        GLint specular_intensity_loc    {-1};
        GLint shininess_loc             {-1};
        GLint point_light_intensity_loc {-1};
        GLint point_light_pos_vs_loc    {-1};
        GLint gamma_loc                 {-1};
        GLint show_normals_loc          {-1};
        GLint show_accuracy_loc         {-1};
        GLint show_radius_dev_loc       {-1};
        GLint show_output_sens_loc      {-1};
        GLint accuracy_loc              {-1};
        GLint average_radius_loc        {-1};
    };
    PointColorLightingShader m_point_color_lighting_shader;

    struct PointProvShader {
        GLuint program{0};
        GLint mvp_matrix_loc            {-1};
        GLint max_radius_loc            {-1};
        GLint min_radius_loc            {-1};
        GLint max_screen_size_loc       {-1};
        GLint min_screen_size_loc       {-1};
        GLint scale_radius_gamma_loc    {-1};
        GLint max_radius_cut_loc        {-1};
        GLint scale_radius_loc          {-1};
        GLint scale_projection_loc      {-1};
        GLint show_normals_loc          {-1}; // bool  show_normals
        GLint show_accuracy_loc         {-1}; // bool  show_accuracy
        GLint show_radius_dev_loc       {-1}; // bool  show_radius_deviation
        GLint show_output_sens_loc      {-1}; // bool  show_output_sensitivity
        GLint accuracy_loc              {-1}; // float accuracy
        GLint average_radius_loc        {-1}; // float average_radius
        GLint channel_loc;
        GLint heatmap_loc;
        GLint heatmap_min_loc;
        GLint heatmap_max_loc;
        GLint heatmap_min_color_loc;
        GLint heatmap_max_color_loc;
    };
    PointProvShader m_point_prov_shader;


    struct SurfelShader {
        GLuint program{0};
        GLint  mvp_matrix_loc{-1};
        GLint  model_view_matrix_loc {-1};
        GLint  max_radius_loc{-1};
        GLint  min_radius_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint  scale_radius_loc{-1};
        GLint scale_projection_loc      {-1};
        GLint scale_radius_gamma_loc    {-1};
        GLint max_radius_cut_loc        {-1};
        GLint viewport_loc              {-1};
        GLint use_aniso_loc             {-1};
    };
    SurfelShader m_surfel_shader;

    struct SurfelColorShader {
        GLuint program                   {0};
        GLint mvp_matrix_loc            {-1}; // mat4  mvp_matrix
        GLint view_matrix_loc           {-1}; // mat4 view_matrix
        GLint model_view_matrix_loc     {-1};
        GLint normal_matrix_loc         {-1}; // mat3 normal_matrix
        GLint min_radius_loc            {-1}; // float min_radius
        GLint max_radius_loc            {-1}; // float max_radius
        GLint max_screen_size_loc       {-1};
        GLint min_screen_size_loc       {-1};
        GLint scale_radius_gamma_loc    {-1};
        GLint max_radius_cut_loc        {-1};
        GLint scale_radius_loc          {-1}; // float scale_radius
        GLint viewport_loc              {-1};
        GLint show_normals_loc          {-1}; // bool  show_normals
        GLint show_accuracy_loc         {-1}; // bool  show_accuracy
        GLint show_radius_dev_loc       {-1}; // bool  show_radius_deviation
        GLint show_output_sens_loc      {-1}; // bool  show_output_sensitivity
        GLint accuracy_loc              {-1}; // float accuracy
        GLint average_radius_loc        {-1}; // float average_radius
        GLint scale_projection_loc      {-1};
        GLint use_aniso_loc             {-1};
    };
    SurfelColorShader m_surfel_color_shader;

    struct SurfelColorLightingShader {
        GLuint program{0};
        GLint mvp_matrix_loc            {-1}; // mat4 mvp_matrix
        GLint view_matrix_loc           {-1}; // mat4 view_matrix
        GLint model_view_matrix_loc     {-1};
        GLint normal_matrix_loc         {-1}; // mat3 normal_matrix
        GLint max_radius_loc            {-1}; // float max_radius
        GLint max_screen_size_loc       {-1};
        GLint min_screen_size_loc       {-1};
        GLint min_radius_loc            {-1}; // float min_radius
        GLint scale_radius_loc          {-1}; // float scale_radius
        GLint scale_radius_gamma_loc    {-1};
        GLint max_radius_cut_loc        {-1};
        GLint viewport_loc              {-1}; // vec2 viewport
        GLint scale_projection_loc      {-1};
        GLint show_normals_loc          {-1}; // bool  show_normals
        GLint show_accuracy_loc         {-1}; // bool  show_accuracy
        GLint show_radius_dev_loc       {-1}; // bool  show_radius_deviation
        GLint show_output_sens_loc      {-1}; // bool  show_output_sensitivity
        GLint accuracy_loc              {-1}; // float accuracy
        GLint average_radius_loc        {-1}; // float average_radius
        // --- Unified float-based lighting uniforms ---
        GLint use_tone_mapping_loc      {-1};
        GLint ambient_intensity_loc     {-1};
        GLint specular_intensity_loc    {-1};
        GLint shininess_loc             {-1};
        GLint point_light_intensity_loc {-1};
        GLint point_light_pos_vs_loc    {-1};
        GLint gamma_loc                 {-1};
        GLint use_aniso_loc             {-1};
    };
    SurfelColorLightingShader m_surfel_color_lighting_shader;

    struct SurfelProvShader {
        GLuint program{0};
        GLint mvp_matrix_loc            {-1}; // mat4  mvp_matrix
        GLint min_radius_loc            {-1}; // float min_radius
        GLint max_radius_loc            {-1}; // float max_radius
        GLint min_screen_size_loc       {-1};
        GLint max_screen_size_loc       {-1};
        GLint scale_radius_loc          {-1}; // float scale_radius
        GLint scale_radius_gamma_loc    {-1}; // float scale_radius_gamma
        GLint max_radius_cut_loc        {-1}; // float max_radius_cut
        GLint viewport_loc              {-1}; // vec2 viewport
        GLint scale_projection_loc      {-1};
        GLint show_normals_loc          {-1}; // bool  show_normals
        GLint show_accuracy_loc         {-1}; // bool  show_accuracy
        GLint show_radius_dev_loc       {-1}; // bool  show_radius_deviation
        GLint show_output_sens_loc      {-1}; // bool  show_output_sensitivity
        GLint accuracy_loc              {-1}; // float accuracy
        GLint average_radius_loc        {-1}; // float average_radius
        GLint channel_loc               {-1}; // int   channel
        GLint heatmap_loc               {-1};
        GLint heatmap_min_loc           {-1};
        GLint heatmap_max_loc           {-1};
        GLint heatmap_min_color_loc     {-1};
        GLint heatmap_max_color_loc     {-1};
    };
    SurfelProvShader m_surfel_prov_shader;

    // Pass 1: Depth-only pass to establish a clean depth buffer and prevent Z-fighting.
    struct SurfelPass1Shader {
        GLuint program{0};
        GLint mvp_matrix_loc{-1};           // mat4 mvp_matrix
        GLint projection_matrix_loc{-1};    
        GLint model_view_matrix_loc{-1};
        GLint model_matrix_loc{-1};         // mat4 model_matrix
        GLint near_plane_loc{-1};           // float near_plane
        GLint far_plane_loc{-1};            // float far_plane
        GLint viewport_loc {-1};
        GLint use_aniso_loc{-1};
        GLint scale_projection_loc      {-1};
        GLint scale_radius_gamma_loc    {-1};
        GLint max_radius_cut_loc        {-1};

        GLint max_radius_loc{-1};           // float max_radius
        GLint min_radius_loc{-1};           // float min_radius
        GLint max_screen_size_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint scale_radius_loc{-1};         // float scale_radius
    };
    SurfelPass1Shader m_surfel_pass1_shader;

    // Pass 2: Accumulation pass to gather color, normal, and position data in off-screen buffers.
    struct SurfelPass2Shader {
        GLuint program{0};
        // Matrix uniforms used in pass 2
        GLint model_view_matrix_loc{-1};
        GLint projection_matrix_loc{-1};
        GLint normal_matrix_loc{-1};
        // Samplers / viewport
        GLint depth_texture_loc{-1};
        GLint viewport_loc{-1};
        GLint use_aniso_loc{-1};
        GLint scale_projection_loc{-1};
        // Scaling uniforms (from VS)
        GLint max_radius_loc{-1};
        GLint min_radius_loc{-1};
        GLint min_screen_size_loc{-1};
        GLint max_screen_size_loc{-1};
        GLint scale_radius_loc{-1};
        GLint scale_radius_gamma_loc{-1};
        GLint max_radius_cut_loc{-1};
        // Visualization uniforms (from vis_color.glsl)
        GLint show_normals_loc{-1};
        GLint show_accuracy_loc{-1};
        GLint show_radius_dev_loc{-1};
        GLint show_output_sens_loc{-1};
        GLint accuracy_loc{-1};
        GLint average_radius_loc{-1};
        // Blending uniforms
        GLint depth_range_loc{-1};
        GLint flank_lift_loc{-1};
        // Misc
        GLint coloring_loc{-1};
    };
    SurfelPass2Shader m_surfel_pass2_shader;

    // Pass 3: Resolve pass to combine accumulated data, apply lighting, and render the final image.
    struct SurfelPass3Shader {
        GLuint program{0};
        // Input textures from FBO
        GLint in_color_texture_loc{-1};
        GLint in_normal_texture_loc{-1};
        GLint in_vs_position_texture_loc{-1};
        GLint in_depth_texture_loc{-1};
        // Lighting uniforms (for Blinn-Phong shading)
        GLint point_light_pos_vs_loc    {-1};
        GLint point_light_intensity_loc {-1};
        GLint ambient_intensity_loc     {-1};
        GLint specular_intensity_loc    {-1};
        GLint shininess_loc             {-1};
        GLint use_tone_mapping_loc      {-1};
        GLint gamma_loc                 {-1};
        GLint lighting_loc {-1};

    };
    SurfelPass3Shader m_surfel_pass3_shader;

    struct LineShader {
        GLuint program{0};
        GLint in_color_location{-1};
        GLint mvp_matrix_location{-1};
    };
    LineShader m_line_shader;


    struct PclResource {
        GLuint screen_quad_vao = 0;
        GLuint screen_quad_vbo = 0;
        std::array<float, 18> screen_quad_vertex = {
            -1.0f,  1.0f, 0.0f,   -1.0f, -1.0f, 0.0f,    1.0f, -1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,    1.0f, -1.0f, 0.0f,    1.0f,  1.0f, 0.0f
        };

        GLuint vao = 0;
    };

    PclResource m_pcl_resource;

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

    std::unordered_map<MultipassTargetKey, MultipassTarget, MultipassTargetKeyHash> m_multipass_targets;


    struct BoxResource {
        GLuint vbo = 0;
        GLuint ibo = 0;
        GLuint vao = 0;
        GLuint program = 0;
        std::array<unsigned short, 24> idx = { 
            0, 1, 2, 3, 4, 5, 6, 7,
            0, 2, 1, 3, 4, 6, 5, 7,
            0, 4, 1, 5, 2, 6, 3, 7,
        };
    };
    BoxResource m_box_resource;

    //struct PlaneResource {
    //    GLuint vbo{ 0 };
    //    GLuint ibo{ 0 };
    //    GLuint vao{ 0 };
    //    GLuint program{ 0 };
    //    std::array<unsigned short, 6> idx = {
    //        1,3,7,5,1,7
    //    };
    //};
    //PlaneResource m_plane_resource;


    struct FrustumResource {
        GLuint vao = 0;
        GLuint vbo = 0;
        GLuint ibo = 0;
        GLuint program = 0;
        std::array<float, 24> vertices;
        std::array<unsigned short, 24> idx = {
            0, 1, 2, 3, 4, 5, 6, 7,
            0, 2, 1, 3, 4, 6, 5, 7,
            0, 4, 1, 5, 2, 6, 3, 7,
        };
    };
    FrustumResource m_frustum_resource;

    struct TextResource {
        GLuint vao{ 0 };
        GLuint vbo{ 0 };
        GLuint program{ 0 };
        GLuint atlas_texture{ 0 };
        std::string text;
        size_t num_vertices{ 0 };
    };
    TextResource m_text_resource;

    struct CameraBinding {
        const osg::Camera* osgCamera{nullptr};
        std::unique_ptr<lamure::ren::camera> lamureCamera;
        lamure::view_t viewDescriptor{0};
        scm::math::mat4d modelview{scm::math::mat4d::identity()};
        scm::math::mat4d projection{scm::math::mat4d::identity()};
    };

    // Matrices
    scm::math::mat4d m_modelview_matrix{scm::math::mat4d::identity()};
    scm::math::mat4d m_projection_matrix{scm::math::mat4d::identity()};

    // Schism objects
    scm::gl::render_device_ptr      m_device;
    scm::gl::render_context_ptr     m_context;

    // Cameras
    lamure::ren::camera* m_scm_camera{nullptr};
    const osg::Camera* m_active_osg_camera{nullptr};
    std::unordered_map<const osg::Camera*, CameraBinding> m_camera_bindings;
    lamure::view_t m_next_view_descriptor{0};
    osg::ref_ptr<osg::Camera>   m_osg_camera;
    osg::ref_ptr<osg::Camera>   m_hud_camera;

    // Geodes
    osg::ref_ptr<osg::Geode> m_init_geode;
    osg::ref_ptr<osg::Geode> m_pointcloud_geode;
    osg::ref_ptr<osg::Geode> m_boundingbox_geode;
    osg::ref_ptr<osg::Geode> m_frustum_geode;
    osg::ref_ptr<osg::Geode> m_text_geode;

    // Stateset
    osg::ref_ptr<osg::StateSet> m_init_stateset;
    osg::ref_ptr<osg::StateSet> m_pointcloud_stateset;
    osg::ref_ptr<osg::StateSet> m_boundingbox_stateset;
    osg::ref_ptr<osg::StateSet> m_frustum_stateset;
    osg::ref_ptr<osg::StateSet> m_text_stateset;

    // Geometry
    osg::ref_ptr<osg::Geometry> m_init_geometry;
    osg::ref_ptr<osg::Geometry> m_pointcloud_geometry;
    osg::ref_ptr<osg::Geometry> m_boundingbox_geometry;
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
    LamureRenderer(Lamure* lamure_plugin);
    ~LamureRenderer();

    void init();
    void shutdown();
    void detachCallbacks();

    bool beginFrame();
    void endFrame();
    bool pauseAndWaitForIdle(uint32_t extraDrainFrames);
    void resumeRendering();
    bool isRendering() const;

    void initFrustumResources();
    void initLamureShader();
    void initSchismObjects();
    void initUniforms();
    void initBoxResources();
    void initPclResources();
    void releaseMultipassTargets();
    void initializeMultipassTarget(MultipassTarget& target, int width, int height);
    void destroyMultipassTarget(MultipassTarget& target);

    bool getRendering() { return m_rendering; };
    void setRendering(bool rendering) { m_rendering = rendering; };

    // Getters for private members
    lamure::ren::camera* getScmCamera() { return m_scm_camera; }
    osg::ref_ptr<osg::Camera> getOsgCamera() { return m_osg_camera; }

    scm::math::mat4d getModelViewMatrix() { return m_modelview_matrix; }
    scm::math::mat4d getProjectionMatrix() { return m_projection_matrix; }
    lamure::ren::camera* activateCameraForDraw(const osg::Camera* osgCamera,
                                               const scm::math::mat4d& modelview,
                                               const scm::math::mat4d& projection,
                                               bool syncActive);
    MultipassTarget& acquireMultipassTarget(lamure::context_t contextID, const osg::Camera* camera, int width, int height);

    void setModelViewMatrix(scm::math::mat4d model_view_matrix) { m_modelview_matrix = model_view_matrix; }
    void setProjectionMatrix(scm::math::mat4d projection_matrix) { m_projection_matrix = projection_matrix; }
    void updateSyncCameraState(const osg::Camera* sourceCamera,
                               const scm::math::mat4d& modelview,
                               const scm::math::mat4d& projection,
                               bool syncActive,
                               bool haveState);

    osg::ref_ptr<osg::Geode> getPointcloudGeode() { return m_pointcloud_geode; }
    osg::ref_ptr<osg::Geode> getBoundingboxGeode() { return m_boundingbox_geode; }
    osg::ref_ptr<osg::Geode> getFrustumGeode() { return m_frustum_geode; }
    osg::ref_ptr<osg::Geode> getTextGeode() { return m_text_geode; }

    scm::gl::render_device_ptr getDevice() { return m_device; }
    scm::gl::render_context_ptr getContext() { return m_context; }

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
    void setFrameUniforms(const scm::math::mat4& projection_matrix, const scm::math::vec2& viewport);
    void setModelUniforms(const scm::math::mat4& mvp_matrix);
    void setNodeUniforms(const lamure::ren::bvh* bvh, uint32_t node_id);
    bool isModelVisible(std::size_t modelIndex) const;


    std::string glTypeToString(GLenum type);
    void print_active_uniforms(GLuint programID, const std::string& shaderName);

    const PointShader&                  getPointShader()                const { return m_point_shader; }
    const PointColorShader&             getPointColorShader()           const { return m_point_color_shader; }
    const PointColorLightingShader&     getPointColorLightingShader()   const { return m_point_color_lighting_shader; }
    const PointProvShader&              getPointProvShader()            const { return m_point_prov_shader; }

    const SurfelShader&                 getSurfelShader()               const { return m_surfel_shader; }
    const SurfelColorShader&            getSurfelColorShader()          const { return m_surfel_color_shader; }
    const SurfelColorLightingShader&    getSurfelColorLightingShader()  const { return m_surfel_color_lighting_shader; }
    const SurfelProvShader&             getSurfelProvShader()           const { return m_surfel_prov_shader; }

    const SurfelPass1Shader&            getSurfelPass1Shader()          const { return m_surfel_pass1_shader; }
    const SurfelPass2Shader&            getSurfelPass2Shader()          const { return m_surfel_pass2_shader; }
    const SurfelPass3Shader&            getSurfelPass3Shader()          const { return m_surfel_pass3_shader; }

    const LineShader&                   getLineShader()                 const { return m_line_shader; }

    PclResource&      getPclResource()      { return m_pcl_resource; }
    BoxResource&      getBoxResource()      { return m_box_resource; }
    FrustumResource&  getFrustumResource()  { return m_frustum_resource; }
    TextResource&     getTextResource()     { return m_text_resource; }

};

#endif // _LAMURE_RENDERER_H
