#ifndef _Lamure_H
#define _Lamure_H

#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>

#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <bitset>
#include <initializer_list>
#include <algorithm>
#include <atomic>

#include <osg/Timer>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/MatrixTransform>
#include <osgGA/GUIEventAdapter>
#include <osgViewer/ViewerBase>

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>

#include <scm/core/math.h>

#include <lamure/ren/data_provenance.h>
#include <lamure/prov/prov_aux.h>

#include "LamureRenderer.h"
#include "LamureUI.h"
#include "LamureUtil.h"
#include "LamureMeasurement.h"
#include <osg/observer_ptr>

namespace opencover {
    namespace ui {
        class Element;
        class Group;
        class Button;
    } }

class Lamure;

struct FrameMarks {
    double draw_cb_ms      = -1.0;
    double dispatch_ms     = -1.0;
    double context_bind_ms = -1.0;
    double estimates_ms    = -1.0;
    double pass1_ms        = -1.0;
    double pass2_ms        = -1.0;
    double pass3_ms        = -1.0;
    double singlepass_ms   = -1.0;

    void reset() { *this = FrameMarks{}; }
};

enum class MarkField : uint8_t {
    DrawCB_Total,
    Dispatch,
    ContextBind,
    Estimates,
    Pass1,
    Pass2,
    Pass3,
    SinglePass,
};

struct ScopedMark {
    Lamure* plugin;
    MarkField field;
    double t0;
    ScopedMark(Lamure* p, MarkField f) noexcept
        : plugin(p), field(f), t0(osg::Timer::instance()->time_s()) {}
    ~ScopedMark();
};

#define LM_CAT_IMPL(a,b) a##b
#define LM_CAT(a,b) LM_CAT_IMPL(a,b)

// für „unused“-Warnungen (trotzdem läuft der Destruktor!)
#if __cplusplus >= 201703L
#define LM_MAYBE_UNUSED [[maybe_unused]]
#else
#define LM_MAYBE_UNUSED
#endif

// Einziger öffentliche Entry-Point: erzeugt eine RAII-Variable.
// Ctor = Begin-Mark, Dtor = End + "time taken" schreiben.
// Wenn kein Measurement aktiv ist, macht ScopedMark intern nichts.
#define LM_SCOPE(pluginPtr, FIELD) \
  LM_MAYBE_UNUSED ::ScopedMark LM_CAT(_lm_scope_, __LINE__){ (pluginPtr), MarkField::FIELD }

// Start-Marke: legt nur eine lokale Tick-Variable an
#define LM_MARK_BEGIN(VARNAME) \
    const osg::Timer_t VARNAME = osg::Timer::instance()->tick()

enum class MeasureMode { Off, Lite, Full };

class Lamure : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    struct Settings
    {
        int32_t frame_div{ 1 };
        int32_t vram{ 1024 };
        int32_t ram{ 4096 };
        int32_t upload{ 32 };
        bool provenance{ 1 };
        bool create_aux_resources{ 1 };
        bool face_eye{ 0 };
        int32_t gui{ 1 };
        int32_t travel{ 2 };
        float travel_speed{ 20.5f };
        int32_t max_brush_size{ 4096 };
        bool lod_update{ 1 };
        float lod_error{ 1.0f };
        LamureRenderer::ShaderType shader_type{ LamureRenderer::ShaderType::Point };
        bool use_pvs{ 0 };
        bool pvs_culling{ 0 };
        float aux_point_size{ 1.0f };
        float aux_point_distance{ 0.5f };
        float aux_point_scale{ 1.0f };
        float aux_focal_length{ 1.0f };
        int32_t vis{ 0 };
        int32_t show_normals{ 1 };
        bool show_accuracy{ 0 };
        bool show_radius_deviation{ 0 };
        bool show_output_sensitivity{ 0 };
        bool show_sparse{ 0 };
        bool show_views{ 0 };
        bool show_photos{ 0 };
        bool show_octrees{ 0 };
        bool show_bvhs{ 0 };
        bool show_pvs{ 0 };
        int32_t channel{ 0 };
        scm::math::vec3f point_light_pos{ 0.0f, 1000.0f, 0.0f };
        float point_light_intensity{ 0.6f };
        float ambient_intensity{ 0.0f };
        float specular_intensity{ 0.1f };
        float shininess{ 1 };
        float gamma{ 1 };
        bool use_tone_mapping{ 0 };
        bool heatmap{ 0 };
        float heatmap_min{ 0.0f };
        float heatmap_max{ 0.05f };
        std::string shader{};
        scm::math::vec3f background_color{ 68.0f / 255.0f, 0.0f, 84.0f / 255.0f };
        scm::math::vec3f heatmap_color_min{ 68.0f / 255.0f, 0.0f, 84.0f / 255.0f };
        scm::math::vec3f heatmap_color_max{ 251.f / 255.f, 231.f / 255.f, 35.f / 255.f };
        std::string atlas_file{};
        std::string json{};
        std::string pvs{};
        std::string background_image{};
        std::vector<std::string> models;
        std::vector<uint32_t> initial_selection;
        float scale_radius{ 0.05f };
        float scale_radius_gamma{ 0.5f };
        float scale_element{ 1.0f };
        float scale_point{ 1.0f };
        float scale_surfel{ 1.75f };
        float min_radius{ 0.0f };
        float max_radius{ std::min(std::numeric_limits<float>::max(), 3.0f) };
        float min_screen_size{ 0.0f };
        float max_screen_size{ std::min(std::numeric_limits<float>::max(), 10000.0f) };
        float max_radius_cut{ 2.5f };
        // Surfel scaling mode: 0=off (isotropic), 1=auto (default), 2=on (anisotropic)
        int32_t anisotropic_surfel_scaling{ 1 };
        // Auto-mode off-axis sensitivity threshold (max(|col2.x|,|col2.y|))
        float anisotropic_auto_threshold{ 0.05f };
        float depth_range{ 2.0f };
        float flank_lift{ 0.0f };
        std::vector<float> bvh_color{ 1.0f, 1.0f, 0.0f, 1.0f };
        std::vector<float> frustum_color{ 1.0f, 0.0f, 0.0f, 1.0f };
        uint16_t num_models{};
        bool show_pointcloud{ true };
        bool show_boundingbox{ false };
        bool show_frustum{ false };
        bool show_coord{ false };
        bool show_text{ false };
        bool show_sync{ true };
        bool show_notify{ true };
        bool use_initial_navigation{ false };
        osg::Matrix initial_navigation;
        bool use_initial_view{ false };
        osg::Matrix initial_view;
        bool initial_tf_overrides{ false };
        std::vector<LamureMeasurement::Segment> measurement_segments;
        std::string measurement_dir;
        std::string measurement_name;
        bool coloring{ false };
        bool lighting{ false };
        bool point{ false };
        bool surfel{ false };
        bool splatting{ false };
        bool measure_off{ false };
        bool measure_light{ false };
        bool measure_full{ true };
        int32_t measure_sample{ 1 };
        int32_t pause_frames{2};
        bool prefer_parent{ true };
    };

    struct ModelInfo
    {
        std::vector<scm::math::mat4d> model_transformations;
        std::vector<scm::math::vec3f> root_bb_min;
        std::vector<scm::math::vec3f> root_bb_max;
        std::vector<scm::math::vec3f> root_center;
        scm::math::vec3f models_min;
        scm::math::vec3f models_max;
        scm::math::vec3d models_center;
        std::vector<bool> model_visible;
        std::vector<scm::math::mat4d> config_transforms;
    };

    struct RenderInfo
    {
        uint64_t rendered_primitives{0};
        uint64_t rendered_nodes{0};
        uint64_t rendered_bounding_boxes{0};

        float est_screen_px       = -1.0f;
        float est_sum_area_px     = -1.0f;
        float est_coverage_px     = -1.0f;
        float est_density         = -1.0f;
        float est_coverage        = -1.0f;
        float est_overdraw        = -1.0f;
        float estimates_ms        = -1.0f;
        float avg_area_px_per_prim= -1.0f;

        float est_density_raw     = -1.0f;
        float est_coverage_raw    = -1.0f;
        float est_coverage_px_raw = -1.0f;
        float est_overdraw_raw    = -1.0f;

        float fps                 = -1.0f;
    };


    struct Trackball
    {
        float dist{0.0f};
        float size{0.0f};
        osg::Vec3 initial_pos;
        osg::Vec3 pos;
    };

    Lamure();
    ~Lamure();

    static Lamure* instance();
    bool init2();
    static int loadBvh(const char* filename, osg::Group* parent, const char* ck = "");
    static int unloadBvh(const char* filename, const char* ck = "");
    void loadSettingsFromCovise();
    void preFrame();
    void perform_system_reset();
    void startMeasurement();
    void stopMeasurement();
    void applyInitialTransforms();

    LamureUI* getUI() { return m_ui.get(); }
    LamureRenderer* getRenderer() { return m_renderer.get(); }
    LamureMeasurement*       getMeasurement()       noexcept { return m_measurement.get(); }
    const LamureMeasurement* getMeasurement() const noexcept { return m_measurement.get(); }

    Settings&   getSettings()   { return m_settings; }
    ModelInfo&  getModelInfo()  { return m_model_info; }
    RenderInfo& getRenderInfo() { return m_render_info; }
    Trackball&  getTrackball()  { return m_trackball; }
    bool initialized = false;
    bool getProvValid() const { return prov_valid; }
    lamure::ren::Data_Provenance& getDataProvenance() { return m_data_provenance; }

    bool writeSettingsJson(const Lamure::Settings& s, const std::string& outPath);
    bool rendering_{false};
    void dumpSettings(const char* tag = "");

    void beginFrameMarks() noexcept { m_marks.reset(); }
    void addMarkMs(MarkField f, double ms) noexcept;
    const FrameMarks& getFrameMarks() const noexcept { return m_marks; }

    bool m_bootstrapLoad = false;
    std::vector<osg::ref_ptr<osg::Group>> m_modelRoots;
    std::unordered_map<std::string,uint16_t> m_pathToIndex;

    void setModelVisible(uint16_t idx, bool v);
    bool isModelVisible(uint16_t idx) const;

    struct PendingModel {
        std::string path;
        uint16_t mid;
    };
    std::deque<PendingModel> m_pending;
    std::unordered_map<std::string,uint16_t> m_model_idx;
    std::map<std::string, osg::ref_ptr<osg::Group>> m_model_nodes;
    osg::ref_ptr<osg::Group> m_pluginRootGroup;

    struct SceneNodes {
        osg::ref_ptr<osg::Group> root;
        osg::ref_ptr<osg::MatrixTransform> config;
        osg::ref_ptr<osg::MatrixTransform> bvh;
        osg::ref_ptr<osg::Group> payload;
    };
    std::unordered_map<std::string, SceneNodes> m_scene_nodes;

    osg::ref_ptr<osg::Group> getGroup() { return m_lamure_grp; }

    std::unordered_map<std::string, osg::observer_ptr<osg::Node>> m_pendingTransformPrint;
    std::unordered_map<std::string, scm::math::mat4d> m_vrmlTransforms;
    std::unordered_set<std::string> m_registeredFiles;
    std::unordered_map<std::string, std::string> m_model_source_keys;

private:
    std::vector<std::string> m_files_to_load;
    bool m_reload_imminent = false;
    int m_frames_to_wait = 0;

    static Lamure* plugin;

    bool m_initialized = false;

    bool m_first_frame = true;
    bool m_models_from_config = false;

    std::map<uint16_t, std::unique_ptr<LamureRenderer>> m_rendererMap;
    std::unique_ptr<LamureRenderer> m_renderer;
    std::unique_ptr<LamureUI>       m_ui;

    Settings   m_settings;
    ModelInfo  m_model_info;
    RenderInfo m_render_info;
    Trackball  m_trackball;

    lamure::ren::Data_Provenance        m_data_provenance;
    osgViewer::ViewerBase::FrameScheme  rendering_scheme{};
    std::unique_ptr<LamureMeasurement>  m_measurement;
    std::vector<osg::Vec3>              _path;
    float                               _speed{1.0f};
    bool                                measurement_running{false};
    bool                                prov_valid{false};
    bool                                m_silenceMeasureToggle{false};
    float                               prev_frame_rate_ = 0.0f;
    unsigned int                        prev_vsync_frames_ = 0;
    bool                                fps_cap_modified_ = false;
    bool                                vsync_modified_   = false;
    FrameMarks                          m_marks;
    osg::ref_ptr<osg::Group>            m_lamure_grp;

    std::bitset<512> m_keyDown_;

    bool m_hotkeyDown_ = false;       // entprellt KEYDOWN (Key-Repeat ignorieren)
    int  m_hotkeyKey_  = 'm';         // Default: Taste 'm'

    void toggleMeasurementButton() {
        if (!m_ui || !m_ui->getMeasureButton()) return;
        const bool newState = !m_ui->getMeasureButton()->state();
        m_ui->getMeasureButton()->setState(newState); // ruft den Button-Callback auf
    }

    static inline int  clampKeyIndex(int sym) { return (sym & 0x1FF); }
    inline bool        held(int sym) const { return m_keyDown_.test(clampKeyIndex(sym)); }
    inline bool        anyHeld(std::initializer_list<int> syms) const {
        for (int s : syms) if (held(s)) return true; return false;
    }

    std::vector<LamureMeasurement::Segment> parseMeasurementSegments(const std::string& cfg) const;
    std::string buildMeasurementOutputPath() const;
    void applyShaderToRendererFromSettings();

    // Centralized model resolution and post-processing
    std::vector<std::string> resolveAndNormalizeModels();
    void updateModelDependentSettings();
    void adjustOsgCameraClipping();
    void ensureFileMenuEntry(const std::string& path);

    // Bootstrap file collection before initialization
    std::vector<std::string> m_bootstrap_files;

    // Reset state machine for safer multi-frame shutdown/rebuild
    bool m_reset_in_progress{false};
    bool m_renderer_paused_for_reset{false};
    int  m_post_shutdown_delay{0};
    bool m_did_initial_build{false};
public:
    bool isResetInProgress() const noexcept { return m_reset_in_progress; }
};

inline ScopedMark::~ScopedMark() {
    if (!plugin || !plugin->getMeasurement()) return;
    const double t1 = osg::Timer::instance()->time_s();
    plugin->addMarkMs(field, (t1 - t0) * 1000.0);
}

#endif
