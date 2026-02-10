#ifndef _Lamure_H
#define _Lamure_H

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <GL/glew.h>

#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <type_traits>
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
#include <chrono>
#include <mutex>

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
#include "LamureEditTool.h"
#include <osg/observer_ptr>

namespace opencover {
    namespace ui {
        class Element;
        class Group;
        class Button;
    } }

class Lamure;
class LamureEditTool;

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
        int32_t min_vram{ 0 };
        int32_t min_ram{ 0 };
        int32_t min_upload{ 0 };
        int32_t size_of_provenance{ 0 };
        bool provenance{ 1 };
        bool create_aux_resources{ 1 };
        int32_t gui{ 1 };
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
        std::string json{};
        std::string pvs{};
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
        bool show_text{ false };
        bool show_sync{ true };
        bool show_notify{ true };
        bool use_initial_navigation{ false };
        osg::Matrix initial_navigation;
        bool use_initial_view{ false };
        osg::Matrix initial_view;
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
        std::vector<scm::math::vec3f> root_bb_min;
        std::vector<scm::math::vec3f> root_bb_max;
        std::vector<scm::math::vec3f> root_center;
        scm::math::vec3f models_min;
        scm::math::vec3f models_max;
        scm::math::vec3d models_center;
        std::vector<bool> model_visible;
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

    struct SceneNodes {
        osg::ref_ptr<osg::MatrixTransform> model_transform;
        osg::ref_ptr<osg::Geode> point_geode;
        osg::ref_ptr<osg::Geode> cut_geode;
        osg::ref_ptr<osg::Geode> box_geode;
    };
    std::vector<SceneNodes> m_scene_nodes;

    Lamure();
    ~Lamure();

    static Lamure* instance();
    bool init2() override;
    static int loadBvh(const char* filename, osg::Group* parent, const char* ck = "");
    static int unloadBvh(const char* filename, const char* ck = "");
    void loadSettingsFromCovise();
    void preFrame() override;
    void rebuildRenderer();
    void startMeasurement();
    void stopMeasurement();
    void applyInitialTransforms();
    void dumpModelParentChains() const;
    void markRebuildEnd();

    LamureUI* getUI() { return m_ui.get(); }
    LamureRenderer* getRenderer() { return m_renderer.get(); }
    LamureMeasurement*       getMeasurement()       noexcept { return m_measurement.get(); }
    const LamureMeasurement* getMeasurement() const noexcept { return m_measurement.get(); }
    LamureEditTool*          getEditTool()          noexcept { return m_edit_tool.get(); }
    const LamureEditTool*    getEditTool()    const noexcept { return m_edit_tool.get(); }

    void setEditMode(bool enabled);
    bool isEditModeActive() const noexcept { return m_edit_mode; }
    void setEditAction(LamureEditTool::BrushAction action);
    LamureEditTool::BrushAction getEditAction() const noexcept { return m_edit_action; }

    Settings&   getSettings()   { return m_settings; }
    ModelInfo&  getModelInfo()  { return m_model_info; }
    RenderInfo& getRenderInfo() { return m_render_info; }
    Trackball&  getTrackball()  { return m_trackball; }
    const std::vector<SceneNodes>& getSceneNodes() const { return m_scene_nodes; }
    bool initialized = false;
    lamure::ren::Data_Provenance& getDataProvenance() { return m_data_provenance; }
    bool isRebuildInProgress() const noexcept { return m_rebuild_in_progress; }

    bool writeSettingsJson(const Lamure::Settings& s, const std::string& outPath);
    bool rendering_{false};
    template <typename... Args>
    void logInfo(Args&&... args) const {
        if (!m_settings.show_notify)
            return;
        std::cout << "[Lamure] ";
        using expander = int[];
        (void)expander{0, ((std::cout << std::forward<Args>(args)), 0)...};
        std::cout << "\n";
    }

    template <typename T1>
    void dump(T1&& first) const {
        dump(std::forward<T1>(first), 0);
    }

    template <typename T1, typename T2,
              typename std::enable_if<
                  std::is_integral<typename std::remove_reference<T2>::type>::value ||
                  std::is_enum<typename std::remove_reference<T2>::type>::value,
                  int>::type = 0>
    void dump(T1&& first, T2&& second) const {
        auto* dumpBtn = (m_ui ? m_ui->getDumpButton() : nullptr);
        if (!dumpBtn || !dumpBtn->state())
            return;
        std::cout << std::forward<T1>(first);
        if (second == 0)
            dumpBtn->setState(false);
    }

    template <typename T1, typename T2,
              typename std::enable_if<
                  !std::is_integral<typename std::remove_reference<T2>::type>::value &&
                  !std::is_enum<typename std::remove_reference<T2>::type>::value,
                  int>::type = 0>
    void dump(T1&& first, T2&& second) const {
        auto* dumpBtn = (m_ui ? m_ui->getDumpButton() : nullptr);
        if (!dumpBtn || !dumpBtn->state())
            return;
        std::cout << std::forward<T1>(first);
        std::cout << std::forward<T2>(second);
    }

    template <typename T1, typename T2, typename... Rest>
    void dump(T1&& first, T2&& second, Rest&&... rest) const {
        auto* dumpBtn = (m_ui ? m_ui->getDumpButton() : nullptr);
        if (!dumpBtn || !dumpBtn->state())
            return;
        std::cout << std::forward<T1>(first);
        std::cout << std::forward<T2>(second);
        using expander = int[];
        (void)expander{0, ((std::cout << std::forward<Rest>(rest)), 0)...};
    }

    void beginFrameMarks() noexcept { m_marks.reset(); }
    void addMarkMs(MarkField f, double ms) noexcept;
    const FrameMarks& getFrameMarks() const noexcept { return m_marks; }

    void setModelVisible(uint16_t idx, bool v);
    bool isModelVisible(uint16_t idx) const;

    void setBrushFrozen(bool f) { m_brush_frozen = f; }
    bool isBrushFrozen() const { return m_brush_frozen; }

    // Scenegraph visibility toggles (Group-based, use NodeMask toggling).
    void setShowPointcloud(bool show);
    void setShowBoundingbox(bool show);

    std::unordered_map<std::string,uint16_t> m_model_idx;

    osg::ref_ptr<osg::Group> getGroup() { return m_lamure_grp; }

    std::unordered_map<std::string, osg::observer_ptr<osg::Node>> m_pendingTransformUpdate;
    std::vector<osg::ref_ptr<osg::Group>> m_model_parents;
    std::vector<osg::ref_ptr<osg::Group>> m_bootstrap_parents;
    std::unordered_set<std::string> m_registeredFiles;
    std::unordered_map<std::string, std::string> m_model_source_keys;


private:
    std::vector<std::string> m_files_to_load;
    std::unordered_set<std::string> m_files_to_load_set;
    bool m_reload_imminent = false;
    int m_frames_to_wait = 0;

    static Lamure* plugin;

    mutable std::mutex m_settings_mutex;
    mutable std::mutex m_load_bvh_mutex;

    bool m_first_frame = true;
    bool m_models_from_config = false;

    std::map<uint16_t, std::unique_ptr<LamureRenderer>> m_rendererMap;
    std::unique_ptr<LamureRenderer> m_renderer;
    std::unique_ptr<LamureUI>       m_ui;
    std::unique_ptr<LamureEditTool> m_edit_tool;

    Settings   m_settings;
    ModelInfo  m_model_info;
    RenderInfo m_render_info;
    Trackball  m_trackball;

    lamure::ren::Data_Provenance        m_data_provenance;
    osgViewer::ViewerBase::FrameScheme  rendering_scheme{};
    std::unique_ptr<LamureMeasurement>  m_measurement;
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
    void resetVrmlRootTransform(osg::Node* node);
    static void detachFromParents(osg::Node* node);

    // Centralized model resolution
    std::vector<std::string> resolveAndNormalizeModels();
    void adjustOsgCameraClipping();
    void ensureFileMenuEntry(const std::string& path, osg::Group *parent = nullptr);

    // Bootstrap file collection before initialization
    std::vector<std::string> m_bootstrap_files;

    // Rebuild state machine for safer multi-frame shutdown/rebuild
    bool m_rebuild_in_progress{false};
    std::atomic<bool> m_is_rebuilding{false};
    bool m_renderer_paused_for_rebuild{false};
    int  m_post_shutdown_delay{0};
    bool m_did_initial_build{false};
    LamureEditTool::BrushAction m_edit_action{LamureEditTool::BrushAction::None};
    bool m_edit_mode{false};
    bool m_brush_frozen{false};
};

inline ScopedMark::~ScopedMark() {
    if (!plugin || !plugin->getMeasurement()) return;
    const double t1 = osg::Timer::instance()->time_s();
    plugin->addMarkMs(field, (t1 - t0) * 1000.0);
}

#endif
