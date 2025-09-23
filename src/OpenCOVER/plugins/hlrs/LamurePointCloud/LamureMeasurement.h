#pragma once

// GLEW zuerst (für glGetString, glGetIntegerv, glewIsSupported, GL_* Konstanten)
#include <GL/glew.h>

// OSG / OpenCOVER
#include <cover/VRViewer.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRPluginSupport.h>
#include <osg/Timer>
#include <osg/Camera>
#include <osg/Matrix>
#include <osg/Quat>
#include <osg/Vec3>
#include <osgViewer/Viewer>

// STL
#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <ostream>

// Fwd
class Lamure;

// -----------------------------------------------------------------------------
// Öffentliche Datenstrukturen, die auch in der .cpp verwendet werden
// -----------------------------------------------------------------------------

struct Synthesis {
    // Basics
    size_t nFrames = 0;
    size_t timelineBlocks = 0;
    double total_duration_ms = 0.0;
    double avg_frame_time_ms = 0.0;
    double avg_cpu_main_ms   = 0.0;
    double avg_gpu_time_ms   = 0.0;
    double avg_wait_ms       = 0.0;

    double fps_avg  = 0.0;
    double fps_p50  = 0.0;
    double fps_p95  = 0.0;
    double fps_p99  = 0.0;

    double prims_per_frame_avg = 0.0;
    double prims_per_frame_p50 = 0.0;
    double prims_per_frame_p95 = 0.0;
    double prims_per_frame_p99 = 0.0;

    // Busy / Wait
    double avg_cpu_busy_pct  = 0.0;
    double avg_gpu_busy_pct  = 0.0;
    double avg_wait_frac_pct = 0.0;

    // Perzentile & Stutter
    double p50 = 0.0, p95 = 0.0, p99 = 0.0, jitter_J = 0.0;
    int stutter_count_2xmedian = 0;
    int stutter_count_33ms     = 0;

    // Coverage / Effizienz
    double avg_rpts_points_per_ms = 0.0;
    double avg_covC1 = 0.0;
    double avg_covD  = 0.0;

    // Renderer Estimates (avg)
    double est_screen_px   = 0.0;
    double est_sum_area_px = 0.0;
    double est_density     = 0.0;
    double est_coverage    = 0.0;
    double est_coverage_px = 0.0;
    double est_overdraw    = 0.0;
    double avg_area_px_per_prim = 0.0;

    // Zeitmarken (avg)
    double mark_draw_impl_ms=0, mark_pass1_ms=0, mark_pass2_ms=0, mark_pass3_ms=0,
           mark_dispatch_ms=0, mark_context_bind_ms=0, mark_estimates_ms=0, mark_singlepass_ms=0;

    // Boundness-Zählung
    size_t cnt_gpu=0, cnt_cpu=0, cnt_wait=0, cnt_mixed=0, cnt_unknown=0;
};

class LamureMeasurement
{
public:
    // --- Öffentliche Strukturen ---
    struct TimeBlock {
        unsigned frame     = 0;   // Basierend auf pickStableFrame()
        unsigned src_frame = 0;   // frame - used_offset
        int      camIndex  = -1;  // -1 = viewer, sonst Kameraindex
        std::string scope;        // "viewer" | "camera"
        std::string name;         // z.B. "Draw traversal", "GPU draw"
        double begin_ms = 0.0;
        double end_ms   = 0.0;
        double taken_ms = 0.0;
        unsigned used_offset = 0;
    };

    struct Segment {
        osg::Vec3 tra;     // Positionsänderung relativ zum Segment-Anfang
        osg::Vec3 rot;     // Drehwinkel-Delta in Grad (Pitch, Yaw, Roll)
        float     transSpeed = 0.f;   // Translationstempo in Einheiten/s
        float     rotSpeed   = 0.f;   // Rotationsgeschwindigkeit in °/s
    };

    struct FrameStats {
        // Ident & Counter
        unsigned int frame_number       = 0;
        uint64_t rendered_primitives    = 0;
        uint64_t rendered_nodes         = 0;
        uint64_t rendered_bounding_boxes= 0;

        // Frame & CPU times
        float frame_rate           = -1.0f;
        float frame_duration_ms    = -1.0f;
        float rendering_traversals_ms = -1.0f;

        float cpu_update_ms        = -1.0f;
        float cpu_cull_ms          = -1.0f;
        float cpu_draw_ms          = -1.0f;

        // GPU Telemetrie (per Frame)
        float gpu_time_ms          = -1.0f;
        float gpu_clock            = -1.0f;
        float gpu_mem_clock        = -1.0f;
        float gpu_util             = -1.0f;
        float gpu_pci              = -1.0f;

        // Weitere Scopes
        float sync_time_ms         = -1.0f;
        float swap_time_ms         = -1.0f;
        float finish_ms            = -1.0f;
        float plugin_ms            = -1.0f;
        float isect_ms             = -1.0f;
        float opencover_ms         = -1.0f;

        // Renderer Estimates (capped)
        float est_screen_px        = -1.0f;
        float est_sum_area_px      = -1.0f;
        float est_density          = -1.0f;
        float est_coverage         = -1.0f;
        float est_coverage_px      = -1.0f;
        float est_overdraw         = -1.0f;

        // Renderer Estimates (raw)
        float est_density_raw      = -1.0f;
        float est_coverage_raw     = -1.0f;
        float est_coverage_px_raw  = -1.0f;
        float est_overdraw_raw     = -1.0f;

        // Per-primitive Statistik
        float avg_area_px_per_prim = -1.0f;

        // Zeitmarken
        float mark_draw_impl_ms    = -1.0f;
        float mark_pass1_ms        = -1.0f;
        float mark_pass2_ms        = -1.0f;
        float mark_pass3_ms        = -1.0f;
        float mark_singlepass_ms   = -1.0f;
        float mark_dispatch_ms     = -1.0f;
        float mark_context_bind_ms = -1.0f;
        float mark_estimates_ms    = -1.0f;

        // Pose
        osg::Vec3d position{0.0,0.0,0.0};
        osg::Quat  orientation{0.0,0.0,0.0,1.0};

        // Backoffs / Segment
        unsigned backoff_cull = 0;
        unsigned backoff_draw = 0;
        unsigned backoff_gpu  = 0;
        int      segment_index = -1;

        // Abgeleitete Metriken
        float cpu_main_ms        = -1.0f;
        float cpu_busy_pct_proxy = -1.0f;
        float gpu_busy_pct_proxy = -1.0f;
        float wait_ms            = -1.0f;
        float wait_frac_pct      = -1.0f;

        std::string boundness;   // "GPU-bound" | "CPU-bound" | "Wait/Sync-bound" | "mixed" | "unknown"
    };

public:
    // --- Lebenszyklus ---
    LamureMeasurement(Lamure* plugin,
                      opencover::VRViewer* viewer,
                      const std::vector<Segment>& segments,
                      const std::string& logfile);
    ~LamureMeasurement();

    // --- Steuerung / Status ---
    bool isActive() const noexcept;  // true, solange m_running
    void stop();

    // --- Sampling / Auswertung ---
    bool wantsSampling(unsigned frameNo) const noexcept;
    bool collectFrameStats(osgViewer::ViewerBase* viewer,
                           const osg::FrameStamp* fs,
                           FrameStats& out,
                           bool debugPrint = false);

    // --- Export / Abschluss ---
    void writeLogAndStop();

    bool writeReportMarkdown(const std::filesystem::path& md_path, const Synthesis& syn, bool mode_full, bool mode_light, bool wrote_frames, bool wrote_timeline, bool wrote_summary_json, const std::filesystem::path& frames_path, const std::filesystem::path& timeline_path, const std::filesystem::path& json_path, const std::filesystem::path& base_path);

    // --- Convenience/Getters ---
    bool isSampleFrame() const noexcept { return m_sampleThisFrame; }
    unsigned currentFrameNo() const noexcept { return m_currFrameNo; }
    const std::vector<TimeBlock>& getTimeline() const { return m_timeline; }

    // Debug-Helfer
    void printDebugStats(unsigned int num);

private:
    // --- interne Helfer ---
    void initCallbacks();
    void drawIncrement(bool preDraw, const osg::FrameStamp* frameStamp);
    void updateCamera(const osg::Vec3& traAbs, const osg::Vec3& rotAbsDeg);

    int findStatsIndexForFrame(uint64_t frame_no) const;

    bool getTimeTakenMsBacksearch(osg::Stats* s,
        unsigned baseFrame, unsigned backSearch,
        const std::string& timeTakenKey,
        const std::string& beginKey,
        const std::string& endKey,
        double& outMs, unsigned& usedOffset,
        double* outBeginMs=nullptr, double* outEndMs=nullptr);

    bool tryAddBlock(osg::Stats* stats, unsigned baseFrame, unsigned backsearch,
        const std::string& statPrefix, const std::string& nameForCSV,
        const std::string& scope, int camIndex, std::vector<TimeBlock>& localBlocks);

    // Exporter
    bool writeFramesCSV(const std::filesystem::path& frames_path,
        bool mode_full, bool mode_light,
        const std::unordered_map<unsigned,double>& gpuMsByFrame);

    bool writeSummaryJSON(const std::filesystem::path& json_path,
        const Synthesis& syn, bool mode_full, bool mode_light, bool hasTimeline);

    void writeTimelineCSV(const std::string& path);

    void writeLamureConfigMarkdown(std::ostream& md);
    void writeLamureConfigCsv(std::ostream& csv);
    void appendPreprocessBuildLogsMarkdown(std::ostream& md);

    // GPU/VRAM-Infos (static snapshot)
    void cacheStaticGpuInfo();

private:
    // --- Konfiguration/Flags ---
    bool     m_exportReport     = true;
    bool     m_verbose          = false;
    unsigned m_logEveryN        = 30;
    bool     m_dumpAttrs        = false;
    unsigned m_gpuBackSearch    = 16;
    bool     m_measure_timeline = true;

    // --- Laufzeitstatus ---
    bool          m_running          = true;
    bool          m_written          = false;
    std::string   m_logfile;
    std::string   m_reportMDPath;

    // --- Zeit / Frames ---
    osg::Timer_t  m_startTick{};
    osg::Timer_t  m_lastFrameTick{};
    unsigned      m_currFrameNo = 0;
    bool          m_sampleThisFrame = false;

    // --- Segmente / Pose ---
    const std::vector<Segment> m_segments;
    size_t       m_currentSegment{0};
    bool         m_haveSegmentStart{false};
    double       m_segmentStartRefTime{0.0};
    osg::Vec3    m_cumulativeTra{0.0f,0.0f,0.0f};
    osg::Vec3    m_cumulativeRot{0.0f,0.0f,0.0f};

    osg::Vec3    m_lastTraApplied{0,0,0};
    osg::Vec3    m_lastRotApplied{0,0,0};
    bool         m_havePoseDeltas = false;

    osg::Quat    m_lastQuat;
    bool         m_haveLastQuat{false};

    // --- Datencontainer ---
    std::vector<FrameStats> m_stats;
    std::vector<TimeBlock>  m_timeline;

    // --- GPU/VRAM (static snapshot; primary = bevorzugte Quelle (NVML dann GL_NVX)) ---
    bool   m_gpu_static_captured = false;
    bool   m_gpu_static_tried     = false;
    double m_gpu_mem_used_mb_static        = 0.0;
    double m_gpu_mem_total_mb_static       = 0.0;
    double m_gpu_mem_used_mb_nvml_static   = 0.0;
    double m_gpu_mem_total_mb_nvml_static  = 0.0;
    double m_gpu_mem_used_mb_gl_static     = 0.0;
    double m_gpu_mem_total_mb_gl_static    = 0.0;

    // --- Plugin/Viewer ---
    Lamure*              m_plugin = nullptr;
    opencover::VRViewer* m_viewer = nullptr;

    // --- Frame-Marks Callback ---
    struct MarkCallback : public osg::Camera::DrawCallback {
        LamureMeasurement* m_meas = nullptr;
        bool               m_pre  = false;
        MarkCallback(LamureMeasurement* m, bool pre) : m_meas(m), m_pre(pre) {}
        void operator()(osg::RenderInfo& ri) const override {
            m_meas->drawIncrement(m_pre, ri.getState()->getFrameStamp());
        }
    };
    osg::ref_ptr<MarkCallback> m_preCB;
    osg::ref_ptr<MarkCallback> m_postCB;
};
