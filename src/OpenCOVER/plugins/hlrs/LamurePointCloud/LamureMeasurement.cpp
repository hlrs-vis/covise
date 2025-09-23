#include "LamureMeasurement.h"

#include <cover/VRViewer.h>
#include <osgViewer/Viewer>
#include <osg/Stats>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <locale>
#include <unordered_map>
#include <cover/coVRPluginSupport.h>
#include <map>
#include <set>
#include <filesystem>
#include <system_error>
#include <thread>
#include <ctime>
#include <cstring>
#include "Lamure.h"

#include <lamure/ren/model_database.h>
#include <lamure/ren/bvh.h>

#ifdef _WIN32
#include <windows.h>
#include <cover/coVRConfig.h>
#else
#include <unistd.h> // für gethostname(), sysconf()
#endif

#ifndef LM_STR_HELPER
#define LM_STR_HELPER(x) #x
#define LM_STR(x) LM_STR_HELPER(x)
#endif

// --- GL_NVX_gpu_memory_info (für VRAM der *aktuellen* GL-GPU) ---
#ifndef GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX
#define GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX      0x9048
#define GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX    0x9049
#endif

#ifdef _WIN32  // <--- FEHLTE
// --- NVML dynamic loader ---
struct NvmlLoader {
    HMODULE h = nullptr; bool ok = false; void* dev = nullptr;
    using nvmlInit_t = int(*)(); using nvmlShutdown_t = int(*)();
    using nvmlDeviceGetHandleByIndex_t = int(*)(unsigned, void**);
    using nvmlDeviceGetUtilizationRates_t = int(*)(void*, void*);
    using nvmlDeviceGetMemoryInfo_t = int(*)(void*, void*);
    using nvmlDeviceGetTemperature_t = int(*)(void*, unsigned, unsigned*);
    using nvmlDeviceGetName_t = int(*)(void*, char*, unsigned);

    nvmlInit_t nvmlInit=nullptr; nvmlShutdown_t nvmlShutdown=nullptr;
    nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleByIndex=nullptr;
    nvmlDeviceGetUtilizationRates_t nvmlDeviceGetUtilizationRates=nullptr;
    nvmlDeviceGetMemoryInfo_t nvmlDeviceGetMemoryInfo=nullptr;
    nvmlDeviceGetTemperature_t nvmlDeviceGetTemperature=nullptr;
    nvmlDeviceGetName_t nvmlDeviceGetName=nullptr;

    bool ensureLoaded(){
        if (ok) return true;
        if (!h) { h = LoadLibraryA("nvml.dll"); if (!h) return false; }
        nvmlInit = (nvmlInit_t)GetProcAddress(h,"nvmlInit_v2"); if(!nvmlInit) nvmlInit=(nvmlInit_t)GetProcAddress(h,"nvmlInit");
        nvmlShutdown=(nvmlShutdown_t)GetProcAddress(h,"nvmlShutdown");
        nvmlDeviceGetHandleByIndex=(nvmlDeviceGetHandleByIndex_t)GetProcAddress(h,"nvmlDeviceGetHandleByIndex_v2");
        if(!nvmlDeviceGetHandleByIndex) nvmlDeviceGetHandleByIndex=(nvmlDeviceGetHandleByIndex_t)GetProcAddress(h,"nvmlDeviceGetHandleByIndex");
        nvmlDeviceGetUtilizationRates=(nvmlDeviceGetUtilizationRates_t)GetProcAddress(h,"nvmlDeviceGetUtilizationRates");
        nvmlDeviceGetMemoryInfo=(nvmlDeviceGetMemoryInfo_t)GetProcAddress(h,"nvmlDeviceGetMemoryInfo");
        nvmlDeviceGetTemperature=(nvmlDeviceGetTemperature_t)GetProcAddress(h,"nvmlDeviceGetTemperature");
        nvmlDeviceGetName=(nvmlDeviceGetName_t)GetProcAddress(h,"nvmlDeviceGetName");
        if(!nvmlInit||!nvmlShutdown||!nvmlDeviceGetHandleByIndex||!nvmlDeviceGetUtilizationRates||!nvmlDeviceGetMemoryInfo) return false;
        if (nvmlInit()!=0) return false;
        if (nvmlDeviceGetHandleByIndex(0,&dev)!=0) return false;
        ok=true; return true;
    }
    ~NvmlLoader(){ if(ok&&nvmlShutdown) nvmlShutdown(); if(h) FreeLibrary(h); }
};

#ifndef NVML_TEMPERATURE_GPU
#define NVML_TEMPERATURE_GPU 0u
#endif

struct NvmlUtilization { unsigned gpu; unsigned memory; };
struct NvmlMemory { unsigned long long total, free, used; };

static bool nvmlPoll(NvmlLoader& nv, double& gpuUtilPct, double& memUsedMB, double& memTotalMB,
    int* gpuTempC=nullptr, std::string* gpuName=nullptr){
    gpuUtilPct=memUsedMB=memTotalMB=0.0;
    if(gpuTempC) *gpuTempC = -1;
    if(gpuName)  gpuName->clear();
    if(!nv.ensureLoaded()) return false;

    NvmlUtilization u{};
    if(nv.nvmlDeviceGetUtilizationRates && nv.nvmlDeviceGetUtilizationRates(nv.dev,&u)==0)
        gpuUtilPct=double(u.gpu);

    NvmlMemory m{};
    if(nv.nvmlDeviceGetMemoryInfo && nv.nvmlDeviceGetMemoryInfo(nv.dev,&m)==0){
        memTotalMB=m.total/1048576.0; memUsedMB=m.used/1048576.0;
    }

    if (gpuTempC && nv.nvmlDeviceGetTemperature){
        unsigned t=0; if(nv.nvmlDeviceGetTemperature(nv.dev,NVML_TEMPERATURE_GPU,&t)==0) *gpuTempC=int(t);
    }
    if (gpuName && nv.nvmlDeviceGetName){
        char nameBuf[96] = {0};
        if(nv.nvmlDeviceGetName(nv.dev,nameBuf,sizeof(nameBuf))==0) *gpuName = nameBuf;
    }
    return true;
}
#endif


namespace {

    static inline double inv_ms_to_fps(double ms) {
        return (ms > 1e-9 && std::isfinite(ms)) ? 1000.0 / ms : 0.0;
    }

    static std::string iso8601_local_now() {
        std::time_t t = std::time(nullptr);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S%z", &tm);
        return buf;
    }

    static std::string iso8601_utc_now() {
        std::time_t t = std::time(nullptr);
        std::tm tm{};
#ifdef _WIN32
        gmtime_s(&tm, &t);
#else
        gmtime_r(&t, &tm);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
        return buf;
    }

    static std::string hostname_string() {
        char buf[256] = {0};
#ifdef _WIN32
        DWORD sz = sizeof(buf);
        if (GetComputerNameA(buf, &sz)) return std::string(buf, sz);
#else
        if (gethostname(buf, sizeof(buf)-1)==0) return std::string(buf);
#endif
        return std::string("unknown-host");
    }

    static std::string username_string() {
#ifdef _WIN32
        char buf[256]; DWORD sz = sizeof(buf);
        if (GetUserNameA(buf, &sz)) return std::string(buf, sz-1);
        return "unknown-user";
#else
        const char* u = getenv("USER");
        return u ? u : "unknown-user";
#endif
    }

    static unsigned cpu_threads() {
        unsigned n = std::thread::hardware_concurrency();
        return n ? n : 0u;
    }

    static uint64_t total_ram_mb() {
#ifdef _WIN32
        MEMORYSTATUSEX ms{}; ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) return static_cast<uint64_t>(ms.ullTotalPhys / (1024ull*1024ull));
        return 0ull;
#else
        long pages = sysconf(_SC_PHYS_PAGES);
        long psize = sysconf(_SC_PAGE_SIZE);
        if (pages>0 && psize>0) return static_cast<uint64_t>((1.0*pages*psize)/(1024.0*1024.0));
        return 0ull;
#endif
    }

    static void get_gl_strings(std::string& vendor, std::string& renderer, std::string& version) {
        const GLubyte* v  = glGetString(GL_VENDOR);
        const GLubyte* r  = glGetString(GL_RENDERER);
        const GLubyte* ve = glGetString(GL_VERSION);
        vendor   = v  ? reinterpret_cast<const char*>(v)  : "";
        renderer = r  ? reinterpret_cast<const char*>(r)  : "";
        version  = ve ? reinterpret_cast<const char*>(ve) : "";
    }

    static bool getF(osg::Stats* s, unsigned f, const std::string& key, double& out) {
        return s && s->getAttribute(f, key, out);
    }

    static bool getDelta(osg::Stats* s, unsigned f,
        const std::string& beginKey,
        const std::string& endKey,
        double& outSeconds)
    {
        double b = 0.0, e = 0.0;
        if (!s) return false;
        if (!s->getAttribute(f, beginKey, b)) return false;
        if (!s->getAttribute(f, endKey, e)) return false;
        outSeconds = e - b;
        return (outSeconds >= 0.0);
    }

    // Priorisiert "... time taken", fällt auf (end-begin) zurück
    static bool getTimeTakenMs(osg::Stats* s, unsigned f,
        const std::string& timeTakenKey,
        const std::string& beginKey,
        const std::string& endKey,
        double& outMs)
    {
        double v = 0.0;
        if (getF(s, f, timeTakenKey, v)) { outMs = v * 1000.0; return true; }
        if (getDelta(s, f, beginKey, endKey, v)) { outMs = v * 1000.0; return true; }
        return false;
    }

    static unsigned pickStableFrame(osgViewer::ViewerBase* viewer,
        osg::Stats* viewerStats,
        const osgViewer::ViewerBase::Cameras& cams,
        const osg::FrameStamp* fs)
    {
        unsigned f = fs ? fs->getFrameNumber()
            : (viewerStats ? viewerStats->getLatestFrameNumber() : 0);

        if (!cams.empty()) {
            if (auto* r = dynamic_cast<osgViewer::Renderer*>(cams.front()->getRenderer())) {
                if (!r->getGraphicsThreadDoesCull())
                    f = (f > 0) ? (f - 1) : 0;
            }
        }

        if (viewerStats)
            f = std::min(f, viewerStats->getLatestFrameNumber());
        for (auto* cam : cams)
            if (auto* cs = cam->getStats())
                f = std::min(f, cs->getLatestFrameNumber());

        return f;
    }

    inline void ensureParentDir(const std::filesystem::path& p) {
        const auto dir = p.has_filename() ? p.parent_path() : p;
        if (dir.empty()) return;
        std::error_code ec;
        (void)std::filesystem::create_directories(dir, ec);
        if (ec) {
            std::cerr << "[Measurement] create_directories failed for "
                << dir.string() << ": " << ec.message() << "\n";
        }
    }

    inline bool openCsv(std::ofstream& out, const std::filesystem::path& p) {
        ensureParentDir(p);
        out.open(p, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            std::error_code ec;
            auto abs = std::filesystem::absolute(p, ec);
            std::filesystem::path cwd;
            std::error_code ec2;
            cwd = std::filesystem::current_path(ec2);
            std::cerr << "[Measurement] Failed to open for write: "
                << (ec ? p.string() : abs.string())
                << " | cwd=" << (ec2 ? std::string("<unk>") : cwd.string())
                << "\n";
            return false;
        }
        out.imbue(std::locale::classic());
        out << std::fixed << std::setprecision(4);
        return true;
    }

    static inline double pi() { return 3.14159265358979323846; }

    // ---- NEU: Helper fürs Set/Unset-Handling ----
    static inline bool is_setD(double v) {
        // -1.0 (oder allgemein <0) bzw. NaN/Inf => unset
        return std::isfinite(v) && v >= 0.0;
    }

    template <class V>
    static inline void safe_push(std::vector<V>& vec, V v) {
        if (is_setD(double(v)) && v > 0) vec.push_back(v);
    }

    static double percentile_select(std::vector<double> v, double p) {
        if (v.empty()) return -1.0; // unset statt 0.0
        p = std::clamp(p, 0.0, 1.0);
        const size_t k = static_cast<size_t>(p * (v.size() - 1));
        std::nth_element(v.begin(), v.begin()+k, v.end());
        return v[k];
    }

    static inline double avg_or_unset(double sum, int cnt) {
        return (cnt > 0) ? (sum / double(cnt)) : -1.0;
    }

    static Synthesis computeSynthesis(
        const std::vector<LamureMeasurement::FrameStats>& stats,
        const std::unordered_map<unsigned,double>& gpuMsByFrame)
    {
        Synthesis syn;
        syn.nFrames = stats.size();
        if (stats.empty()) return syn;

        std::vector<double> ft_ms; ft_ms.reserve(stats.size());

        // Summen + Zähler nur über gesetzte Werte
        double sum_cpu_main=0, sum_wait=0, sum_gpu_from_tl=0;
        int cnt_cpu_main=0, cnt_wait=0, cnt_gpu_from_tl=0;

        double sum_cpu_bp=0, sum_gpu_bp=0, sum_wait_pct=0;
        int cnt_cpu_bp=0, cnt_gpu_bp=0, cnt_wait_pct=0;

        double sum_covC1=0, sum_covD=0; int cnt_cov=0;
        double sum_rpts=0; int cnt_rpts=0;

        // Marks/Estimates Summen + Zähler
        double m_draw_impl=0, m_p1=0, m_p2=0, m_p3=0, m_disp=0, m_bind=0, m_est=0, m_sp=0;
        int c_draw_impl=0, c_p1=0, c_p2=0, c_p3=0, c_disp=0, c_bind=0, c_est=0, c_sp=0;

        double t_est_screen_px=0, t_est_sum_area_px=0, t_est_density=0, t_est_coverage=0,
            t_est_coverage_px=0, t_est_overdraw=0, t_avg_area_pp=0;
        int    c_est_screen_px=0, c_est_sum_area_px=0, c_est_density=0, c_est_coverage=0,
            c_est_coverage_px=0, c_est_overdraw=0, c_avg_area_pp=0;

        for (const auto& s : stats) {
            // Frametime-Kandidat (nur >0 und gesetzt)
            const double ft_choice = is_setD(s.frame_duration_ms) ? s.frame_duration_ms
                : is_setD(s.rendering_traversals_ms) ? s.rendering_traversals_ms
                : -1.0;
            safe_push(ft_ms, ft_choice);
            if (is_setD(ft_choice)) syn.total_duration_ms += ft_choice;

            // CPU/GPU/Wait (nur gesetzte Werte addieren)
            if (is_setD(s.cpu_main_ms)) { sum_cpu_main += s.cpu_main_ms; ++cnt_cpu_main; }
            if (is_setD(s.wait_ms))     { sum_wait     += s.wait_ms;     ++cnt_wait;     }

            // busy/percent
            if (is_setD(s.cpu_busy_pct_proxy)) { sum_cpu_bp += s.cpu_busy_pct_proxy; ++cnt_cpu_bp; }
            if (is_setD(s.gpu_busy_pct_proxy)) { sum_gpu_bp += s.gpu_busy_pct_proxy; ++cnt_gpu_bp; }
            if (is_setD(s.wait_frac_pct))      { sum_wait_pct += s.wait_frac_pct;     ++cnt_wait_pct; }

            // Boundness (Stringzählung bleibt wie gehabt)
            if      (s.boundness == "GPU-bound")       ++syn.cnt_gpu;
            else if (s.boundness == "CPU-bound")       ++syn.cnt_cpu;
            else if (s.boundness == "Wait/Sync-bound") ++syn.cnt_wait;
            else if (s.boundness == "mixed")           ++syn.cnt_mixed;
            else                                       ++syn.cnt_unknown;

            // GPU-Zeit pro Frame aus Timeline
            if (auto it = gpuMsByFrame.find(s.frame_number); it != gpuMsByFrame.end()) {
                const double gms = it->second;
                if (is_setD(gms)) { sum_gpu_from_tl += gms; ++cnt_gpu_from_tl; }
            }

            // Coverage/Effizienz – nur wenn Inputs gesetzt/valide
            if (is_setD(s.est_density) || is_setD(s.est_coverage)) {
                const double D  = is_setD(s.est_density) ? std::max(0.0, double(s.est_density)) : 0.0;
                const double C1 = is_setD(s.est_coverage) ? double(s.est_coverage)
                    : (D > 0.0 ? 1.0 - std::exp(-D) : 0.0);
                if (C1 > 0.0 || D > 0.0) { sum_covC1 += C1; sum_covD += D; ++cnt_cov; }
            }

            // RPTS (points/ms) nur falls GPU>0 und Splats>0
            if (auto it = gpuMsByFrame.find(s.frame_number); it!=gpuMsByFrame.end()) {
                const double gms = it->second;
                if (is_setD(gms) && gms > 0.0 && s.rendered_primitives > 0) {
                    sum_rpts += double(s.rendered_primitives) / gms; ++cnt_rpts;
                }
            }

            // Renderer-Estimates (avg) – nur gesetzte übernehmen
            if (is_setD(s.est_screen_px))       { t_est_screen_px   += s.est_screen_px;   ++c_est_screen_px; }
            if (is_setD(s.est_sum_area_px))     { t_est_sum_area_px += s.est_sum_area_px; ++c_est_sum_area_px; }
            if (is_setD(s.est_density))         { t_est_density     += s.est_density;     ++c_est_density; }
            if (is_setD(s.est_coverage))        { t_est_coverage    += s.est_coverage;    ++c_est_coverage; }
            if (is_setD(s.est_coverage_px))     { t_est_coverage_px += s.est_coverage_px; ++c_est_coverage_px; }
            if (is_setD(s.est_overdraw))        { t_est_overdraw    += s.est_overdraw;    ++c_est_overdraw; }
            if (is_setD(s.avg_area_px_per_prim)){ t_avg_area_pp     += double(s.avg_area_px_per_prim); ++c_avg_area_pp; }

            // Marks: Heuristik Sekunden→ms, nur übernehmen wenn gesetzt (>=0, finite)
            const float mark_vals[] = {
                s.mark_draw_impl_ms, s.mark_pass1_ms, s.mark_pass2_ms, s.mark_pass3_ms,
                s.mark_dispatch_ms, s.mark_context_bind_ms, s.mark_estimates_ms, s.mark_singlepass_ms
            };
            float mx = 0.0f;
            for (float v : mark_vals) mx = std::max(mx, v);
            const bool marks_look_like_seconds = (mx > 0.0f && mx < 0.01f);
            auto norm = [&](float v)->double {
                if (!std::isfinite(v) || v < 0.0f) return -1.0;
                return marks_look_like_seconds ? (double(v) * 1000.0) : double(v);
                };
            auto add_mark = [&](double nv, double& sum, int& cnt){
                if (is_setD(nv)) { sum += nv; ++cnt; }
                };
            add_mark(norm(s.mark_draw_impl_ms),    m_draw_impl, c_draw_impl);
            add_mark(norm(s.mark_pass1_ms),        m_p1,        c_p1);
            add_mark(norm(s.mark_pass2_ms),        m_p2,        c_p2);
            add_mark(norm(s.mark_pass3_ms),        m_p3,        c_p3);
            add_mark(norm(s.mark_dispatch_ms),     m_disp,      c_disp);
            add_mark(norm(s.mark_context_bind_ms), m_bind,      c_bind);
            add_mark(norm(s.mark_estimates_ms),    m_est,       c_est);
            add_mark(norm(s.mark_singlepass_ms),   m_sp,        c_sp);
        }

        // Perzentile & Stutter
        syn.p50 = percentile_select(ft_ms, 0.50);
        syn.p95 = percentile_select(ft_ms, 0.95);
        syn.p99 = percentile_select(ft_ms, 0.99);
        syn.jitter_J = (is_setD(syn.p99) && is_setD(syn.p50)) ? std::max(0.0, syn.p99 - syn.p50) : -1.0;

        if (is_setD(syn.p50)) {
            for (double ft : ft_ms) {
                if (ft > 2.0 * syn.p50) ++syn.stutter_count_2xmedian;
                if (ft > 33.333)        ++syn.stutter_count_33ms;
            }
        } else {
            syn.stutter_count_2xmedian = -1;
            syn.stutter_count_33ms     = -1;
        }

        // Mittelwerte (nur über gesetzte Werte)
        syn.avg_frame_time_ms = avg_or_unset(syn.total_duration_ms, int(ft_ms.size()));
        syn.avg_cpu_main_ms   = avg_or_unset(sum_cpu_main,    cnt_cpu_main);
        syn.avg_gpu_time_ms   = avg_or_unset(sum_gpu_from_tl, cnt_gpu_from_tl);
        syn.avg_wait_ms       = avg_or_unset(sum_wait,        cnt_wait);

        // FPS (nur wenn Basis gesetzt)
        syn.fps_avg = is_setD(syn.avg_frame_time_ms) ? inv_ms_to_fps(syn.avg_frame_time_ms) : -1.0;
        syn.fps_p50 = is_setD(syn.p50) ? inv_ms_to_fps(syn.p50) : -1.0;
        syn.fps_p95 = is_setD(syn.p95) ? inv_ms_to_fps(syn.p95) : -1.0;
        syn.fps_p99 = is_setD(syn.p99) ? inv_ms_to_fps(syn.p99) : -1.0;

        syn.avg_cpu_busy_pct  = avg_or_unset(sum_cpu_bp,  cnt_cpu_bp);
        syn.avg_gpu_busy_pct  = avg_or_unset(sum_gpu_bp,  cnt_gpu_bp);
        syn.avg_wait_frac_pct = avg_or_unset(sum_wait_pct, cnt_wait_pct);

        syn.avg_rpts_points_per_ms = avg_or_unset(sum_rpts, cnt_rpts);
        syn.avg_covC1 = avg_or_unset(sum_covC1, cnt_cov);
        syn.avg_covD  = avg_or_unset(sum_covD,  cnt_cov);

        // Primitive/Frame (nur wenn Basis gesetzt)
        auto prod_or_unset = [](double a, double b)->double {
            return (is_setD(a) && is_setD(b)) ? (a*b) : -1.0;
            };
        syn.prims_per_frame_avg = prod_or_unset(syn.avg_rpts_points_per_ms, syn.avg_frame_time_ms);
        syn.prims_per_frame_p50 = prod_or_unset(syn.avg_rpts_points_per_ms, syn.p50);
        syn.prims_per_frame_p95 = prod_or_unset(syn.avg_rpts_points_per_ms, syn.p95);
        syn.prims_per_frame_p99 = prod_or_unset(syn.avg_rpts_points_per_ms, syn.p99);

        // Estimates averages
        syn.est_screen_px        = avg_or_unset(t_est_screen_px,   c_est_screen_px);
        syn.est_sum_area_px      = avg_or_unset(t_est_sum_area_px, c_est_sum_area_px);
        syn.est_density          = avg_or_unset(t_est_density,     c_est_density);
        syn.est_coverage         = avg_or_unset(t_est_coverage,    c_est_coverage);
        syn.est_coverage_px      = avg_or_unset(t_est_coverage_px, c_est_coverage_px);
        syn.est_overdraw         = avg_or_unset(t_est_overdraw,    c_est_overdraw);
        syn.avg_area_px_per_prim = avg_or_unset(t_avg_area_pp,     c_avg_area_pp);

        // Marks averages
        syn.mark_draw_impl_ms    = avg_or_unset(m_draw_impl, c_draw_impl);
        syn.mark_pass1_ms        = avg_or_unset(m_p1,        c_p1);
        syn.mark_pass2_ms        = avg_or_unset(m_p2,        c_p2);
        syn.mark_pass3_ms        = avg_or_unset(m_p3,        c_p3);
        syn.mark_dispatch_ms     = avg_or_unset(m_disp,      c_disp);
        syn.mark_context_bind_ms = avg_or_unset(m_bind,      c_bind);
        syn.mark_estimates_ms    = avg_or_unset(m_est,       c_est);
        syn.mark_singlepass_ms   = avg_or_unset(m_sp,        c_sp);

        return syn;
    }

} // namespace



LamureMeasurement::LamureMeasurement(
    Lamure*                      plugin,
    opencover::VRViewer*         viewer,
    const std::vector<Segment>&  segments,
    const std::string&           logfile)
    : m_plugin(plugin)
    , m_viewer(viewer)
    , m_segments(segments)
    , m_logfile(logfile)
{
    m_timeline.clear();


    m_exportReport    = true;

    // --- Logging/Verbosity ---
    m_verbose   = false;
    m_logEveryN = 30;
    m_dumpAttrs = false;

    // --- Backsearch & State ---
    m_gpuBackSearch = 16;
    m_running = true;
    m_written = false;
    m_haveSegmentStart = false;
    m_havePoseDeltas = false;
    m_gpu_static_captured = false;
    m_gpu_static_tried = false;
    m_measure_timeline = m_plugin->getSettings().measure_full;

    m_gpu_mem_used_mb_static = m_gpu_mem_total_mb_static = 0.0;
    m_gpu_mem_used_mb_nvml_static = m_gpu_mem_total_mb_nvml_static = 0.0;
    m_gpu_mem_used_mb_gl_static = m_gpu_mem_total_mb_gl_static = 0.0;
    m_startTick = osg::Timer::instance()->tick();
    m_lastFrameTick = m_startTick;

    initCallbacks();

    if (auto* vs = m_viewer ? m_viewer->getViewerStats() : nullptr) {
        // Für Debug/Grundwerte minimal:
        vs->collectStats("frame_rate", true);
        vs->collectStats("update",     true);

        if (m_plugin->getSettings().measure_full) {
            vs->collectStats("sync",   true);
            vs->collectStats("swap",   true);
            vs->collectStats("finish", true);
            vs->collectStats("isect",  true);
            vs->collectStats("plugin", true);
            vs->collectStats("opencover", true);
            vs->collectStats("gpu",    true);
        }
    }

    osgViewer::ViewerBase::Cameras cams;
    if (m_viewer) m_viewer->getCameras(cams);
    for (auto* cam : cams)
        if (auto* cs = cam->getStats()) {
            cs->collectStats("rendering", true);
            cs->collectStats("gpu",       true);
            cs->collectStats("cull",      true);
            cs->collectStats("draw",      true);
        }

    std::cout << "[Measurement] Measurement started\n";
}


LamureMeasurement::~LamureMeasurement()
{
    std::cout << "[Measurement] Measurement finished.\n";
}

bool LamureMeasurement::isActive() const noexcept {
    return m_running;
}

int LamureMeasurement::findStatsIndexForFrame(uint64_t frame_no) const {
    // Suche rückwärts im letzten Fenster – meist ist das der letzte Eintrag
    for (int i = int(m_stats.size()) - 1; i >= 0 && i >= int(m_stats.size()) - 32; --i) {
        if (m_stats[size_t(i)].frame_number == frame_no) return i;
    }
    return m_stats.empty() ? -1 : int(m_stats.size()) - 1;
}


bool LamureMeasurement::wantsSampling(unsigned frameNo) const noexcept {
    if (m_plugin->getSettings().measure_off) return false;
    const unsigned N = m_plugin->getSettings().measure_sample;
    if (N == 0u) return true;            // oder false – je nach gewünschtem Default
    return (frameNo % N) == 0u;
}


void LamureMeasurement::stop() {
    m_running = false;
    if (!m_written) {
        writeLogAndStop();
    }
}

void LamureMeasurement::initCallbacks()
{
    auto cam = m_viewer->getCamera();
    m_preCB = new MarkCallback(this, true);
    m_postCB = new MarkCallback(this, false);
    cam->setPreDrawCallback(m_preCB);
    cam->setPostDrawCallback(m_postCB);
}

void LamureMeasurement::cacheStaticGpuInfo()
{
    if (m_gpu_static_captured)
        return;
    double used_nvml = 0.0, total_nvml = 0.0;
    double used_gl   = 0.0, total_gl   = 0.0;
    bool haveNvml = false, haveGl = false;
#ifdef _WIN32
    {
        static NvmlLoader g_nvml_once;
        double util = 0.0, used = 0.0, total = 0.0;
        if (nvmlPoll(g_nvml_once, util, used, total)) {
            used_nvml  = used;
            total_nvml = total;
            haveNvml   = true;
        }
    }
#endif
    if (glewIsSupported("GL_NVX_gpu_memory_info")) {
        GLint totalKB = 0, freeKB = 0;
        glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX,   &totalKB);
        glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &freeKB);
        if (totalKB > 0 && freeKB >= 0) {
            total_gl = totalKB / 1024.0;
            const double free_gl = freeKB / 1024.0;
            used_gl   = std::max(0.0, total_gl - free_gl);
            haveGl    = true;
        }
    }
    if (haveNvml) {
        m_gpu_mem_used_mb_nvml_static  = used_nvml;
        m_gpu_mem_total_mb_nvml_static = total_nvml;
    }
    if (haveGl) {
        m_gpu_mem_used_mb_gl_static    = used_gl;
        m_gpu_mem_total_mb_gl_static   = total_gl;
    }
    if (haveNvml) {
        m_gpu_mem_used_mb_static  = m_gpu_mem_used_mb_nvml_static;
        m_gpu_mem_total_mb_static = m_gpu_mem_total_mb_nvml_static;
    } else if (haveGl) {
        m_gpu_mem_used_mb_static  = m_gpu_mem_used_mb_gl_static;
        m_gpu_mem_total_mb_static = m_gpu_mem_total_mb_gl_static;
    } else {
        m_gpu_mem_used_mb_static  = 0.0;
        m_gpu_mem_total_mb_static = 0.0;
    }
    m_gpu_static_captured = true;
    std::cout << "[Measurement] Cached static GPU memory: primary used/total="
        << m_gpu_mem_used_mb_static << "/"
        << m_gpu_mem_total_mb_static << " MB"
        << (haveNvml ? " (NVML)" : (haveGl ? " (GL_NVX)" : " (none)"))
        << "\n";
}

bool LamureMeasurement::getTimeTakenMsBacksearch(
    osg::Stats* s,
    unsigned baseFrame,
    unsigned backSearch,
    const std::string& timeTakenKey,
    const std::string& beginKey,
    const std::string& endKey,
    double& outMs,
    unsigned& usedOffset,
    double* outBeginMs,
    double* outEndMs)
{
    outMs = 0.0; usedOffset = 0;
    if (outBeginMs) *outBeginMs = 0.0;
    if (outEndMs)   *outEndMs   = 0.0;
    if (!s) return false;

    const unsigned earliest = s->getEarliestFrameNumber();
    const unsigned latest   = s->getLatestFrameNumber();
    if (baseFrame > latest) baseFrame = latest;

    for (unsigned off = 0; off <= backSearch; ++off)
    {
        if (baseFrame < off) break;
        const unsigned f = baseFrame - off;
        if (f < earliest) break;

        double vTaken = 0.0;
        if (s->getAttribute(f, timeTakenKey, vTaken)) {
            outMs = vTaken * 1000.0;
            usedOffset = off;
            if (outBeginMs || outEndMs) {
                double b=0.0, e=0.0;
                if (outBeginMs && s->getAttribute(f, beginKey, b)) *outBeginMs = b * 1000.0;
                if (outEndMs   && s->getAttribute(f, endKey,   e)) *outEndMs   = e * 1000.0;
            }
            return true;
        }

        double b=0.0, e=0.0;
        if (s->getAttribute(f, beginKey, b) && s->getAttribute(f, endKey, e) && e>=b) {
            outMs = (e-b) * 1000.0;
            usedOffset = off;
            if (outBeginMs) *outBeginMs = b * 1000.0;
            if (outEndMs)   *outEndMs   = e * 1000.0;
            return true;
        }
    }
    return false;
}


bool LamureMeasurement::tryAddBlock(
    osg::Stats* stats,
    unsigned int baseFrame,
    unsigned int backsearch,
    const std::string& statPrefix,
    const std::string& nameForCSV,
    const std::string& scope,
    int camIndex,
    std::vector<TimeBlock>& localBlocks)
{
    if (!m_measure_timeline || !stats) return false;

    double taken_ms = 0.0, begin_ms = 0.0, end_ms = 0.0;
    unsigned usedOffset = 0;

    if (!getTimeTakenMsBacksearch(stats, baseFrame, backsearch,
        statPrefix + " time taken",
        statPrefix + " begin time",
        statPrefix + " end time",
        taken_ms, usedOffset, &begin_ms, &end_ms))
        return false;

    TimeBlock tb;
    tb.frame       = baseFrame;  // bleibt auf baseFrame
    tb.src_frame   = (usedOffset <= baseFrame) ? baseFrame - usedOffset : baseFrame;
    tb.camIndex    = camIndex;
    tb.scope       = scope;
    tb.name        = nameForCSV;
    tb.begin_ms    = begin_ms;
    tb.end_ms      = end_ms;
    tb.taken_ms    = taken_ms;
    tb.used_offset = usedOffset;

    localBlocks.push_back(std::move(tb));
    return true;
}

void LamureMeasurement::drawIncrement(bool preDraw, const osg::FrameStamp* frameStamp)
{
    if (!m_running) return;

    if (preDraw) {
        if (!m_gpu_static_tried) { cacheStaticGpuInfo(); m_gpu_static_tried = true; }
        return;
    }

    if (frameStamp) {
        const unsigned cur = frameStamp->getFrameNumber();
        if (cur != m_currFrameNo) {
            m_currFrameNo = cur;
            m_sampleThisFrame = !m_plugin->getSettings().measure_off && wantsSampling(m_currFrameNo);
        }
    }

    // --- FrameStamp optional behandeln
    const bool haveFS = (frameStamp != nullptr);
    const unsigned f  = haveFS ? frameStamp->getFrameNumber() : m_currFrameNo;

    // Nur wenn dieses Frame gesampelt werden soll, Stats sammeln.
    if (haveFS && m_sampleThisFrame) {
        FrameStats s{};
        s.frame_number = m_currFrameNo; // Eine Quelle der Wahrheit
        const bool dbg = (m_verbose && ((m_currFrameNo % m_logEveryN) == 0));
        if (collectFrameStats(m_viewer, frameStamp, s, dbg)) {
            if (m_plugin->getSettings().measure_light) {
                FrameStats slim{};
                slim.frame_number            = s.frame_number;
                slim.frame_duration_ms       = s.frame_duration_ms;
                slim.rendering_traversals_ms = s.rendering_traversals_ms;
                slim.cpu_main_ms             = s.cpu_main_ms;
                slim.gpu_time_ms             = s.gpu_time_ms;
                slim.rendered_primitives     = s.rendered_primitives;
                slim.rendered_nodes          = s.rendered_nodes;
                slim.est_density             = s.est_density;
                slim.est_coverage            = s.est_coverage;
                slim.est_overdraw            = s.est_overdraw;
                slim.segment_index           = s.segment_index;
                m_stats.push_back(std::move(slim));
            } else {
                m_stats.push_back(std::move(s));
            }
        }
    }

    // Zeitbasis
    const double tNow = (frameStamp)
        ? frameStamp->getReferenceTime()
        : osg::Timer::instance()->delta_s(m_startTick, osg::Timer::instance()->tick());

    if (!m_haveSegmentStart) { m_segmentStartRefTime = tNow; m_haveSegmentStart = true; }


    // --- (ab hier Deine vorhandene Segment-/Kamera-Logik unverändert) ---
    auto colLen = [](const osg::Matrix& M, int c){
        osg::Vec3 v(M(0,c), M(1,c), M(2,c));
        return std::max(1e-9, (double)v.length());
        };
    auto computeScale = [&](){
        const osg::Matrix dcsNow  = opencover::cover->getXformMat();
        const osg::Matrix baseM   = opencover::cover->getBaseMat();
        double sx = colLen(baseM, 0), sy = colLen(baseM, 1), sz = colLen(baseM, 2);
        const double avgBase = (sx + sy + sz) / 3.0;
        if (avgBase > 0.999 && avgBase < 1.001) {
            const osg::Matrix worldM = baseM * dcsNow;
            sx = colLen(worldM, 0);
            sy = colLen(worldM, 1);
            sz = colLen(worldM, 2);
        }
        return osg::Vec3d(sx, sy, sz);
        };
    auto dirScale = [&](const osg::Vec3& traMeters, const osg::Vec3d& S){
        const double L = osg::Vec3d(traMeters).length();
        if (L <= 1e-12) return 1.0;
        const osg::Vec3d u = osg::Vec3d(traMeters) / L;
        const double sx = S.x(), sy = S.y(), sz = S.z();
        return std::sqrt( (u.x()*sx)*(u.x()*sx) + (u.y()*sy)*(u.y()*sy) + (u.z()*sz)*(u.z()*sz) );
        };

    // Segmente vorziehen falls fertig
    while (m_currentSegment < m_segments.size())
    {
        const auto& seg = m_segments[m_currentSegment];

        const float  dist = seg.tra.length();
        const osg::Vec3d S = computeScale();
        const double sEff  = dirScale(seg.tra, S);
        const double transSpeedEff = (seg.transSpeed > 0.f ? seg.transSpeed * sEff : 0.0);
        const double transDur = (transSpeedEff > 0.0 && dist > 0.0) ? (dist / transSpeedEff) : 0.0;

        const double angleMag = std::max({ std::abs((double)seg.rot.x()),
            std::abs((double)seg.rot.y()),
            std::abs((double)seg.rot.z()) });
        const double rotDur   = (seg.rotSpeed > 0.f && angleMag > 0.0) ? (angleMag / seg.rotSpeed) : 0.0;

        const double segDuration =
            (transDur > 0.0 && rotDur > 0.0) ? std::max(transDur, rotDur)
            : ((transDur > 0.0) ? transDur : rotDur);

        const double elapsed = tNow - m_segmentStartRefTime;
        if (elapsed < segDuration) break;

        m_cumulativeTra += seg.tra;
        m_cumulativeRot += seg.rot;
        ++m_currentSegment;
        m_segmentStartRefTime += segDuration;
    }

    if (m_currentSegment >= m_segments.size()) { stop(); return; }

    // Aktuelles Segment interpolieren
    const auto& seg = m_segments[m_currentSegment];

    const float  dist = seg.tra.length();
    const osg::Vec3d S = computeScale();
    const double sEff  = dirScale(seg.tra, S);
    const double transSpeedEff = (seg.transSpeed > 0.f ? seg.transSpeed * sEff : 0.0);
    const double transDur = (transSpeedEff > 0.0 && dist > 0.0) ? (dist / transSpeedEff) : 0.0;

    const double angleMag = std::max({ std::abs((double)seg.rot.x()),
        std::abs((double)seg.rot.y()),
        std::abs((double)seg.rot.z()) });
    const double rotDur   = (seg.rotSpeed > 0.f && angleMag > 0.0) ? (angleMag / seg.rotSpeed) : 0.0;

    const double segDuration =
        (transDur > 0.0 && rotDur > 0.0) ? std::max(transDur, rotDur)
        : ((transDur > 0.0) ? transDur : rotDur);

    const double elapsed = std::max(0.0, tNow - m_segmentStartRefTime);
    const double frac = (segDuration > 0.0) ? std::clamp(elapsed / segDuration, 0.0, 1.0) : 1.0;

    const osg::Vec3 tra = m_cumulativeTra + seg.tra * static_cast<float>(frac);
    const osg::Vec3 rot = m_cumulativeRot + seg.rot * static_cast<float>(frac);

    updateCamera(tra, rot);
}


void LamureMeasurement::updateCamera(const osg::Vec3& traAbs, const osg::Vec3& rotAbsDeg)
{
    if (!m_havePoseDeltas) {
        m_lastTraApplied = traAbs;
        m_lastRotApplied = rotAbsDeg;
        m_havePoseDeltas = true;
        return;
    }

    // Inkremente seit letztem Frame (Meter bzw. Grad)
    const osg::Vec3 dTra = traAbs - m_lastTraApplied;
    const osg::Vec3 dRot = rotAbsDeg - m_lastRotApplied;

    m_lastTraApplied = traAbs;
    m_lastRotApplied = rotAbsDeg;

    const double dPitch = osg::DegreesToRadians(dRot.x());
    const double dRoll  = osg::DegreesToRadians(dRot.y());
    const double dYaw = osg::DegreesToRadians(dRot.z());

    osg::Matrix dcs = opencover::cover->getXformMat();

    // Lokale Achsen aus aktueller Viewer-Orientierung holen
    const osg::Matrixd viewerM = opencover::cover->getViewerMat();
    const osg::Vec3 viewerPos  = viewerM.getTrans();
    osg::Vec3 axisX(viewerM(0,0), viewerM(1,0), viewerM(2,0));
    osg::Vec3 axisY(viewerM(0,1), viewerM(1,1), viewerM(2,1));
    osg::Vec3 axisZ(viewerM(0,2), viewerM(1,2), viewerM(2,2));
    axisX.normalize(); axisY.normalize(); axisZ.normalize();

    // Rotation um eigene Achsen (in dieser Reihenfolge: Yaw -> Pitch -> Roll)
    dcs.postMult(osg::Matrix::translate(-viewerPos));
    if (std::abs(dYaw)   > 0.0) dcs.postMult(osg::Matrix::rotate(dYaw,   axisZ));
    if (std::abs(dPitch) > 0.0) dcs.postMult(osg::Matrix::rotate(dPitch, axisX));
    if (std::abs(dRoll)  > 0.0) dcs.postMult(osg::Matrix::rotate(dRoll,  axisY));
    dcs.postMult(osg::Matrix::translate(viewerPos));

    // ---- Translation: Meter -> Welt (Skalierung berücksichtigen, Multiplikations-Variante) ----
    auto colLen = [](const osg::Matrix& M, int c){
        osg::Vec3 v(M(0,c), M(1,c), M(2,c));
        return std::max(1e-9, (double)v.length());
        };

    const osg::Matrix baseM  = opencover::cover->getBaseMat();
    double sx = colLen(baseM, 0), sy = colLen(baseM, 1), sz = colLen(baseM, 2);
    const double avgBase = (sx + sy + sz) / 3.0;
    if (avgBase > 0.999 && avgBase < 1.001) {
        const osg::Matrix worldM = baseM * dcs;
        sx = colLen(worldM, 0);
        sy = colLen(worldM, 1);
        sz = colLen(worldM, 2);
    }

    const osg::Vec3 dTraLocal(
        static_cast<float>(dTra.x() * sx),
        static_cast<float>(dTra.y() * sy),
        static_cast<float>(dTra.z() * sz)
    );

    dcs.postMult(osg::Matrix::translate(dTraLocal));
    opencover::cover->setXformMat(dcs);
}

bool LamureMeasurement::collectFrameStats(osgViewer::ViewerBase* viewer,
    const osg::FrameStamp* fs, FrameStats& stats, bool debugPrint)
{
    if (!viewer) return false;

    osg::Stats* viewerStats = viewer->getViewerStats();
    osgViewer::ViewerBase::Cameras cams;
    viewer->getCameras(cams);

    // EINHEITLICH dieselbe Frame-ID verwenden:
    unsigned f = stats.frame_number;
    if (f == 0u) { // Fallback, falls doch mal leer reinkommt
        f = fs ? fs->getFrameNumber()
            : (viewerStats ? viewerStats->getLatestFrameNumber() : 0u);
        stats.frame_number = f;
    }

    // --- Viewer
    if (viewerStats) {
        double hz=0, ft=0, rtv=0; unsigned rtoff=0;
        if (viewerStats->getAttribute(f, "Frame rate", hz))
            stats.frame_rate = static_cast<float>(hz);
        if (viewerStats->getAttribute(f, "Frame duration", ft))
            stats.frame_duration_ms = static_cast<float>(ft * 1000.0);

        if (getTimeTakenMsBacksearch(viewerStats, f, m_gpuBackSearch,
            "Rendering traversals time taken",
            "Rendering traversals begin time",
            "Rendering traversals end time",
            rtv, rtoff))
        {
            stats.rendering_traversals_ms = static_cast<float>(rtv);
            if (stats.frame_duration_ms <= 0.0f)
                stats.frame_duration_ms = static_cast<float>(rtv);
        }

        // Fallback for frame_rate from frame_duration_ms
        if (stats.frame_rate <= 0.f && stats.frame_duration_ms > 0.f)
            stats.frame_rate = static_cast<float>(inv_ms_to_fps(stats.frame_duration_ms));

        auto addViewer = [&](const std::string& prefix, const char* name){
            double ms=0; unsigned off=0;
            if (getTimeTakenMsBacksearch(viewerStats, f, m_gpuBackSearch,
                prefix + " time taken", prefix + " begin time", prefix + " end time",
                ms, off))
            {
                if      (std::strcmp(name, "Update traversal")==0) stats.cpu_update_ms = static_cast<float>(ms);
                else if (std::strcmp(name, "sync")==0)             stats.sync_time_ms  = static_cast<float>(ms);
                else if (std::strcmp(name, "swap")==0)             stats.swap_time_ms  = static_cast<float>(ms);
                else if (std::strcmp(name, "finish")==0)           stats.finish_ms     = static_cast<float>(ms);
                else if (std::strcmp(name, "Plugin")==0)           stats.plugin_ms     = static_cast<float>(ms);
                else if (std::strcmp(name, "Isect")==0)            stats.isect_ms      = static_cast<float>(ms);
                else if (std::strcmp(name, "opencover")==0)        stats.opencover_ms  = static_cast<float>(ms);
            }
            if (m_measure_timeline) {
                (void)tryAddBlock(viewerStats, f, m_gpuBackSearch, prefix, name, "viewer", -1, m_timeline);
            }
            };

        addViewer("Update traversal", "Update traversal");
        addViewer("sync",            "sync");
        addViewer("swap",            "swap");
        addViewer("finish",          "finish");
        addViewer("Plugin",          "Plugin");
        addViewer("Isect",           "Isect");
        addViewer("opencover",       "opencover");

        double v=0.0;
        if (viewerStats->getAttribute(f, "GPU Clock MHz", v))     stats.gpu_clock     = static_cast<float>(v);
        if (viewerStats->getAttribute(f, "GPU Mem Clock MHz", v)) stats.gpu_mem_clock = static_cast<float>(v);
        if (viewerStats->getAttribute(f, "GPU Utilization", v))   stats.gpu_util      = static_cast<float>(v);
        if (viewerStats->getAttribute(f, "GPU PCIe rx KB/s", v))  stats.gpu_pci       = static_cast<float>(v);
    }

    // --- Kamera-Aggregate (akkumulieren in double, 1x cast)
    double sumCull=0, sumDraw=0, sumGpu=0;
    std::vector<TimeBlock> blocks;
    for (size_t i=0; i<cams.size(); ++i) {
        osg::Camera* cam = cams[i];
        osg::Stats*  cs  = cam ? cam->getStats() : nullptr;
        if (!cs) continue;

        unsigned base = stats.frame_number;
        if (auto* rnd = dynamic_cast<osgViewer::Renderer*>(cam->getRenderer()))
            if (!rnd->getGraphicsThreadDoesCull() && base>0) --base;

        double cull=0, draw=0, gpu=0; unsigned offC=0, offD=0, offG=0;

        if (getTimeTakenMsBacksearch(cs, base, m_gpuBackSearch,
            "Cull traversal time taken", "Cull traversal begin time", "Cull traversal end time",
            cull, offC))
        { stats.backoff_cull = std::max(stats.backoff_cull, offC); sumCull += cull; }
        if (m_measure_timeline)
            (void)tryAddBlock(cs, base, m_gpuBackSearch, "Cull traversal", "Cull traversal", "camera", int(i), blocks);

        if (getTimeTakenMsBacksearch(cs, base, m_gpuBackSearch,
            "Draw traversal time taken", "Draw traversal begin time", "Draw traversal end time",
            draw, offD))
        { stats.backoff_draw = std::max(stats.backoff_draw, offD); sumDraw += draw; }
        if (m_measure_timeline)
            (void)tryAddBlock(cs, base, m_gpuBackSearch, "Draw traversal", "Draw traversal", "camera", int(i), blocks);

        if (!getTimeTakenMsBacksearch(cs, base, m_gpuBackSearch,
            "GPU draw time taken", "GPU draw begin time", "GPU draw end time",
            gpu, offG))
        {
            getTimeTakenMsBacksearch(cs, base, m_gpuBackSearch,
                "GPU time taken", "GPU begin time", "GPU end time",
                gpu, offG);
        }
        if (gpu > 0.0) { stats.backoff_gpu = std::max(stats.backoff_gpu, offG); sumGpu += gpu; }

        if (m_measure_timeline) {
            if (!tryAddBlock(cs, base, m_gpuBackSearch, "GPU draw", "GPU draw", "camera", int(i), blocks))
                (void)tryAddBlock(cs, base, m_gpuBackSearch, "GPU", "GPU time", "camera", int(i), blocks);
        }
    }
    if (m_measure_timeline && !blocks.empty())
        m_timeline.insert(m_timeline.end(), blocks.begin(), blocks.end());

    stats.cpu_cull_ms = static_cast<float>(sumCull);
    stats.cpu_draw_ms = static_cast<float>(sumDraw);
    stats.gpu_time_ms = static_cast<float>(sumGpu);

    // --- Pose (nur in Full)
    if (m_plugin->getSettings().measure_full) {
        cacheStaticGpuInfo();
        const osg::Camera* camPose = (!cams.empty() && cams.front()) ? cams.front() : (m_viewer ? m_viewer->getCamera() : nullptr);
        if (camPose) {
            const osg::Matrixd w2v = opencover::cover->getBaseMat() * camPose->getViewMatrix();
            osg::Matrixd v2w;
            if (v2w.invert(w2v)) {
                osg::Vec3d T,S; osg::Quat R,SO;
                v2w.decompose(T, R, S, SO);
                if (m_haveLastQuat) {
                    const double dot = R.x()*m_lastQuat.x() + R.y()*m_lastQuat.y() + R.z()*m_lastQuat.z() + R.w()*m_lastQuat.w();
                    if (dot < 0.0) R.set(-R.x(), -R.y(), -R.z(), -R.w());
                }
                m_lastQuat = R; m_haveLastQuat = true;
                stats.position = T; stats.orientation = R;
            }
        }
    }

    // --- Renderer-Estimates (capped + raw)
    const auto ri = m_plugin->getRenderInfo();
    stats.rendered_primitives         = ri.rendered_primitives;
    stats.rendered_nodes              = ri.rendered_nodes;
    stats.rendered_bounding_boxes     = ri.rendered_bounding_boxes;

    stats.est_screen_px    = ri.est_screen_px;
    stats.est_sum_area_px  = ri.est_sum_area_px;
    stats.est_density      = ri.est_density;
    stats.est_coverage     = ri.est_coverage;
    stats.est_coverage_px  = ri.est_coverage_px;
    stats.est_overdraw     = ri.est_overdraw;
    stats.avg_area_px_per_prim = ri.avg_area_px_per_prim;

    stats.est_density_raw      = ri.est_density_raw;
    stats.est_coverage_raw     = ri.est_coverage_raw;
    stats.est_coverage_px_raw  = ri.est_coverage_px_raw;
    stats.est_overdraw_raw     = ri.est_overdraw_raw;

    // Fallback, falls nur Fläche/Screen bekannt (capped)
    if ((stats.est_density <= 0.0f || stats.est_coverage <= 0.0f) &&
        (stats.est_sum_area_px > 0.0f && stats.est_screen_px > 0.0f))
    {
        const float D = stats.est_sum_area_px / std::max(1.0f, stats.est_screen_px);
        stats.est_density     = std::max(0.0f, D);
        stats.est_coverage    = 1.0f - static_cast<float>(std::exp(-D));
        stats.est_coverage_px = stats.est_coverage * stats.est_screen_px;
        stats.est_overdraw    = (stats.est_coverage_px > 0.0f) ? (stats.est_sum_area_px / stats.est_coverage_px) : 0.0f;
    }

    // --- Debug
    if (debugPrint || (m_verbose && ((stats.frame_number % m_logEveryN) == 0))) {
        std::cout << std::fixed << std::setprecision(3)
            << "[LamureMeasurement] f=" << stats.frame_number
            << " | frame=" << stats.frame_duration_ms
            << " | upd="   << stats.cpu_update_ms
            << " | cull="  << stats.cpu_cull_ms
            << " | draw="  << stats.cpu_draw_ms
            << " | gpu="   << stats.gpu_time_ms
            << " | swap="  << stats.swap_time_ms
            << " | sync="  << stats.sync_time_ms
            << " | plug="  << stats.plugin_ms
            << " | cover=" << stats.opencover_ms
            << "\n";
    }

    // === Abgeleitete Proxys & Boundness (KORRIGIERTES SET/UNSET-HANDLING) ===
    const float ft = (stats.frame_duration_ms > 0.0f)
        ? stats.frame_duration_ms
        : stats.rendering_traversals_ms;

    auto add_if_set = [](float acc, float v)->float {
        return (std::isfinite(v) && v >= 0.0f) ? (acc + v) : acc;
        };

    // cpu_main_ms: Summe nur über gesetzte Komponenten
    bool any_cpu_known = false;
    float cpu_main = 0.0f;
    auto add_cpu = [&](float v){ if (std::isfinite(v) && v >= 0.0f) { cpu_main += v; any_cpu_known = true; } };
    add_cpu(stats.cpu_update_ms);
    add_cpu(stats.cpu_cull_ms);
    add_cpu(stats.cpu_draw_ms);
    add_cpu(stats.plugin_ms);
    add_cpu(stats.isect_ms);
    add_cpu(stats.opencover_ms);
    stats.cpu_main_ms = any_cpu_known ? cpu_main : -1.0f;

    // wait_ms: Summe nur über gesetzte Wait-Komponenten
    bool any_wait_known = false;
    float waits = 0.0f;
    auto add_wait = [&](float v){ if (std::isfinite(v) && v >= 0.0f) { waits += v; any_wait_known = true; } };
    add_wait(stats.sync_time_ms);
    add_wait(stats.swap_time_ms);
    add_wait(stats.finish_ms);
    stats.wait_ms = any_wait_known ? waits : -1.0f;

    // Prozente nur berechnen, wenn ft > 0
    if (ft > 0.0f) {
        const float cpu_clip  = std::max(0.0f, stats.cpu_main_ms);
        const float gpu_clip  = std::min(std::max(0.0f, stats.gpu_time_ms), ft);
        const float wait_clip = std::max(0.0f, stats.wait_ms);

        stats.cpu_busy_pct_proxy = std::clamp(100.0f * (cpu_clip / ft), 0.0f, 100.0f);
        stats.gpu_busy_pct_proxy = std::clamp(100.0f * (gpu_clip / ft), 0.0f, 100.0f);
        stats.wait_frac_pct      = 100.0f * (wait_clip / ft);
    } else {
        stats.cpu_busy_pct_proxy = -1.0f;
        stats.gpu_busy_pct_proxy = -1.0f;
        stats.wait_frac_pct      = -1.0f;
    }

    // Boundness
    constexpr float GPU_HI=70.f, CPU_HI=70.f, CPU_LO=60.f, GPU_LO=60.f, WAIT_HI=20.f;
    auto bad = [](float v){ return !std::isfinite(v) || v < 0.0f; };

    if (ft <= 0.0f || bad(stats.cpu_busy_pct_proxy) || bad(stats.gpu_busy_pct_proxy) || bad(stats.wait_frac_pct))
        stats.boundness = "unknown";
    else if (stats.wait_frac_pct > WAIT_HI)
        stats.boundness = "Wait/Sync-bound";
    else if (stats.gpu_busy_pct_proxy > GPU_HI && stats.cpu_busy_pct_proxy < CPU_LO)
        stats.boundness = "GPU-bound";
    else if (stats.cpu_busy_pct_proxy > CPU_HI && stats.gpu_busy_pct_proxy < GPU_LO)
        stats.boundness = "CPU-bound";
    else
        stats.boundness = "mixed";

    // --- FrameMarks (nur in Full)
    if (m_plugin->getSettings().measure_full) {
        const FrameMarks& fm = m_plugin->getFrameMarks();
        stats.mark_draw_impl_ms     = fm.draw_cb_ms;
        stats.mark_pass1_ms         = fm.pass1_ms;
        stats.mark_pass2_ms         = fm.pass2_ms;
        stats.mark_pass3_ms         = fm.pass3_ms;
        stats.mark_singlepass_ms    = fm.singlepass_ms;
        stats.mark_dispatch_ms      = fm.dispatch_ms;
        stats.mark_context_bind_ms  = fm.context_bind_ms;
        stats.mark_estimates_ms     = fm.estimates_ms;
    }

    stats.segment_index = static_cast<int>(m_currentSegment);
    return true;
}


static inline std::string csvQuote(const std::string& s) {
    if (s.find_first_of(",;\t\" \n\r") == std::string::npos) return s;
    std::string r; r.reserve(s.size()+8);
    r.push_back('"');
    for (char c : s) r += (c=='"') ? std::string("\"\"") : std::string(1,c);
    r.push_back('"');
    return r;
}

void LamureMeasurement::writeLogAndStop()
{
    if (m_written) { std::cerr << "[Measurement] writeLogAndStop() already executed, skipping.\n"; return; }

    const bool mode_off   = m_plugin->getSettings().measure_off;
    const bool mode_light = m_plugin->getSettings().measure_light;
    const bool mode_full  = m_plugin->getSettings().measure_full;

    const bool hasStats    = !m_stats.empty();
    const bool hasTimeline = (m_measure_timeline && !m_timeline.empty());

    // Basename + Pfade
    std::filesystem::path base_path(m_logfile);
    if (base_path.empty())         base_path = std::filesystem::path("lamure_measurement");
    if (base_path.has_extension()) base_path.replace_extension();

    const auto frames_path   = base_path.string() + "_frames.csv";
    const auto json_path     = base_path.string() + "_summary.json";
    const auto timeline_path = base_path.string() + "_timeline.csv";
    const auto md_path       = base_path.string() + "_report.md";
    m_reportMDPath = md_path;

    cacheStaticGpuInfo();

    // Wenn wirklich gar keine Daten: trotzdem Settings + (kleines) MD schreiben.
    if (mode_off || (!hasStats && !hasTimeline)) {
        m_written = true;
        return;
    }

    // 1) GPU/Timeline-Aggregat pro Frame
    std::unordered_map<unsigned,double> gpuMsByFrame;
    if (hasTimeline) {
        gpuMsByFrame.reserve(m_timeline.size());
        for (const auto& tb : m_timeline)
            if (tb.name.find("GPU") != std::string::npos)
                gpuMsByFrame[tb.frame] += tb.taken_ms;
    }

    // 2) Frames.csv
    bool wrote_frames = false;
    if (hasStats) wrote_frames = writeFramesCSV(frames_path, mode_full, mode_light, gpuMsByFrame);

    // 3) Timeline.csv
    bool wrote_timeline = false;
    if (hasTimeline) { writeTimelineCSV(timeline_path); wrote_timeline = true; }

    // 4) ZENTRAL: Synthesis
    Synthesis syn = computeSynthesis(m_stats, gpuMsByFrame);
    syn.timelineBlocks = hasTimeline ? m_timeline.size() : 0;

    // 5) JSON Summary (maschinenlesbar)
    bool wrote_summary_json = writeSummaryJSON(json_path, syn, mode_full, mode_light, hasTimeline);

    if (mode_full || mode_light)
    {
        (void)writeReportMarkdown(
            md_path,
            syn,
            mode_full,
            mode_light,
            wrote_frames,
            wrote_timeline,
            wrote_summary_json,
            frames_path,
            timeline_path,
            json_path,
            base_path
        );
    }

    // 7) Settings JSON immer
    if (mode_full || mode_light) {
        const auto settings_path = base_path.string() + "_settings.json";
        m_plugin->writeSettingsJson(m_plugin->getSettings(), settings_path);
    }

    m_written = true;
}


bool LamureMeasurement::writeReportMarkdown(
    const std::filesystem::path& md_path,
    const Synthesis& syn,
    bool mode_full,
    bool mode_light,
    bool wrote_frames,
    bool wrote_timeline,
    bool wrote_summary_json,
    const std::filesystem::path& frames_path,
    const std::filesystem::path& timeline_path,
    const std::filesystem::path& json_path,
    const std::filesystem::path& base_path)
{
    ensureParentDir(md_path);

    std::ofstream md(md_path, std::ios::out | std::ios::trunc);
    if (!md) {
        std::cerr << "[Measurement] Could not write Markdown report: " << md_path << "\n";
        return false;
    }

    // Keine Exponentialnotation im Report
    md.setf(std::ios::fixed);
    md << std::setprecision(3);

    // --- Helpers für "nur wenn gesetzt" ---
    auto isSetD = [](double v){ return std::isfinite(v) && v >= 0.0; };
    auto isSetI = [](long long v){ return v >= 0; };

    auto yamlD = [&](const char* key, double v){
        if (isSetD(v)) md << "  " << key << ": " << v << "\n";
        };
    auto yamlI = [&](const char* key, long long v){
        if (isSetI(v)) md << "  " << key << ": " << v << "\n";
        };

    auto lineD = [&](const char* label, double v){
        if (isSetD(v)) md << "- " << label << ": " << v << "\n";
        };
    auto lineI = [&](const char* label, long long v){
        if (isSetI(v)) md << "- " << label << ": " << v << "\n";
        };
    auto lineS = [&](const char* label, const std::string& s){
        if (!s.empty()) md << "- " << label << ": " << s << "\n";
        };

    // --- YAML Front-Matter
    md << "---\n";
    md << "generated_at_local: \"" << iso8601_local_now() << "\"\n";
    md << "generated_at_utc: \""  << iso8601_utc_now()   << "\"\n";
    md << "mode: " << (mode_full ? "Full" : mode_light ? "Light" : "Off") << "\n";
    md << "sample_n: " << m_plugin->getSettings().measure_sample << "\n";
    md << "frames_collected: " << syn.nFrames << "\n";
    md << "timeline_blocks: "  << syn.timelineBlocks << "\n";
    md << "metrics:\n";
    // Achtung: Keys nur bei gesetzten Werten
    yamlD("acg_fps",             syn.fps_avg);
    yamlD("avg_frame_time_ms",   syn.avg_frame_time_ms);
    yamlD("avg_cpu_main_ms",     syn.avg_cpu_main_ms);
    yamlD("avg_gpu_time_ms",     syn.avg_gpu_time_ms);
    yamlD("avg_wait_ms",         syn.avg_wait_ms);
    yamlD("p50_ms",              syn.p50);
    yamlD("p95_ms",              syn.p95);
    yamlD("p99_ms",              syn.p99);
    yamlD("jitter_p99_minus_p50_ms", syn.jitter_J);
    yamlI("stutter_count_gt_2x_median", syn.stutter_count_2xmedian);
    yamlI("stutter_count_gt_33ms",      syn.stutter_count_33ms);
    yamlD("avg_rpts_points_per_ms", syn.avg_rpts_points_per_ms);
    yamlD("avg_coverage_c1",      syn.avg_covC1);
    yamlD("avg_screen_density_d", syn.avg_covD);
    md << "---\n\n";

    // --- Body
    md << "# Lamure Measurement Report\n\n";
    if (wrote_frames)       md << "- Frames CSV: `"   << std::filesystem::path(frames_path).filename().string()   << "`\n";
    if (wrote_timeline)     md << "- Timeline CSV: `" << std::filesystem::path(timeline_path).filename().string() << "`\n";
    if (wrote_summary_json) md << "- Summary JSON: `" << std::filesystem::path(json_path).filename().string()    << "`\n";

    const auto preprocess_log_path = base_path.string() + "_preprocess_buildlog.txt";
    md << "- Preprocess Build Log: "
        << (std::filesystem::exists(preprocess_log_path)
            ? ("`" + std::filesystem::path(preprocess_log_path).filename().string() + "`")
            : "(not found)") << "\n";

    md << "\n## Run Metadata\n";
    md << "- Local time: " << iso8601_local_now() << "\n";
    md << "- UTC time:   " << iso8601_utc_now()   << "\n";
    md << "- Host:       " << hostname_string()   << "\n";
    md << "- User:       " << username_string()   << "\n";
    std::error_code cwd_ec;
    auto cwd = std::filesystem::current_path(cwd_ec);
    md << "- CWD:        " << (cwd_ec ? std::string("<unknown>") : cwd.string()) << "\n";
    std::string glVendor, glRenderer, glVersion;
    get_gl_strings(glVendor, glRenderer, glVersion);
    if (!glVendor.empty() || !glRenderer.empty() || !glVersion.empty()) {
        md << "- OpenGL Vendor:   " << (glVendor.empty() ? "n/a" : glVendor)   << "\n";
        md << "- OpenGL Renderer: " << (glRenderer.empty()? "n/a" : glRenderer)<< "\n";
        md << "- OpenGL Version:  " << (glVersion.empty() ? "n/a" : glVersion) << "\n";
    }

    md << "\n## System Snapshot\n";
    md << "- CPU threads: " << cpu_threads() << "\n";
    md << "- RAM total:   " << total_ram_mb() << " MB\n";
#ifdef _WIN32
    {
        static NvmlLoader g_nvml_once;
        double util=0, used=0, total=0;
        int tempC = -1; std::string gpuName;
        bool haveNvml = nvmlPoll(g_nvml_once, util, used, total, &tempC, &gpuName);
        if (haveNvml) {
            lineS("GPU (NVML)", gpuName);
            lineD("GPU Util (NVML) %", util);
            if (tempC >= 0) md << "- GPU Temp (NVML): " << tempC << " °C\n";
            md << "- VRAM used/total (primary): " << m_gpu_mem_used_mb_static
                << " / " << m_gpu_mem_total_mb_static << " MB\n";
            if (m_gpu_mem_total_mb_nvml_static > 0.0)
                md << "- VRAM (NVML) used/total:   " << m_gpu_mem_used_mb_nvml_static
                << " / " << m_gpu_mem_total_mb_nvml_static << " MB\n";
        } else {
            md << "- NVML: not available\n";
        }
    }
#else
    md << "- VRAM used/total (primary): " << m_gpu_mem_used_mb_static
        << " / " << m_gpu_mem_total_mb_static << " MB\n";
    if (m_gpu_mem_total_mb_gl_static > 0.0)
        md << "- VRAM (GL_NVX) used/total:  " << m_gpu_mem_used_mb_gl_static
        << " / " << m_gpu_mem_total_mb_gl_static << " MB\n";
#endif

    md << "\n## Summary (inline)\n";
    md << "- Frames collected: " << syn.nFrames << "\n";
    md << "- Timeline blocks: "  << syn.timelineBlocks << "\n";
    md << "- Mode: "             << (mode_full ? "Full" : mode_light ? "Light" : "Off") << "\n";
    md << "- Sample N: "         << m_plugin->getSettings().measure_sample << "\n";

    lineD("Avg Frame Time (ms)", syn.avg_frame_time_ms);
    lineD("Avg CPU Main (ms)",   syn.avg_cpu_main_ms);
    lineD("Avg GPU Time (ms)",   syn.avg_gpu_time_ms);
    lineD("Avg Wait (ms)",       syn.avg_wait_ms);
    lineD("Avg CPU Busy (%)",    syn.avg_cpu_busy_pct);
    lineD("Avg GPU Busy (%)",    syn.avg_gpu_busy_pct);
    lineD("Avg Wait Fraction (%)", syn.avg_wait_frac_pct);

    md << "\n## FPS & Primitive/Frame (derived)\n";
    lineD("Avg FPS", syn.fps_avg);
    if (isSetD(syn.fps_p50) && isSetD(syn.fps_p95) && isSetD(syn.fps_p99)) {
        md << "- FPS p50/p95/p99: " << syn.fps_p50 << " / " << syn.fps_p95 << " / " << syn.fps_p99 << "\n";
    } else {
        lineD("FPS p50", syn.fps_p50);
        lineD("FPS p95", syn.fps_p95);
        lineD("FPS p99", syn.fps_p99);
    }
    lineD("Ø Primitive pro Frame", syn.prims_per_frame_avg);
    if (isSetD(syn.prims_per_frame_p50) && isSetD(syn.prims_per_frame_p95) && isSetD(syn.prims_per_frame_p99)) {
        md << "- Primitive/Frame p50/p95/p99: "
            << syn.prims_per_frame_p50 << " / "
            << syn.prims_per_frame_p95 << " / "
            << syn.prims_per_frame_p99 << "\n";
    } else {
        lineD("Primitive/Frame p50", syn.prims_per_frame_p50);
        lineD("Primitive/Frame p95", syn.prims_per_frame_p95);
        lineD("Primitive/Frame p99", syn.prims_per_frame_p99);
    }

    md << "\n## Frametime-Perzentile & Jitter\n";
    if (isSetD(syn.p50) && isSetD(syn.p95) && isSetD(syn.p99)) {
        md << "- p50/p95/p99 (ms): " << syn.p50 << " / " << syn.p95 << " / " << syn.p99 << "\n";
    } else {
        lineD("p50 (ms)", syn.p50);
        lineD("p95 (ms)", syn.p95);
        lineD("p99 (ms)", syn.p99);
    }
    lineD("Jitter J (p99 - p50) (ms)", syn.jitter_J);
    lineI("Stutter >2×Median (Count)", syn.stutter_count_2xmedian);
    lineI("Stutter >33.33 ms (Count)", syn.stutter_count_33ms);

    md << "\n## Durchsatz/Effizienz\n";
    lineD("Avg Rpts (points/ms)", syn.avg_rpts_points_per_ms);
    lineD("Avg avg_area_px_per_prim", syn.avg_area_px_per_prim);

    md << "\n## Coverage (aus Estimates)\n";
    lineD("Avg Coverage C1",      syn.avg_covC1);
    lineD("Avg Screen Density D", syn.avg_covD);

    md << "\n## Renderer Estimates (avg)\n";
    lineD("est_screen_px",   syn.est_screen_px);
    lineD("est_sum_area_px", syn.est_sum_area_px);
    lineD("est_density",     syn.est_density);
    lineD("est_coverage",    syn.est_coverage);
    lineD("est_coverage_px", syn.est_coverage_px);
    lineD("est_overdraw",    syn.est_overdraw);

    md << "\n## Renderer-Zeitmarken (Durchschnitt pro Frame)\n";
    lineD("draw_impl_ms",     syn.mark_draw_impl_ms);
    lineD("pass1_ms",         syn.mark_pass1_ms);
    lineD("pass2_ms",         syn.mark_pass2_ms);
    lineD("pass3_ms",         syn.mark_pass3_ms);
    lineD("dispatch_ms",      syn.mark_dispatch_ms);
    lineD("context_bind_ms",  syn.mark_context_bind_ms);
    lineD("estimates_ms",     syn.mark_estimates_ms);
    lineD("singlepass_ms",    syn.mark_singlepass_ms);

    md << "## Boundness Verteilung\n";
    // Diese Zähler sind nie -1 (sie werden aus Strings hochgezählt)
    md << "- GPU-bound: "       << syn.cnt_gpu    << "\n";
    md << "- CPU-bound: "       << syn.cnt_cpu    << "\n";
    md << "- Wait/Sync-bound: " << syn.cnt_wait   << "\n";
    md << "- mixed: "           << syn.cnt_mixed  << "\n";
    md << "- unknown: "         << syn.cnt_unknown<< "\n";

    if (auto* pol = lamure::ren::policy::get_instance()) {
        md << "\n## Policy Budgets\n";
        md << "- max_upload_budget_in_mb: "  << pol->max_upload_budget_in_mb()  << "\n";
        md << "- render_budget_in_mb: "      << pol->render_budget_in_mb()      << "\n";
        md << "- out_of_core_budget_in_mb: " << pol->out_of_core_budget_in_mb() << "\n";
        md << "- size_of_provenance: "       << pol->size_of_provenance()       << "\n";
        const bool provenance_enabled = (pol->size_of_provenance() > 0);
        md << "- provenance_enabled: "       << (provenance_enabled ? "true" : "false") << "\n";
    }

    md << "\n## GPU Memory (static)\n";
    md << "- Primary used/total (MB): " << m_gpu_mem_used_mb_static
        << " / " << m_gpu_mem_total_mb_static << "\n";
    md << "- NVML used/total (MB):   " << m_gpu_mem_used_mb_nvml_static
        << " / " << m_gpu_mem_total_mb_nvml_static << "\n";
    md << "- GL_NVX used/total (MB): " << m_gpu_mem_used_mb_gl_static
        << " / " << m_gpu_mem_total_mb_gl_static << "\n";

    writeLamureConfigMarkdown(md);
    appendPreprocessBuildLogsMarkdown(md);

    md.flush();
    return true;
}


bool LamureMeasurement::writeFramesCSV(
    const std::filesystem::path& frames_path,
    bool mode_full,
    bool mode_light,
    const std::unordered_map<unsigned,double>& gpuMsByFrame)
{
    if (m_stats.empty()) return false;

    std::ofstream out;
    if (!openCsv(out, frames_path)) return false;
    out << std::fixed << std::setprecision(6);

    // --- Helper ---
    auto is_set_num = [](double v) -> bool {
        return std::isfinite(v) && v >= 0.0;
        };
    auto emitD = [&](double v) {
        if (is_set_num(v)) out << v;
        };
    auto emitDAnySign = [&](double v) {
        if (std::isfinite(v)) out << v;
        };
    auto emitI = [&](long long v) {
        if (v >= 0) out << v;
        };
    auto emitS = [&](const std::string& s) {
        out << s;
        };
    auto emitSep = [&](){ out << ';'; };

    const bool lite = (!mode_full && mode_light);

    // --- Header ---
    if (lite) {
        out <<
            "frame_number;frame_time_ms;rendering_traversals_ms;cpu_main_ms;gpu_time_ms;"
            "rendered_primitives;rendered_nodes;est_density;est_coverage;est_overdraw;segment_index\n";
    } else {
        out <<
            "frame_number;frame_rate;frame_time_ms;cpu_known_sum_ms;residual_time_ms;"
            "rendering_traversals_ms;cpu_update_ms;cpu_cull_ms;cpu_draw_ms;gpu_time_ms;"
            "gpu_clock;gpu_mem_clock;gpu_util;gpu_pci;sync_time_ms;swap_time_ms;finish_ms;"
            "isect_ms;plugin_ms;opencover_ms;cpu_main_ms;cpu_busy_pct_proxy;gpu_busy_pct_proxy;"
            "wait_ms;wait_frac_pct;boundness;rendered_primitives;rendered_nodes;rendered_bounding_boxes;"
            "est_screen_px;est_sum_area_px;est_density;est_coverage;est_coverage_px;est_overdraw;"
            "est_density_raw;est_coverage_raw;est_coverage_px_raw;est_overdraw_raw;"
            "pos_x;pos_y;pos_z;quat_x;quat_y;quat_z;quat_w;backoff_cull;backoff_draw;backoff_gpu;"
            "segment_index;rpts_points_per_ms;avg_area_px_per_prim;k_orient_used;"
            "mark_draw_impl_ms;mark_pass1_ms;mark_pass2_ms;mark_pass3_ms;mark_singlepass_ms;"
            "mark_dispatch_ms;mark_context_bind_ms;mark_estimates_ms\n";
    }

    // Wir schreiben **nur** Frames, für die echte FrameStats existieren
    std::map<unsigned, FrameStats> statsByFrame;
    for (const auto& ms : m_stats) statsByFrame.emplace(ms.frame_number, ms);

    // Settings einmal bestimmen (für k_orient_used)
    float k_orient_used = 0.70f;
    const auto& st = m_plugin->getSettings();
    if (st.point) k_orient_used = 1.0f;
    else if (st.surfel || st.splatting) k_orient_used = 0.70f;

    for (const auto& kv : statsByFrame) {
        const unsigned fid = kv.first;
        FrameStats s = kv.second;

        // GPU-Zeit aus Timeline (falls vorhanden)
        const double gms = (gpuMsByFrame.count(fid) ? gpuMsByFrame.at(fid)
            : std::numeric_limits<double>::quiet_NaN());

        if (lite) {
            emitI(static_cast<long long>(s.frame_number)); emitSep();
            emitD(s.frame_duration_ms); emitSep();
            emitD(s.rendering_traversals_ms); emitSep();
            emitD(s.cpu_main_ms); emitSep();
            emitD(s.gpu_time_ms); emitSep();
            emitD( (s.rendered_primitives > 0) ? double(s.rendered_primitives) : -1.0 ); emitSep();
            emitD( (s.rendered_nodes      > 0) ? double(s.rendered_nodes)      : -1.0 ); emitSep();
            emitD(s.est_density); emitSep();
            emitD(s.est_coverage); emitSep();
            emitD(s.est_overdraw); emitSep();
            emitI(s.segment_index);
            out << "\n";
            continue;
        }

        // --- Full-Mode: cpu_known/residual nur wenn Inputs gesetzt
        double cpu_known = 0.0;
        bool have_any_cpu_known = false;
        auto try_add = [&](double v){
            if (is_set_num(v)) { cpu_known += v; have_any_cpu_known = true; }
            };
        try_add(s.cpu_update_ms);
        try_add(s.cpu_cull_ms);
        try_add(s.cpu_draw_ms);
        try_add(s.sync_time_ms);
        try_add(s.swap_time_ms);
        try_add(s.finish_ms);
        try_add(s.isect_ms);
        try_add(s.plugin_ms);
        try_add(s.opencover_ms);

        double residual = std::numeric_limits<double>::quiet_NaN();
        if (is_set_num(s.frame_duration_ms) && have_any_cpu_known)
            residual = std::max(0.0, s.frame_duration_ms - cpu_known);

        double rpts_points_per_ms = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(gms) && gms > 1e-6 && s.rendered_primitives > 0)
            rpts_points_per_ms = double(s.rendered_primitives) / gms;

        // --- Zeile schreiben ---
        emitI(static_cast<long long>(s.frame_number)); emitSep();
        emitD(s.frame_rate); emitSep();
        emitD(s.frame_duration_ms); emitSep();
        if (have_any_cpu_known) { emitD(cpu_known);} emitSep();
        emitD(residual); emitSep();
        emitD(s.rendering_traversals_ms); emitSep();
        emitD(s.cpu_update_ms); emitSep();
        emitD(s.cpu_cull_ms);   emitSep();
        emitD(s.cpu_draw_ms);   emitSep();
        emitD(s.gpu_time_ms);   emitSep();
        emitD(s.gpu_clock);     emitSep();
        emitD(s.gpu_mem_clock); emitSep();
        emitD(s.gpu_util);      emitSep();
        emitD(s.gpu_pci);       emitSep();
        emitD(s.sync_time_ms);  emitSep();
        emitD(s.swap_time_ms);  emitSep();
        emitD(s.finish_ms);     emitSep();
        emitD(s.isect_ms);      emitSep();
        emitD(s.plugin_ms);     emitSep();
        emitD(s.opencover_ms);  emitSep();
        emitD(s.cpu_main_ms);       emitSep();
        emitD(s.cpu_busy_pct_proxy);emitSep();
        emitD(s.gpu_busy_pct_proxy);emitSep();
        emitD(s.wait_ms);           emitSep();
        emitD(s.wait_frac_pct);     emitSep();
        emitS(s.boundness);         emitSep();
        emitD( (s.rendered_primitives > 0) ? double(s.rendered_primitives) : -1.0 ); emitSep();
        emitD( (s.rendered_nodes      > 0) ? double(s.rendered_nodes)      : -1.0 ); emitSep();
        emitD( (s.rendered_bounding_boxes > 0) ? double(s.rendered_bounding_boxes) : -1.0 ); emitSep();
        emitD(s.est_screen_px);   emitSep();
        emitD(s.est_sum_area_px); emitSep();
        emitD(s.est_density);     emitSep();
        emitD(s.est_coverage);    emitSep();
        emitD(s.est_coverage_px); emitSep();
        emitD(s.est_overdraw);    emitSep();
        emitD(s.est_density_raw);     emitSep();
        emitD(s.est_coverage_raw);    emitSep();
        emitD(s.est_coverage_px_raw); emitSep();
        emitD(s.est_overdraw_raw);    emitSep();
        emitDAnySign(s.position.x()); emitSep();
        emitDAnySign(s.position.y()); emitSep();
        emitDAnySign(s.position.z()); emitSep();
        emitDAnySign(s.orientation.x()); emitSep();
        emitDAnySign(s.orientation.y()); emitSep();
        emitDAnySign(s.orientation.z()); emitSep();
        emitDAnySign(s.orientation.w()); emitSep();
        emitI(s.backoff_cull); emitSep();
        emitI(s.backoff_draw); emitSep();
        emitI(s.backoff_gpu);  emitSep();
        emitI(s.segment_index); emitSep();
        emitD(rpts_points_per_ms); emitSep();
        emitD(s.avg_area_px_per_prim); emitSep();
        emitD(k_orient_used); emitSep();
        emitD(s.mark_draw_impl_ms);    emitSep();
        emitD(s.mark_pass1_ms);        emitSep();
        emitD(s.mark_pass2_ms);        emitSep();
        emitD(s.mark_pass3_ms);        emitSep();
        emitD(s.mark_singlepass_ms);   emitSep();
        emitD(s.mark_dispatch_ms);     emitSep();
        emitD(s.mark_context_bind_ms); emitSep();
        emitD(s.mark_estimates_ms);
        out << "\n";
    }

    out.flush();
    return true;
}



bool LamureMeasurement::writeSummaryJSON(const std::filesystem::path& json_path,
    const Synthesis& syn, bool mode_full, bool mode_light, bool hasTimeline)
{
    std::ofstream js(json_path, std::ios::out | std::ios::trunc);
    if (!js) return false;

    js << std::fixed << std::setprecision(6);

    auto is_set_num = [](double v){ return std::isfinite(v) && v >= 0.0; };
    auto is_set_i   = [](long long v){ return v >= 0; };

    // kleiner Writer für ein Objekt: schreibt Kommas korrekt und lässt "unset" weg
    struct ObjWriter {
        std::ostream& os;
        bool first = true;
        void add(const char* k, double v, bool set) {
            if (!set) return;
            if (!first) os << ",\n";
            first = false;
            os << "    \"" << k << "\": " << v;
        }
        void addS(const char* k, const std::string& v, bool set=true) {
            if (!set) return;
            if (!first) os << ",\n";
            first = false;
            os << "    \"" << k << "\": \"" << v << "\"";
        }
        void addI(const char* k, long long v, bool set) {
            if (!set) return;
            if (!first) os << ",\n";
            first = false;
            os << "    \"" << k << "\": " << v;
        }
    };

    js << "{\n";

    // Top-level simple Felder
    {
        ObjWriter top{js};
        top.addS("generated_at_local", iso8601_local_now());
        top.addS("generated_at_utc",   iso8601_utc_now());
        top.addS("mode", (mode_full ? "Full" : mode_light ? "Light" : "Off"));
        top.addI("sample_n", m_plugin->getSettings().measure_sample, true);
        top.addI("frames_collected", syn.nFrames, is_set_i(syn.nFrames));
        top.addI("timeline_blocks",  hasTimeline ? syn.timelineBlocks : 0, true);
        js << "\n";
    }

    // metrics
    js << "  ,\"metrics\": {\n";
    {
        ObjWriter m{js};
        m.add("avg_frame_time_ms",    syn.avg_frame_time_ms, is_set_num(syn.avg_frame_time_ms));
        m.add("avg_cpu_main_ms",      syn.avg_cpu_main_ms,   is_set_num(syn.avg_cpu_main_ms));
        m.add("avg_gpu_time_ms",      syn.avg_gpu_time_ms,   is_set_num(syn.avg_gpu_time_ms));
        m.add("avg_wait_ms",          syn.avg_wait_ms,       is_set_num(syn.avg_wait_ms));
        m.add("p50_ms",               syn.p50,               is_set_num(syn.p50));
        m.add("p95_ms",               syn.p95,               is_set_num(syn.p95));
        m.add("p99_ms",               syn.p99,               is_set_num(syn.p99));
        m.add("jitter_p99_minus_p50_ms", syn.jitter_J,       is_set_num(syn.jitter_J));
        m.addI("stutter_count_gt_2x_median", syn.stutter_count_2xmedian, is_set_i(syn.stutter_count_2xmedian));
        m.addI("stutter_count_gt_33ms",      syn.stutter_count_33ms,     is_set_i(syn.stutter_count_33ms));
        m.add("avg_rpts_points_per_ms", syn.avg_rpts_points_per_ms, is_set_num(syn.avg_rpts_points_per_ms));
        m.add("avg_coverage_c1",       syn.avg_covC1, is_set_num(syn.avg_covC1));
        m.add("avg_screen_density_d",  syn.avg_covD,  is_set_num(syn.avg_covD));
        js << "\n";
    }
    js << "  }\n";

    // derived  (fix: kein führendes Komma vor dem Key; Komma kommt nach dem metrics-Objekt)
    js << "  ,\"derived\": {\n";
    {
        ObjWriter d{js};
        d.add("avg_fps",              syn.fps_avg,            is_set_num(syn.fps_avg));
        d.add("fps_p50",              syn.fps_p50,            is_set_num(syn.fps_p50));
        d.add("fps_p95",              syn.fps_p95,            is_set_num(syn.fps_p95));
        d.add("fps_p99",              syn.fps_p99,            is_set_num(syn.fps_p99));
        d.add("prims_per_frame_avg",  syn.prims_per_frame_avg,is_set_num(syn.prims_per_frame_avg));
        d.add("prims_per_frame_p50",  syn.prims_per_frame_p50,is_set_num(syn.prims_per_frame_p50));
        d.add("prims_per_frame_p95",  syn.prims_per_frame_p95,is_set_num(syn.prims_per_frame_p95));
        d.add("prims_per_frame_p99",  syn.prims_per_frame_p99,is_set_num(syn.prims_per_frame_p99));
        js << "\n";
    }
    js << "  }\n";

    // gpu_memory_static (fix: Komma vor diesem Objekt)
    js << "  ,\"gpu_memory_static\": {\n";
    {
        ObjWriter g{js};
        g.add("primary_used_mb",  m_gpu_mem_used_mb_static,       is_set_num(m_gpu_mem_used_mb_static));
        g.add("primary_total_mb", m_gpu_mem_total_mb_static,      is_set_num(m_gpu_mem_total_mb_static));
        g.add("nvml_used_mb",     m_gpu_mem_used_mb_nvml_static,  is_set_num(m_gpu_mem_used_mb_nvml_static));
        g.add("nvml_total_mb",    m_gpu_mem_total_mb_nvml_static, is_set_num(m_gpu_mem_total_mb_nvml_static));
        g.add("glnvx_used_mb",    m_gpu_mem_used_mb_gl_static,    is_set_num(m_gpu_mem_used_mb_gl_static));
        g.add("glnvx_total_mb",   m_gpu_mem_total_mb_gl_static,   is_set_num(m_gpu_mem_total_mb_gl_static));
        js << "\n";
    }
    js << "}\n"; // close object gpu_memory_static

    js << "}\n"; // close root
    js.flush();
    return true;
}




void LamureMeasurement::writeTimelineCSV(const std::string& path)
{
    if (m_timeline.empty()) {
        std::cout << "[Measurement] Timeline disabled or empty, skipping CSV.\n";
        return;
    }

    try {
        std::filesystem::path p(path);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());
    } catch (...) {}

    cacheStaticGpuInfo();

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "[Measurement] Failed to open timeline CSV: " << path << "\n";
        return;
    }

    out.setf(std::ios::fixed);
    out << std::setprecision(3);

    std::vector<TimeBlock> blocks = m_timeline;
    std::sort(blocks.begin(), blocks.end(), [](const TimeBlock& a, const TimeBlock& b){
        if (a.src_frame != b.src_frame) return a.src_frame < b.src_frame;
        if (a.begin_ms  != b.begin_ms)  return a.begin_ms  < b.begin_ms;
        if (a.camIndex  != b.camIndex)  return a.camIndex  < b.camIndex;
        return a.name < b.name;
        });

    out << "frame;src_frame;cam;scope;name;begin_ms;end_ms;taken_ms;used_offset\n";
    for (const auto& b : blocks) {
        out << b.frame << ';'
            << b.src_frame << ';'
            << b.camIndex << ';'
            << csvQuote(b.scope) << ';'
            << csvQuote(b.name)  << ';'
            << b.begin_ms << ';'
            << b.end_ms   << ';'
            << b.taken_ms << ';'
            << b.used_offset
            << '\n';
    }

    out.flush();
    std::cout << "[Measurement] Timeline CSV written: " << path << " (" << blocks.size() << " rows)\n";
}


void LamureMeasurement::writeLamureConfigMarkdown(std::ostream& md) {
    md << "\n## Lamure Config (compile-time)\n";

#ifdef LAMURE_ENABLE_INFO
    md << "- LAMURE_ENABLE_INFO: ON\n";
#else
    md << "- LAMURE_ENABLE_INFO: OFF\n";
#endif

#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
    md << "- LAMURE_RENDERING_USE_SPLIT_SCREEN: ON\n";
#else
    md << "- LAMURE_RENDERING_USE_SPLIT_SCREEN: OFF\n";
#endif

#ifdef LAMURE_RENDERING_ENABLE_MULTI_VIEW_TEST
    md << "- LAMURE_RENDERING_ENABLE_MULTI_VIEW_TEST: ON\n";
#else
    md << "- LAMURE_RENDERING_ENABLE_MULTI_VIEW_TEST: OFF\n";
#endif

#ifdef LAMURE_RENDERING_ENABLE_LAZY_MODELS_TEST
    md << "- LAMURE_RENDERING_ENABLE_LAZY_MODELS_TEST: ON\n";
#else
    md << "- LAMURE_RENDERING_ENABLE_LAZY_MODELS_TEST: OFF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_MEASURE_SYSTEM_PERFORMANCE
    md << "- LAMURE_CUT_UPDATE_ENABLE_MEASURE_SYSTEM_PERFORMANCE: ON\n";
#else
    md << "- LAMURE_CUT_UPDATE_ENABLE_MEASURE_SYSTEM_PERFORMANCE: OFF\n";
#endif

#ifdef LAMURE_DEFAULT_COLOR_R
    md << "- LAMURE_DEFAULT_COLOR_R: " << LAMURE_DEFAULT_COLOR_R << "\n";
#else
    md << "- LAMURE_DEFAULT_COLOR_R: UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_COLOR_G
    md << "- LAMURE_DEFAULT_COLOR_G: " << LAMURE_DEFAULT_COLOR_G << "\n";
#else
    md << "- LAMURE_DEFAULT_COLOR_G: UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_COLOR_B
    md << "- LAMURE_DEFAULT_COLOR_B: " << LAMURE_DEFAULT_COLOR_B << "\n";
#else
    md << "- LAMURE_DEFAULT_COLOR_B: UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_MODEL_TIMEOUT
    md << "- LAMURE_CUT_UPDATE_ENABLE_MODEL_TIMEOUT: ON\n";
#else
    md << "- LAMURE_CUT_UPDATE_ENABLE_MODEL_TIMEOUT: OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_MAX_MODEL_TIMEOUT
    md << "- LAMURE_CUT_UPDATE_MAX_MODEL_TIMEOUT: " << LAMURE_CUT_UPDATE_MAX_MODEL_TIMEOUT << "\n";
#else
    md << "- LAMURE_CUT_UPDATE_MAX_MODEL_TIMEOUT: UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_CUT_UPDATE_EXPERIMENTAL_MODE
    md << "- LAMURE_CUT_UPDATE_ENABLE_CUT_UPDATE_EXPERIMENTAL_MODE: ON\n";
#else
    md << "- LAMURE_CUT_UPDATE_ENABLE_CUT_UPDATE_EXPERIMENTAL_MODE: OFF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_NUM_CUT_UPDATE_THREADS
    md << "- LAMURE_CUT_UPDATE_NUM_CUT_UPDATE_THREADS: " << LAMURE_CUT_UPDATE_NUM_CUT_UPDATE_THREADS << "\n";
#else
    md << "- LAMURE_CUT_UPDATE_NUM_CUT_UPDATE_THREADS: UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_SHOW_OOC_CACHE_USAGE
    md << "- LAMURE_CUT_UPDATE_ENABLE_SHOW_OOC_CACHE_USAGE: ON\n";
#else
    md << "- LAMURE_CUT_UPDATE_ENABLE_SHOW_OOC_CACHE_USAGE: OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_ENABLE_SHOW_GPU_CACHE_USAGE
    md << "- LAMURE_CUT_UPDATE_ENABLE_SHOW_GPU_CACHE_USAGE: ON\n";
#else
    md << "- LAMURE_CUT_UPDATE_ENABLE_SHOW_GPU_CACHE_USAGE: OFF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_REPEAT_MODE
    md << "- LAMURE_CUT_UPDATE_ENABLE_REPEAT_MODE: ON\n";
#else
    md << "- LAMURE_CUT_UPDATE_ENABLE_REPEAT_MODE: OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_MAX_NUM_UPDATES_PER_FRAME
    md << "- LAMURE_CUT_UPDATE_MAX_NUM_UPDATES_PER_FRAME: " << LAMURE_CUT_UPDATE_MAX_NUM_UPDATES_PER_FRAME << "\n";
#else
    md << "- LAMURE_CUT_UPDATE_MAX_NUM_UPDATES_PER_FRAME: UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_SPLIT_AGAIN_MODE
    md << "- LAMURE_CUT_UPDATE_ENABLE_SPLIT_AGAIN_MODE: ON\n";
#else
    md << "- LAMURE_CUT_UPDATE_ENABLE_SPLIT_AGAIN_MODE: OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_MUST_COLLAPSE_OUTSIDE_FRUSTUM
    md << "- LAMURE_CUT_UPDATE_MUST_COLLAPSE_OUTSIDE_FRUSTUM: ON\n";
#else
    md << "- LAMURE_CUT_UPDATE_MUST_COLLAPSE_OUTSIDE_FRUSTUM: OFF\n";
#endif

#ifdef LAMURE_DATABASE_SAFE_MODE
    md << "- LAMURE_DATABASE_SAFE_MODE: ON\n";
#else
    md << "- LAMURE_DATABASE_SAFE_MODE: OFF\n";
#endif

#ifdef LAMURE_DEFAULT_IMPORTANCE
    md << "- LAMURE_DEFAULT_IMPORTANCE: " << LAMURE_DEFAULT_IMPORTANCE << "\n";
#else
    md << "- LAMURE_DEFAULT_IMPORTANCE: UNDEF\n";
#endif
#ifdef LAMURE_MIN_IMPORTANCE
    md << "- LAMURE_MIN_IMPORTANCE: " << LAMURE_MIN_IMPORTANCE << "\n";
#else
    md << "- LAMURE_MIN_IMPORTANCE: UNDEF\n";
#endif
#ifdef LAMURE_MAX_IMPORTANCE
    md << "- LAMURE_MAX_IMPORTANCE: " << LAMURE_MAX_IMPORTANCE << "\n";
#else
    md << "- LAMURE_MAX_IMPORTANCE: UNDEF\n";
#endif

#ifdef LAMURE_DEFAULT_THRESHOLD
    md << "- LAMURE_DEFAULT_THRESHOLD: " << LAMURE_DEFAULT_THRESHOLD << "\n";
#else
    md << "- LAMURE_DEFAULT_THRESHOLD: UNDEF\n";
#endif
#ifdef LAMURE_MIN_THRESHOLD
    md << "- LAMURE_MIN_THRESHOLD: " << LAMURE_MIN_THRESHOLD << "\n";
#else
    md << "- LAMURE_MIN_THRESHOLD: UNDEF\n";
#endif
#ifdef LAMURE_MAX_THRESHOLD
    md << "- LAMURE_MAX_THRESHOLD: " << LAMURE_MAX_THRESHOLD << "\n";
#else
    md << "- LAMURE_MAX_THRESHOLD: UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_PREFETCHING
    md << "- LAMURE_CUT_UPDATE_ENABLE_PREFETCHING: ON\n";
#else
    md << "- LAMURE_CUT_UPDATE_ENABLE_PREFETCHING: OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_PREFETCH_FACTOR
    md << "- LAMURE_CUT_UPDATE_PREFETCH_FACTOR: " << LAMURE_CUT_UPDATE_PREFETCH_FACTOR << "\n";
#else
    md << "- LAMURE_CUT_UPDATE_PREFETCH_FACTOR: UNDEF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_PREFETCH_BUDGET
    md << "- LAMURE_CUT_UPDATE_PREFETCH_BUDGET: " << LAMURE_CUT_UPDATE_PREFETCH_BUDGET << "\n";
#else
    md << "- LAMURE_CUT_UPDATE_PREFETCH_BUDGET: UNDEF\n";
#endif

#ifdef LAMURE_MIN_UPLOAD_BUDGET
    md << "- LAMURE_MIN_UPLOAD_BUDGET: " << LAMURE_MIN_UPLOAD_BUDGET << "\n";
#else
    md << "- LAMURE_MIN_UPLOAD_BUDGET: UNDEF\n";
#endif
#ifdef LAMURE_MIN_VIDEO_MEMORY_BUDGET
    md << "- LAMURE_MIN_VIDEO_MEMORY_BUDGET: " << LAMURE_MIN_VIDEO_MEMORY_BUDGET << "\n";
#else
    md << "- LAMURE_MIN_VIDEO_MEMORY_BUDGET: UNDEF\n";
#endif
#ifdef LAMURE_MIN_MAIN_MEMORY_BUDGET
    md << "- LAMURE_MIN_MAIN_MEMORY_BUDGET: " << LAMURE_MIN_MAIN_MEMORY_BUDGET << "\n";
#else
    md << "- LAMURE_MIN_MAIN_MEMORY_BUDGET: UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_UPLOAD_BUDGET
    md << "- LAMURE_DEFAULT_UPLOAD_BUDGET: " << LAMURE_DEFAULT_UPLOAD_BUDGET << "\n";
#else
    md << "- LAMURE_DEFAULT_UPLOAD_BUDGET: UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET
    md << "- LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET: " << LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET << "\n";
#else
    md << "- LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET: UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_MAIN_MEMORY_BUDGET
    md << "- LAMURE_DEFAULT_MAIN_MEMORY_BUDGET: " << LAMURE_DEFAULT_MAIN_MEMORY_BUDGET << "\n";
#else
    md << "- LAMURE_DEFAULT_MAIN_MEMORY_BUDGET: UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_SIZE_OF_PROVENANCE
    md << "- LAMURE_DEFAULT_SIZE_OF_PROVENANCE: " << LAMURE_DEFAULT_SIZE_OF_PROVENANCE << "\n";
#else
    md << "- LAMURE_DEFAULT_SIZE_OF_PROVENANCE: UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_NUM_LOADING_THREADS
    md << "- LAMURE_CUT_UPDATE_NUM_LOADING_THREADS: " << LAMURE_CUT_UPDATE_NUM_LOADING_THREADS << "\n";
#else
    md << "- LAMURE_CUT_UPDATE_NUM_LOADING_THREADS: UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_CACHE_MAINTENANCE
    md << "- LAMURE_CUT_UPDATE_ENABLE_CACHE_MAINTENANCE: ON\n";
#else
    md << "- LAMURE_CUT_UPDATE_ENABLE_CACHE_MAINTENANCE: OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_CACHE_MAINTENANCE_COUNTER
    md << "- LAMURE_CUT_UPDATE_CACHE_MAINTENANCE_COUNTER: " << LAMURE_CUT_UPDATE_CACHE_MAINTENANCE_COUNTER << "\n";
#else
    md << "- LAMURE_CUT_UPDATE_CACHE_MAINTENANCE_COUNTER: UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE
    md << "- LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE: " << LM_STR(LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE) << "\n";
#else
    md << "- LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE: UNDEF\n";
#endif

#ifdef LAMURE_WYSIWYG_SPLAT_SCALE
    md << "- LAMURE_WYSIWYG_SPLAT_SCALE: " << LAMURE_WYSIWYG_SPLAT_SCALE << "\n";
#else
    md << "- LAMURE_WYSIWYG_SPLAT_SCALE: UNDEF\n";
#endif

}

void LamureMeasurement::writeLamureConfigCsv(std::ostream& csv)
{
#ifdef LAMURE_ENABLE_INFO
    csv << "LAMURE_ENABLE_INFO;ON\n";
#else
    csv << "LAMURE_ENABLE_INFO;OFF\n";
#endif

#ifdef LAMURE_RENDERING_USE_SPLIT_SCREEN
    csv << "LAMURE_RENDERING_USE_SPLIT_SCREEN;ON\n";
#else
    csv << "LAMURE_RENDERING_USE_SPLIT_SCREEN;OFF\n";
#endif

#ifdef LAMURE_RENDERING_ENABLE_MULTI_VIEW_TEST
    csv << "LAMURE_RENDERING_ENABLE_MULTI_VIEW_TEST;ON\n";
#else
    csv << "LAMURE_RENDERING_ENABLE_MULTI_VIEW_TEST;OFF\n";
#endif

#ifdef LAMURE_RENDERING_ENABLE_LAZY_MODELS_TEST
    csv << "LAMURE_RENDERING_ENABLE_LAZY_MODELS_TEST;ON\n";
#else
    csv << "LAMURE_RENDERING_ENABLE_LAZY_MODELS_TEST;OFF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_MEASURE_SYSTEM_PERFORMANCE
    csv << "LAMURE_CUT_UPDATE_ENABLE_MEASURE_SYSTEM_PERFORMANCE;ON\n";
#else
    csv << "LAMURE_CUT_UPDATE_ENABLE_MEASURE_SYSTEM_PERFORMANCE;OFF\n";
#endif

#ifdef LAMURE_DEFAULT_COLOR_R
    csv << "LAMURE_DEFAULT_COLOR_R;" << LAMURE_DEFAULT_COLOR_R << "\n";
#else
    csv << "LAMURE_DEFAULT_COLOR_R;UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_COLOR_G
    csv << "LAMURE_DEFAULT_COLOR_G;" << LAMURE_DEFAULT_COLOR_G << "\n";
#else
    csv << "LAMURE_DEFAULT_COLOR_G;UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_COLOR_B
    csv << "LAMURE_DEFAULT_COLOR_B;" << LAMURE_DEFAULT_COLOR_B << "\n";
#else
    csv << "LAMURE_DEFAULT_COLOR_B;UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_MODEL_TIMEOUT
    csv << "LAMURE_CUT_UPDATE_ENABLE_MODEL_TIMEOUT;ON\n";
#else
    csv << "LAMURE_CUT_UPDATE_ENABLE_MODEL_TIMEOUT;OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_MAX_MODEL_TIMEOUT
    csv << "LAMURE_CUT_UPDATE_MAX_MODEL_TIMEOUT;" << LAMURE_CUT_UPDATE_MAX_MODEL_TIMEOUT << "\n";
#else
    csv << "LAMURE_CUT_UPDATE_MAX_MODEL_TIMEOUT;UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_CUT_UPDATE_EXPERIMENTAL_MODE
    csv << "LAMURE_CUT_UPDATE_ENABLE_CUT_UPDATE_EXPERIMENTAL_MODE;ON\n";
#else
    csv << "LAMURE_CUT_UPDATE_ENABLE_CUT_UPDATE_EXPERIMENTAL_MODE;OFF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_NUM_CUT_UPDATE_THREADS
    csv << "LAMURE_CUT_UPDATE_NUM_CUT_UPDATE_THREADS;" << LAMURE_CUT_UPDATE_NUM_CUT_UPDATE_THREADS << "\n";
#else
    csv << "LAMURE_CUT_UPDATE_NUM_CUT_UPDATE_THREADS;UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_SHOW_OOC_CACHE_USAGE
    csv << "LAMURE_CUT_UPDATE_ENABLE_SHOW_OOC_CACHE_USAGE;ON\n";
#else
    csv << "LAMURE_CUT_UPDATE_ENABLE_SHOW_OOC_CACHE_USAGE;OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_ENABLE_SHOW_GPU_CACHE_USAGE
    csv << "LAMURE_CUT_UPDATE_ENABLE_SHOW_GPU_CACHE_USAGE;ON\n";
#else
    csv << "LAMURE_CUT_UPDATE_ENABLE_SHOW_GPU_CACHE_USAGE;OFF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_REPEAT_MODE
    csv << "LAMURE_CUT_UPDATE_ENABLE_REPEAT_MODE;ON\n";
#else
    csv << "LAMURE_CUT_UPDATE_ENABLE_REPEAT_MODE;OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_MAX_NUM_UPDATES_PER_FRAME
    csv << "LAMURE_CUT_UPDATE_MAX_NUM_UPDATES_PER_FRAME;" << LAMURE_CUT_UPDATE_MAX_NUM_UPDATES_PER_FRAME << "\n";
#else
    csv << "LAMURE_CUT_UPDATE_MAX_NUM_UPDATES_PER_FRAME;UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_SPLIT_AGAIN_MODE
    csv << "LAMURE_CUT_UPDATE_ENABLE_SPLIT_AGAIN_MODE;ON\n";
#else
    csv << "LAMURE_CUT_UPDATE_ENABLE_SPLIT_AGAIN_MODE;OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_MUST_COLLAPSE_OUTSIDE_FRUSTUM
    csv << "LAMURE_CUT_UPDATE_MUST_COLLAPSE_OUTSIDE_FRUSTUM;ON\n";
#else
    csv << "LAMURE_CUT_UPDATE_MUST_COLLAPSE_OUTSIDE_FRUSTUM;OFF\n";
#endif

#ifdef LAMURE_DATABASE_SAFE_MODE
    csv << "LAMURE_DATABASE_SAFE_MODE;ON\n";
#else
    csv << "LAMURE_DATABASE_SAFE_MODE;OFF\n";
#endif

#ifdef LAMURE_DEFAULT_IMPORTANCE
    csv << "LAMURE_DEFAULT_IMPORTANCE;" << LAMURE_DEFAULT_IMPORTANCE << "\n";
#else
    csv << "LAMURE_DEFAULT_IMPORTANCE;UNDEF\n";
#endif
#ifdef LAMURE_MIN_IMPORTANCE
    csv << "LAMURE_MIN_IMPORTANCE;" << LAMURE_MIN_IMPORTANCE << "\n";
#else
    csv << "LAMURE_MIN_IMPORTANCE;UNDEF\n";
#endif
#ifdef LAMURE_MAX_IMPORTANCE
    csv << "LAMURE_MAX_IMPORTANCE;" << LAMURE_MAX_IMPORTANCE << "\n";
#else
    csv << "LAMURE_MAX_IMPORTANCE;UNDEF\n";
#endif

#ifdef LAMURE_DEFAULT_THRESHOLD
    csv << "LAMURE_DEFAULT_THRESHOLD;" << LAMURE_DEFAULT_THRESHOLD << "\n";
#else
    csv << "LAMURE_DEFAULT_THRESHOLD;UNDEF\n";
#endif
#ifdef LAMURE_MIN_THRESHOLD
    csv << "LAMURE_MIN_THRESHOLD;" << LAMURE_MIN_THRESHOLD << "\n";
#else
    csv << "LAMURE_MIN_THRESHOLD;UNDEF\n";
#endif
#ifdef LAMURE_MAX_THRESHOLD
    csv << "LAMURE_MAX_THRESHOLD;" << LAMURE_MAX_THRESHOLD << "\n";
#else
    csv << "LAMURE_MAX_THRESHOLD;UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_PREFETCHING
    csv << "LAMURE_CUT_UPDATE_ENABLE_PREFETCHING;ON\n";
#else
    csv << "LAMURE_CUT_UPDATE_ENABLE_PREFETCHING;OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_PREFETCH_FACTOR
    csv << "LAMURE_CUT_UPDATE_PREFETCH_FACTOR;" << LAMURE_CUT_UPDATE_PREFETCH_FACTOR << "\n";
#else
    csv << "LAMURE_CUT_UPDATE_PREFETCH_FACTOR;UNDEF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_PREFETCH_BUDGET
    csv << "LAMURE_CUT_UPDATE_PREFETCH_BUDGET;" << LAMURE_CUT_UPDATE_PREFETCH_BUDGET << "\n";
#else
    csv << "LAMURE_CUT_UPDATE_PREFETCH_BUDGET;UNDEF\n";
#endif

#ifdef LAMURE_MIN_UPLOAD_BUDGET
    csv << "LAMURE_MIN_UPLOAD_BUDGET;" << LAMURE_MIN_UPLOAD_BUDGET << "\n";
#else
    csv << "LAMURE_MIN_UPLOAD_BUDGET;UNDEF\n";
#endif
#ifdef LAMURE_MIN_VIDEO_MEMORY_BUDGET
    csv << "LAMURE_MIN_VIDEO_MEMORY_BUDGET;" << LAMURE_MIN_VIDEO_MEMORY_BUDGET << "\n";
#else
    csv << "LAMURE_MIN_VIDEO_MEMORY_BUDGET;UNDEF\n";
#endif
#ifdef LAMURE_MIN_MAIN_MEMORY_BUDGET
    csv << "LAMURE_MIN_MAIN_MEMORY_BUDGET;" << LAMURE_MIN_MAIN_MEMORY_BUDGET << "\n";
#else
    csv << "LAMURE_MIN_MAIN_MEMORY_BUDGET;UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_UPLOAD_BUDGET
    csv << "LAMURE_DEFAULT_UPLOAD_BUDGET;" << LAMURE_DEFAULT_UPLOAD_BUDGET << "\n";
#else
    csv << "LAMURE_DEFAULT_UPLOAD_BUDGET;UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET
    csv << "LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET;" << LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET << "\n";
#else
    csv << "LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET;UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_MAIN_MEMORY_BUDGET
    csv << "LAMURE_DEFAULT_MAIN_MEMORY_BUDGET;" << LAMURE_DEFAULT_MAIN_MEMORY_BUDGET << "\n";
#else
    csv << "LAMURE_DEFAULT_MAIN_MEMORY_BUDGET;UNDEF\n";
#endif
#ifdef LAMURE_DEFAULT_SIZE_OF_PROVENANCE
    csv << "LAMURE_DEFAULT_SIZE_OF_PROVENANCE;" << LAMURE_DEFAULT_SIZE_OF_PROVENANCE << "\n";
#else
    csv << "LAMURE_DEFAULT_SIZE_OF_PROVENANCE;UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_NUM_LOADING_THREADS
    csv << "LAMURE_CUT_UPDATE_NUM_LOADING_THREADS;" << LAMURE_CUT_UPDATE_NUM_LOADING_THREADS << "\n";
#else
    csv << "LAMURE_CUT_UPDATE_NUM_LOADING_THREADS;UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_ENABLE_CACHE_MAINTENANCE
    csv << "LAMURE_CUT_UPDATE_ENABLE_CACHE_MAINTENANCE;ON\n";
#else
    csv << "LAMURE_CUT_UPDATE_ENABLE_CACHE_MAINTENANCE;OFF\n";
#endif
#ifdef LAMURE_CUT_UPDATE_CACHE_MAINTENANCE_COUNTER
    csv << "LAMURE_CUT_UPDATE_CACHE_MAINTENANCE_COUNTER;" << LAMURE_CUT_UPDATE_CACHE_MAINTENANCE_COUNTER << "\n";
#else
    csv << "LAMURE_CUT_UPDATE_CACHE_MAINTENANCE_COUNTER;UNDEF\n";
#endif

#ifdef LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE
    csv << "LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE;" << LM_STR(LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE) << "\n";
#else
    csv << "LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE;UNDEF\n";
#endif

#ifdef LAMURE_WYSIWYG_SPLAT_SCALE
    csv << "LAMURE_WYSIWYG_SPLAT_SCALE;" << LAMURE_WYSIWYG_SPLAT_SCALE << "\n";
#else
    csv << "LAMURE_WYSIWYG_SPLAT_SCALE;UNDEF\n";
#endif
}


void LamureMeasurement::printDebugStats(unsigned int num)
{
    if (num == 0) return;
    if (m_stats.empty()) {
        std::cout << "[Measurement] No stats collected to print." << std::endl;
        return;
    }

    auto print_frame = [](const FrameStats& s, int frame_num) {
        std::cout << std::fixed << std::setprecision(4)
            << "Frame #"          << std::setw(4) << s.frame_number << "\n"
            << "FrameRate: "      << std::setw(8) << s.frame_rate << "\n"
            << "FrameDuration: "  << std::setw(8) << s.frame_duration_ms << "\n"
            << "Splats: "         << std::setw(8) << s.rendered_primitives << "\n"
            << "Nodes: "          << std::setw(8) << s.rendered_nodes << "\n"
            << "Boxes: "          << std::setw(8) << s.rendered_bounding_boxes << "\n"
            << "CPU Cull: "       << std::setw(8) << s.cpu_cull_ms << "\n"
            << "CPU Draw: "       << std::setw(8) << s.cpu_draw_ms << "\n"
            << "CPU Update: "         << std::setw(8) << s.cpu_update_ms << "\n"
            << "GPU Time: "        << std::setw(8) << s.gpu_time_ms << "\n"
            << "GPU Clock: "         << std::setw(8) << s.gpu_clock << "\n"
            << "GPU Mem Clock: "      << std::setw(8) << s.gpu_mem_clock << "\n"
            << "GPU Util: "        << std::setw(8) << s.gpu_util << "\n"
            << "GPU PCI: "         << std::setw(8) << s.gpu_pci << "\n"
            << "Sync: "           << std::setw(8) << s.sync_time_ms << "\n"
            << "Swap: "           << std::setw(8) << s.swap_time_ms << "\n"
            << "Isect: "          << std::setw(8) << s.isect_ms << "\n"
            << "Plugin: "         << std::setw(8) << s.plugin_ms << "\n"
            << "OpenCov: "        << std::setw(8) << s.opencover_ms << "\n"
            << "Pos: ("           << std::setw(6) << s.position.x() 
            << ", "               << std::setw(6) << s.position.y() 
            << ", "               << std::setw(6) << s.position.z() << ")\n"
            << "Quat: ("          << std::setw(6) << s.orientation.x() 
            << ", "               << std::setw(6) << s.orientation.y() 
            << ", "               << std::setw(6) << s.orientation.z() 
            << ", "               << std::setw(6) << s.orientation.w() << ")\n"
            << std::endl;
        };
    std::cout << "--- Measurement Debug Stats ---" << std::endl;
    size_t count = m_stats.size();

    if (count <= num*2) {
        for (size_t i = 0; i < count; ++i) {
            print_frame(m_stats[i], static_cast<int>(i + 1));
        }
    } else {
        // First 3 frames
        for (size_t i = 0; i < num; ++i) {
            print_frame(m_stats[i], static_cast<int>(i + 1));
        }
        std::cout << "..." << std::endl;
        // Last 3 frames
        for (size_t i = count - num; i < count; ++i) {
            print_frame(m_stats[i], static_cast<int>(i + 1));
        }
    }
    std::cout << "-------------------------------" << std::endl;
}


void LamureMeasurement::appendPreprocessBuildLogsMarkdown(std::ostream& md)
{
    auto& S = m_plugin->getSettings();

    // Verzeichnisse aus den absoluten Model-Pfaden ableiten
    std::error_code ec;
    std::set<std::filesystem::path> modelDirs;
    for (const auto& m : S.models) {
        std::filesystem::path mp(m);
        const auto d = mp.has_parent_path() ? mp.parent_path() : mp;
        modelDirs.insert(std::filesystem::weakly_canonical(d, ec));
    }

    auto toLower = [](std::string s){
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c){ return (char)std::tolower(c); });
        return s;
        };
    const std::vector<std::string> preferred_exact = {
        "build_config.log","build_config.txt",
        "preprocess_config.log","preprocess_config.txt",
        "lamure_preprocess.log","lamure_build.log"
    };
    auto looksLikePreprocessLog = [&](const std::filesystem::directory_entry& e)->bool {
        if (!e.is_regular_file()) return false;
        const auto ext = toLower(e.path().extension().string());
        // <-- NEU: akzeptiere generell alle .log/.txt
        if (ext != ".log" && ext != ".txt") return false;
        return true;
        };

    // Gewichtung für Sortierung: zuerst exakte Matches, dann solche mit Keywords, dann der Rest alphabetisch
    auto scoreName = [&](const std::string& lower)->int {
        if (std::find(preferred_exact.begin(), preferred_exact.end(), lower) != preferred_exact.end())
            return 2; // top
        if (lower.find("preprocess") != std::string::npos ||
            lower.find("build")      != std::string::npos ||
            lower.find("lamure")     != std::string::npos)
            return 1; // gut
        return 0;     // sonstige .log/.txt (z.B. naturkundemuseum_School000.log)
        };

    md << "\n## Preprocess / Build Logs (aggregated)\n";
    if (modelDirs.empty()) {
        md << "_Keine Model-Verzeichnisse gefunden; keine Logsuche durchgeführt._\n";
        return;
    }

    md << "- Modelle gescannt (" << S.models.size() << "):\n";
    for (const auto& m : S.models) md << "  - " << m << "\n";

    size_t totalDirs = 0, totalFiles = 0;
    constexpr std::uintmax_t MAX_FILE_BYTES = 10ull * 1024ull * 1024ull; // 10 MB/File
    constexpr std::size_t    MAX_LINE_CHARS = 16 * 1024;                 // 16 KB/Zeile

    struct Cand { std::string name_lower; std::filesystem::path path; };
    auto isExact = [&](const std::string& n){
        for (const auto& ex : preferred_exact) if (n == ex) return true;
        return false;
        };

    for (const auto& dir : modelDirs) {
        ++totalDirs;

        std::filesystem::path d_abs = std::filesystem::weakly_canonical(dir, ec);
        const std::string d_show = ec ? dir.string() : d_abs.string();

        std::vector<Cand> found;
        try {
            for (auto& ent : std::filesystem::directory_iterator(d_abs))
                if (looksLikePreprocessLog(ent))
                    found.push_back({ toLower(ent.path().filename().string()), ent.path() });
        } catch (...) {
            md << "\n**Directory:** " << d_show << "\n\n";
            md << "_Warnung: Directory-Iteration fehlgeschlagen._\n";
            continue;
        }

        md << "\n**Directory:** " << d_show << "\n\n";
        if (found.empty()) {
            md << "_Keine preprocess/build-Logs gefunden._\n";
            continue;
        }

        std::stable_sort(found.begin(), found.end(),
            [&](const Cand& a, const Cand& b){
                const int sa = scoreName(a.name_lower);
                const int sb = scoreName(b.name_lower);
                if (sa != sb) return sa > sb;               // höherer Score zuerst
                return a.name_lower < b.name_lower;         // sonst alphabetisch
            });

        for (const auto& c : found) {
            ++totalFiles;
            std::error_code fsec;
            const auto sz = std::filesystem::file_size(c.path, fsec);
            const bool large = (!fsec && sz > MAX_FILE_BYTES);

            md << "<details>\n<summary><code>"
                << c.path.filename().string()
                << "</code> ("
                << (fsec ? "size: ?" : std::to_string(static_cast<long long>(sz)) + " bytes")
                << ")</summary>\n\n```text\n";

            std::ifstream in(c.path, std::ios::in | std::ios::binary);
            if (!in.is_open()) {
                md << "[Could not open file]\n```\n</details>\n";
                continue;
            }

            std::size_t to_read = large ? static_cast<std::size_t>(MAX_FILE_BYTES)
                : static_cast<std::size_t>(sz);
            std::string line; line.reserve(4096);
            std::size_t read_bytes = 0;
            while (in.good() && read_bytes < to_read) {
                if (!std::getline(in, line)) break;
                if (line.size() > MAX_LINE_CHARS) line.resize(MAX_LINE_CHARS);
                md << line << " ";
                read_bytes += line.size() + 1;
            }
            if (large) {
                md << "\n[... truncated at " << (MAX_FILE_BYTES/1024/1024)
                    << " MB to keep report compact ...]\n";
            }
            md << "```\n</details>\n";
        }
    }

    md << "\n_Scanned " << totalDirs << " directories; aggregated "
        << totalFiles << " log files._\n";
}
