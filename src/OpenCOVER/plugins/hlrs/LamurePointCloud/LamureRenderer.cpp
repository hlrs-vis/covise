#include "LamureRenderer.h"
#include "Lamure.h"
#include "LamureUtil.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRPluginSupport.h>

#include <osg/BlendFunc>
#include <osg/CullFace>
#include <osg/Depth>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osg/State>
#include <osg/GraphicsContext>
#include <osg/ClipNode>
#include <osg/ClipPlane>
#include <osg/Stats>
#include <osgViewer/Viewer>
#include <osgViewer/Renderer>
#include <osgText/Text>
#include <osg/NodeVisitor>
#include <osgUtil/CullVisitor>

#include <lamure/ren/model_database.h>
#include <lamure/ren/cut_database.h>
#include <lamure/ren/controller.h>
#include <lamure/pvs/pvs_database.h>
#include <lamure/ren/policy.h>

#include <chrono>
#include <algorithm>
#include <limits>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <thread>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <gl_state.h>
#include <config/CoviseConfig.h>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

namespace {
    constexpr uint64_t kInvalidTimingFrame = std::numeric_limits<uint64_t>::max();

    inline uint64_t frameNumberFromRenderInfo(const osg::RenderInfo& renderInfo)
    {
        const osg::State* state = renderInfo.getState();
        const osg::FrameStamp* fs = state ? state->getFrameStamp() : nullptr;
        return fs ? static_cast<uint64_t>(fs->getFrameNumber()) : kInvalidTimingFrame;
    }

    inline double elapsedMs(const std::chrono::steady_clock::time_point& t0)
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }


    inline void accumulateMs(double& dst, double v)
    {
        if (!LamureUtil::isValidValue(v)) return;
        if (!LamureUtil::isValidValue(dst)) dst = 0.0;
        dst += v;
    }

    inline double sumKnownMs(std::initializer_list<double> vals)
    {
        double sum = 0.0;
        bool any = false;
        for (double v : vals) {
            if (LamureUtil::isValidValue(v)) {
                sum += v;
                any = true;
            }
        }
        return any ? sum : -1.0;
    }

    // Prebuilt stat key triplets for hot paths.
    struct StatKeys {
        std::string taken, begin, end;
    };

    static const StatKeys kUpdateTraversalKeys{
        "Update traversal time taken",
        "Update traversal begin time",
        "Update traversal end time"
    };
    static const StatKeys kSyncKeys{
        "sync time taken",
        "sync begin time",
        "sync end time"
    };
    static const StatKeys kSwapKeys{
        "swap time taken",
        "swap begin time",
        "swap end time"
    };
    static const StatKeys kFinishKeys{
        "finish time taken",
        "finish begin time",
        "finish end time"
    };
    static const StatKeys kCullTraversalKeys{
        "Cull traversal time taken",
        "Cull traversal begin time",
        "Cull traversal end time"
    };
    static const StatKeys kDrawTraversalKeys{
        "Draw traversal time taken",
        "Draw traversal begin time",
        "Draw traversal end time"
    };
    static const StatKeys kGpuDrawKeys{
        "GPU draw time taken",
        "GPU draw begin time",
        "GPU draw end time"
    };

    inline bool queryTimeTakenMsBacksearch(
        osg::Stats* stats,
        unsigned baseFrame,
        unsigned backsearch,
        const StatKeys& keys,
        double& outMs)
    {
        outMs = -1.0;
        if (!stats) return false;

        const unsigned earliest = stats->getEarliestFrameNumber();
        const unsigned latest = stats->getLatestFrameNumber();
        if (baseFrame > latest) baseFrame = latest;

        for (unsigned off = 0; off <= backsearch; ++off) {
            if (baseFrame < off) break;
            const unsigned f = baseFrame - off;
            if (f < earliest) break;

            double vTaken = 0.0;
            if (stats->getAttribute(f, keys.taken, vTaken)) {
                outMs = vTaken * 1000.0;
                return true;
            }

            double b = 0.0, e = 0.0;
            if (stats->getAttribute(f, keys.begin, b) && stats->getAttribute(f, keys.end, e) && e >= b) {
                outMs = (e - b) * 1000.0;
                return true;
            }
        }
        return false;
    }

    inline void ensureCameraStatsEnabled(osg::Camera* cam)
    {
        if (!cam) return;
        if (!cam->getStats()) {
            cam->setStats(new osg::Stats("LamureCameraStats"));
        }
        if (auto* stats = cam->getStats()) {
            stats->collectStats("rendering", true);
            stats->collectStats("gpu", true);
        }
    }

    inline void ensureViewerCameraStatsEnabled(opencover::VRViewer* viewer)
    {
        if (!viewer) return;
        osgViewer::ViewerBase::Cameras cams;
        viewer->getCameras(cams);
        for (osg::Camera* cam : cams) {
            ensureCameraStatsEnabled(cam);
        }
    }

    inline std::mutex& statsTextDrawMutex()
    {
        static std::mutex s_mutex;
        return s_mutex;
    }

    inline bool resolveMappedViewAndCamera(
        const osg::Camera* camera,
        const std::unordered_map<const osg::Camera*, int>& viewIds,
        const std::unordered_map<const osg::Camera*, std::shared_ptr<lamure::ren::camera>>& scmCameras,
        int& outViewId,
        std::shared_ptr<lamure::ren::camera>* outCamera = nullptr)
    {
        if (!camera) return false;

        const auto viewIt = viewIds.find(camera);
        if (viewIt == viewIds.end() || viewIt->second < 0) return false;
        outViewId = viewIt->second;

        if (!outCamera) return true;
        const auto camIt = scmCameras.find(camera);
        *outCamera = (camIt != scmCameras.end()) ? camIt->second : nullptr;
        return true;
    }

    inline bool resolveMappedCamera(
        const osg::Camera* camera,
        const std::unordered_map<const osg::Camera*, int>& viewIds,
        const std::unordered_map<const osg::Camera*, std::shared_ptr<lamure::ren::camera>>& scmCameras,
        std::shared_ptr<lamure::ren::camera>& outCamera)
    {
        int viewId = -1;
        return resolveMappedViewAndCamera(camera, viewIds, scmCameras, viewId, &outCamera);
    }

    class StatsTextDrawLockCallback : public osg::Drawable::DrawCallback
    {
    public:
        void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override
        {
            std::lock_guard<std::mutex> lock(statsTextDrawMutex());
            if (drawable)
                drawable->drawImplementation(renderInfo);
        }
    };

    class ClipDistanceScope {
    public:
        ClipDistanceScope(LamureRenderer* renderer, bool enable)
            : m_renderer(renderer)
            , m_enabled(enable && renderer)
        {
            if (m_enabled)
                m_renderer->enableClipDistances();
        }

        ~ClipDistanceScope()
        {
            if (m_enabled)
                m_renderer->disableClipDistances();
        }

        ClipDistanceScope(const ClipDistanceScope&) = delete;
        ClipDistanceScope& operator=(const ClipDistanceScope&) = delete;

    private:
        LamureRenderer* m_renderer{nullptr};
        bool m_enabled{false};
    };

    struct FastState {
        GLint program = 0;
        GLint vao = 0;
        GLint fbo = 0;
        GLint active_tex = 0;
        GLint tex_binding = 0;
        GLint arrayBuffer = 0;
        GLint elem_buf = 0;
        GLboolean blend = GL_FALSE;
        GLboolean depth = GL_FALSE;
        GLboolean cull = GL_FALSE;
        GLint blend_src_rgb = GL_ONE;
        GLint blend_dst_rgb = GL_ZERO;
        GLint blend_src_alpha = GL_ONE;
        GLint blend_dst_alpha = GL_ZERO;
        GLint blend_eq_rgb = GL_FUNC_ADD;
        GLint blend_eq_alpha = GL_FUNC_ADD;
        GLint depth_func = GL_LESS;
        GLboolean depth_mask = GL_TRUE;
        GLint viewport[4] = { 0, 0, 0, 0 };

        bool isFullCapture = false;

        static FastState capture(bool full = false) {
            FastState s;
            s.isFullCapture = full;
            
            // Common state
            glGetIntegerv(GL_CURRENT_PROGRAM, &s.program);
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &s.vao);
            glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &s.arrayBuffer);
            glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &s.elem_buf);
            s.cull  = glIsEnabled(GL_CULL_FACE);

            if (full) {
                glGetIntegerv(GL_FRAMEBUFFER_BINDING, &s.fbo);
                glGetIntegerv(GL_ACTIVE_TEXTURE, &s.active_tex);
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &s.tex_binding);

                s.blend = glIsEnabled(GL_BLEND);
                s.depth = glIsEnabled(GL_DEPTH_TEST);

                glGetIntegerv(GL_BLEND_SRC_RGB, &s.blend_src_rgb);
                glGetIntegerv(GL_BLEND_DST_RGB, &s.blend_dst_rgb);
                glGetIntegerv(GL_BLEND_SRC_ALPHA, &s.blend_src_alpha);
                glGetIntegerv(GL_BLEND_DST_ALPHA, &s.blend_dst_alpha);
                glGetIntegerv(GL_BLEND_EQUATION_RGB, &s.blend_eq_rgb);
                glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &s.blend_eq_alpha);

                glGetIntegerv(GL_DEPTH_FUNC, &s.depth_func);
                glGetBooleanv(GL_DEPTH_WRITEMASK, &s.depth_mask);
                glGetIntegerv(GL_VIEWPORT, s.viewport);
            }
            return s;
        }

        void restore() const {
            glUseProgram(program);
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, arrayBuffer);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elem_buf);
            if (cull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);

            if (isFullCapture) {
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glActiveTexture(active_tex);
                glBindTexture(GL_TEXTURE_2D, tex_binding);

                if (blend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
                if (depth) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);

                glBlendFuncSeparate(blend_src_rgb, blend_dst_rgb, blend_src_alpha, blend_dst_alpha);
                glBlendEquationSeparate(blend_eq_rgb, blend_eq_alpha);

                glDepthFunc(depth_func);
                glDepthMask(depth_mask);
                glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            }
        }
    };

} // namespace

DispatchDrawCallback::DispatchDrawCallback(Lamure* plugin)
    : _plugin(plugin)
    , _renderer(plugin ? plugin->getRenderer() : nullptr)
{
}

void DispatchDrawCallback::drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
{
    if (!_plugin || !_renderer) return;

    int ctx = renderInfo.getContextID();
    const uint64_t frameNo = frameNumberFromRenderInfo(renderInfo);
    const osg::Camera* camera = renderInfo.getCurrentCamera();
    int viewId = -1;
    const bool timingEnabled = _renderer->isTimingModeActive();
    const bool allowLodUpdate = _plugin->getSettings().lod_update && !_plugin->isRebuildInProgress();
    auto& res = _renderer->getResources(ctx);
    std::shared_ptr<lamure::ren::camera> scmCameraHolder;
    bool initialized = false;
    bool dispatchReserved = false;
    {
        std::lock_guard<std::mutex> callbackLock(res.callback_mutex);
        if (!resolveMappedViewAndCamera(camera, res.view_ids, res.scm_cameras, viewId, &scmCameraHolder)) return;
        initialized = res.initialized;
        if (allowLodUpdate && !_plugin->getSettings().models.empty()) {
            int maxViewId = viewId;
            for (const auto& kv : res.view_ids) {
                if (kv.second >= 0) {
                    maxViewId = std::max(maxViewId, kv.second);
                }
            }
            if (res.dispatch_frame != frameNo) {
                res.dispatch_frame = frameNo;
                res.dispatch_done = false;
            }
            if (initialized && scmCameraHolder && viewId == maxViewId && !res.dispatch_done) {
                // Reserve dispatch for this frame before leaving the lock.
                res.dispatch_done = true;
                dispatchReserved = true;
            }
        }
    }
    lamure::ren::camera* scmCamera = scmCameraHolder.get();

    // Ensure initialization happened
    if (!initialized || !scmCamera) return;
    if (!_renderer->gpuOrganizationReady()) return;

    if (allowLodUpdate) {
        lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
        lamure::ren::controller* controller = lamure::ren::controller::get_instance();

        lamure::context_t context_id = controller->deduce_context_id(ctx);
        lamure::view_t view_id = static_cast<lamure::view_t>(viewId);

        const auto tContextStart = std::chrono::steady_clock::now();
        cuts->send_camera(context_id, view_id, *scmCamera);

        const auto corner_values = scmCamera->get_frustum_corners();
        if (corner_values.size() >= 3) {
            const double top_minus_bottom = scm::math::length((corner_values[2]) - (corner_values[0]));
            if (top_minus_bottom > 1e-9) {
                const osg::Camera* cam = renderInfo.getCurrentCamera();
                const osg::Viewport* vp = cam ? cam->getViewport() : nullptr;
                if (vp && vp->height() > 0.0) {
                    const float viewportHeight = static_cast<float>(vp->height());
                    const float height_divided_by_top_minus_bottom = viewportHeight / static_cast<float>(top_minus_bottom);
                    cuts->send_height_divided_by_top_minus_bottom(context_id, view_id, height_divided_by_top_minus_bottom);
                }
            }
        }

        if (_plugin->getSettings().use_pvs) {
            lamure::pvs::pvs_database::get_instance()->set_viewer_position(scmCamera->get_cam_pos());
        }
        if (timingEnabled) {
            _renderer->noteContextUpdateMs(ctx, frameNo, elapsedMs(tContextStart));
        }

        if (dispatchReserved) {
            const auto tDispatchStart = std::chrono::steady_clock::now();
            bool dispatchSucceeded = false;
            try {
                if (lamure::ren::policy::get_instance()->size_of_provenance() > 0) {
                    controller->dispatch(context_id, _renderer->getDevice(ctx), _plugin->getDataProvenance());
                }
                else {
                    controller->dispatch(context_id, _renderer->getDevice(ctx));
                }
                dispatchSucceeded = true;
            }
            catch (const std::exception& e) {
                if (_renderer->notifyOn()) std::cerr << "[Lamure][WARN] dispatch skipped: " << e.what() << "\n";
            }
            {
                std::lock_guard<std::mutex> callbackLock(res.callback_mutex);
                if (res.dispatch_frame == frameNo) {
                    // Keep reserved state on success, allow retry on failure.
                    if (!dispatchSucceeded) {
                        res.dispatch_done = false;
                    }
                }
            }
            if (timingEnabled) {
                _renderer->noteDispatchMs(ctx, frameNo, elapsedMs(tDispatchStart));
            }
        }
    }

    if (drawable) drawable->drawImplementation(renderInfo);
}

void CutsDrawCallback::drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
{
    auto* data = dynamic_cast<const LamureModelData*>(drawable->getUserData());
    if (!data || !m_renderer) return;

    Lamure* plugin = m_renderer->getPlugin();
    if (!plugin) return;

    if (!plugin->getSettings().lod_update || plugin->isRebuildInProgress()) {
        return;
    }

    int ctx = renderInfo.getContextID();
    const uint64_t frameNo = frameNumberFromRenderInfo(renderInfo);
    const osg::Camera* camera = renderInfo.getCurrentCamera();
    int viewId = -1;
    auto& res = m_renderer->getResources(ctx);
    {
        std::lock_guard<std::mutex> callbackLock(res.callback_mutex);
        if (!resolveMappedViewAndCamera(camera, res.view_ids, res.scm_cameras, viewId)) return;
    }
    const bool timingEnabled = m_renderer->isTimingModeActive();
    
    lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    lamure::ren::controller* controller = lamure::ren::controller::get_instance();

    osg::Matrixd view_osg;
    osg::Matrixd proj_osg;
    osg::Matrixd model_osg;

    const osg::Node* drawableParent = drawable ? drawable->getParent(0) : nullptr;
    if (!m_renderer->getModelViewProjectionFromRenderInfo(renderInfo, drawableParent, model_osg, view_osg, proj_osg)) {
        return;
    }
    
    const scm::math::mat4 model_matrix = LamureUtil::matConv4F(model_osg);
    lamure::context_t context_id = controller->deduce_context_id(ctx);

    const auto tContextStart = std::chrono::steady_clock::now();
    cuts->send_transform(context_id, data->modelId, model_matrix);
    if (Lamure::instance()) {
        cuts->send_threshold(context_id, data->modelId, Lamure::instance()->getSettings().lod_error);
    }
    cuts->send_rendered(context_id, data->modelId);
    database->get_model(data->modelId)->set_transform(model_matrix);
    if (timingEnabled) {
        m_renderer->noteContextUpdateMs(ctx, frameNo, elapsedMs(tContextStart));
    }
    
    if (drawable) drawable->drawImplementation(renderInfo);
}


void PointsDrawCallback::drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
{
    auto* data = dynamic_cast<const LamureModelData*>(drawable->getUserData());
    if (!data || !m_renderer) return;

    Lamure* plugin = m_renderer->getPlugin();
    if (!plugin) return;

    const uint64_t frameNo = frameNumberFromRenderInfo(renderInfo);
    int ctx = renderInfo.getContextID();
    const osg::Camera* cam = renderInfo.getCurrentCamera();

    if (!m_renderer->beginFrame(ctx)) return;

    bool pixelMetricsActive = false;
    bool stateCaptured = false;
    FastState before;
    auto cleanup = [&]() {
        if (pixelMetricsActive) {
            m_renderer->endPixelMetricsCapture(ctx);
            pixelMetricsActive = false;
        }
        m_renderer->endFrame(ctx);
        if (stateCaptured) {
            before.restore();
            stateCaptured = false;
        }
    };

    const auto& settings = plugin->getSettings();
    if (settings.models.empty() || !cam) {
        cleanup();
        return;
    }

    int viewId = -1;
    std::shared_ptr<lamure::ren::camera> scmCameraHolder;
    bool initialized = false;
    bool shadersInitialized = false;
    bool resourcesInitialized = false;
    bool hasMappedView = false;
    auto& res = m_renderer->getResources(ctx);
    {
        std::lock_guard<std::mutex> callbackLock(res.callback_mutex);
        hasMappedView = resolveMappedViewAndCamera(cam, res.view_ids, res.scm_cameras, viewId, &scmCameraHolder);
        initialized = res.initialized;
        shadersInitialized = res.shaders_initialized;
        resourcesInitialized = res.resources_initialized;
    }
    if (!hasMappedView) {
        cleanup();
        return;
    }

    lamure::ren::camera* scmCamera = scmCameraHolder.get();
    const bool readyForRender = m_renderer->gpuOrganizationReady() && initialized && scmCamera &&
                                shadersInitialized && resourcesInitialized;
    if (!readyForRender) {
        cleanup();
        return;
    }

    const osg::GraphicsContext* currentGc = m_renderer->getGC(renderInfo);
    if (!currentGc) {
        cleanup();
        return;
    }

    osg::State* state = renderInfo.getState();
    const bool wantsMultipass = (settings.shader_type == LamureRenderer::ShaderType::SurfelMultipass);
    before = FastState::capture(wantsMultipass);
    stateCaptured = true;

    glDisable(GL_CULL_FACE);
    m_renderer->updateActiveClipPlanes();
    ClipDistanceScope clipScope(m_renderer, m_renderer->clipPlaneCount() > 0);
    if (state) {
        state->setCheckForGLErrors(osg::State::CheckForGLErrors::NEVER_CHECK_GL_ERRORS);
    }

    lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    lamure::ren::controller* controller = lamure::ren::controller::get_instance();

    osg::Matrixd view_osg;
    osg::Matrixd proj_osg;
    osg::Matrixd model_osg;

    const osg::Node* drawableParent = drawable ? drawable->getParent(0) : nullptr;

    static bool loggedOpenMP = false;
    if (!loggedOpenMP) {
        loggedOpenMP = true;
#ifdef _OPENMP
        if (m_renderer->notifyOn()) { 
             std::cout << "[Lamure] OpenMP is ENABLED. Max threads: " << omp_get_max_threads() << std::endl; 
        }
#else
        if (m_renderer->notifyOn()) { 
            std::cout << "[Lamure] OpenMP is DISABLED." << std::endl; 
        }
#endif
    }

    if (!m_renderer->getModelViewProjectionFromRenderInfo(renderInfo, drawableParent, model_osg, view_osg, proj_osg)) {
        cleanup();
        return;
    }
    const scm::math::mat4 view = LamureUtil::matConv4F(view_osg);
    const scm::math::mat4 proj = LamureUtil::matConv4F(proj_osg);

    const osg::Viewport* vp = cam ? cam->getViewport() : nullptr;
    const double vpW = vp ? vp->width() : 0.0;
    const double vpH = vp ? vp->height() : 0.0;
    const scm::math::vec2 viewport((float)vpW, (float)vpH);
    const bool timingEnabled = m_renderer->isTimingModeActive();
    if (timingEnabled && frameNo != kInvalidTimingFrame && vpW > 0.0 && vpH > 0.0) {
        // Ensure this context participates in the frame snapshot even when nothing is rendered.
        m_renderer->noteContextRenderCounts(ctx, frameNo, 0, 0, 0);
    }

    m_renderer->setFrameUniforms(proj, view, viewport, res);

    lamure::context_t context_id = controller->deduce_context_id(ctx);
    lamure::view_t view_id = static_cast<lamure::view_t>(viewId);
    
    // NOTE: Cuts and Dispatch are now handled in separate callbacks!
    
    scm::math::mat4 model_matrix = LamureUtil::matConv4F(model_osg);

    if (plugin->getUI() && plugin->getUI()->getDumpButton() && plugin->getUI()->getDumpButton()->state()) {
        const osg::FrameStamp* fs = state ? state->getFrameStamp() : nullptr;
        const uint64_t frameNumber = fs ? fs->getFrameNumber() : 0;
        static uint64_t pendingFrame = std::numeric_limits<uint64_t>::max();
        static std::map<lamure::model_t, scm::math::mat4f> pendingModels;

        if (pendingFrame == std::numeric_limits<uint64_t>::max()) {
            pendingFrame = frameNumber;
        }

        if (frameNumber != pendingFrame) {
            if (scmCamera) {
                const scm::math::mat4f view_scm = scmCamera->get_view_matrix();
                const scm::math::mat4f proj_scm = scmCamera->get_projection_matrix();
                plugin->dump("[Lamure] dump(ctx=", ctx, "): scm_view\n",
                             view_scm, "\n\n");
                plugin->dump("[Lamure] dump(ctx=", ctx, "): scm_proj\n",
                             proj_scm, "\n\n");
            }
            plugin->dump("[Lamure] dump: model\n", 1);
            for (const auto& entry : pendingModels) {
                plugin->dump("model=", entry.first, "\n",
                             entry.second, "\n\n");
            }
            pendingModels.clear();
            pendingFrame = frameNumber;
            plugin->dump("", 0);
        }

        pendingModels[data->modelId] = model_matrix;
    }

    lamure::ren::cut& cut = cuts->get_cut(context_id, view_id, data->modelId);
    const auto& renderable = cut.complete_set();

    if (renderable.empty()) {
        cleanup();
        return;
    }

    const scm::math::mat4 mvp = proj * view * model_matrix;
    const scm::math::mat4 m = model_matrix;

    m_renderer->setModelUniforms(mvp, m, res);

    const lamure::ren::bvh* bvh = database->get_model(data->modelId)->get_bvh();
    size_t surfels_per_node = database->get_primitives_per_node();
    const std::vector<scm::gl::boxf>& bbv = bvh->get_bounding_boxes();
    scm::gl::frustum frustum = scmCamera->get_frustum_by_model(m);
    auto initPointVaoForCurrentGraphicsContext = [&]() -> bool {
        if (!currentGc) {
            return false;
        }

        auto cachedIt = res.point_vaos.find(currentGc);
        if (cachedIt != res.point_vaos.end()) {
            if (cachedIt->second != 0) {
                return true;
            }
            res.point_vaos.erase(cachedIt);
        }

        auto* ctrl = lamure::ren::controller::get_instance();
        auto device = m_renderer->getDevice(ctx);
        if (!ctrl || !device) {
            return false;
        }

        const bool hasProvenance = (lamure::ren::policy::get_instance()->size_of_provenance() > 0 && m_renderer->getPlugin());
        if (hasProvenance) {
            auto scmContext = m_renderer->getSchismContext(ctx);
            if (!scmContext) {
                return false;
            }
            scmContext->apply_vertex_input();
            scmContext->bind_vertex_array(
                ctrl->get_context_memory(context_id, lamure::ren::bvh::primitive_type::POINTCLOUD, device, m_renderer->getPlugin()->getDataProvenance()));

            GLint currentPointVao = 0;
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &currentPointVao);
            if (currentPointVao <= 0) {
                return false;
            }
            res.point_vaos[currentGc] = static_cast<GLuint>(currentPointVao);
            return true;
        }

        scm::gl::buffer_ptr pointBuffer = ctrl->get_context_buffer(context_id, device);
        if (!pointBuffer) {
            return false;
        }

        const GLuint pointVbo = static_cast<GLuint>(pointBuffer->object_id());
        if (pointVbo == 0) {
            return false;
        }

        GLuint vao = 0;
        glGenVertexArrays(1, &vao);
        if (vao == 0) {
            return false;
        }

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, pointVbo);

        constexpr GLsizei stride = static_cast<GLsizei>(8u * sizeof(float));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<const GLvoid*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 1, GL_UNSIGNED_BYTE, GL_TRUE, stride, reinterpret_cast<const GLvoid*>(12));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 1, GL_UNSIGNED_BYTE, GL_TRUE, stride, reinterpret_cast<const GLvoid*>(13));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 1, GL_UNSIGNED_BYTE, GL_TRUE, stride, reinterpret_cast<const GLvoid*>(14));
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 1, GL_UNSIGNED_BYTE, GL_TRUE, stride, reinterpret_cast<const GLvoid*>(15));
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<const GLvoid*>(16));
        glEnableVertexAttribArray(6);
        glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<const GLvoid*>(20));

        glBindVertexArray(0);

        res.point_vaos[currentGc] = vao;
        return true;
    };

    auto setPointVaoForCurrentGraphicsContext = [&]() -> GLuint {
        if (!currentGc) {
            return 0;
        }
        auto it = res.point_vaos.find(currentGc);
        if (it == res.point_vaos.end() || it->second == 0) {
            if (it != res.point_vaos.end()) {
                res.point_vaos.erase(it);
            }
            return 0;
        }
        glBindVertexArray(it->second);
        return it->second;
    };

    auto setScreenQuadVaoForCurrentGraphicsContext = [&]() -> GLuint {
        if (!currentGc || res.geo_screen_quad.vbo == 0) {
            return 0;
        }
        auto it = res.screen_quad_vaos.find(currentGc);
        if (it != res.screen_quad_vaos.end() && it->second != 0) {
            glBindVertexArray(it->second);
            return it->second;
        }

        GLuint vao = 0;
        glGenVertexArrays(1, &vao);
        if (vao == 0) {
            return 0;
        }
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, res.geo_screen_quad.vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glBindVertexArray(0);
        res.screen_quad_vaos[currentGc] = vao;
        glBindVertexArray(vao);
        return vao;
    };

    if (!initPointVaoForCurrentGraphicsContext()) {
        cleanup();
        return;
    }
    const GLuint cachedPointVao = setPointVaoForCurrentGraphicsContext();
    if (cachedPointVao == 0) {
        cleanup();
        return;
    }

    const bool isMultipass = wantsMultipass;
    pixelMetricsActive =
        (!isMultipass) && m_renderer->beginPixelMetricsCapture(ctx, frameNo, viewId, static_cast<double>(vpW * vpH));

    const bool useAnisoThisPass = LamureUtil::decideUseAniso(proj, settings.anisotropic_surfel_scaling, settings.anisotropic_auto_threshold);
    uint64_t rendered_primitives = 0;
    uint64_t rendered_nodes = 0;

    const bool enableColorDebug = settings.coloring;
    const bool showNormalsDebug = enableColorDebug && settings.show_normals;
    const bool showAccuracyDebug = enableColorDebug && settings.show_accuracy;
    const bool showRadiusDeviationDebug = enableColorDebug && settings.show_radius_deviation;
    const bool showOutputSensitivityDebug = enableColorDebug && settings.show_output_sensitivity;

    // Check if we need per-node uniforms
    const bool needNodeUniforms = (showRadiusDeviationDebug || showAccuracyDebug);

    auto cullAndBatch = [&](bool useOpenMP, bool updateCounters) {
        auto& firsts = res.batch_firsts;
        auto& counts = res.batch_counts;
        firsts.clear();
        counts.clear();

#ifdef _OPENMP
        if (useOpenMP) {
            int max_threads = omp_get_max_threads();
            auto& tls_firsts = res.tls_firsts;
            auto& tls_counts = res.tls_counts;

            if (tls_firsts.size() < static_cast<size_t>(max_threads)) {
                tls_firsts.resize(max_threads);
                tls_counts.resize(max_threads);
            }

            for (int t = 0; t < max_threads; ++t) {
                tls_firsts[t].clear();
                tls_counts[t].clear();
                tls_firsts[t].reserve((renderable.size() / max_threads) + 1);
                tls_counts[t].reserve((renderable.size() / max_threads) + 1);
            }

            if (updateCounters) {
                uint64_t local_rendered_primitives = 0;
                uint64_t local_rendered_nodes = 0;
                #pragma omp parallel
                {
                    int t = omp_get_thread_num();
                    auto& local_firsts = tls_firsts[t];
                    auto& local_counts = tls_counts[t];

                    #pragma omp for schedule(dynamic, 64) reduction(+:local_rendered_primitives, local_rendered_nodes)
                    for (int i = 0; i < (int)renderable.size(); ++i) {
                        const auto& node_slot = renderable[i];
                        if (scmCamera->cull_against_frustum(frustum, bbv[node_slot.node_id_]) != 1) {
                            local_firsts.push_back((GLint)(node_slot.slot_id_ * surfels_per_node));
                            local_counts.push_back((GLsizei)surfels_per_node);
                            local_rendered_primitives += surfels_per_node;
                            ++local_rendered_nodes;
                        }
                    }
                }
                rendered_primitives += local_rendered_primitives;
                rendered_nodes += local_rendered_nodes;
            } else {
                #pragma omp parallel
                {
                    int t = omp_get_thread_num();
                    auto& local_firsts = tls_firsts[t];
                    auto& local_counts = tls_counts[t];

                    #pragma omp for schedule(dynamic, 64)
                    for (int i = 0; i < (int)renderable.size(); ++i) {
                        const auto& node_slot = renderable[i];
                        if (scmCamera->cull_against_frustum(frustum, bbv[node_slot.node_id_]) != 1) {
                            local_firsts.push_back((GLint)(node_slot.slot_id_ * surfels_per_node));
                            local_counts.push_back((GLsizei)surfels_per_node);
                        }
                    }
                }
            }

            size_t total_size = 0;
            for (int t = 0; t < max_threads; ++t) total_size += tls_firsts[t].size();
            firsts.reserve(total_size);
            counts.reserve(total_size);
            for (int t = 0; t < max_threads; ++t) {
                firsts.insert(firsts.end(), tls_firsts[t].begin(), tls_firsts[t].end());
                counts.insert(counts.end(), tls_counts[t].begin(), tls_counts[t].end());
            }

            if (!firsts.empty()) {
                glMultiDrawArrays(scm::gl::PRIMITIVE_POINT_LIST, firsts.data(), counts.data(), (GLsizei)firsts.size());
            }
            return;
        }
#endif

        firsts.reserve(renderable.size());
        counts.reserve(renderable.size());
        for (const auto& node_slot : renderable) {
            if (scmCamera->cull_against_frustum(frustum, bbv[node_slot.node_id_]) != 1) {
                firsts.push_back((GLint)(node_slot.slot_id_ * surfels_per_node));
                counts.push_back((GLsizei)surfels_per_node);
                if (updateCounters) {
                    rendered_primitives += surfels_per_node;
                    ++rendered_nodes;
                }
            }
        }
        if (!firsts.empty()) {
            glMultiDrawArrays(scm::gl::PRIMITIVE_POINT_LIST, firsts.data(), counts.data(), (GLsizei)firsts.size());
        }
    };

    if (isMultipass) {
        GLint prev_fbo = 0;
        GLint prev_draw_buffer = 0;
        GLint prev_read_buffer = 0;
        GLint prev_viewport[4] = {0,0,0,0};
        GLint prev_scissor_box[4] = {0,0,0,0};
        const GLboolean prev_scissor_test = glIsEnabled(GL_SCISSOR_TEST);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
        glGetIntegerv(GL_DRAW_BUFFER, &prev_draw_buffer);
        glGetIntegerv(GL_READ_BUFFER, &prev_read_buffer);
        glGetIntegerv(GL_VIEWPORT, prev_viewport);
        glGetIntegerv(GL_SCISSOR_BOX, prev_scissor_box);
        const int vpWidth  = std::max(1, prev_viewport[2]);
        const int vpHeight = std::max(1, prev_viewport[3]);
        auto& target = m_renderer->acquireMultipassTarget(ctx, viewId, vpWidth, vpHeight);
        const scm::math::vec2 passViewport(static_cast<float>(vpWidth), static_cast<float>(vpHeight));
        const float scale_radius_combined = settings.scale_radius * settings.scale_element;
        const float scale_proj_pass = opencover::cover->getScale() * static_cast<float>(vpHeight) * 0.5f * proj.data_array[5];
        const auto bindTexture2DToUnit = [](GLuint unit, GLuint texture) {
#if defined(GL_VERSION_4_5)
            if (GLEW_VERSION_4_5) {
                glBindTextureUnit(unit, texture);
                return;
            }
#endif
#if defined(GL_ARB_direct_state_access)
            if (GLEW_ARB_direct_state_access) {
                glBindTextureUnit(unit, texture);
                return;
            }
#endif
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, texture);
        };

        // --- PASS 1: Depth pre-pass
        glBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
        glViewport(0, 0, vpWidth, vpHeight);
        glDisable(GL_SCISSOR_TEST);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glClear(GL_DEPTH_BUFFER_BIT);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LEQUAL);

        glUseProgram(res.sh_surfel_pass1.program);
        if (res.sh_surfel_pass1.viewport_loc          >= 0) glUniform2f(res.sh_surfel_pass1.viewport_loc, static_cast<float>(vpWidth), static_cast<float>(vpHeight));
        if (res.sh_surfel_pass1.max_radius_loc        >= 0) glUniform1f(res.sh_surfel_pass1.max_radius_loc, settings.max_radius);
        if (res.sh_surfel_pass1.min_radius_loc        >= 0) glUniform1f(res.sh_surfel_pass1.min_radius_loc, settings.min_radius);
        if (res.sh_surfel_pass1.min_screen_size_loc   >= 0) glUniform1f(res.sh_surfel_pass1.min_screen_size_loc, settings.min_screen_size);
        if (res.sh_surfel_pass1.max_screen_size_loc   >= 0) glUniform1f(res.sh_surfel_pass1.max_screen_size_loc, settings.max_screen_size);
        if (res.sh_surfel_pass1.scale_radius_loc      >= 0) glUniform1f(res.sh_surfel_pass1.scale_radius_loc, scale_radius_combined);
        if (res.sh_surfel_pass1.scale_radius_gamma_loc  >= 0) glUniform1f(res.sh_surfel_pass1.scale_radius_gamma_loc, settings.scale_radius_gamma);
        if (res.sh_surfel_pass1.max_radius_cut_loc      >= 0) glUniform1f(res.sh_surfel_pass1.max_radius_cut_loc, settings.max_radius_cut);
        if (res.sh_surfel_pass1.scale_projection_loc    >= 0) glUniform1f(res.sh_surfel_pass1.scale_projection_loc, scale_proj_pass);
        if (res.sh_surfel_pass1.projection_matrix_loc >= 0) glUniformMatrix4fv(res.sh_surfel_pass1.projection_matrix_loc, 1, GL_FALSE, proj.data_array);
        if (res.sh_surfel_pass1.use_aniso_loc          >= 0) glUniform1i(res.sh_surfel_pass1.use_aniso_loc, useAnisoThisPass ? 1 : 0);

        const scm::math::mat4 model_view_matrix = view * model_matrix;
        if (res.sh_surfel_pass1.model_view_matrix_loc >= 0)
            glUniformMatrix4fv(res.sh_surfel_pass1.model_view_matrix_loc, 1, GL_FALSE, model_view_matrix.data_array);

        // Batch Rendering for Pass 1 (no node uniforms here)
        cullAndBatch(true, true);

        // --- PASS 2: Accumulation
        GLenum accumBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
        glDrawBuffers(3, accumBuffers);
        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glDepthFunc(GL_ALWAYS);

        glUseProgram(res.sh_surfel_pass2.program);
        glViewport(0, 0, vpWidth, vpHeight);
        bindTexture2DToUnit(0, target.depth_texture);

        if (res.sh_surfel_pass2.viewport_loc            >= 0) glUniform2f(res.sh_surfel_pass2.viewport_loc, passViewport.x, passViewport.y);
        if (res.sh_surfel_pass2.max_radius_loc          >= 0) glUniform1f(res.sh_surfel_pass2.max_radius_loc, settings.max_radius);
        if (res.sh_surfel_pass2.min_radius_loc          >= 0) glUniform1f(res.sh_surfel_pass2.min_radius_loc, settings.min_radius);
        if (res.sh_surfel_pass2.scale_radius_loc        >= 0) glUniform1f(res.sh_surfel_pass2.scale_radius_loc, scale_radius_combined);
        if (res.sh_surfel_pass2.scale_radius_gamma_loc  >= 0) glUniform1f(res.sh_surfel_pass2.scale_radius_gamma_loc, settings.scale_radius_gamma);
        if (res.sh_surfel_pass2.max_radius_cut_loc      >= 0) glUniform1f(res.sh_surfel_pass2.max_radius_cut_loc, settings.max_radius_cut);
        if (res.sh_surfel_pass2.coloring_loc            >= 0) glUniform1i(res.sh_surfel_pass2.coloring_loc, enableColorDebug ? 1 : 0);
        if (res.sh_surfel_pass2.show_normals_loc        >= 0) glUniform1i(res.sh_surfel_pass2.show_normals_loc, showNormalsDebug ? 1 : 0);
        if (res.sh_surfel_pass2.show_output_sens_loc    >= 0) glUniform1i(res.sh_surfel_pass2.show_output_sens_loc, showOutputSensitivityDebug ? 1 : 0);
        if (res.sh_surfel_pass2.show_radius_dev_loc     >= 0) glUniform1i(res.sh_surfel_pass2.show_radius_dev_loc, showRadiusDeviationDebug ? 1 : 0);
        if (res.sh_surfel_pass2.show_accuracy_loc       >= 0) glUniform1i(res.sh_surfel_pass2.show_accuracy_loc, showAccuracyDebug ? 1 : 0);
        if (res.sh_surfel_pass2.projection_matrix_loc   >= 0) glUniformMatrix4fv(res.sh_surfel_pass2.projection_matrix_loc, 1, GL_FALSE, proj.data_array);
        if (res.sh_surfel_pass2.min_screen_size_loc     >= 0) glUniform1f(res.sh_surfel_pass2.min_screen_size_loc, settings.min_screen_size);
        if (res.sh_surfel_pass2.max_screen_size_loc     >= 0) glUniform1f(res.sh_surfel_pass2.max_screen_size_loc, settings.max_screen_size);
        if (res.sh_surfel_pass2.scale_projection_loc    >= 0) glUniform1f(res.sh_surfel_pass2.scale_projection_loc, scale_proj_pass);
        if (res.sh_surfel_pass2.use_aniso_loc           >= 0) glUniform1i(res.sh_surfel_pass2.use_aniso_loc, useAnisoThisPass ? 1 : 0);
        if (res.sh_surfel_pass2.depth_range_loc         >= 0) glUniform1f(res.sh_surfel_pass2.depth_range_loc, settings.depth_range);
        if (res.sh_surfel_pass2.flank_lift_loc          >= 0) glUniform1f(res.sh_surfel_pass2.flank_lift_loc, settings.flank_lift);

        scm::math::mat4 mv_copy = model_view_matrix;
        const scm::math::mat3 normal_matrix = scm::math::transpose(scm::math::inverse(LamureUtil::matConv4to3F(mv_copy)));
        if (res.sh_surfel_pass2.model_view_matrix_loc >= 0)
            glUniformMatrix4fv(res.sh_surfel_pass2.model_view_matrix_loc, 1, GL_FALSE, model_view_matrix.data_array);
        if (res.sh_surfel_pass2.normal_matrix_loc >= 0)
            glUniformMatrix3fv(res.sh_surfel_pass2.normal_matrix_loc, 1, GL_FALSE, normal_matrix.data_array);

        if (needNodeUniforms) {
            for (const auto& node_slot : renderable) {
                if (scmCamera->cull_against_frustum(frustum, bbv[node_slot.node_id_]) != 1) {
                    m_renderer->setNodeUniforms(bvh, node_slot.node_id_, res);
                    glDrawArrays(scm::gl::PRIMITIVE_POINT_LIST, (node_slot.slot_id_) * (GLsizei)surfels_per_node, (GLsizei)surfels_per_node);
                }
            }
        } else {
             // Reuse Pass-1 cull+batch result: same renderable/frustum in this frame.
             auto& firsts = res.batch_firsts;
             auto& counts = res.batch_counts;
             if (!firsts.empty()) {
                 glMultiDrawArrays(scm::gl::PRIMITIVE_POINT_LIST, firsts.data(), counts.data(), (GLsizei)firsts.size());
             }
        }

        // --- PASS 3: Resolve / Lighting
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
        glDrawBuffer(prev_draw_buffer);
        glReadBuffer(prev_read_buffer);
        glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);
        if (prev_scissor_test) {
            glEnable(GL_SCISSOR_TEST);
            glScissor(prev_scissor_box[0], prev_scissor_box[1], prev_scissor_box[2], prev_scissor_box[3]);
        } else {
            glDisable(GL_SCISSOR_TEST);
        }

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LEQUAL);
        glDisable(GL_BLEND);

        glUseProgram(res.sh_surfel_pass3.program);
        bindTexture2DToUnit(0, target.texture_color);
        bindTexture2DToUnit(1, target.texture_normal);
        bindTexture2DToUnit(2, target.texture_position);
        bindTexture2DToUnit(3, target.depth_texture);

        scm::math::vec4 light_ws(settings.point_light_pos[0], settings.point_light_pos[1], settings.point_light_pos[2], 1.0f);
        scm::math::vec4 light_vs4 = view * light_ws;
        scm::math::vec3 light_vs(light_vs4[0], light_vs4[1], light_vs4[2]);

        if (res.sh_surfel_pass3.point_light_pos_vs_loc      >= 0) glUniform3fv(res.sh_surfel_pass3.point_light_pos_vs_loc, 1, light_vs.data_array);
        if (res.sh_surfel_pass3.point_light_intensity_loc   >= 0) glUniform1f(res.sh_surfel_pass3.point_light_intensity_loc, settings.point_light_intensity);
        if (res.sh_surfel_pass3.ambient_intensity_loc       >= 0) glUniform1f(res.sh_surfel_pass3.ambient_intensity_loc, settings.ambient_intensity);
        if (res.sh_surfel_pass3.specular_intensity_loc      >= 0) glUniform1f(res.sh_surfel_pass3.specular_intensity_loc, settings.specular_intensity);
        if (res.sh_surfel_pass3.shininess_loc               >= 0) glUniform1f(res.sh_surfel_pass3.shininess_loc, settings.shininess);
        if (res.sh_surfel_pass3.gamma_loc                   >= 0) glUniform1f(res.sh_surfel_pass3.gamma_loc, settings.gamma);
        if (res.sh_surfel_pass3.use_tone_mapping_loc        >= 0) glUniform1i(res.sh_surfel_pass3.use_tone_mapping_loc, settings.use_tone_mapping ? 1 : 0);
        if (res.sh_surfel_pass3.lighting_loc                >= 0) glUniform1f(res.sh_surfel_pass3.lighting_loc, settings.lighting);

        if (setScreenQuadVaoForCurrentGraphicsContext() == 0) {
            cleanup();
            return;
        }
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        glActiveTexture(GL_TEXTURE0);
        glDepthMask(GL_FALSE);

        m_renderer->noteContextRenderCounts(ctx, frameNo, rendered_primitives, rendered_nodes, 0);
        cleanup();
        return;
    }

    if (needNodeUniforms) {
        for (const auto& node_slot : renderable) {
            if (scmCamera->cull_against_frustum(frustum, bbv[node_slot.node_id_]) != 1) {
                m_renderer->setNodeUniforms(bvh, node_slot.node_id_, res);
                glDrawArrays(scm::gl::PRIMITIVE_POINT_LIST, (node_slot.slot_id_) * (GLsizei)surfels_per_node, (GLsizei)surfels_per_node);
                rendered_primitives += surfels_per_node;
                ++rendered_nodes;
            }
        }
    } else {
        // Optimized Batch Rendering
        cullAndBatch(true, true);
    }

    m_renderer->noteContextRenderCounts(ctx, frameNo, rendered_primitives, rendered_nodes, 0);

    cleanup();
}
void BoundingBoxDrawCallback::drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
{
    auto* data = drawable ? dynamic_cast<const LamureModelData*>(drawable->getUserData()) : nullptr;
    if (!data || !m_renderer)
        return;

    Lamure* plugin = m_renderer->getPlugin();
    if (!plugin)
        return;
    const int ctx = renderInfo.getContextID();
    const uint64_t frameNo = frameNumberFromRenderInfo(renderInfo);
    const osg::Camera* camera = renderInfo.getCurrentCamera();
    int viewId = -1;
    auto& res = m_renderer->getResources(ctx);
    std::shared_ptr<lamure::ren::camera> scmCameraHolder;
    GLuint boxVbo = 0;
    GLuint boxIbo = 0;
    GLuint lineProgram = 0;
    GLint mvpLocation = -1;
    GLint colorLocation = -1;
    GLsizei boxIndexCount = 0;

    {
        std::lock_guard<std::mutex> callbackLock(res.callback_mutex);
        if (!resolveMappedViewAndCamera(camera, res.view_ids, res.scm_cameras, viewId, &scmCameraHolder)) return;
        lineProgram = res.sh_line.program;
        mvpLocation = res.sh_line.mvp_matrix_location;
        colorLocation = res.sh_line.in_color_location;
        boxVbo = res.geo_box.vbo;
        boxIbo = res.geo_box.ibo;
        boxIndexCount = static_cast<GLsizei>(res.box_idx.size());
    }

    lamure::ren::camera* scmCamera = scmCameraHolder.get();
    if (!m_renderer->gpuOrganizationReady())
        return;
    const osg::GraphicsContext* currentGc = m_renderer->getGC(renderInfo);
    if (!currentGc)
        return;

    if (!scmCamera || lineProgram == 0 ||
        boxVbo == 0 || boxIbo == 0) {
        return;
    }

    GLuint boxVao = 0;
    {
        std::lock_guard<std::mutex> callbackLock(res.callback_mutex);
        auto vaoIt = res.box_vaos.find(currentGc);
        if (vaoIt != res.box_vaos.end() && vaoIt->second != 0) {
            boxVao = vaoIt->second;
        } else {
            glGenVertexArrays(1, &boxVao);
            glBindVertexArray(boxVao);
            glBindBuffer(GL_ARRAY_BUFFER, boxVbo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boxIbo);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glBindVertexArray(0);
            if (boxVao != 0) {
                res.box_vaos[currentGc] = boxVao;
            }
        }
    }
    if (boxVao == 0) {
        return;
    }
    
    osg::State* state = renderInfo.getState();
    if (state) {
        state->setCheckForGLErrors(osg::State::ONCE_PER_FRAME);
    }

    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
    lamure::ren::controller* controller = lamure::ren::controller::get_instance();
    lamure::context_t context_id = controller->deduce_context_id(ctx);
    lamure::view_t view_id = static_cast<lamure::view_t>(viewId);

    const osg::Node* drawableParent = drawable ? drawable->getParent(0) : nullptr;
    osg::Matrixd model_osg;
    osg::Matrixd view_osg;
    osg::Matrixd proj_osg;
    if (!m_renderer->getModelViewProjectionFromRenderInfo(renderInfo, drawableParent, model_osg, view_osg, proj_osg)) {
        return;
    }
    scm::math::mat4 model_matrix = LamureUtil::matConv4F(model_osg);

    lamure::ren::cut& cut = cuts->get_cut(context_id, view_id, data->modelId);
    const auto& renderable = cut.complete_set();
    if (renderable.empty()) {
        return;
    }

    const auto it = m_renderer->m_bvh_node_vertex_offsets.find(data->modelId);
    if (it == m_renderer->m_bvh_node_vertex_offsets.end()) {
        return;
    }

    const lamure::ren::bvh* bvh = database->get_model(data->modelId)->get_bvh();
    const std::vector<scm::gl::boxf>& bbv = bvh->get_bounding_boxes();
    const std::vector<uint32_t>& node_offsets = it->second;

    const scm::math::mat4 view_matrix = LamureUtil::matConv4F(view_osg);
    const scm::math::mat4 projection_matrix = LamureUtil::matConv4F(proj_osg);
    const scm::math::mat4 mvp_matrix = projection_matrix * view_matrix * model_matrix;

    // Capture only states this callback mutates.
    GLint prevProgram = 0;
    GLint prevVao = 0;
    GLint prevArrayBuffer = 0;
    GLint prevElementBuffer = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArrayBuffer);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &prevElementBuffer);

    glBindVertexArray(boxVao);
    glUseProgram(lineProgram);
    glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, mvp_matrix.data_array);
    glUniform4f(colorLocation,
        plugin->getSettings().bvh_color[0], 
        plugin->getSettings().bvh_color[1],
        plugin->getSettings().bvh_color[2],
        plugin->getSettings().bvh_color[3]);

    glBindBuffer(GL_ARRAY_BUFFER, boxVbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boxIbo);

    GLint boxVboSize = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &boxVboSize);
    const GLsizeiptr requiredBoxBytes = static_cast<GLsizeiptr>(sizeof(float) * 8 * 3);
    if (boxVboSize < requiredBoxBytes) {
        glBufferData(GL_ARRAY_BUFFER, requiredBoxBytes, nullptr, GL_DYNAMIC_DRAW);
    }

    uint64_t rendered_bounding_boxes = 0;
    for (const auto& node_slot : renderable) {
        const uint32_t node_id = node_slot.node_id_;
        if (node_id >= bbv.size() || node_id >= node_offsets.size())
            continue;

        const auto corners = LamureUtil::getBoxCorners(bbv[node_id]);
        if (corners.size() >= 8 * 3) {
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * (8 * 3), corners.data());
            glDrawElements(GL_LINES, boxIndexCount, GL_UNSIGNED_SHORT, nullptr);
            ++rendered_bounding_boxes;
        }
    }

    glUseProgram(static_cast<GLuint>(prevProgram));
    glBindVertexArray(static_cast<GLuint>(prevVao));
    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(prevArrayBuffer));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLuint>(prevElementBuffer));

    m_renderer->noteContextRenderCounts(ctx, frameNo, 0, 0, rendered_bounding_boxes);
}



LamureRenderer::LamureRenderer(Lamure *plugin)
: m_plugin(plugin)
{
}

LamureRenderer::~LamureRenderer()
{
    releaseMultipassTargets();
}

bool LamureRenderer::notifyOn() const
{
    return m_plugin && m_plugin->getSettings().show_notify;
}

bool LamureRenderer::isTimingModeActive() const
{
    if (!m_plugin) return false;
    const auto& s = m_plugin->getSettings();
    const auto* meas = m_plugin->getMeasurement();
    const bool measurementActive = (meas && meas->isActive() && s.measure_full && !s.measure_off);
    const bool statisticsOverlayActive = s.show_stats;
    return statisticsOverlayActive || measurementActive;
}

LamureRenderer::ContextTimingSample& LamureRenderer::upsertTimingSampleLocked(int ctxId, uint64_t frameNo)
{
    auto& perCtx = m_timing_frames[frameNo];
    auto& sample = perCtx[ctxId];
    if (sample.frame_number != frameNo) {
        sample = ContextTimingSample{};
        sample.frame_number = frameNo;
    }
    return sample;
}

void LamureRenderer::trimTimingHistoryLocked()
{
    while (m_timing_frames.size() > kTimingHistoryLimit) {
        const uint64_t dropped = m_timing_frames.begin()->first;
        m_timing_frames.erase(m_timing_frames.begin());
        m_timing_global_by_frame.erase(dropped);
        if (m_last_complete_timing_frame == dropped) {
            m_last_complete_timing_frame = kInvalidTimingFrame;
        }
    }

    while (m_timing_global_by_frame.size() > kTimingHistoryLimit) {
        m_timing_global_by_frame.erase(m_timing_global_by_frame.begin());
    }
}

void LamureRenderer::noteContextUpdateMs(int ctxId, uint64_t frameNo, double ms)
{
    if (ctxId < 0 || frameNo == kInvalidTimingFrame) return;
    std::lock_guard<std::mutex> lock(m_timing_mutex);
    auto& t = upsertTimingSampleLocked(ctxId, frameNo);
    accumulateMs(t.context_update_ms, ms);
    trimTimingHistoryLocked();
}

void LamureRenderer::noteDispatchMs(int ctxId, uint64_t frameNo, double ms)
{
    if (ctxId < 0 || frameNo == kInvalidTimingFrame) return;
    std::lock_guard<std::mutex> lock(m_timing_mutex);
    auto& t = upsertTimingSampleLocked(ctxId, frameNo);
    accumulateMs(t.dispatch_ms, ms);
    trimTimingHistoryLocked();
}

void LamureRenderer::noteContextStats(int ctxId, uint64_t frameNo, double cullMs, double drawMs, double gpuMs)
{
    if (ctxId < 0 || frameNo == kInvalidTimingFrame) return;
    std::lock_guard<std::mutex> lock(m_timing_mutex);
    auto& t = upsertTimingSampleLocked(ctxId, frameNo);
    accumulateMs(t.cpu_cull_ms, cullMs);
    accumulateMs(t.cpu_draw_ms, drawMs);
    accumulateMs(t.gpu_ms, gpuMs);
    t.render_cpu_ms = sumKnownMs({t.context_update_ms, t.dispatch_ms, t.cpu_cull_ms, t.cpu_draw_ms});
    trimTimingHistoryLocked();
}

void LamureRenderer::noteContextPixelStats(int ctxId, uint64_t frameNo, double samplesPassed, double coveredSamples, double viewportPixels)
{
    if (ctxId < 0 || frameNo == kInvalidTimingFrame) return;
    std::lock_guard<std::mutex> lock(m_timing_mutex);
    auto& t = upsertTimingSampleLocked(ctxId, frameNo);
    if (LamureUtil::isValidValue(samplesPassed)) {
        accumulateMs(t.samples_passed, samplesPassed);
    }
    if (LamureUtil::isValidValue(coveredSamples)) {
        t.covered_samples = coveredSamples;
    }
    if (LamureUtil::isValidValue(viewportPixels)) {
        t.viewport_pixels = viewportPixels;
    }
    if (LamureUtil::isValidValue(t.covered_samples) && LamureUtil::isValidValue(t.viewport_pixels) && t.viewport_pixels > 0.0) {
        t.coverage = std::clamp(t.covered_samples / t.viewport_pixels, 0.0, 1.0);
    }
    if (LamureUtil::isValidValue(t.samples_passed) && LamureUtil::isValidValue(t.covered_samples)) {
        if (t.covered_samples > 0.0) {
            t.overdraw = t.samples_passed / t.covered_samples;
        } else if (t.covered_samples == 0.0) {
            t.overdraw = 0.0;
        }
    }
    trimTimingHistoryLocked();
}

void LamureRenderer::noteContextRenderCounts(int ctxId, uint64_t frameNo, uint64_t renderedPrimitives, uint64_t renderedNodes, uint64_t renderedBoundingBoxes)
{
    if (ctxId < 0 || frameNo == kInvalidTimingFrame) return;
    std::lock_guard<std::mutex> lock(m_timing_mutex);
    auto& t = upsertTimingSampleLocked(ctxId, frameNo);
    auto it = m_timing_latest_frame_by_ctx.find(ctxId);
    if (it == m_timing_latest_frame_by_ctx.end()) {
        m_timing_latest_frame_by_ctx.emplace(ctxId, frameNo);
    } else {
        it->second = std::max(it->second, frameNo);
    }
    t.rendered_primitives += renderedPrimitives;
    t.rendered_nodes += renderedNodes;
    t.rendered_bounding_boxes += renderedBoundingBoxes;
    trimTimingHistoryLocked();
}

void LamureRenderer::noteGlobalStats(uint64_t frameNo, double cpuUpdateMs, double waitMs)
{
    if (frameNo == kInvalidTimingFrame) return;
    std::lock_guard<std::mutex> lock(m_timing_mutex);
    auto& g = m_timing_global_by_frame[frameNo];
    if (LamureUtil::isValidValue(cpuUpdateMs)) g.cpu_update_ms = cpuUpdateMs;
    if (LamureUtil::isValidValue(waitMs))      g.wait_ms = waitMs;
    trimTimingHistoryLocked();
}

void LamureRenderer::commitFrameTiming(int ctxId, uint64_t frameNo, const ContextTimingSample& sample)
{
    if (ctxId < 0 || frameNo == kInvalidTimingFrame) return;
    std::lock_guard<std::mutex> lock(m_timing_mutex);
    auto& t = upsertTimingSampleLocked(ctxId, frameNo);

    accumulateMs(t.dispatch_ms, sample.dispatch_ms);
    accumulateMs(t.context_update_ms, sample.context_update_ms);
    accumulateMs(t.cpu_cull_ms, sample.cpu_cull_ms);
    accumulateMs(t.cpu_draw_ms, sample.cpu_draw_ms);
    accumulateMs(t.gpu_ms, sample.gpu_ms);
    accumulateMs(t.samples_passed, sample.samples_passed);

    if (LamureUtil::isValidValue(sample.covered_samples)) {
        if (!LamureUtil::isValidValue(t.covered_samples)) t.covered_samples = sample.covered_samples;
        else t.covered_samples = std::max(t.covered_samples, sample.covered_samples);
    }
    if (LamureUtil::isValidValue(sample.viewport_pixels)) {
        t.viewport_pixels = sample.viewport_pixels;
    }
    if (LamureUtil::isValidValue(sample.coverage)) {
        t.coverage = sample.coverage;
    } else if (LamureUtil::isValidValue(t.covered_samples) && LamureUtil::isValidValue(t.viewport_pixels) && t.viewport_pixels > 0.0) {
        t.coverage = std::clamp(t.covered_samples / t.viewport_pixels, 0.0, 1.0);
    }
    if (LamureUtil::isValidValue(sample.overdraw)) {
        t.overdraw = sample.overdraw;
    } else if (LamureUtil::isValidValue(t.samples_passed) && LamureUtil::isValidValue(t.covered_samples)) {
        if (t.covered_samples > 0.0) {
            t.overdraw = t.samples_passed / t.covered_samples;
        } else if (t.covered_samples == 0.0) {
            t.overdraw = 0.0;
        }
    }

    t.render_cpu_ms = sumKnownMs({t.context_update_ms, t.dispatch_ms, t.cpu_cull_ms, t.cpu_draw_ms});
    trimTimingHistoryLocked();
}

void LamureRenderer::releasePixelMetricsQueries(ContextResources& res)
{
    for (auto& slot : res.pixel_query_slots) {
        if (slot.query_id) {
            glDeleteQueries(1, &slot.query_id);
            slot.query_id = 0;
        }
        slot.issued = false;
        slot.frame_number = std::numeric_limits<uint64_t>::max();
        slot.viewport_pixels = -1.0;
    }
    res.pixel_queries_ready = false;
    res.pixel_metrics_checked = false;
    res.pixel_metrics_supported = false;
    res.pixel_capture_active = false;
    res.pixel_total_query_active = 0;
    res.pixel_stencil_needs_clear = false;
    res.pixel_capture_view_id = -1;
    res.pixel_capture_used_stencil = false;
    res.pixel_warned_no_stencil = false;
    res.pixel_aggregate_frame = std::numeric_limits<uint64_t>::max();
    res.pixel_viewport_sum = 0.0;
    res.pixel_viewport_accounted.clear();
    res.pixel_view_covered_samples.clear();
}

bool LamureRenderer::beginPixelMetricsCapture(int ctxId, uint64_t frameNo, int viewId, double viewportPixels)
{
    if (!m_plugin) return false;
    if (!isTimingModeActive() || frameNo == kInvalidTimingFrame || viewportPixels <= 0.0) return false;
    auto& res = getResources(ctxId);

    if (!res.pixel_metrics_checked) {
        GLint stencilBits = 0;
        glGetIntegerv(GL_STENCIL_BITS, &stencilBits);
        if (stencilBits > 0) {
            const GLint bitIndex = std::max(0, std::min(stencilBits - 1, 7));
            res.pixel_stencil_bit = static_cast<GLuint>(1u << bitIndex);
            res.pixel_capture_used_stencil = true;
        } else {
            res.pixel_stencil_bit = 0;
            res.pixel_capture_used_stencil = false;
            if (!res.pixel_warned_no_stencil && notifyOn()) {
                std::cout << "[Lamure][WARN] Coverage fallback active (no stencil buffer)." << std::endl;
                res.pixel_warned_no_stencil = true;
            }
        }
        res.pixel_metrics_supported = true;
        res.pixel_metrics_checked = true;
    }
    if (!res.pixel_metrics_supported) return false;

    if (!res.pixel_queries_ready) {
        for (auto& slot : res.pixel_query_slots) {
            glGenQueries(1, &slot.query_id);
        }
        res.pixel_queries_ready = true;
    }

    if (res.pixel_current_frame != frameNo) {
        res.pixel_current_frame = frameNo;
        res.pixel_stencil_needs_clear = true;
    }

    if (res.pixel_aggregate_frame != frameNo) {
        res.pixel_aggregate_frame = frameNo;
        res.pixel_viewport_sum = 0.0;
        res.pixel_viewport_accounted.clear();
        res.pixel_view_covered_samples.clear();
    }
    if (viewId >= 0 && res.pixel_viewport_accounted.insert(viewId).second) {
        res.pixel_viewport_sum += viewportPixels;
    }

    if (res.pixel_capture_active) return false;

    ContextResources::PixelQuerySlot* totalSlot = nullptr;
    for (size_t n = 0; n < res.pixel_query_slots.size(); ++n) {
        auto& slot = res.pixel_query_slots[(res.pixel_query_next_slot + n) % res.pixel_query_slots.size()];
        if (!slot.issued) {
            totalSlot = &slot;
            res.pixel_query_next_slot = static_cast<uint32_t>((res.pixel_query_next_slot + n + 1) % res.pixel_query_slots.size());
            break;
        }
    }
    if (!totalSlot || !totalSlot->query_id) return false;

    const bool useStencil = (res.pixel_stencil_bit != 0);
    res.pixel_capture_used_stencil = useStencil;
    if (useStencil) {
        res.pixel_prev_stencil_test = glIsEnabled(GL_STENCIL_TEST);
        glGetIntegerv(GL_STENCIL_FUNC, &res.pixel_prev_stencil_func);
        glGetIntegerv(GL_STENCIL_REF, &res.pixel_prev_stencil_ref);
        glGetIntegerv(GL_STENCIL_VALUE_MASK, &res.pixel_prev_stencil_value_mask);
        glGetIntegerv(GL_STENCIL_WRITEMASK, &res.pixel_prev_stencil_writemask);
        glGetIntegerv(GL_STENCIL_FAIL, &res.pixel_prev_stencil_fail);
        glGetIntegerv(GL_STENCIL_PASS_DEPTH_FAIL, &res.pixel_prev_stencil_zfail);
        glGetIntegerv(GL_STENCIL_PASS_DEPTH_PASS, &res.pixel_prev_stencil_zpass);

        if (res.pixel_stencil_needs_clear) {
            glEnable(GL_STENCIL_TEST);
            glStencilMask(static_cast<GLuint>(res.pixel_stencil_bit));
            glClearStencil(0);
            glClear(GL_STENCIL_BUFFER_BIT);
            res.pixel_stencil_needs_clear = false;
        }

        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, static_cast<GLint>(res.pixel_stencil_bit), static_cast<GLuint>(res.pixel_stencil_bit));
        glStencilMask(static_cast<GLuint>(res.pixel_stencil_bit));
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    }

    glBeginQuery(GL_SAMPLES_PASSED, totalSlot->query_id);
    totalSlot->frame_number = frameNo;
    totalSlot->viewport_pixels = viewportPixels;
    totalSlot->is_coverage = false;
    totalSlot->issued = true;

    res.pixel_capture_active = true;
    res.pixel_total_query_active = totalSlot->query_id;
    res.pixel_capture_viewport_pixels = viewportPixels;
    res.pixel_capture_frame = frameNo;
    res.pixel_capture_view_id = viewId;
    return true;
}

void LamureRenderer::endPixelMetricsCapture(int ctxId)
{
    auto& res = getResources(ctxId);
    if (!res.pixel_capture_active || !res.pixel_total_query_active) return;

    ContextResources::PixelQuerySlot* totalSlot = nullptr;
    for (auto& slot : res.pixel_query_slots) {
        if (slot.query_id == res.pixel_total_query_active) {
            totalSlot = &slot;
            break;
        }
    }

    const uint64_t captureFrame = res.pixel_capture_frame;
    const double captureViewportPixels = res.pixel_capture_viewport_pixels;
    const int captureViewId = res.pixel_capture_view_id;
    const GLuint totalQueryId = res.pixel_total_query_active;

    glEndQuery(GL_SAMPLES_PASSED);

    GLuint totalResult = 0;
    glGetQueryObjectuiv(totalQueryId, GL_QUERY_RESULT, &totalResult);
    double coveredResultForView = -1.0;

    if (totalSlot) {
        totalSlot->issued = false;
        totalSlot->is_coverage = false;
        totalSlot->frame_number = std::numeric_limits<uint64_t>::max();
        totalSlot->viewport_pixels = -1.0;
    }

    if (res.pixel_capture_used_stencil && res.sh_coverage_query.program && res.geo_screen_quad.vbo) {
        GLint prevProgram = 0, prevVAO = 0;
        GLboolean prevBlend = glIsEnabled(GL_BLEND);
        GLboolean prevCull = glIsEnabled(GL_CULL_FACE);
        GLboolean prevDepth = glIsEnabled(GL_DEPTH_TEST);
        GLboolean prevStencil = glIsEnabled(GL_STENCIL_TEST);
        GLboolean prevColorMask[4] = {GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE};
        GLboolean prevDepthMask = GL_TRUE;
        GLint prevStencilFunc = GL_ALWAYS, prevStencilRef = 0, prevStencilValMask = ~0, prevStencilWriteMask = ~0;
        GLint prevStencilFail = GL_KEEP, prevStencilZFail = GL_KEEP, prevStencilZPass = GL_KEEP;

        glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
        glGetBooleanv(GL_COLOR_WRITEMASK, prevColorMask);
        glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
        glGetIntegerv(GL_STENCIL_FUNC, &prevStencilFunc);
        glGetIntegerv(GL_STENCIL_REF, &prevStencilRef);
        glGetIntegerv(GL_STENCIL_VALUE_MASK, &prevStencilValMask);
        glGetIntegerv(GL_STENCIL_WRITEMASK, &prevStencilWriteMask);
        glGetIntegerv(GL_STENCIL_FAIL, &prevStencilFail);
        glGetIntegerv(GL_STENCIL_PASS_DEPTH_FAIL, &prevStencilZFail);
        glGetIntegerv(GL_STENCIL_PASS_DEPTH_PASS, &prevStencilZPass);

        glUseProgram(res.sh_coverage_query.program);
        GLuint coverageVao = 0;
        glGenVertexArrays(1, &coverageVao);
        glBindVertexArray(coverageVao);
        glBindBuffer(GL_ARRAY_BUFFER, res.geo_screen_quad.vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<const GLvoid*>(0));
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_EQUAL, static_cast<GLint>(res.pixel_stencil_bit), static_cast<GLuint>(res.pixel_stencil_bit));
        glStencilMask(0);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

        glBeginQuery(GL_SAMPLES_PASSED, totalQueryId);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glEndQuery(GL_SAMPLES_PASSED);

        GLuint coveredResult = 0;
        glGetQueryObjectuiv(totalQueryId, GL_QUERY_RESULT, &coveredResult);
        coveredResultForView = static_cast<double>(coveredResult);

        if (prevBlend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
        if (prevCull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
        if (prevDepth) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
        glDepthMask(prevDepthMask);
        glColorMask(prevColorMask[0], prevColorMask[1], prevColorMask[2], prevColorMask[3]);
        if (prevStencil) glEnable(GL_STENCIL_TEST); else glDisable(GL_STENCIL_TEST);
        glStencilFunc(prevStencilFunc, prevStencilRef, static_cast<GLuint>(prevStencilValMask));
        glStencilMask(static_cast<GLuint>(prevStencilWriteMask));
        glStencilOp(prevStencilFail, prevStencilZFail, prevStencilZPass);
        if (coverageVao != 0) {
            glDeleteVertexArrays(1, &coverageVao);
        }
        glBindVertexArray(static_cast<GLuint>(prevVAO));
        glUseProgram(static_cast<GLuint>(prevProgram));
    }

    if (res.pixel_capture_used_stencil) {
        if (res.pixel_prev_stencil_test) glEnable(GL_STENCIL_TEST); else glDisable(GL_STENCIL_TEST);
        glStencilFunc(res.pixel_prev_stencil_func, res.pixel_prev_stencil_ref, static_cast<GLuint>(res.pixel_prev_stencil_value_mask));
        glStencilMask(static_cast<GLuint>(res.pixel_prev_stencil_writemask));
        glStencilOp(res.pixel_prev_stencil_fail, res.pixel_prev_stencil_zfail, res.pixel_prev_stencil_zpass);
    }

    double aggregatedCovered = -1.0;
    if (captureViewId >= 0) {
        if (!LamureUtil::isValidValue(coveredResultForView)) {
            // Fallback without stencil: covered <= total and <= viewport
            coveredResultForView = std::min<double>(static_cast<double>(totalResult), captureViewportPixels);
        }
        auto& prev = res.pixel_view_covered_samples[captureViewId];
        prev = std::max(prev, coveredResultForView);
        aggregatedCovered = 0.0;
        for (const auto& kv : res.pixel_view_covered_samples) {
            aggregatedCovered += kv.second;
        }
    }
    const double aggregatedViewport =
        (res.pixel_viewport_sum > 0.0) ? res.pixel_viewport_sum : captureViewportPixels;
    noteContextPixelStats(ctxId, captureFrame, static_cast<double>(totalResult), aggregatedCovered, aggregatedViewport);

    res.pixel_capture_active = false;
    res.pixel_total_query_active = 0;
    res.pixel_capture_viewport_pixels = -1.0;
    res.pixel_capture_frame = std::numeric_limits<uint64_t>::max();
    res.pixel_capture_view_id = -1;
    res.pixel_capture_used_stencil = false;
}

LamureRenderer::TimingSnapshot LamureRenderer::getTimingSnapshot(uint64_t preferredFrame) const
{
    TimingSnapshot snap{};
    if (!isTimingModeActive()) return snap;

    std::lock_guard<std::mutex> lock(m_timing_mutex);
    uint64_t target = kInvalidTimingFrame;

    if (preferredFrame != kInvalidTimingFrame) {
        auto it = m_timing_frames.find(preferredFrame);
        if (it == m_timing_frames.end()) return snap;
        target = preferredFrame;
    } else {
        auto isCompleteFrame = [&](uint64_t frameNo, const std::unordered_map<int, ContextTimingSample>& perCtx) -> bool {
            if (perCtx.empty()) return false;
            for (const auto& latestByCtx : m_timing_latest_frame_by_ctx) {
                const int ctxId = latestByCtx.first;
                const uint64_t latestFrame = latestByCtx.second;
                if (latestFrame < frameNo) continue;
                if (perCtx.find(ctxId) == perCtx.end()) {
                    return false;
                }
            }
            return true;
        };

        for (auto it = m_timing_frames.rbegin(); it != m_timing_frames.rend(); ++it) {
            if (isCompleteFrame(it->first, it->second)) {
                target = it->first;
                m_last_complete_timing_frame = target;
                break;
            }
        }

        if (target == kInvalidTimingFrame && m_last_complete_timing_frame != kInvalidTimingFrame) {
            if (m_timing_frames.find(m_last_complete_timing_frame) != m_timing_frames.end()) {
                target = m_last_complete_timing_frame;
            }
        }
    }

    if (target == kInvalidTimingFrame) return snap;
    auto frameIt = m_timing_frames.find(target);
    if (frameIt == m_timing_frames.end()) return snap;

    snap.frame_number = target;

    auto globalIt = m_timing_global_by_frame.find(target);
    if (globalIt != m_timing_global_by_frame.end()) {
        snap.cpu_update_ms = globalIt->second.cpu_update_ms;
        snap.wait_ms = globalIt->second.wait_ms;
    }

    for (const auto& kv : frameIt->second) {
        auto per = kv.second;
        per.render_cpu_ms = sumKnownMs({per.context_update_ms, per.dispatch_ms, per.cpu_cull_ms, per.cpu_draw_ms});
        snap.per_context.emplace_back(kv.first, per);
        snap.rendered_primitives += per.rendered_primitives;
        snap.rendered_nodes += per.rendered_nodes;
        snap.rendered_bounding_boxes += per.rendered_bounding_boxes;
        accumulateMs(snap.dispatch_ms, per.dispatch_ms);
        accumulateMs(snap.context_update_ms, per.context_update_ms);
        accumulateMs(snap.cpu_cull_ms, per.cpu_cull_ms);
        accumulateMs(snap.cpu_draw_ms, per.cpu_draw_ms);
        accumulateMs(snap.gpu_ms, per.gpu_ms);
        accumulateMs(snap.render_cpu_ms, per.render_cpu_ms);
        if (LamureUtil::isValidValue(per.viewport_pixels) && per.viewport_pixels > 0.0) {
            accumulateMs(snap.viewport_pixels, per.viewport_pixels);
            const double samples = LamureUtil::isValidValue(per.samples_passed) ? per.samples_passed : 0.0;
            const double covered = LamureUtil::isValidValue(per.covered_samples) ? per.covered_samples : 0.0;
            accumulateMs(snap.samples_passed, samples);
            accumulateMs(snap.covered_samples, covered);
        } else {
            accumulateMs(snap.samples_passed, per.samples_passed);
            accumulateMs(snap.covered_samples, per.covered_samples);
        }
    }

    if (LamureUtil::isValidValue(snap.covered_samples) && LamureUtil::isValidValue(snap.viewport_pixels) && snap.viewport_pixels > 0.0) {
        snap.coverage = std::clamp(snap.covered_samples / snap.viewport_pixels, 0.0, 1.0);
    }
    if (LamureUtil::isValidValue(snap.samples_passed) && LamureUtil::isValidValue(snap.covered_samples) && snap.covered_samples > 0.0) {
        snap.overdraw = snap.samples_passed / snap.covered_samples;
    }

    std::sort(snap.per_context.begin(), snap.per_context.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    return snap;
}

bool LamureRenderer::getDisplayTimingData(int ctxId, uint64_t currentFrameNo, ContextTimingSample& outContext, TimingSnapshot& outSummed) const
{
    outContext = ContextTimingSample{};
    outSummed = TimingSnapshot{};
    if (!isTimingModeActive()) return false;

    std::lock_guard<std::mutex> lock(m_timing_mutex);
    if (m_timing_frames.empty()) return false;

    uint64_t displayFrameNo = kInvalidTimingFrame;
    if (currentFrameNo != kInvalidTimingFrame && currentFrameNo > 0) {
        const uint64_t prev = currentFrameNo - 1;
        if (m_timing_frames.find(prev) != m_timing_frames.end()) {
            displayFrameNo = prev;
        }
    }
    if (displayFrameNo == kInvalidTimingFrame) {
        for (auto it = m_timing_frames.rbegin(); it != m_timing_frames.rend(); ++it) {
            if (currentFrameNo != kInvalidTimingFrame && it->first >= currentFrameNo) continue;
            displayFrameNo = it->first;
            break;
        }
    }
    if (displayFrameNo == kInvalidTimingFrame) {
        displayFrameNo = m_timing_frames.rbegin()->first;
    }

    auto frameIt = m_timing_frames.find(displayFrameNo);
    if (frameIt == m_timing_frames.end()) {
        return false;
    }

    auto ctxIt = frameIt->second.find(ctxId);
    if (ctxIt != frameIt->second.end()) {
        outContext = ctxIt->second;
    }

    outSummed.frame_number = displayFrameNo;
    auto globalIt = m_timing_global_by_frame.find(displayFrameNo);
    if (globalIt != m_timing_global_by_frame.end()) {
        outSummed.cpu_update_ms = globalIt->second.cpu_update_ms;
        outSummed.wait_ms = globalIt->second.wait_ms;
    }

    for (const auto& kv : frameIt->second) {
        auto per = kv.second;
        per.render_cpu_ms = sumKnownMs({per.context_update_ms, per.dispatch_ms, per.cpu_cull_ms, per.cpu_draw_ms});
        outSummed.per_context.emplace_back(kv.first, per);
        outSummed.rendered_primitives += per.rendered_primitives;
        outSummed.rendered_nodes += per.rendered_nodes;
        outSummed.rendered_bounding_boxes += per.rendered_bounding_boxes;
        accumulateMs(outSummed.dispatch_ms, per.dispatch_ms);
        accumulateMs(outSummed.context_update_ms, per.context_update_ms);
        accumulateMs(outSummed.cpu_cull_ms, per.cpu_cull_ms);
        accumulateMs(outSummed.cpu_draw_ms, per.cpu_draw_ms);
        accumulateMs(outSummed.gpu_ms, per.gpu_ms);
        accumulateMs(outSummed.render_cpu_ms, per.render_cpu_ms);
        if (LamureUtil::isValidValue(per.viewport_pixels) && per.viewport_pixels > 0.0) {
            accumulateMs(outSummed.viewport_pixels, per.viewport_pixels);
            const double samples = LamureUtil::isValidValue(per.samples_passed) ? per.samples_passed : 0.0;
            const double covered = LamureUtil::isValidValue(per.covered_samples) ? per.covered_samples : 0.0;
            accumulateMs(outSummed.samples_passed, samples);
            accumulateMs(outSummed.covered_samples, covered);
        } else {
            accumulateMs(outSummed.samples_passed, per.samples_passed);
            accumulateMs(outSummed.covered_samples, per.covered_samples);
        }
    }

    if (LamureUtil::isValidValue(outSummed.covered_samples) &&
        LamureUtil::isValidValue(outSummed.viewport_pixels) &&
        outSummed.viewport_pixels > 0.0) {
        outSummed.coverage = std::clamp(outSummed.covered_samples / outSummed.viewport_pixels, 0.0, 1.0);
    }
    if (LamureUtil::isValidValue(outSummed.samples_passed) &&
        LamureUtil::isValidValue(outSummed.covered_samples)) {
        if (outSummed.covered_samples > 0.0) {
            outSummed.overdraw = outSummed.samples_passed / outSummed.covered_samples;
        } else if (outSummed.covered_samples == 0.0) {
            outSummed.overdraw = 0.0;
        }
    }

    std::sort(outSummed.per_context.begin(), outSummed.per_context.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    return true;
}

std::string LamureRenderer::getTimingCompactString(uint64_t preferredFrame) const
{
    const auto snap = getTimingSnapshot(preferredFrame);
    if (snap.per_context.empty()) return std::string();

    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    bool first = true;
    for (const auto& kv : snap.per_context) {
        if (!first) os << '|';
        first = false;
        const auto& t = kv.second;
        os << "ctx" << kv.first
           << "(disp=" << (LamureUtil::isValidValue(t.dispatch_ms) ? t.dispatch_ms : -1.0)
           << ",ctx=" << (LamureUtil::isValidValue(t.context_update_ms) ? t.context_update_ms : -1.0)
           << ",cull=" << (LamureUtil::isValidValue(t.cpu_cull_ms) ? t.cpu_cull_ms : -1.0)
           << ",draw=" << (LamureUtil::isValidValue(t.cpu_draw_ms) ? t.cpu_draw_ms : -1.0)
           << ",gpu=" << (LamureUtil::isValidValue(t.gpu_ms) ? t.gpu_ms : -1.0)
           << ",rcpu=" << (LamureUtil::isValidValue(t.render_cpu_ms) ? t.render_cpu_ms : -1.0)
           << ')';
    }
    return os.str();
}

void LamureRenderer::updateLiveTimingFromRenderInfo(osg::RenderInfo& renderInfo, int ctxId)
{
    if (!isTimingModeActive()) return;

    const uint64_t frameNo = frameNumberFromRenderInfo(renderInfo);
    if (frameNo == kInvalidTimingFrame) return;

    if (m_plugin) {
        const auto& s = m_plugin->getSettings();
        const auto* meas = m_plugin->getMeasurement();
        const bool measurementOwnsTiming = (meas && meas->isActive() && s.measure_full && !s.measure_off);
        if (measurementOwnsTiming) {
            return;
        }
    }

    {
        std::lock_guard<std::mutex> lock(m_timing_mutex);
        if (m_live_timing_last_scanned_frame == frameNo) {
            return;
        }
        m_live_timing_last_scanned_frame = frameNo;
    }

    auto* viewer = opencover::VRViewer::instance();
    if (!viewer) return;
    if (!m_stats_initialized) {
        ensureViewerCameraStatsEnabled(viewer);
    }

    osg::Stats* viewerStats = viewer->getViewerStats();
    if (viewerStats) {
        if (!m_stats_initialized) {
            viewerStats->collectStats("update", true);
            viewerStats->collectStats("sync", true);
            viewerStats->collectStats("swap", true);
            viewerStats->collectStats("finish", true);
        }

        double updMs = -1.0, syncMs = -1.0, swapMs = -1.0, finishMs = -1.0;
        (void)queryTimeTakenMsBacksearch(viewerStats, static_cast<unsigned>(frameNo), 8, kUpdateTraversalKeys, updMs);
        (void)queryTimeTakenMsBacksearch(viewerStats, static_cast<unsigned>(frameNo), 8, kSyncKeys, syncMs);
        (void)queryTimeTakenMsBacksearch(viewerStats, static_cast<unsigned>(frameNo), 8, kSwapKeys, swapMs);
        (void)queryTimeTakenMsBacksearch(viewerStats, static_cast<unsigned>(frameNo), 8, kFinishKeys, finishMs);

        const double waitMs = sumKnownMs({syncMs, swapMs, finishMs});
        noteGlobalStats(frameNo, updMs, waitMs);
    }

    osgViewer::ViewerBase::Cameras cams;
    viewer->getCameras(cams);
    for (osg::Camera* cam : cams) {
        if (!cam) continue;
        if (!m_stats_initialized) ensureCameraStatsEnabled(cam);
        osg::Stats* camStats = cam->getStats();
        if (!camStats) continue;

        if (!m_stats_initialized) {
            camStats->collectStats("rendering", true);
            camStats->collectStats("gpu", true);
        }

        unsigned base = static_cast<unsigned>(frameNo);
        if (auto* rnd = dynamic_cast<osgViewer::Renderer*>(cam->getRenderer())) {
            if (!rnd->getGraphicsThreadDoesCull() && base > 0u) --base;
        }

        double cullMs = -1.0, drawMs = -1.0, gpuMs = -1.0;
        (void)queryTimeTakenMsBacksearch(camStats, base, 8, kCullTraversalKeys, cullMs);
        (void)queryTimeTakenMsBacksearch(camStats, base, 8, kDrawTraversalKeys, drawMs);
        (void)queryTimeTakenMsBacksearch(camStats, base, 8, kGpuDrawKeys, gpuMs);

        int camCtxId = ctxId;
        if (auto* gc = cam->getGraphicsContext()) {
            if (auto* gs = gc->getState()) {
                camCtxId = static_cast<int>(gs->getContextID());
            }
        }
        ContextTimingSample sample{};
        sample.frame_number = frameNo;
        sample.cpu_cull_ms = cullMs;
        sample.cpu_draw_ms = drawMs;
        sample.gpu_ms = gpuMs;
        commitFrameTiming(camCtxId, frameNo, sample);
    }
    m_stats_initialized = true;
}

void LamureRenderer::updateActiveClipPlanes()
{
    m_clip_plane_count = 0;
    std::fill(m_clip_planes.begin(), m_clip_planes.end(), scm::math::vec4f(0.f, 0.f, 0.f, 0.f));

    if (!m_plugin || !opencover::cover || !opencover::cover->isClippingOn())
        return;

    osg::ClipNode* clipNode = dynamic_cast<osg::ClipNode*>(opencover::cover->getObjectsRoot());
    if (!clipNode)
        return;

    const int available = std::min<int>(clipNode->getNumClipPlanes(), kMaxClipPlanes);
    for (int i = 0; i < available; ++i) {
        osg::ClipPlane* plane = clipNode->getClipPlane(i);
        if (!plane)
            continue;
        const osg::Vec4d eq = plane->getClipPlane();
        m_clip_planes[m_clip_plane_count++] = scm::math::vec4f(static_cast<float>(eq.x()),
                                                               static_cast<float>(eq.y()),
                                                               static_cast<float>(eq.z()),
                                                               static_cast<float>(eq.w()));
        if (m_clip_plane_count >= kMaxClipPlanes)
            break;
    }
}

void LamureRenderer::enableClipDistances()
{
    const int desired = (std::min)(m_clip_plane_count, kMaxClipPlanes);
    for (int i = 0; i < desired; ++i)
        glEnable(GL_CLIP_DISTANCE0 + i);
    m_enabled_clip_distances = desired;
}

void LamureRenderer::disableClipDistances()
{
    for (int i = 0; i < m_enabled_clip_distances; ++i)
        glDisable(GL_CLIP_DISTANCE0 + i);
    m_enabled_clip_distances = 0;
}

void LamureRenderer::uploadClipPlanes(GLint countLocation, GLint dataLocation) const
{
    const GLint planeCount = (std::min)(m_clip_plane_count, kMaxClipPlanes);
    if (countLocation >= 0)
        glUniform1i(countLocation, planeCount);
    if (dataLocation >= 0 && planeCount > 0)
        glUniform4fv(dataLocation, planeCount, reinterpret_cast<const GLfloat*>(m_clip_planes.data()));
}

LamureRenderer::ContextResources& LamureRenderer::getResources(int ctxId)
{
    std::scoped_lock lock(m_renderMutex, m_ctx_mutex);
    return m_ctx_res[ctxId];
}

int LamureRenderer::resolveViewId(osg::RenderInfo& renderInfo) const
{
    const osg::Camera* currentCamera = renderInfo.getCurrentCamera();
    if (!currentCamera)
        return -1;

    // Deterministic-only mapping: no guessed fallback IDs.
    const auto* cfg = opencover::coVRConfig::instance();
    if (!cfg)
        return -1;
    for (size_t i = 0; i < cfg->channels.size(); ++i) {
        const auto& ch = cfg->channels[i];
        if (ch.camera != currentCamera)
            continue;
        if (ch.screenNum >= 0)
            return ch.screenNum;
        return static_cast<int>(i);
    }

    return -1;
}

InitDrawCallback::InitDrawCallback(Lamure* plugin)
    : _plugin(plugin)
    , _renderer(plugin ? plugin->getRenderer() : nullptr)
{
}

void InitDrawCallback::drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
{
    int ctx = renderInfo.getContextID();
    osg::Camera* cam = renderInfo.getCurrentCamera();
    if (!cam) {
        if (drawable) { drawable->drawImplementation(renderInfo); }
        return;
    }

    auto& res = _renderer->getResources(ctx);
    std::lock_guard<std::mutex> callbackLock(res.callback_mutex);

    const int viewId = _renderer->resolveViewId(renderInfo);
    if (viewId < 0) {
        if (drawable) { drawable->drawImplementation(renderInfo); }
        return;
    }
    
    // Context-level initialization (shared GL resources).
    if (!res.initialized) {
        res.ctx = ctx;
        _renderer->initSchismObjects(res);
        res.initialized = (res.scm_device != nullptr && res.scm_context != nullptr);
        if (!res.initialized) {
            if (drawable) { drawable->drawImplementation(renderInfo); }
            return;
        }
    }

    // View-level initialization (camera per view on the same context).
    auto viewIdIt = res.view_ids.find(cam);
    if (viewIdIt == res.view_ids.end() || viewIdIt->second != viewId) {
        res.view_ids[cam] = viewId;
        res.scm_cameras.erase(cam);
    }

    auto camIt = res.scm_cameras.find(cam);
    if (camIt == res.scm_cameras.end() || !camIt->second) {
        if (!_renderer->initCamera(res, cam, viewId)) {
            if (drawable) { drawable->drawImplementation(renderInfo); }
            return;
        }
        camIt = res.scm_cameras.find(cam);
        if (camIt == res.scm_cameras.end() || !camIt->second) {
            if (drawable) { drawable->drawImplementation(renderInfo); }
            return;
        }
    }
    auto& scmCamera = camIt->second;

    osg::Matrixd view_osg;
    osg::Matrixd proj_osg;
    _renderer->getMatricesFromRenderInfo(renderInfo, view_osg, proj_osg);
    osg::Matrix mv_matrix = view_osg;
    scm::math::mat4d modelview_matrix = LamureUtil::matConv4D(mv_matrix);
    scm::math::mat4d projection_matrix = LamureUtil::matConv4D(proj_osg);

    if (scmCamera) {
        scmCamera->set_projection_matrix(projection_matrix);
        if (_plugin->getUI()->getSyncButton()->state()) {
            scmCamera->set_view_matrix(modelview_matrix);
        }
    }

    if (!_renderer->initGpus(res)) {
        if (drawable) { drawable->drawImplementation(renderInfo); }
        return;
    }

    if (!res.shaders_initialized) {
        if (!_renderer->initLamureShader(res)) {
            if (drawable) { drawable->drawImplementation(renderInfo); }
            return;
        }
        _renderer->initUniforms(res);
    }

    if (!res.resources_initialized) {
        GLState before = GLState::capture();

        _renderer->initFrustumResources(res);
        _renderer->initBoxResources(res);
        _renderer->initPclResources(res);
        
        before.restore();
        res.resources_initialized = true;

    }
    
    if (drawable) { drawable->drawImplementation(renderInfo); }
}

StatsDrawCallback::StatsDrawCallback(Lamure *plugin, osgText::Text *label, osgText::Text *values)
    : _plugin(plugin)
    , _label(label)
    , _values(values)
    , _renderer(plugin ? plugin->getRenderer() : nullptr)
{
}

void StatsDrawCallback::drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const
{
    // One shared stats text drawable is rendered by multiple contexts. Serialize updates/draw to
    // avoid races in osgText glyph/shader state initialization.
    std::lock_guard<std::mutex> statsDrawLock(statsTextDrawMutex());

    if (!_renderer) {
        if (drawable) { drawable->drawImplementation(renderInfo); }
        return;
    }
    const int ctx = renderInfo.getContextID();
    const osg::Camera* camera = renderInfo.getCurrentCamera();
    int statsViewId = -1;
    {
        auto& res = _renderer->getResources(ctx);
        std::lock_guard<std::mutex> callbackLock(res.callback_mutex);

        if (camera) {
            auto itDirect = res.view_ids.find(camera);
            if (itDirect != res.view_ids.end()) {
                statsViewId = itDirect->second;
            } else {
                for (const auto& hudPair : res.hud_cameras) {
                    if (hudPair.second.get() == camera) {
                        auto itMapped = res.view_ids.find(hudPair.first);
                        if (itMapped != res.view_ids.end()) {
                            statsViewId = itMapped->second;
                        }
                        break;
                    }
                }
            }
        }
    }

    // Keep the same relative anchor per window/context instead of using window[0] dimensions.
    {
        int width = 0;
        int height = 0;
        if (auto* cam = renderInfo.getCurrentCamera()) {
            if (const auto* vp = cam->getViewport()) {
                width = static_cast<int>(vp->width());
                height = static_cast<int>(vp->height());
            }
        }
        if (width > 0 && height > 0) {
            constexpr float marginX = 12.0f;
            constexpr float marginY = 12.0f;
            constexpr float labelColumnOffset = 120.0f;
            const osg::Vec3 labelPos(static_cast<float>(width) - marginX - labelColumnOffset,
                                     static_cast<float>(height) - marginY,
                                     0.0f);
            const osg::Vec3 valuePos(static_cast<float>(width) - marginX,
                                     static_cast<float>(height) - marginY,
                                     0.0f);
            if (_label.valid()) {
                _label->setPosition(labelPos);
            }
            if (_values.valid()) {
                _values->setPosition(valuePos);
            }
        }
    }

    {
        const bool haveState = (renderInfo.getState() != nullptr) || (camera != nullptr);

        if (haveState && _values.valid() && _plugin)
        {
            _renderer->updateLiveTimingFromRenderInfo(renderInfo, ctx);
            const uint64_t frameNo = frameNumberFromRenderInfo(renderInfo);

            std::stringstream label_ss;
            std::stringstream value_ss;

            double fpsAvg = 0.0;
            double frameTotalMs = -1.0;
            if (auto *vs = opencover::VRViewer::instance()->getViewerStats())
            {
                (void)vs->getAveragedAttribute("Frame rate", fpsAvg);
                double frameDurationAvg = 0.0;
                if (vs->getAveragedAttribute("Frame duration", frameDurationAvg) && frameDurationAvg > 0.0) {
                    frameTotalMs = frameDurationAvg * 1000.0;
                }
            }
            if (fpsAvg <= 0.0)
            {
                const double fd = std::max(1e-6, opencover::cover->frameDuration());
                fpsAvg = 1.0 / fd;
                frameTotalMs = fd * 1000.0;
            }
            if (!LamureUtil::isValidValue(frameTotalMs)) {
                const double fd = opencover::cover->frameDuration();
                if (fd > 0.0) frameTotalMs = fd * 1000.0;
            }

            const bool showSummedMetrics = (statsViewId == 0);
            LamureRenderer::ContextTimingSample ctxTiming{};
            LamureRenderer::TimingSnapshot timingSum{};
            (void)_renderer->getDisplayTimingData(ctx, frameNo, ctxTiming, timingSum);

            const double primMio = static_cast<double>(ctxTiming.rendered_primitives) / 1e6;
            const double primPerSecMio = primMio * fpsAvg;
            const double sumPrimMio = static_cast<double>(timingSum.rendered_primitives) / 1e6;
            const double sumPrimPerSecMio = sumPrimMio * fpsAvg;
            auto fmt = [](double v, int prec, double scale = 1.0) -> std::string {
                if (!LamureUtil::isValidValue(v)) return std::string("-");
                char buf[32];
                snprintf(buf, sizeof(buf), "%.*f", prec, v * scale);
                return buf;
            };
            auto fmtMs      = [&](double v) { return fmt(v, 2); };
            auto fmtRatio   = [&](double v) { return fmt(v, 2); };
            auto fmtPercent = [&](double v) { return fmt(v, 1, 100.0); };
            auto fmtMio     = [&](double v) { return fmt(v, 2, 1.0 / 1e6); };

            label_ss
                << "FPS:" << "\n"
                << "LOD Error:" << "\n"
                << "Nodes:" << "\n"
                << "Primitives (Mio):" << "\n"
                << "Primitives / s (Mio):" << "\n"
                << "Coverage (%):" << "\n"
                << "Overdraw (x):" << "\n"
                << "Samples Passed (Mio):" << "\n"
                << "Covered Pixels (Mio):" << "\n"
                << "Boxes:" << "\n"
                << "Frame Total (ms):" << "\n"
                << "CPU Update (ms):" << "\n"
                << "Wait (ms):" << "\n";

            label_ss
                << "\n"
                << "Context Metrics:" << "\n"
                << "Dispatch (ms):" << "\n"
                << "Context (ms):" << "\n"
                << "Cull (ms):" << "\n"
                << "Draw (ms):" << "\n"
                << "GPU (ms):" << "\n"
                << "Render CPU (ms):" << "\n";

            if (showSummedMetrics) {
                label_ss
                    << "\n"
                    << "Summed Metrics:" << "\n"
                    << "Sum Nodes:" << "\n"
                    << "Sum Primitives (Mio):" << "\n"
                    << "Sum Primitives / s (Mio):" << "\n"
                    << "Sum Boxes:" << "\n"
                    << "Sum Coverage (%):" << "\n"
                    << "Sum Overdraw (x):" << "\n"
                    << "Sum Samples Passed (Mio):" << "\n"
                    << "Sum Covered Pixels (Mio):" << "\n"
                    << "Sum Dispatch (ms):" << "\n"
                    << "Sum Context (ms):" << "\n"
                    << "Sum Render CPU (ms):" << "\n"
                    << "Sum GPU (ms):" << "\n";
            }

            value_ss << "\n"
                << std::fixed << std::setprecision(2)
                << fpsAvg << "\n"
                << _plugin->getSettings().lod_error << "\n"
                << ctxTiming.rendered_nodes << "\n"
                << primMio << "\n"
                << primPerSecMio << "\n"
                << fmtPercent(ctxTiming.coverage) << "\n"
                << fmtRatio(ctxTiming.overdraw) << "\n"
                << fmtMio(ctxTiming.samples_passed) << "\n"
                << fmtMio(ctxTiming.covered_samples) << "\n"
                << ctxTiming.rendered_bounding_boxes << "\n"
                << fmtMs(frameTotalMs) << "\n"
                << fmtMs(timingSum.cpu_update_ms) << "\n"
                << fmtMs(timingSum.wait_ms) << "\n";

            value_ss
                << "\n"
                << "\n"
                << fmtMs(ctxTiming.dispatch_ms) << "\n"
                << fmtMs(ctxTiming.context_update_ms) << "\n"
                << fmtMs(ctxTiming.cpu_cull_ms) << "\n"
                << fmtMs(ctxTiming.cpu_draw_ms) << "\n"
                << fmtMs(ctxTiming.gpu_ms) << "\n"
                << fmtMs(ctxTiming.render_cpu_ms) << "\n";

            if (showSummedMetrics) {
                value_ss
                    << "\n"
                    << "\n"
                    << timingSum.rendered_nodes << "\n"
                    << sumPrimMio << "\n"
                    << sumPrimPerSecMio << "\n"
                    << timingSum.rendered_bounding_boxes << "\n"
                    << fmtPercent(timingSum.coverage) << "\n"
                    << fmtRatio(timingSum.overdraw) << "\n"
                    << fmtMio(timingSum.samples_passed) << "\n"
                    << fmtMio(timingSum.covered_samples) << "\n"
                    << fmtMs(timingSum.dispatch_ms) << "\n"
                    << fmtMs(timingSum.context_update_ms) << "\n"
                    << fmtMs(timingSum.render_cpu_ms) << "\n"
                    << fmtMs(timingSum.gpu_ms) << "\n";
            }

            if (_label.valid()) {
                _label->setText(label_ss.str(), osgText::String::ENCODING_UTF8);
            }
            _values->setText(value_ss.str(), osgText::String::ENCODING_UTF8);
        }
    }
    if (drawable) { drawable->drawImplementation(renderInfo); }
}

FrustumDrawCallback::FrustumDrawCallback(Lamure* plugin)
    : _plugin(plugin)
    , _renderer(plugin ? plugin->getRenderer() : nullptr)
{
}

void FrustumDrawCallback::drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
{
    if (!_plugin || !_renderer) {
        return;
    }

    const int ctx = renderInfo.getContextID();
    const osg::Camera* camera = renderInfo.getCurrentCamera();
    auto& res = _renderer->getResources(ctx);
    std::shared_ptr<lamure::ren::camera> scmCameraHolder;
    GLuint frustumVao = 0;
    GLuint frustumVbo = 0;
    GLuint frustumIbo = 0;
    GLuint lineProgram = 0;
    GLint mvpLocation = -1;
    GLint colorLocation = -1;
    GLsizei frustumIndexCount = 0;
    {
        std::lock_guard<std::mutex> callbackLock(res.callback_mutex);
        if (!resolveMappedCamera(camera, res.view_ids, res.scm_cameras, scmCameraHolder)) return;
        frustumVao = res.geo_frustum.vao;
        frustumVbo = res.geo_frustum.vbo;
        frustumIbo = res.geo_frustum.ibo;
        lineProgram = res.sh_line.program;
        mvpLocation = res.sh_line.mvp_matrix_location;
        colorLocation = res.sh_line.in_color_location;
        frustumIndexCount = static_cast<GLsizei>(res.frustum_idx.size());
    }

    lamure::ren::camera* scmCamera = scmCameraHolder.get();
    if (!_renderer->gpuOrganizationReady() || !renderInfo.getState())
        return;
    if (!scmCamera || lineProgram == 0 || frustumVao == 0 || frustumVbo == 0 || frustumIbo == 0 || frustumIndexCount <= 0)
        return;

    GLint prevProgram = 0;
    GLint prevVao = 0;
    GLint prevArrayBuffer = 0;
    GLint prevElementBuffer = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArrayBuffer);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &prevElementBuffer);

    const auto corner_values = scmCamera->get_frustum_corners();
    std::array<float, 24> frustumVertices{};
    const size_t corner_count = (std::min)(corner_values.size(), frustumVertices.size() / 3);
    for (size_t i = 0; i < corner_count; ++i) {
        const auto vv = scm::math::vec3f(corner_values[i]);
        frustumVertices[i * 3 + 0] = vv.x;
        frustumVertices[i * 3 + 1] = vv.y;
        frustumVertices[i * 3 + 2] = vv.z;
    }

    osg::Matrixd view_osg, proj_osg;
    _renderer->getMatricesFromRenderInfo(renderInfo, view_osg, proj_osg);
    
    scm::math::mat4f view = LamureUtil::matConv4F(view_osg);
    scm::math::mat4f proj = LamureUtil::matConv4F(proj_osg);
    const scm::math::mat4f mvp_matrix = proj * view;

    glLineWidth(1);
    glBindVertexArray(frustumVao);
    glBindBuffer(GL_ARRAY_BUFFER, frustumVbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, frustumIbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * frustumVertices.size(), frustumVertices.data());

    glUseProgram(lineProgram);
    if (mvpLocation >= 0)
        glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, mvp_matrix.data_array);
    const auto& frustumColor = _plugin->getSettings().frustum_color;
    if (colorLocation >= 0 && frustumColor.size() >= 4) {
        glUniform4f(colorLocation, frustumColor[0], frustumColor[1], frustumColor[2], frustumColor[3]);
    }
    glDrawElements(GL_LINES, frustumIndexCount, GL_UNSIGNED_SHORT, nullptr);

    glUseProgram(static_cast<GLuint>(prevProgram));
    glBindVertexArray(static_cast<GLuint>(prevVao));
    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(prevArrayBuffer));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLuint>(prevElementBuffer));
}

void LamureRenderer::syncEditBrushGeometry()
{
    if (!m_edit_brush_geode)
        return;

    m_edit_brush_geode->removeDrawables(0, m_edit_brush_geode->getNumDrawables());

    osg::ref_ptr<osg::Shape> shape;
    shape = new osg::Sphere(osg::Vec3(), 1.0f);

    osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(shape.get());
    drawable->setName("EditBrushDrawable");
    drawable->setColor(osg::Vec4(1.0f, 0.5f, 0.1f, 0.3f));
    auto* ss = drawable->getOrCreateStateSet();
    ss->setMode(GL_BLEND, osg::StateAttribute::ON);
    ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    ss->setAttributeAndModes(new osg::BlendFunc(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA));

    osg::ref_ptr<osg::Depth> depth = new osg::Depth;
    depth->setWriteMask(false);
    ss->setAttributeAndModes(depth.get(), osg::StateAttribute::ON);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);

    m_edit_brush_geode->addDrawable(drawable.get());
}

osg::MatrixTransform* LamureRenderer::ensureEditBrushNode(osg::Group* parent)
{
    if (!m_edit_brush_transform) {
        m_edit_brush_transform = new osg::MatrixTransform();
        m_edit_brush_transform->setName("EditBrushTransform");
    }
    if (parent && m_edit_brush_transform->getNumParents() == 0) {
        parent->addChild(m_edit_brush_transform.get());
    }

    if (!m_edit_brush_geode) {
        m_edit_brush_geode = new osg::Geode();
        m_edit_brush_transform->addChild(m_edit_brush_geode.get());
    } else if (m_edit_brush_geode->getNumParents() == 0) {
        m_edit_brush_transform->addChild(m_edit_brush_geode.get());
    }

    // Brush should be visible but not intercept pick/intersection.
    const unsigned int mask = ~opencover::Isect::Intersection & ~opencover::Isect::Pick;
    m_edit_brush_transform->setNodeMask(m_edit_brush_transform->getNodeMask() & mask);
    m_edit_brush_geode->setNodeMask(m_edit_brush_geode->getNodeMask() & mask);

    syncEditBrushGeometry();
    return m_edit_brush_transform.get();
}

void LamureRenderer::destroyEditBrushNode(osg::Group* parent)
{
    if (parent && m_edit_brush_transform) {
        parent->removeChild(m_edit_brush_transform.get());
    }
    m_edit_brush_geode = nullptr;
    m_edit_brush_transform = nullptr;
}

osg::Matrixd LamureRenderer::composeEditBrushMatrix(const osg::Matrixd& interactorMat,
                                                    const osg::Matrixd& parentWorld,
                                                    const osg::Matrixd& invRoot,
                                                    double hullScale) const
{
    osg::Matrixd brushMat;
    brushMat.makeScale(osg::Vec3d(hullScale, hullScale, hullScale));
    brushMat.postMult(interactorMat);
    brushMat.postMult(parentWorld);
    brushMat.postMult(invRoot);
    return brushMat;
}

osg::Matrixd LamureRenderer::updateEditBrushFromInteractor(const osg::Matrixd& interactorMat,
                                                           const osg::Matrixd& invRoot,
                                                           double hullScale)
{
    osg::Matrixd parentWorld;
    parentWorld.makeIdentity();
    if (auto* objScale = opencover::cover->getObjectsScale()) {
        osg::MatrixList parentList = objScale->getWorldMatrices();
        if (!parentList.empty())
            parentWorld = parentList.front();
    }
    osg::Matrixd brushMat = composeEditBrushMatrix(interactorMat, parentWorld, invRoot, hullScale);
    setEditBrushTransform(brushMat);
    return brushMat;
}

void LamureRenderer::setEditBrushTransform(const osg::Matrixd& brushMat)
{
    if (m_edit_brush_transform)
        m_edit_brush_transform->setMatrix(brushMat);
    syncEditBrushGeometry();
}

void LamureRenderer::init()
{
    if (notifyOn()) { std::cout << "[Lamure] LamureRenderer::init()" << std::endl; }
    std::lock_guard<std::mutex> sceneLock(m_sceneMutex);

    if (auto* vs = opencover::VRViewer::instance()->getViewerStats()) {
        vs->collectStats("frame_rate", true);
    }
    ensureViewerCameraStatsEnabled(opencover::VRViewer::instance());

    m_init_geode = new osg::Geode();
    m_init_geode->setName("InitGeode");
    m_init_geode->setCullingActive(false);
    m_init_stateset = new osg::StateSet();
    m_init_geode->setStateSet(m_init_stateset.get());
    m_plugin->getGroup()->addChild(m_init_geode);
    m_init_geometry = new osg::Geometry();
    m_init_geometry->setName("InitGeometry");
    m_init_geometry->setUseDisplayList(false);
    m_init_geometry->setUseVertexBufferObjects(true);
    m_init_geometry->setUseVertexArrayObject(false);
    m_init_geometry->setCullingActive(false);
    m_init_geometry->setDrawCallback(new InitDrawCallback(m_plugin));
    if (auto *stateSet = m_init_geometry->getOrCreateStateSet()) {
        stateSet->setRenderBinDetails(-10, "RenderBin");
    }
    m_init_geode->addDrawable(m_init_geometry);

    m_dispatch_geode = new osg::Geode();
    m_dispatch_geode->setName("DispatchGeode");
    m_dispatch_geode->setCullingActive(false);
    m_dispatch_stateset = new osg::StateSet();
    m_dispatch_stateset->setRenderBinDetails(-5, "RenderBin");
    m_dispatch_geode->setStateSet(m_dispatch_stateset.get());
    m_plugin->getGroup()->addChild(m_dispatch_geode);
    m_dispatch_geometry = new osg::Geometry();
    m_dispatch_geometry->setName("DispatchGeometry");
    m_dispatch_geometry->setUseDisplayList(false);
    m_dispatch_geometry->setUseVertexBufferObjects(true);
    m_dispatch_geometry->setUseVertexArrayObject(false);
    m_dispatch_geometry->setCullingActive(false);
    m_dispatch_geometry->setDrawCallback(new DispatchDrawCallback(m_plugin));
    m_dispatch_geode->addDrawable(m_dispatch_geometry);

    if (notifyOn()) { std::cout << "[Lamure] StatsGeode()" << std::endl; }
    m_stats_geode = new osg::Geode();
    m_stats_geode->setName("StatsGeode");
    m_stats_geode->setCullingActive(false);
    {
        osg::Vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
        std::string font = opencover::coVRFileManager::instance()->getFontFile(NULL);
        float characterSize = 18.0f;
        const osg::GraphicsContext::Traits* traits = opencover::coVRConfig::instance()->windows[0].context->getTraits();
        const float marginX = 12.f, marginY = 12.f;
        const float labelColumnOffset = 120.f;
        osg::Vec3 pos_label(traits->width - marginX - labelColumnOffset, traits->height - marginY, 0.0f);
        osg::Vec3 pos_value(traits->width - marginX,                      traits->height - marginY, 0.0f);
        osg::ref_ptr<osgText::Text> label = new osgText::Text();
        label->setName("LabelStats");
        label->setColor(osg::Vec4(1.f, 1.f, 1.f, 0.95f));
        label->setBackdropType(osgText::Text::OUTLINE);
        label->setBackdropColor(osg::Vec4(0.f, 0.f, 0.f, 0.9f));
        label->setFont(font);
        label->setCharacterSizeMode(osgText::TextBase::SCREEN_COORDS);
        label->setCharacterSize(characterSize);
        label->setAlignment(osgText::TextBase::RIGHT_TOP);
        label->setAutoRotateToScreen(false);
        label->setDataVariance(osg::Object::DYNAMIC);
        label->setCullingActive(false);
        label->setPosition(pos_label);
        std::stringstream label_ss;
        label_ss << "FPS:" << "\n"
            << "LOD Error:" << "\n"
            << "Nodes:" << "\n"
            << "Primitives (Mio):" << "\n"
            << "Primitives / s (Mio):" << "\n"
            << "Coverage (%):" << "\n"
            << "Overdraw (x):" << "\n"
            << "Samples Passed (Mio):" << "\n"
            << "Covered Pixels (Mio):" << "\n"
            << "Boxes:" << "\n"
            << "Frame Total (ms):" << "\n"
            << "CPU Update (ms):" << "\n"
            << "Wait (ms):" << "\n"
            << "\n"
            << "Context Metrics:" << "\n"
            << "Dispatch (ms):" << "\n"
            << "Context (ms):" << "\n"
            << "Cull (ms):" << "\n"
            << "Draw (ms):" << "\n"
            << "GPU (ms):" << "\n"
            << "Render CPU (ms):" << "\n"
            << "\n"
            << "Summed Metrics:" << "\n"
            << "Sum Nodes:" << "\n"
            << "Sum Primitives (Mio):" << "\n"
            << "Sum Primitives / s (Mio):" << "\n"
            << "Sum Boxes:" << "\n"
            << "Sum Coverage (%):" << "\n"
            << "Sum Overdraw (x):" << "\n"
            << "Sum Samples Passed (Mio):" << "\n"
            << "Sum Covered Pixels (Mio):" << "\n"
            << "Sum Dispatch (ms):" << "\n"
            << "Sum Context (ms):" << "\n"
            << "Sum Render CPU (ms):" << "\n"
            << "Sum GPU (ms):" << "\n";
        label->setText(label_ss.str(), osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> value = new osgText::Text();
        value->setName("ValueStats");
        value->setColor(osg::Vec4(1.f, 1.f, 1.f, 0.95f));
        value->setBackdropType(osgText::Text::OUTLINE);
        value->setBackdropColor(osg::Vec4(0.f, 0.f, 0.f, 0.9f));
        value->setFont(font);
        value->setCharacterSizeMode(osgText::TextBase::SCREEN_COORDS);
        value->setCharacterSize(characterSize);
        value->setAlignment(osgText::TextBase::RIGHT_TOP);
        value->setAutoRotateToScreen(false);
        value->setDataVariance(osg::Object::DYNAMIC);
        value->setCullingActive(false);
        value->setPosition(pos_value);
        std::stringstream value_ss;
        value_ss << "\n"
            << "0.00" << "\n"
            << "0.00" << "\n"
            << "0" << "\n"
            << "0.00" << "\n"
            << "0.00" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "0" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "\n"
            << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "\n"
            << "\n"
            << "0" << "\n"
            << "0.00" << "\n"
            << "0.00" << "\n"
            << "0" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n"
            << "-" << "\n";
        value->setText(value_ss.str(), osgText::String::ENCODING_UTF8);
        m_stats_geode->addDrawable(label.get());
        m_stats_geode->addDrawable(value.get());
        label->setDrawCallback(new StatsTextDrawLockCallback());
        value->setDrawCallback(new StatsDrawCallback(m_plugin, label.get(), value.get()));
    }
    m_stats_geode->setNodeMask(0x0);

    m_frustum_geode      = new osg::Geode();
    m_frustum_geode->setName("FrustumGeode");
    m_frustum_geometry   = new osg::Geometry();
    m_frustum_geometry->setName("FrustumGeometry");
    m_frustum_geometry->setUseDisplayList(false);
    m_frustum_geometry->setUseVertexBufferObjects(true);
    m_frustum_geometry->setUseVertexArrayObject(false);
    m_frustum_geometry->setCullingActive(false);
    m_frustum_geometry->setDrawCallback(new FrustumDrawCallback(m_plugin));
    m_frustum_geode->addDrawable(m_frustum_geometry.get());
    m_frustum_group = new osg::Group();
    m_frustum_group->setName("FrustumGroup");
    const bool show_frustum = m_plugin->getSettings().show_frustum;
    m_frustum_group->setNodeMask(show_frustum ? 0xFFFFFFFF : 0x0);
    m_frustum_group->addChild(m_frustum_geode.get());

    m_edit_brush_transform = new osg::MatrixTransform();
    m_edit_brush_transform->setName("EditBrushTransform");
    m_edit_brush_geode = new osg::Geode();
    m_edit_brush_geode->setName("EditBrushGeode");
    m_edit_brush_transform->addChild(m_edit_brush_geode.get());

    m_stats_stateset = new osg::StateSet();
    m_stats_stateset->setRenderBinDetails(1000, "RenderBin");
    m_stats_stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    m_stats_stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
    m_stats_stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
    m_stats_stateset->setMode(GL_CULL_FACE, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
    m_stats_stateset->setMode(GL_BLEND, osg::StateAttribute::ON | osg::StateAttribute::PROTECTED);
    m_stats_stateset->setAttributeAndModes(
        new osg::BlendFunc(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA),
        osg::StateAttribute::ON | osg::StateAttribute::PROTECTED);
    osg::ref_ptr<osg::Depth> statsDepth = new osg::Depth();
    statsDepth->setFunction(osg::Depth::ALWAYS);
    statsDepth->setWriteMask(false);
    m_stats_stateset->setAttributeAndModes(statsDepth.get(), osg::StateAttribute::ON | osg::StateAttribute::PROTECTED);
    m_stats_geode->setStateSet(m_stats_stateset.get());

    m_frustum_stateset = new osg::StateSet();
    m_frustum_stateset->setRenderBinDetails(10, "RenderBin");
    m_frustum_geode->setStateSet(m_frustum_stateset.get());
    m_frustum_geode->setCullingActive(false);
    
    auto ui = m_plugin->getUI();

    const bool show_stats = m_plugin->getSettings().show_stats;
    ui->getPointcloudButton()->setState(   m_plugin->getSettings().show_pointcloud );
    ui->getBoundingboxButton()->setState(  m_plugin->getSettings().show_boundingbox );
    ui->getFrustumButton()->setState(      m_plugin->getSettings().show_frustum );
    ui->getStatsButton()->setState(         show_stats );
    ui->getSyncButton()->setState(         m_plugin->getSettings().show_sync );
    ui->getNotifyButton()->setState(       m_plugin->getSettings().show_notify );
    if (m_stats_geode.valid()) {
        m_stats_geode->setNodeMask(show_stats ? 0xFFFFFFFF : 0x0);
    }

    ui->getPointcloudButton()->setCallback([this](bool state) {
        if (!m_plugin) return;
        m_plugin->setShowPointcloud(state);

        if (!state) {
            m_plugin->getRenderInfo().rendered_primitives = 0;
            m_plugin->getRenderInfo().rendered_nodes = 0;
        }
    });

    ui->getBoundingboxButton()->setCallback([this](bool state) {
        if (!m_plugin) return;
        m_plugin->setShowBoundingbox(state);

        if (!state) {
            m_plugin->getRenderInfo().rendered_bounding_boxes = 0;
        }
    });

    ui->getFrustumButton()->setCallback([this](bool state) {
        if (!m_plugin) return;
        m_plugin->getSettings().show_frustum = state;
        if (m_frustum_group.valid()) {
            m_frustum_group->setNodeMask(state ? 0xFFFFFFFF : 0x0);
        }
    });

    ui->getStatsButton()->setCallback([this](bool state) {
        if (!m_plugin) return;
        m_plugin->getSettings().show_stats = state;
        if (m_stats_geode.valid()) {
            m_stats_geode->setNodeMask(state ? 0xFFFFFFFF : 0x0);
        }
    });
    ui->getSyncButton()->setCallback([this](bool state) {
        if (!m_plugin) return;
        m_plugin->getSettings().show_sync = state;
        if (notifyOn()) {
            std::cout << "[Lamure] Sync " << (state ? "on" : "off") << std::endl;
        }
    });
    ui->getDumpButton()->setCallback([this, ui](bool state) {
        if (!m_plugin) return;
        if (!state) return;
        m_plugin->dumpModelParentChains();
    });

    m_plugin->getGroup()->addChild(m_edit_brush_transform.get());
    m_plugin->getGroup()->addChild(m_frustum_group.get());
}

void LamureRenderer::detachCallbacks()
{
    if (m_init_geometry.valid())        m_init_geometry->setDrawCallback(nullptr);
    if (m_dispatch_geometry.valid())    m_dispatch_geometry->setDrawCallback(nullptr);
    if (m_frustum_geometry.valid())     m_frustum_geometry->setDrawCallback(nullptr);
    if (m_stats_geode.valid()) {
        const unsigned int numDrawables = m_stats_geode->getNumDrawables();
        for (unsigned int i = 0; i < numDrawables; ++i) {
            if (auto* drawable = m_stats_geode->getDrawable(i))
                drawable->setDrawCallback(nullptr);
        }
    }
}

void LamureRenderer::syncHudCameras()
{
    if (!m_stats_geode.valid())
        return;

    auto* viewer = opencover::VRViewer::instance();
    if (!viewer)
        return;

    osgViewer::ViewerBase::Cameras cameras;
    viewer->getCameras(cameras, true);

    for (osg::Camera* cam : cameras) {
        if (!cam)
            continue;

        osg::GraphicsContext* gc = cam->getGraphicsContext();
        if (!gc)
            continue;

        osg::State* state = gc->getState();
        if (!state)
            continue;

        const int ctx = static_cast<int>(state->getContextID());
        if (ctx < 0)
            continue;

        auto& res = getResources(ctx);
        auto scmIt = res.scm_cameras.find(cam);
        if (scmIt == res.scm_cameras.end() || !scmIt->second)
            continue;

        const osg::Viewport* vp = cam->getViewport();
        const osg::GraphicsContext::Traits* traits = gc->getTraits();
        const int W = vp ? static_cast<int>(vp->width()) : (traits ? traits->width : 0);
        const int H = vp ? static_cast<int>(vp->height()) : (traits ? traits->height : 0);
        if (W <= 0 || H <= 0)
            continue;

        auto hudIt = res.hud_cameras.find(cam);
        osg::ref_ptr<osg::Camera> hudCamera = (hudIt != res.hud_cameras.end()) ? hudIt->second : nullptr;
        if (!hudCamera.valid()) {
            hudCamera = new osg::Camera();
            hudCamera->setName("hud_camera");
            hudCamera->setProjectionResizePolicy(osg::Camera::FIXED);
            hudCamera->setRenderOrder(osg::Camera::POST_RENDER, 10);
            hudCamera->setClearMask(0);
            hudCamera->setAllowEventFocus(false);
            hudCamera->setCullingActive(false);
            hudCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
            cam->addChild(hudCamera.get());
            res.hud_cameras[cam] = hudCamera;
        }

        hudCamera->setGraphicsContext(gc);
        hudCamera->setViewport(0, 0, W, H);
        hudCamera->setViewMatrix(osg::Matrix::identity());
        hudCamera->setProjectionMatrix(osg::Matrix::ortho2D(0.0, static_cast<double>(W), 0.0, static_cast<double>(H)));

        bool hasStatsGeode = false;
        for (unsigned i = 0; i < hudCamera->getNumChildren(); ++i) {
            if (hudCamera->getChild(i) == m_stats_geode.get()) {
                hasStatsGeode = true;
                break;
            }
        }
        if (!hasStatsGeode) {
            hudCamera->addChild(m_stats_geode.get());
        }
    }
}

void LamureRenderer::shutdown()
{
    detachCallbacks();

    // Wait for pools to drain before shutdown to avoid races
    if (auto* ctrl = lamure::ren::controller::get_instance()) {
        //if (m_osg_camera.valid() && m_osg_camera->getGraphicsContext() && m_osg_camera->getGraphicsContext()->getState()) {
        //    lamure::context_t ctx = m_osg_camera->getGraphicsContext()->getState()->getContextID();
        //    ctrl->wait_for_idle(ctx);
        //}
        ctrl->shutdown_pools();
    }
    if (auto* cache = lamure::ren::ooc_cache::get_instance()) {
        cache->wait_for_idle();
        cache->shutdown_pool();
    }

    releaseMultipassTargets();

    {
        std::scoped_lock lock(m_renderMutex, m_ctx_mutex);
        for (auto& kv : m_ctx_res) {
            auto& ctxRes = kv.second;
            for (auto& hudEntry : ctxRes.hud_cameras) {
                osg::ref_ptr<osg::Camera> hudCamera = hudEntry.second;
                if (!hudCamera.valid())
                    continue;
                while (hudCamera->getNumParents() > 0) {
                    osg::Group* parent = hudCamera->getParent(hudCamera->getNumParents() - 1);
                    if (!parent)
                        break;
                    parent->removeChild(hudCamera.get());
                }
                hudCamera->removeChildren(0, hudCamera->getNumChildren());
            }
            ctxRes.hud_cameras.clear();
            ctxRes.scm_cameras.clear();
            ctxRes.view_ids.clear();
        }
        //this crashes because the gpu_context has not been deleted yet
        //m_ctx_res.clear();
    }
    {
        std::lock_guard<std::mutex> lock(m_timing_mutex);
        m_timing_frames.clear();
        m_timing_global_by_frame.clear();
        m_timing_latest_frame_by_ctx.clear();
        m_last_complete_timing_frame = kInvalidTimingFrame;
        m_live_timing_last_scanned_frame = kInvalidTimingFrame;
    }
    m_stats_initialized = false;
    m_gpu_org_ready.store(false, std::memory_order_release);


    if (m_plugin && m_plugin->getGroup()) {
        if (m_init_geode.valid())       m_plugin->getGroup()->removeChild(m_init_geode);
        if (m_dispatch_geode.valid())   m_plugin->getGroup()->removeChild(m_dispatch_geode);
        if (m_frustum_group.valid())    m_plugin->getGroup()->removeChild(m_frustum_group);
    }

    if (m_stats_geode.valid()) {
        while (m_stats_geode->getNumParents() > 0) {
            osg::Group* parent = m_stats_geode->getParent(m_stats_geode->getNumParents() - 1);
            if (!parent)
                break;
            parent->removeChild(m_stats_geode.get());
        }
    }

    m_init_geode = nullptr;
    m_dispatch_geode = nullptr;
    m_stats_geode = nullptr;
    m_frustum_geode = nullptr;
    m_frustum_group = nullptr;
    m_init_stateset = nullptr;
    m_dispatch_stateset = nullptr;
    m_stats_stateset = nullptr;
    m_frustum_stateset = nullptr;
    m_init_geometry = nullptr;
    m_dispatch_geometry = nullptr;
    m_frustum_geometry = nullptr;
    m_osg_camera = nullptr;

    lamure::ren::controller::destroy_instance();
    lamure::ren::ooc_cache::destroy_instance();
    lamure::ren::cut_database::destroy_instance();
    lamure::ren::model_database::destroy_instance();
}

bool LamureRenderer::beginFrame(int ctxId)
{
    auto& res = getResources(ctxId);
    std::unique_lock<std::mutex> lock(m_renderMutex);

    if (!res.rendering_allowed)
        return false;

    // We only track that THIS context is active.
    // If it's already active (re-entry), we allow it.
    if (res.rendering) {
       return true;
    }
    
    res.rendering = true;
    return true;
}

void LamureRenderer::endFrame(int ctxId)
{
    bool performFinish = false;
    {
        auto& res = getResources(ctxId);
        
        std::lock_guard<std::mutex> lock(m_renderMutex);
        res.rendering = false;

        if (m_pauseRequested)
        {
            if (res.frames_pending_drain > 0)
            {
                --res.frames_pending_drain;
                performFinish = true;
            }

            bool allDrained = true;
            for (const auto& kv : m_ctx_res) {
                if (kv.second.frames_pending_drain > 0) {
                    allDrained = false;
                    break;
                }
            }
            if (allDrained)
            {
                for (auto& kv : m_ctx_res) {
                    kv.second.rendering_allowed = false;
                }
                m_pauseRequested = false;
            }
        }
    }
    if (performFinish) { glFinish(); }
    m_renderCondition.notify_all();
}

bool LamureRenderer::pauseAndDrainFrames(uint32_t extraDrainFrames)
{
    std::unique_lock<std::mutex> lock(m_renderMutex);

    bool allPaused = true;
    for (const auto& kv : m_ctx_res) {
        if (kv.second.rendering_allowed) { allPaused = false; break; }
    }
    if (allPaused && !m_pauseRequested)
    {
        m_renderCondition.wait(lock, [this]() {
            for (const auto& kv : m_ctx_res) {
                if (kv.second.rendering) return false;
            }
            return true;
        });
        return false;
    }

    if (m_pauseRequested)
    {
        const uint32_t framesToDrain = std::max<uint32_t>(extraDrainFrames, 1u);
        for (auto& kv : m_ctx_res) {
            if (!kv.second.rendering_allowed && kv.second.frames_pending_drain == 0)
                continue;
            if (kv.second.frames_pending_drain < framesToDrain)
                kv.second.frames_pending_drain = framesToDrain;
        }
        return true;
    }

    m_pauseRequested = true;
    uint32_t framesToDrain = std::max<uint32_t>(extraDrainFrames, 1u);
    for (auto& kv : m_ctx_res) {
        if (kv.second.frames_pending_drain < framesToDrain)
            kv.second.frames_pending_drain = framesToDrain;
    }

    // Check if any context is currently rendering
    bool anyRendering = false;
    for (const auto& kv : m_ctx_res) {
        if (kv.second.rendering) { anyRendering = true; break; }
    }

    if (!anyRendering)
    {
        for (auto& kv : m_ctx_res) {
            kv.second.frames_pending_drain = 0;
            kv.second.rendering_allowed = false;
        }
        m_pauseRequested = false;
        
        osg::ref_ptr<osg::GraphicsContext> gc = m_osg_camera.valid() ? m_osg_camera->getGraphicsContext() : nullptr;
        lock.unlock();
        if (gc.valid())
        {
            const bool needMakeCurrent = !gc->isCurrent();
            if (!needMakeCurrent || gc->makeCurrent())
            {
                glFinish();
                if (needMakeCurrent)
                    gc->releaseContext();
            }
        }
        else
        {
            glFinish();
        }
        return true;
    }

    m_renderCondition.wait(lock, [this]() { 
        for (const auto& kv : m_ctx_res) {
            if (kv.second.rendering) return false; 
        }
        return true;
    });

    return true;
}

void LamureRenderer::resumeRendering()
{
    {
        std::lock_guard<std::mutex> lock(m_renderMutex);
        m_pauseRequested = false;
        for (auto& kv : m_ctx_res) {
            kv.second.frames_pending_drain = 0;
            kv.second.rendering_allowed = true;
        }
    }
    m_renderCondition.notify_all();
}


bool LamureRenderer::isRendering() const
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    for (const auto& kv : m_ctx_res) {
        if (kv.second.rendering) return true;
    }
    return false;
}

void LamureRenderer::initUniforms(ContextResources& ctx) 
{
    if (notifyOn()) { std::cout << "[Lamure] LamureRenderer::initUniforms()" << std::endl; }
    glUseProgram(ctx.sh_point.program);
    ctx.sh_point.mvp_matrix_loc             = glGetUniformLocation(ctx.sh_point.program, "mvp_matrix");
    ctx.sh_point.model_matrix_loc           = glGetUniformLocation(ctx.sh_point.program, "model_matrix");
    ctx.sh_point.clip_plane_count_loc       = glGetUniformLocation(ctx.sh_point.program, "clip_plane_count");
    ctx.sh_point.clip_plane_data_loc        = glGetUniformLocation(ctx.sh_point.program, "clip_planes");
    ctx.sh_point.max_radius_loc             = glGetUniformLocation(ctx.sh_point.program, "max_radius");
    ctx.sh_point.min_radius_loc             = glGetUniformLocation(ctx.sh_point.program, "min_radius");
    ctx.sh_point.max_screen_size_loc        = glGetUniformLocation(ctx.sh_point.program, "max_screen_size");
    ctx.sh_point.min_screen_size_loc        = glGetUniformLocation(ctx.sh_point.program, "min_screen_size");
    ctx.sh_point.scale_radius_loc           = glGetUniformLocation(ctx.sh_point.program, "scale_radius");
    ctx.sh_point.scale_projection_loc       = glGetUniformLocation(ctx.sh_point.program, "scale_projection");
    ctx.sh_point.max_radius_cut_loc         = glGetUniformLocation(ctx.sh_point.program, "max_radius_cut");
    ctx.sh_point.scale_radius_gamma_loc     = glGetUniformLocation(ctx.sh_point.program, "scale_radius_gamma");
    ctx.sh_point.proj_col0_loc              = glGetUniformLocation(ctx.sh_point.program, "Pcol0");
    ctx.sh_point.proj_col1_loc              = glGetUniformLocation(ctx.sh_point.program, "Pcol1");
    ctx.sh_point.viewport_half_y_loc        = glGetUniformLocation(ctx.sh_point.program, "viewport_half_y");
    ctx.sh_point.use_aniso_loc              = glGetUniformLocation(ctx.sh_point.program, "use_aniso");
    ctx.sh_point.aniso_normalize_loc        = glGetUniformLocation(ctx.sh_point.program, "aniso_normalize");

    glUseProgram(ctx.sh_point_color.program);
    ctx.sh_point_color.mvp_matrix_loc            = glGetUniformLocation(ctx.sh_point_color.program, "mvp_matrix");
    ctx.sh_point_color.model_matrix_loc          = glGetUniformLocation(ctx.sh_point_color.program, "model_matrix");
    ctx.sh_point_color.clip_plane_count_loc      = glGetUniformLocation(ctx.sh_point_color.program, "clip_plane_count");
    ctx.sh_point_color.clip_plane_data_loc       = glGetUniformLocation(ctx.sh_point_color.program, "clip_planes");
    ctx.sh_point_color.view_matrix_loc           = glGetUniformLocation(ctx.sh_point_color.program, "view_matrix");
    ctx.sh_point_color.normal_matrix_loc         = glGetUniformLocation(ctx.sh_point_color.program, "normal_matrix");
    ctx.sh_point_color.max_radius_loc            = glGetUniformLocation(ctx.sh_point_color.program, "max_radius");
    ctx.sh_point_color.min_radius_loc            = glGetUniformLocation(ctx.sh_point_color.program, "min_radius");
    ctx.sh_point_color.max_screen_size_loc       = glGetUniformLocation(ctx.sh_point_color.program, "max_screen_size");
    ctx.sh_point_color.min_screen_size_loc       = glGetUniformLocation(ctx.sh_point_color.program, "min_screen_size");
    ctx.sh_point_color.max_radius_cut_loc        = glGetUniformLocation(ctx.sh_point_color.program, "max_radius_cut");
    ctx.sh_point_color.scale_radius_gamma_loc    = glGetUniformLocation(ctx.sh_point_color.program, "scale_radius_gamma");
    ctx.sh_point_color.scale_radius_loc          = glGetUniformLocation(ctx.sh_point_color.program, "scale_radius");
    ctx.sh_point_color.scale_projection_loc      = glGetUniformLocation(ctx.sh_point_color.program, "scale_projection");
    ctx.sh_point_color.proj_col0_loc             = glGetUniformLocation(ctx.sh_point_color.program, "Pcol0");
    ctx.sh_point_color.proj_col1_loc             = glGetUniformLocation(ctx.sh_point_color.program, "Pcol1");
    ctx.sh_point_color.viewport_half_y_loc       = glGetUniformLocation(ctx.sh_point_color.program, "viewport_half_y");
    ctx.sh_point_color.use_aniso_loc             = glGetUniformLocation(ctx.sh_point_color.program, "use_aniso");
    ctx.sh_point_color.aniso_normalize_loc       = glGetUniformLocation(ctx.sh_point_color.program, "aniso_normalize");
    ctx.sh_point_color.show_normals_loc          = glGetUniformLocation(ctx.sh_point_color.program, "show_normals");
    ctx.sh_point_color.show_accuracy_loc         = glGetUniformLocation(ctx.sh_point_color.program, "show_accuracy");
    ctx.sh_point_color.show_radius_dev_loc       = glGetUniformLocation(ctx.sh_point_color.program, "show_radius_deviation");
    ctx.sh_point_color.show_output_sens_loc      = glGetUniformLocation(ctx.sh_point_color.program, "show_output_sensitivity");
    ctx.sh_point_color.accuracy_loc              = glGetUniformLocation(ctx.sh_point_color.program, "accuracy");
    ctx.sh_point_color.average_radius_loc        = glGetUniformLocation(ctx.sh_point_color.program, "average_radius");

    glUseProgram(ctx.sh_point_color_lighting.program);
    ctx.sh_point_color_lighting.mvp_matrix_loc          = glGetUniformLocation(ctx.sh_point_color_lighting.program, "mvp_matrix");
    ctx.sh_point_color_lighting.model_matrix_loc        = glGetUniformLocation(ctx.sh_point_color_lighting.program, "model_matrix");
    ctx.sh_point_color_lighting.clip_plane_count_loc    = glGetUniformLocation(ctx.sh_point_color_lighting.program, "clip_plane_count");
    ctx.sh_point_color_lighting.clip_plane_data_loc     = glGetUniformLocation(ctx.sh_point_color_lighting.program, "clip_planes");
    ctx.sh_point_color_lighting.view_matrix_loc         = glGetUniformLocation(ctx.sh_point_color_lighting.program, "view_matrix");
    ctx.sh_point_color_lighting.normal_matrix_loc       = glGetUniformLocation(ctx.sh_point_color_lighting.program, "normal_matrix");
    ctx.sh_point_color_lighting.max_radius_loc          = glGetUniformLocation(ctx.sh_point_color_lighting.program, "max_radius");
    ctx.sh_point_color_lighting.min_radius_loc          = glGetUniformLocation(ctx.sh_point_color_lighting.program, "min_radius");
    ctx.sh_point_color_lighting.max_screen_size_loc     = glGetUniformLocation(ctx.sh_point_color_lighting.program, "max_screen_size");
    ctx.sh_point_color_lighting.min_screen_size_loc     = glGetUniformLocation(ctx.sh_point_color_lighting.program, "min_screen_size");
    ctx.sh_point_color_lighting.scale_radius_loc        = glGetUniformLocation(ctx.sh_point_color_lighting.program, "scale_radius");
    ctx.sh_point_color_lighting.scale_projection_loc    = glGetUniformLocation(ctx.sh_point_color_lighting.program, "scale_projection");
    ctx.sh_point_color_lighting.proj_col0_loc           = glGetUniformLocation(ctx.sh_point_color_lighting.program, "Pcol0");
    ctx.sh_point_color_lighting.proj_col1_loc           = glGetUniformLocation(ctx.sh_point_color_lighting.program, "Pcol1");
    ctx.sh_point_color_lighting.viewport_half_y_loc     = glGetUniformLocation(ctx.sh_point_color_lighting.program, "viewport_half_y");
    ctx.sh_point_color_lighting.use_aniso_loc           = glGetUniformLocation(ctx.sh_point_color_lighting.program, "use_aniso");
    ctx.sh_point_color_lighting.aniso_normalize_loc     = glGetUniformLocation(ctx.sh_point_color_lighting.program, "aniso_normalize");
    ctx.sh_point_color_lighting.max_radius_cut_loc      = glGetUniformLocation(ctx.sh_point_color_lighting.program, "max_radius_cut");
    ctx.sh_point_color_lighting.scale_radius_gamma_loc  = glGetUniformLocation(ctx.sh_point_color_lighting.program, "scale_radius_gamma");
    ctx.sh_point_color_lighting.show_normals_loc        = glGetUniformLocation(ctx.sh_point_color_lighting.program, "show_normals");
    ctx.sh_point_color_lighting.show_accuracy_loc       = glGetUniformLocation(ctx.sh_point_color_lighting.program, "show_accuracy");
    ctx.sh_point_color_lighting.show_radius_dev_loc     = glGetUniformLocation(ctx.sh_point_color_lighting.program, "show_radius_deviation");
    ctx.sh_point_color_lighting.show_output_sens_loc    = glGetUniformLocation(ctx.sh_point_color_lighting.program, "show_output_sensitivity");
    ctx.sh_point_color_lighting.accuracy_loc            = glGetUniformLocation(ctx.sh_point_color_lighting.program, "accuracy");
    ctx.sh_point_color_lighting.average_radius_loc      = glGetUniformLocation(ctx.sh_point_color_lighting.program, "average_radius");
    ctx.sh_point_color_lighting.point_light_pos_vs_loc  = glGetUniformLocation(ctx.sh_point_color_lighting.program, "point_light_pos_vs");
    ctx.sh_point_color_lighting.point_light_intensity_loc   = glGetUniformLocation(ctx.sh_point_color_lighting.program, "point_light_intensity");
    ctx.sh_point_color_lighting.ambient_intensity_loc       = glGetUniformLocation(ctx.sh_point_color_lighting.program, "ambient_intensity");
    ctx.sh_point_color_lighting.specular_intensity_loc      = glGetUniformLocation(ctx.sh_point_color_lighting.program, "specular_intensity");
    ctx.sh_point_color_lighting.shininess_loc               = glGetUniformLocation(ctx.sh_point_color_lighting.program, "shininess");
    ctx.sh_point_color_lighting.gamma_loc                   = glGetUniformLocation(ctx.sh_point_color_lighting.program, "gamma");
    ctx.sh_point_color_lighting.use_tone_mapping_loc        = glGetUniformLocation(ctx.sh_point_color_lighting.program, "use_tone_mapping");

    glUseProgram(ctx.sh_point_prov.program);
    ctx.sh_point_prov.mvp_matrix_loc         = glGetUniformLocation(ctx.sh_point_prov.program, "mvp_matrix");
    ctx.sh_point_prov.model_matrix_loc       = glGetUniformLocation(ctx.sh_point_prov.program, "model_matrix");
    ctx.sh_point_prov.clip_plane_count_loc   = glGetUniformLocation(ctx.sh_point_prov.program, "clip_plane_count");
    ctx.sh_point_prov.clip_plane_data_loc    = glGetUniformLocation(ctx.sh_point_prov.program, "clip_planes");
    ctx.sh_point_prov.max_radius_loc         = glGetUniformLocation(ctx.sh_point_prov.program, "max_radius");
    ctx.sh_point_prov.min_radius_loc         = glGetUniformLocation(ctx.sh_point_prov.program, "min_radius");
    ctx.sh_point_prov.max_screen_size_loc    = glGetUniformLocation(ctx.sh_point_prov.program, "max_screen_size");
    ctx.sh_point_prov.min_screen_size_loc    = glGetUniformLocation(ctx.sh_point_prov.program, "min_screen_size");
    ctx.sh_point_prov.scale_radius_loc       = glGetUniformLocation(ctx.sh_point_prov.program, "scale_radius");
    ctx.sh_point_prov.max_radius_cut_loc     = glGetUniformLocation(ctx.sh_point_prov.program, "max_radius_cut");
    ctx.sh_point_prov.scale_radius_gamma_loc = glGetUniformLocation(ctx.sh_point_prov.program, "scale_radius_gamma");
    ctx.sh_point_prov.scale_projection_loc   = glGetUniformLocation(ctx.sh_point_prov.program, "scale_projection");
    ctx.sh_point_prov.show_normals_loc       = glGetUniformLocation(ctx.sh_point_prov.program, "show_normals");
    ctx.sh_point_prov.show_accuracy_loc      = glGetUniformLocation(ctx.sh_point_prov.program, "show_accuracy");
    ctx.sh_point_prov.show_radius_dev_loc    = glGetUniformLocation(ctx.sh_point_prov.program, "show_radius_deviation");
    ctx.sh_point_prov.show_output_sens_loc   = glGetUniformLocation(ctx.sh_point_prov.program, "show_output_sensitivity");
    ctx.sh_point_prov.accuracy_loc           = glGetUniformLocation(ctx.sh_point_prov.program, "accuracy");
    ctx.sh_point_prov.average_radius_loc     = glGetUniformLocation(ctx.sh_point_prov.program, "average_radius");
    ctx.sh_point_prov.channel_loc            = glGetUniformLocation(ctx.sh_point_prov.program, "channel");
    ctx.sh_point_prov.heatmap_loc            = glGetUniformLocation(ctx.sh_point_prov.program, "heatmap");
    ctx.sh_point_prov.heatmap_min_loc        = glGetUniformLocation(ctx.sh_point_prov.program, "heatmap_min");
    ctx.sh_point_prov.heatmap_max_loc        = glGetUniformLocation(ctx.sh_point_prov.program, "heatmap_max");
    ctx.sh_point_prov.heatmap_min_color_loc  = glGetUniformLocation(ctx.sh_point_prov.program, "heatmap_min_color");
    ctx.sh_point_prov.heatmap_max_color_loc  = glGetUniformLocation(ctx.sh_point_prov.program, "heatmap_max_color");

    glUseProgram(ctx.sh_surfel.program);
    ctx.sh_surfel.mvp_matrix_loc          = glGetUniformLocation(ctx.sh_surfel.program, "mvp_matrix");
    ctx.sh_surfel.model_matrix_loc        = glGetUniformLocation(ctx.sh_surfel.program, "model_matrix");
    ctx.sh_surfel.clip_plane_count_loc    = glGetUniformLocation(ctx.sh_surfel.program, "clip_plane_count");
    ctx.sh_surfel.clip_plane_data_loc     = glGetUniformLocation(ctx.sh_surfel.program, "clip_planes");
    ctx.sh_surfel.max_radius_loc          = glGetUniformLocation(ctx.sh_surfel.program, "max_radius");
    ctx.sh_surfel.min_radius_loc          = glGetUniformLocation(ctx.sh_surfel.program, "min_radius");
    ctx.sh_surfel.max_screen_size_loc     = glGetUniformLocation(ctx.sh_surfel.program, "max_screen_size");
    ctx.sh_surfel.min_screen_size_loc     = glGetUniformLocation(ctx.sh_surfel.program, "min_screen_size");
    ctx.sh_surfel.scale_radius_loc        = glGetUniformLocation(ctx.sh_surfel.program, "scale_radius");
    ctx.sh_surfel.scale_projection_loc    = glGetUniformLocation(ctx.sh_surfel.program, "scale_projection");
    ctx.sh_surfel.max_radius_cut_loc      = glGetUniformLocation(ctx.sh_surfel.program, "max_radius_cut");
    ctx.sh_surfel.scale_radius_gamma_loc  = glGetUniformLocation(ctx.sh_surfel.program, "scale_radius_gamma");
    ctx.sh_surfel.viewport_loc            = glGetUniformLocation(ctx.sh_surfel.program, "viewport");
    ctx.sh_surfel.use_aniso_loc           = glGetUniformLocation(ctx.sh_surfel.program, "use_aniso");

    glUseProgram(ctx.sh_surfel_color.program);
    ctx.sh_surfel_color.mvp_matrix_loc        = glGetUniformLocation(ctx.sh_surfel_color.program, "mvp_matrix");
    ctx.sh_surfel_color.model_matrix_loc      = glGetUniformLocation(ctx.sh_surfel_color.program, "model_matrix");
    ctx.sh_surfel_color.clip_plane_count_loc  = glGetUniformLocation(ctx.sh_surfel_color.program, "clip_plane_count");
    ctx.sh_surfel_color.clip_plane_data_loc   = glGetUniformLocation(ctx.sh_surfel_color.program, "clip_planes");
    ctx.sh_surfel_color.view_matrix_loc       = glGetUniformLocation(ctx.sh_surfel_color.program, "view_matrix");
    ctx.sh_surfel_color.normal_matrix_loc     = glGetUniformLocation(ctx.sh_surfel_color.program, "normal_matrix");
    ctx.sh_surfel_color.max_radius_loc        = glGetUniformLocation(ctx.sh_surfel_color.program, "max_radius");
    ctx.sh_surfel_color.min_radius_loc        = glGetUniformLocation(ctx.sh_surfel_color.program, "min_radius");
    ctx.sh_surfel_color.max_screen_size_loc   = glGetUniformLocation(ctx.sh_surfel_color.program, "max_screen_size");
    ctx.sh_surfel_color.min_screen_size_loc   = glGetUniformLocation(ctx.sh_surfel_color.program, "min_screen_size");
    ctx.sh_surfel_color.scale_radius_loc      = glGetUniformLocation(ctx.sh_surfel_color.program, "scale_radius");
    ctx.sh_surfel_color.max_radius_cut_loc    = glGetUniformLocation(ctx.sh_surfel_color.program, "max_radius_cut");
    ctx.sh_surfel_color.scale_radius_gamma_loc    = glGetUniformLocation(ctx.sh_surfel_color.program, "scale_radius_gamma");
    ctx.sh_surfel_color.viewport_loc          = glGetUniformLocation(ctx.sh_surfel_color.program, "viewport");
    ctx.sh_surfel_color.scale_projection_loc  = glGetUniformLocation(ctx.sh_surfel_color.program, "scale_projection");
    ctx.sh_surfel_color.use_aniso_loc         = glGetUniformLocation(ctx.sh_surfel_color.program, "use_aniso");
    ctx.sh_surfel_color.show_normals_loc      = glGetUniformLocation(ctx.sh_surfel_color.program, "show_normals");
    ctx.sh_surfel_color.show_accuracy_loc     = glGetUniformLocation(ctx.sh_surfel_color.program, "show_accuracy");
    ctx.sh_surfel_color.show_radius_dev_loc   = glGetUniformLocation(ctx.sh_surfel_color.program, "show_radius_deviation");
    ctx.sh_surfel_color.show_output_sens_loc  = glGetUniformLocation(ctx.sh_surfel_color.program, "show_output_sensitivity");
    ctx.sh_surfel_color.accuracy_loc          = glGetUniformLocation(ctx.sh_surfel_color.program, "accuracy");
    ctx.sh_surfel_color.average_radius_loc    = glGetUniformLocation(ctx.sh_surfel_color.program, "average_radius");

    glUseProgram(ctx.sh_surfel_color_lighting.program);
    ctx.sh_surfel_color_lighting.mvp_matrix_loc          = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "mvp_matrix");
    ctx.sh_surfel_color_lighting.model_matrix_loc        = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "model_matrix");
    ctx.sh_surfel_color_lighting.clip_plane_count_loc    = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "clip_plane_count");
    ctx.sh_surfel_color_lighting.clip_plane_data_loc     = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "clip_planes");
    ctx.sh_surfel_color_lighting.view_matrix_loc         = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "view_matrix");
    ctx.sh_surfel_color_lighting.normal_matrix_loc       = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "normal_matrix");
    ctx.sh_surfel_color_lighting.max_radius_loc          = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "max_radius");
    ctx.sh_surfel_color_lighting.min_radius_loc          = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "min_radius");
    ctx.sh_surfel_color_lighting.max_screen_size_loc     = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "max_screen_size");
    ctx.sh_surfel_color_lighting.min_screen_size_loc     = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "min_screen_size");
    ctx.sh_surfel_color_lighting.scale_radius_loc        = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "scale_radius");
    ctx.sh_surfel_color_lighting.max_radius_cut_loc      = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "max_radius_cut");
    ctx.sh_surfel_color_lighting.scale_radius_gamma_loc  = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "scale_radius_gamma");
    ctx.sh_surfel_color_lighting.viewport_loc            = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "viewport");
    ctx.sh_surfel_color_lighting.scale_projection_loc    = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "scale_projection");
    ctx.sh_surfel_color_lighting.use_aniso_loc           = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "use_aniso");
    ctx.sh_surfel_color_lighting.show_normals_loc        = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "show_normals");
    ctx.sh_surfel_color_lighting.show_accuracy_loc       = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "show_accuracy");
    ctx.sh_surfel_color_lighting.show_radius_dev_loc     = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "show_radius_deviation");
    ctx.sh_surfel_color_lighting.show_output_sens_loc    = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "show_output_sensitivity");
    ctx.sh_surfel_color_lighting.accuracy_loc            = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "accuracy");
    ctx.sh_surfel_color_lighting.average_radius_loc      = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "average_radius");
    ctx.sh_surfel_color_lighting.point_light_pos_vs_loc  = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "point_light_pos_vs");
    ctx.sh_surfel_color_lighting.point_light_intensity_loc   = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "point_light_intensity");
    ctx.sh_surfel_color_lighting.ambient_intensity_loc       = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "ambient_intensity");
    ctx.sh_surfel_color_lighting.specular_intensity_loc      = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "specular_intensity");
    ctx.sh_surfel_color_lighting.shininess_loc               = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "shininess");
    ctx.sh_surfel_color_lighting.gamma_loc                   = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "gamma");
    ctx.sh_surfel_color_lighting.use_tone_mapping_loc        = glGetUniformLocation(ctx.sh_surfel_color_lighting.program, "use_tone_mapping");

    glUseProgram(ctx.sh_surfel_prov.program);
    ctx.sh_surfel_prov.mvp_matrix_loc        = glGetUniformLocation(ctx.sh_surfel_prov.program, "mvp_matrix");
    ctx.sh_surfel_prov.model_matrix_loc      = glGetUniformLocation(ctx.sh_surfel_prov.program, "model_matrix");
    ctx.sh_surfel_prov.clip_plane_count_loc  = glGetUniformLocation(ctx.sh_surfel_prov.program, "clip_plane_count");
    ctx.sh_surfel_prov.clip_plane_data_loc   = glGetUniformLocation(ctx.sh_surfel_prov.program, "clip_planes");
    ctx.sh_surfel_prov.max_radius_loc        = glGetUniformLocation(ctx.sh_surfel_prov.program, "max_radius");
    ctx.sh_surfel_prov.min_radius_loc        = glGetUniformLocation(ctx.sh_surfel_prov.program, "min_radius");
    ctx.sh_surfel_prov.min_screen_size_loc   = glGetUniformLocation(ctx.sh_surfel_prov.program, "min_screen_size");
    ctx.sh_surfel_prov.max_screen_size_loc   = glGetUniformLocation(ctx.sh_surfel_prov.program, "max_screen_size");
    ctx.sh_surfel_prov.scale_radius_loc      = glGetUniformLocation(ctx.sh_surfel_prov.program, "scale_radius");
    ctx.sh_surfel_prov.scale_radius_gamma_loc = glGetUniformLocation(ctx.sh_surfel_prov.program, "scale_radius_gamma");
    ctx.sh_surfel_prov.max_radius_cut_loc    = glGetUniformLocation(ctx.sh_surfel_prov.program, "max_radius_cut");
    ctx.sh_surfel_prov.viewport_loc          = glGetUniformLocation(ctx.sh_surfel_prov.program, "viewport");
    ctx.sh_surfel_prov.scale_projection_loc  = glGetUniformLocation(ctx.sh_surfel_prov.program, "scale_projection");
    ctx.sh_surfel_prov.show_normals_loc      = glGetUniformLocation(ctx.sh_surfel_prov.program, "show_normals");
    ctx.sh_surfel_prov.show_accuracy_loc     = glGetUniformLocation(ctx.sh_surfel_prov.program, "show_accuracy");
    ctx.sh_surfel_prov.show_radius_dev_loc   = glGetUniformLocation(ctx.sh_surfel_prov.program, "show_radius_deviation");
    ctx.sh_surfel_prov.show_output_sens_loc  = glGetUniformLocation(ctx.sh_surfel_prov.program, "show_output_sensitivity");
    ctx.sh_surfel_prov.accuracy_loc          = glGetUniformLocation(ctx.sh_surfel_prov.program, "accuracy");
    ctx.sh_surfel_prov.average_radius_loc    = glGetUniformLocation(ctx.sh_surfel_prov.program, "average_radius");
    ctx.sh_surfel_prov.channel_loc           = glGetUniformLocation(ctx.sh_surfel_prov.program, "channel");
    ctx.sh_surfel_prov.heatmap_loc           = glGetUniformLocation(ctx.sh_surfel_prov.program, "heatmap");
    ctx.sh_surfel_prov.heatmap_min_loc       = glGetUniformLocation(ctx.sh_surfel_prov.program, "heatmap_min");
    ctx.sh_surfel_prov.heatmap_max_loc       = glGetUniformLocation(ctx.sh_surfel_prov.program, "heatmap_max");
    ctx.sh_surfel_prov.heatmap_min_color_loc = glGetUniformLocation(ctx.sh_surfel_prov.program, "heatmap_min_color");
    ctx.sh_surfel_prov.heatmap_max_color_loc = glGetUniformLocation(ctx.sh_surfel_prov.program, "heatmap_max_color");

    glUseProgram(ctx.sh_line.program);
    ctx.sh_line.mvp_matrix_location = glGetUniformLocation(ctx.sh_line.program, "mvp_matrix");
    ctx.sh_line.in_color_location   = glGetUniformLocation(ctx.sh_line.program, "in_color");

    glUseProgram(ctx.sh_surfel_pass1.program);
    ctx.sh_surfel_pass1.mvp_matrix_loc         = glGetUniformLocation(ctx.sh_surfel_pass1.program, "mvp_matrix");
    ctx.sh_surfel_pass1.projection_matrix_loc  = glGetUniformLocation(ctx.sh_surfel_pass1.program, "projection_matrix");
    ctx.sh_surfel_pass1.model_view_matrix_loc  = glGetUniformLocation(ctx.sh_surfel_pass1.program, "model_view_matrix");
    ctx.sh_surfel_pass1.model_matrix_loc       = glGetUniformLocation(ctx.sh_surfel_pass1.program, "model_matrix");
    ctx.sh_surfel_pass1.viewport_loc           = glGetUniformLocation(ctx.sh_surfel_pass1.program, "viewport");
    ctx.sh_surfel_pass1.max_radius_loc         = glGetUniformLocation(ctx.sh_surfel_pass1.program, "max_radius");
    ctx.sh_surfel_pass1.min_radius_loc         = glGetUniformLocation(ctx.sh_surfel_pass1.program, "min_radius");
    ctx.sh_surfel_pass1.min_screen_size_loc    = glGetUniformLocation(ctx.sh_surfel_pass1.program, "min_screen_size");
    ctx.sh_surfel_pass1.max_screen_size_loc    = glGetUniformLocation(ctx.sh_surfel_pass1.program, "max_screen_size");
    ctx.sh_surfel_pass1.scale_radius_loc       = glGetUniformLocation(ctx.sh_surfel_pass1.program, "scale_radius");
    ctx.sh_surfel_pass1.scale_projection_loc   = glGetUniformLocation(ctx.sh_surfel_pass1.program, "scale_projection");
    ctx.sh_surfel_pass1.max_radius_cut_loc     = glGetUniformLocation(ctx.sh_surfel_pass1.program, "max_radius_cut");
    ctx.sh_surfel_pass1.scale_radius_gamma_loc = glGetUniformLocation(ctx.sh_surfel_pass1.program, "scale_radius_gamma");
    ctx.sh_surfel_pass1.use_aniso_loc          = glGetUniformLocation(ctx.sh_surfel_pass1.program, "use_aniso");

    glUseProgram(ctx.sh_surfel_pass2.program);
    ctx.sh_surfel_pass2.model_view_matrix_loc = glGetUniformLocation(ctx.sh_surfel_pass2.program, "model_view_matrix");
    ctx.sh_surfel_pass2.projection_matrix_loc = glGetUniformLocation(ctx.sh_surfel_pass2.program, "projection_matrix");
    ctx.sh_surfel_pass2.normal_matrix_loc     = glGetUniformLocation(ctx.sh_surfel_pass2.program, "normal_matrix");
    ctx.sh_surfel_pass2.depth_texture_loc     = glGetUniformLocation(ctx.sh_surfel_pass2.program, "depth_texture");
    ctx.sh_surfel_pass2.viewport_loc          = glGetUniformLocation(ctx.sh_surfel_pass2.program, "viewport");
    ctx.sh_surfel_pass2.scale_projection_loc  = glGetUniformLocation(ctx.sh_surfel_pass2.program, "scale_projection");
    ctx.sh_surfel_pass2.max_radius_loc        = glGetUniformLocation(ctx.sh_surfel_pass2.program, "max_radius");
    ctx.sh_surfel_pass2.min_radius_loc        = glGetUniformLocation(ctx.sh_surfel_pass2.program, "min_radius");
    ctx.sh_surfel_pass2.max_screen_size_loc   = glGetUniformLocation(ctx.sh_surfel_pass2.program, "max_screen_size");
    ctx.sh_surfel_pass2.min_screen_size_loc   = glGetUniformLocation(ctx.sh_surfel_pass2.program, "min_screen_size");
    ctx.sh_surfel_pass2.scale_radius_loc      = glGetUniformLocation(ctx.sh_surfel_pass2.program, "scale_radius");
    ctx.sh_surfel_pass2.scale_radius_gamma_loc = glGetUniformLocation(ctx.sh_surfel_pass2.program, "scale_radius_gamma");
    ctx.sh_surfel_pass2.max_radius_cut_loc    = glGetUniformLocation(ctx.sh_surfel_pass2.program, "max_radius_cut");
    ctx.sh_surfel_pass2.use_aniso_loc         = glGetUniformLocation(ctx.sh_surfel_pass2.program, "use_aniso");
    ctx.sh_surfel_pass2.show_normals_loc      = glGetUniformLocation(ctx.sh_surfel_pass2.program, "show_normals");
    ctx.sh_surfel_pass2.show_accuracy_loc     = glGetUniformLocation(ctx.sh_surfel_pass2.program, "show_accuracy");
    ctx.sh_surfel_pass2.show_radius_dev_loc   = glGetUniformLocation(ctx.sh_surfel_pass2.program, "show_radius_deviation");
    ctx.sh_surfel_pass2.show_output_sens_loc  = glGetUniformLocation(ctx.sh_surfel_pass2.program, "show_output_sensitivity");
    ctx.sh_surfel_pass2.accuracy_loc          = glGetUniformLocation(ctx.sh_surfel_pass2.program, "accuracy");
    ctx.sh_surfel_pass2.average_radius_loc    = glGetUniformLocation(ctx.sh_surfel_pass2.program, "average_radius");
    ctx.sh_surfel_pass2.depth_range_loc       = glGetUniformLocation(ctx.sh_surfel_pass2.program, "depth_range");
    ctx.sh_surfel_pass2.flank_lift_loc        = glGetUniformLocation(ctx.sh_surfel_pass2.program, "flank_lift");
    ctx.sh_surfel_pass2.coloring_loc          = glGetUniformLocation(ctx.sh_surfel_pass2.program, "coloring");
    if (ctx.sh_surfel_pass2.depth_texture_loc >= 0) {
        glUniform1i(ctx.sh_surfel_pass2.depth_texture_loc, 0);
    }

    glUseProgram(ctx.sh_surfel_pass3.program);
    ctx.sh_surfel_pass3.in_color_texture_loc        = glGetUniformLocation(ctx.sh_surfel_pass3.program, "in_color_texture");
    ctx.sh_surfel_pass3.in_normal_texture_loc       = glGetUniformLocation(ctx.sh_surfel_pass3.program, "in_normal_texture");
    ctx.sh_surfel_pass3.in_vs_position_texture_loc  = glGetUniformLocation(ctx.sh_surfel_pass3.program, "in_vs_position_texture");
    ctx.sh_surfel_pass3.in_depth_texture_loc        = glGetUniformLocation(ctx.sh_surfel_pass3.program, "in_depth_texture");
    ctx.sh_surfel_pass3.point_light_pos_vs_loc      = glGetUniformLocation(ctx.sh_surfel_pass3.program, "point_light_pos_vs");
    ctx.sh_surfel_pass3.point_light_intensity_loc   = glGetUniformLocation(ctx.sh_surfel_pass3.program, "point_light_intensity");
    ctx.sh_surfel_pass3.ambient_intensity_loc       = glGetUniformLocation(ctx.sh_surfel_pass3.program, "ambient_intensity");
    ctx.sh_surfel_pass3.specular_intensity_loc      = glGetUniformLocation(ctx.sh_surfel_pass3.program, "specular_intensity");
    ctx.sh_surfel_pass3.shininess_loc               = glGetUniformLocation(ctx.sh_surfel_pass3.program, "shininess");
    ctx.sh_surfel_pass3.gamma_loc                   = glGetUniformLocation(ctx.sh_surfel_pass3.program, "gamma");
    ctx.sh_surfel_pass3.use_tone_mapping_loc        = glGetUniformLocation(ctx.sh_surfel_pass3.program, "use_tone_mapping");
    ctx.sh_surfel_pass3.lighting_loc                = glGetUniformLocation(ctx.sh_surfel_pass3.program, "lighting");
    if (ctx.sh_surfel_pass3.in_color_texture_loc >= 0)       glUniform1i(ctx.sh_surfel_pass3.in_color_texture_loc, 0);
    if (ctx.sh_surfel_pass3.in_normal_texture_loc >= 0)      glUniform1i(ctx.sh_surfel_pass3.in_normal_texture_loc, 1);
    if (ctx.sh_surfel_pass3.in_vs_position_texture_loc >= 0) glUniform1i(ctx.sh_surfel_pass3.in_vs_position_texture_loc, 2);
    if (ctx.sh_surfel_pass3.in_depth_texture_loc >= 0)       glUniform1i(ctx.sh_surfel_pass3.in_depth_texture_loc, 3);

    glUseProgram(0);
}
bool LamureRenderer::isModelVisible(std::size_t modelIndex) const
{
    if (!m_plugin)
        return false;

    const auto& settings = m_plugin->getSettings();
    const auto& modelInfo = m_plugin->getModelInfo();
    if (modelIndex >= settings.models.size())
        return false;

    if (modelIndex < modelInfo.model_visible.size() && !modelInfo.model_visible[modelIndex])
        return false;

    if (modelIndex >= m_plugin->m_scene_nodes.size())
        return false;

    const auto& sn = m_plugin->m_scene_nodes[modelIndex];

    auto hasPathToSceneRoot = [](osg::Node* start) {
        if (!start)
            return false;
        osg::Node* scene_root = nullptr;
        if (opencover::cover)
            scene_root = opencover::cover->getObjectsRoot();
        std::vector<osg::Node*> stack;
        stack.push_back(start);
        std::unordered_set<const osg::Node*> visited;
        while (!stack.empty()) {
            osg::Node* current = stack.back();
            stack.pop_back();
            if (!current)
                continue;
            if (visited.find(current) != visited.end())
                continue;
            visited.insert(current);
            if (current == scene_root)
                return true;
            if (current->getNodeMask() == 0)
                continue;
            for (unsigned int i = 0; i < current->getNumParents(); ++i) {
                stack.push_back(current->getParent(i));
            }
        }
        return false;
    };

    if (settings.show_pointcloud && sn.point_geode.valid()) {
        osg::Node* node = sn.point_geode.get();
        if (node && node->getNodeMask() != 0 && hasPathToSceneRoot(node))
            return true;
    }

    if (settings.show_boundingbox && sn.box_geode.valid()) {
        osg::Node* node = sn.box_geode.get();
        if (node && node->getNodeMask() != 0 && hasPathToSceneRoot(node))
            return true;
    }

    return false;
}

void LamureRenderer::setFrameUniforms(const scm::math::mat4& projection_matrix, const scm::math::mat4& view_matrix, const scm::math::vec2& viewport, ContextResources& ctxRes) {
    const auto &s = m_plugin->getSettings();

    // Decide anisotropic usage based on current projection and mode (0=off,1=auto,2=on)
    const bool useAnisoThisPass = LamureUtil::decideUseAniso(projection_matrix, s.anisotropic_surfel_scaling, s.anisotropic_auto_threshold);

    // Precompute common scalars once per frame
    const float scale_radius_combined = s.scale_radius * s.scale_element;
    const float scale_projection_val = opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5];
    const scm::math::vec4f proj_col0(
        projection_matrix.data_array[0],
        projection_matrix.data_array[1],
        projection_matrix.data_array[2],
        projection_matrix.data_array[3]);
    const scm::math::vec4f proj_col1(
        projection_matrix.data_array[4],
        projection_matrix.data_array[5],
        projection_matrix.data_array[6],
        projection_matrix.data_array[7]);
    const float viewport_half_y = opencover::cover->getScale() * viewport.y * 0.5f;
    const float len0 = std::sqrt(std::max(0.0f, proj_col0[0] * proj_col0[0] + proj_col0[1] * proj_col0[1]));
    const float len1 = std::sqrt(std::max(0.0f, proj_col1[0] * proj_col1[0] + proj_col1[1] * proj_col1[1]));
    const float rms_proj = std::sqrt(std::max(1e-8f, 0.5f * (len0 * len0 + len1 * len1)));
    const float aniso_normalize = (rms_proj > 1e-8f) ? (len1 / rms_proj) : 1.0f;
    const bool enableColorDebug = s.coloring;
    const bool showNormalsDebug = enableColorDebug && s.show_normals;
    const bool showAccuracyDebug = enableColorDebug && s.show_accuracy;
    const bool showRadiusDeviationDebug = enableColorDebug && s.show_radius_deviation;
    const bool showOutputSensitivityDebug = enableColorDebug && s.show_output_sensitivity;

    struct FrameViewContext {
        scm::math::mat4 viewMat;
        scm::math::mat3 normalMat;
        scm::math::vec3 light_vs;
    };

    auto makeFrameViewContext = [&]() -> FrameViewContext {
        FrameViewContext ctx{};
        ctx.viewMat = view_matrix;
        scm::math::mat4 viewMatCopy = view_matrix;
        scm::math::mat3 viewMat3 = LamureUtil::matConv4to3F(viewMatCopy);
        ctx.normalMat = scm::math::transpose(scm::math::inverse(viewMat3));
        scm::math::vec4 light_ws(s.point_light_pos[0], s.point_light_pos[1], s.point_light_pos[2], 1.0f);
        scm::math::vec4 light_vs4 = ctx.viewMat * light_ws;
        ctx.light_vs = scm::math::vec3(light_vs4[0], light_vs4[1], light_vs4[2]);
        return ctx;
    };

    const bool needsViewContext =
        s.shader_type == ShaderType::PointColor ||
        s.shader_type == ShaderType::PointColorLighting ||
        s.shader_type == ShaderType::SurfelColor ||
        s.shader_type == ShaderType::SurfelColorLighting;
    FrameViewContext frameView{};
    if (needsViewContext) {
        frameView = makeFrameViewContext();
    }

    switch (s.shader_type) {
    case ShaderType::Point: {
        auto& prog = ctxRes.sh_point;
        glUseProgram(prog.program);
        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUniform1f(prog.min_radius_loc, s.min_radius);
        glUniform1f(prog.max_radius_loc, s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, s.scale_radius);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform1f(prog.scale_projection_loc, scale_projection_val);
        if (prog.proj_col0_loc >= 0)       glUniform4fv(prog.proj_col0_loc, 1, proj_col0.data_array);
        if (prog.proj_col1_loc >= 0)       glUniform4fv(prog.proj_col1_loc, 1, proj_col1.data_array);
        if (prog.viewport_half_y_loc >= 0) glUniform1f(prog.viewport_half_y_loc, viewport_half_y);
        if (prog.use_aniso_loc >= 0)       glUniform1i(prog.use_aniso_loc, useAnisoThisPass ? 1 : 0);
        if (prog.aniso_normalize_loc >= 0) glUniform1f(prog.aniso_normalize_loc, aniso_normalize);
        uploadClipPlanes(prog.clip_plane_count_loc, prog.clip_plane_data_loc);
        break;
    }
    case ShaderType::PointColor: {
        auto& prog = ctxRes.sh_point_color;
        glUseProgram(prog.program);
        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUniformMatrix4fv(prog.view_matrix_loc,        1, GL_FALSE, frameView.viewMat.data_array);
        glUniformMatrix3fv(prog.normal_matrix_loc, 1, GL_FALSE, frameView.normalMat.data_array);
        glUniform1f(prog.min_radius_loc, s.min_radius);
        glUniform1f(prog.max_radius_loc, s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, s.scale_radius);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform1f(prog.scale_projection_loc, scale_projection_val);
        if (prog.proj_col0_loc >= 0)       glUniform4fv(prog.proj_col0_loc, 1, proj_col0.data_array);
        if (prog.proj_col1_loc >= 0)       glUniform4fv(prog.proj_col1_loc, 1, proj_col1.data_array);
        if (prog.viewport_half_y_loc >= 0) glUniform1f(prog.viewport_half_y_loc, viewport_half_y);
        if (prog.use_aniso_loc >= 0)       glUniform1i(prog.use_aniso_loc, useAnisoThisPass ? 1 : 0);
        if (prog.aniso_normalize_loc >= 0) glUniform1f(prog.aniso_normalize_loc, aniso_normalize);
        if (prog.show_normals_loc >= 0)     glUniform1i(prog.show_normals_loc,     showNormalsDebug ? 1 : 0);
        if (prog.show_radius_dev_loc >= 0)  glUniform1i(prog.show_radius_dev_loc,  showRadiusDeviationDebug ? 1 : 0);
        if (prog.show_output_sens_loc >= 0) glUniform1i(prog.show_output_sens_loc, showOutputSensitivityDebug ? 1 : 0);
        if (prog.show_accuracy_loc >= 0)    glUniform1i(prog.show_accuracy_loc,    showAccuracyDebug ? 1 : 0);
        uploadClipPlanes(prog.clip_plane_count_loc, prog.clip_plane_data_loc);
        break;
    }
    case ShaderType::PointColorLighting: {
        auto& prog = ctxRes.sh_point_color_lighting;
        glUseProgram(prog.program);
        glUniformMatrix4fv(prog.view_matrix_loc, 1, GL_FALSE, frameView.viewMat.data_array);
        glUniformMatrix3fv(prog.normal_matrix_loc, 1, GL_FALSE, frameView.normalMat.data_array);
        glUniform1f(prog.min_radius_loc, s.min_radius);
        glUniform1f(prog.max_radius_loc, s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, s.scale_radius);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform1f(prog.scale_projection_loc, scale_projection_val);
        if (prog.proj_col0_loc >= 0)       glUniform4fv(prog.proj_col0_loc, 1, proj_col0.data_array);
        if (prog.proj_col1_loc >= 0)       glUniform4fv(prog.proj_col1_loc, 1, proj_col1.data_array);
        if (prog.viewport_half_y_loc >= 0) glUniform1f(prog.viewport_half_y_loc, viewport_half_y);
        if (prog.use_aniso_loc >= 0)       glUniform1i(prog.use_aniso_loc, useAnisoThisPass ? 1 : 0);
        if (prog.aniso_normalize_loc >= 0) glUniform1f(prog.aniso_normalize_loc, aniso_normalize);
        if (prog.show_normals_loc >= 0)     glUniform1i(prog.show_normals_loc,     showNormalsDebug ? 1 : 0);
        if (prog.show_radius_dev_loc >= 0)  glUniform1i(prog.show_radius_dev_loc,  showRadiusDeviationDebug ? 1 : 0);
        if (prog.show_output_sens_loc >= 0) glUniform1i(prog.show_output_sens_loc, showOutputSensitivityDebug ? 1 : 0);
        if (prog.show_accuracy_loc >= 0)    glUniform1i(prog.show_accuracy_loc,    showAccuracyDebug ? 1 : 0);
        glUniform3fv(prog.point_light_pos_vs_loc, 1, frameView.light_vs.data_array);
        glUniform1f(prog.point_light_intensity_loc, s.point_light_intensity);
        glUniform1f(prog.ambient_intensity_loc,     s.ambient_intensity);
        glUniform1f(prog.specular_intensity_loc,    s.specular_intensity);
        glUniform1f(prog.shininess_loc,             s.shininess);
        glUniform1f(prog.gamma_loc,                 s.gamma);
        glUniform1i(prog.use_tone_mapping_loc,      s.use_tone_mapping ? 1 : 0);
        uploadClipPlanes(prog.clip_plane_count_loc, prog.clip_plane_data_loc);
        break;
    }
    case ShaderType::PointProv: {
        auto& prog = ctxRes.sh_point_prov;
        glUseProgram(prog.program);
        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUniform1f(prog.min_radius_loc, s.min_radius);
        glUniform1f(prog.max_radius_loc, s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, s.scale_radius);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform1f(prog.scale_projection_loc, scale_projection_val);
        glUniform1i(prog.show_normals_loc, showNormalsDebug ? 1 : 0);
        glUniform1i(prog.show_radius_dev_loc, showRadiusDeviationDebug ? 1 : 0);
        glUniform1i(prog.show_output_sens_loc, showOutputSensitivityDebug ? 1 : 0);
        glUniform1i(prog.show_accuracy_loc, showAccuracyDebug ? 1 : 0);
        glUniform1i(prog.channel_loc, s.channel);
        glUniform1i(prog.heatmap_loc, s.heatmap);
        glUniform1f(prog.heatmap_min_loc, s.heatmap_min);
        glUniform1f(prog.heatmap_max_loc, s.heatmap_max);
        glUniform3fv(prog.heatmap_min_color_loc, 1, s.heatmap_color_min.data_array);
        glUniform3fv(prog.heatmap_max_color_loc, 1, s.heatmap_color_max.data_array);
        uploadClipPlanes(prog.clip_plane_count_loc, prog.clip_plane_data_loc);
        break;
    }
    case ShaderType::Surfel: {
        auto& prog = ctxRes.sh_surfel;
        glEnable(GL_DEPTH_TEST);
        glUseProgram(prog.program);
        glUniform1f(prog.min_radius_loc, s.min_radius);
        glUniform1f(prog.max_radius_loc, s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, scale_radius_combined);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform1f(prog.scale_projection_loc, scale_projection_val);
        if (prog.viewport_loc >= 0) glUniform2fv(prog.viewport_loc, 1, viewport.data_array);
        if (prog.use_aniso_loc >= 0) glUniform1i(prog.use_aniso_loc, useAnisoThisPass ? 1 : 0);
        uploadClipPlanes(prog.clip_plane_count_loc, prog.clip_plane_data_loc);
        break;
    }
    case ShaderType::SurfelColor: {
        auto& prog = ctxRes.sh_surfel_color;
        //glEnable(GL_DEPTH_TEST);
        glUseProgram(prog.program);
        glUniformMatrix4fv(prog.view_matrix_loc,        1, GL_FALSE, frameView.viewMat.data_array);
        glUniformMatrix3fv(prog.normal_matrix_loc, 1, GL_FALSE, frameView.normalMat.data_array);
        glUniform1f(prog.min_radius_loc, s.min_radius);
        glUniform1f(prog.max_radius_loc, s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, scale_radius_combined);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        if (prog.viewport_loc >= 0) glUniform2fv(prog.viewport_loc, 1, viewport.data_array);
        glUniform1f(prog.scale_projection_loc, scale_projection_val);
        // Anisotrope Skalierung (optional/auto)
        if (prog.use_aniso_loc >= 0) glUniform1i(prog.use_aniso_loc, useAnisoThisPass ? 1 : 0);
        if (prog.show_normals_loc >= 0)     glUniform1i(prog.show_normals_loc,     showNormalsDebug ? 1 : 0);
        if (prog.show_radius_dev_loc >= 0)  glUniform1i(prog.show_radius_dev_loc,  showRadiusDeviationDebug ? 1 : 0);
        if (prog.show_output_sens_loc >= 0) glUniform1i(prog.show_output_sens_loc, showOutputSensitivityDebug ? 1 : 0);
        if (prog.show_accuracy_loc >= 0)    glUniform1i(prog.show_accuracy_loc,    showAccuracyDebug ? 1 : 0);
        uploadClipPlanes(prog.clip_plane_count_loc, prog.clip_plane_data_loc);
        break;
    }
    case ShaderType::SurfelColorLighting: {
        auto& prog = ctxRes.sh_surfel_color_lighting;
        glEnable(GL_DEPTH_TEST);
        glUseProgram(prog.program);
        glUniformMatrix4fv(prog.view_matrix_loc,        1, GL_FALSE, frameView.viewMat.data_array);
        glUniformMatrix3fv(prog.normal_matrix_loc, 1, GL_FALSE, frameView.normalMat.data_array);
        glUniform1f(prog.min_radius_loc,   s.min_radius);
        glUniform1f(prog.max_radius_loc,   s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, scale_radius_combined);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        if (prog.viewport_loc >= 0) glUniform2fv(prog.viewport_loc, 1, viewport.data_array);
        glUniform1f(prog.scale_projection_loc, scale_projection_val);
        // Anisotrope Skalierung (optional/auto)
        if (prog.use_aniso_loc >= 0) glUniform1i(prog.use_aniso_loc, useAnisoThisPass ? 1 : 0);

        if (prog.show_normals_loc >= 0)     glUniform1i(prog.show_normals_loc,     showNormalsDebug ? 1 : 0);
        if (prog.show_radius_dev_loc >= 0)  glUniform1i(prog.show_radius_dev_loc,  showRadiusDeviationDebug ? 1 : 0);
        if (prog.show_output_sens_loc >= 0) glUniform1i(prog.show_output_sens_loc, showOutputSensitivityDebug ? 1 : 0);
        if (prog.show_accuracy_loc >= 0)    glUniform1i(prog.show_accuracy_loc,    showAccuracyDebug ? 1 : 0);

        glUniform3fv(prog.point_light_pos_vs_loc, 1, frameView.light_vs.data_array);
        glUniform1f(prog.point_light_intensity_loc, s.point_light_intensity);
        glUniform1f(prog.ambient_intensity_loc,     s.ambient_intensity);
        glUniform1f(prog.specular_intensity_loc,    s.specular_intensity);
        glUniform1f(prog.shininess_loc,             s.shininess);
        glUniform1f(prog.gamma_loc,                 s.gamma);
        glUniform1i(prog.use_tone_mapping_loc,      s.use_tone_mapping ? 1 : 0);
        uploadClipPlanes(prog.clip_plane_count_loc, prog.clip_plane_data_loc);
        break;
    }
    case ShaderType::SurfelProv: {
        auto& prog = ctxRes.sh_surfel_prov;
        glEnable(GL_DEPTH_TEST);
        glUseProgram(prog.program);
        glUniform1f(prog.min_radius_loc, s.min_radius);
        glUniform1f(prog.max_radius_loc, s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, scale_radius_combined);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        if (prog.viewport_loc >= 0) glUniform2fv(prog.viewport_loc, 1, viewport.data_array);
        glUniform1f(prog.scale_projection_loc, scale_projection_val);
        if (prog.show_normals_loc >= 0)     glUniform1i(prog.show_normals_loc,     showNormalsDebug ? 1 : 0);
        if (prog.show_radius_dev_loc >= 0)  glUniform1i(prog.show_radius_dev_loc,  showRadiusDeviationDebug ? 1 : 0);
        if (prog.show_output_sens_loc >= 0) glUniform1i(prog.show_output_sens_loc, showOutputSensitivityDebug ? 1 : 0);
        if (prog.show_accuracy_loc >= 0)    glUniform1i(prog.show_accuracy_loc,    showAccuracyDebug ? 1 : 0);
        glUniform1i(prog.channel_loc, s.channel);
        glUniform1i(prog.heatmap_loc, s.heatmap);
        glUniform1f(prog.heatmap_min_loc, s.heatmap_min);
        glUniform1f(prog.heatmap_max_loc, s.heatmap_max);
        glUniform3fv(prog.heatmap_min_color_loc, 1, s.heatmap_color_min.data_array);
        glUniform3fv(prog.heatmap_max_color_loc, 1, s.heatmap_color_max.data_array);
        uploadClipPlanes(prog.clip_plane_count_loc, prog.clip_plane_data_loc);
        break;
    }
    // Multipass uniforms are set during the draw callback; nothing to do here.
    case ShaderType::SurfelMultipass: {
        break;
    }
    }
}

void LamureRenderer::setModelUniforms(const scm::math::mat4& mvp_matrix, const scm::math::mat4& model_matrix, ContextResources& ctxRes) {
    switch (m_plugin->getSettings().shader_type) {
    case ShaderType::Point:
        glUniformMatrix4fv(ctxRes.sh_point.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (ctxRes.sh_point.model_matrix_loc >= 0)
            glUniformMatrix4fv(ctxRes.sh_point.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::PointColor:
        glUniformMatrix4fv(ctxRes.sh_point_color.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (ctxRes.sh_point_color.model_matrix_loc >= 0)
            glUniformMatrix4fv(ctxRes.sh_point_color.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::PointColorLighting:
        glUniformMatrix4fv(ctxRes.sh_point_color_lighting.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (ctxRes.sh_point_color_lighting.model_matrix_loc >= 0)
            glUniformMatrix4fv(ctxRes.sh_point_color_lighting.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::PointProv:
        glUniformMatrix4fv(ctxRes.sh_point_prov.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (ctxRes.sh_point_prov.model_matrix_loc >= 0)
            glUniformMatrix4fv(ctxRes.sh_point_prov.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::Surfel:
        glUniformMatrix4fv(ctxRes.sh_surfel.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (ctxRes.sh_surfel.model_matrix_loc >= 0)
            glUniformMatrix4fv(ctxRes.sh_surfel.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::SurfelColor:
        glUniformMatrix4fv(ctxRes.sh_surfel_color.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (ctxRes.sh_surfel_color.model_matrix_loc >= 0)
            glUniformMatrix4fv(ctxRes.sh_surfel_color.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::SurfelColorLighting:
        glUniformMatrix4fv(ctxRes.sh_surfel_color_lighting.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (ctxRes.sh_surfel_color_lighting.model_matrix_loc >= 0)
            glUniformMatrix4fv(ctxRes.sh_surfel_color_lighting.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::SurfelProv:
        glUniformMatrix4fv(ctxRes.sh_surfel_prov.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (ctxRes.sh_surfel_prov.model_matrix_loc >= 0)
            glUniformMatrix4fv(ctxRes.sh_surfel_prov.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::SurfelMultipass:
        break;
    }
}

void LamureRenderer::setNodeUniforms(const lamure::ren::bvh* bvh, uint32_t node_id, ContextResources& ctxRes) {
    const auto &s = m_plugin->getSettings();
    const float gamma = (s.scale_radius_gamma > 0.0f) ? s.scale_radius_gamma : 1.0f;
    auto calc_avg_ws = [&](float avg_raw, float scale_factor) {
        return std::pow(std::max(0.0f, avg_raw), gamma) * scale_factor;
    };
    const bool enableColorDebug = s.coloring;
    const bool showAcc = enableColorDebug && s.show_accuracy;
    const bool showRadiusDev = enableColorDebug && s.show_radius_deviation;

    switch (s.shader_type) {
    case ShaderType::PointColor: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(ctxRes.sh_point_color.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius);
            glUniform1f(ctxRes.sh_point_color.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::PointColorLighting: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(ctxRes.sh_point_color_lighting.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius);
            glUniform1f(ctxRes.sh_point_color_lighting.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::PointProv: {
        if (s.show_accuracy) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(ctxRes.sh_point_prov.accuracy_loc, accuracy);
        }
        if (s.show_radius_deviation) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius);
            glUniform1f(ctxRes.sh_point_prov.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelColor: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(ctxRes.sh_surfel_color.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius * s.scale_element);
            glUniform1f(ctxRes.sh_surfel_color.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelProv: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(ctxRes.sh_surfel_prov.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius * s.scale_element);
            glUniform1f(ctxRes.sh_surfel_prov.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelColorLighting: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(ctxRes.sh_surfel_color_lighting.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius * s.scale_element);
            glUniform1f(ctxRes.sh_surfel_color_lighting.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelMultipass: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(ctxRes.sh_surfel_pass2.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius * s.scale_element);
            glUniform1f(ctxRes.sh_surfel_pass2.average_radius_loc, avg_ws);
        }
        break;
    }
    default: break;
    }
}

bool LamureRenderer::readShader(std::string const &filename_string, std::string &shader_string, bool keep_optional_shader_code = false)
{
    const char *covisedir = getenv("COVISEDIR");
    if (!covisedir)
    {
        std::cerr << "[Lamure][ERR] COVISEDIR not set, cannot load shaders!\n";
        return false;
    }
    std::string shader_root_path = covisedir;
    shader_root_path += "/share/covise/shaders";
    shader_root_path += "/vis";
    std::string path_string = shader_root_path + "/" + filename_string;
#ifdef __APPLE__
    path_string = shader_root_path + "/legacy/" + filename_string;
    if (!boost::filesystem::exists(path_string))
    {
        path_string = shader_root_path + "/" + filename_string;
    }
#endif
    if (!boost::filesystem::exists(path_string))
    {
        std::cerr << "[Lamure][WARN] File " << path_string << " does not exist.\n";
        return false;
    }
    std::ifstream shader_source(path_string, std::ios::in);
    std::string line_buffer;
    std::string include_prefix("INCLUDE");
    std::string optional_begin_prefix("OPTIONAL_BEGIN");
    std::string optional_end_prefix("OPTIONAL_END");

    bool disregard_code = false;
    while (std::getline(shader_source, line_buffer))
    {
        line_buffer = LamureUtil::stripWhitespace(line_buffer);
        if (LamureUtil::parsePrefix(line_buffer, include_prefix))
        {
            if (!disregard_code || keep_optional_shader_code)
            {
                std::string filename_string = line_buffer;
                readShader(filename_string, shader_string);
            }
        }
        else if (LamureUtil::parsePrefix(line_buffer, optional_begin_prefix))
        {
            disregard_code = true;
        }
        else if (LamureUtil::parsePrefix(line_buffer, optional_end_prefix))
        {
            disregard_code = false;
        }
        else
        {
            if ((!disregard_code) || keep_optional_shader_code)
            {
                shader_string += line_buffer + "\n";
            }
        }
    }
    return true;
}


bool LamureRenderer::initLamureShader(ContextResources& res)
{
    if (notifyOn()) { std::cout << "[Lamure] LamureRenderer::initLamureShader() for ctx " << (int)res.ctx << std::endl; }

    {
        std::lock_guard<std::mutex> shader_lock(m_shader_mutex);
        if (!m_shader_sources_loaded) {
            if (notifyOn()) { std::cout << "[Lamure] Loading shader sources from disk..." << std::endl; }
            try
            {
                std::array<std::pair<const char*, std::string*>, 32> shared_sources = {{
                    {"vis_point.glslv", &vis_point_vs_source},
                    {"vis_point.glslf", &vis_point_fs_source},
                    {"vis_point_color.glslv", &vis_point_color_vs_source},
                    {"vis_point_color.glslf", &vis_point_color_fs_source},
                    {"vis_point_color_lighting.glslv", &vis_point_color_lighting_vs_source},
                    {"vis_point_color_lighting.glslf", &vis_point_color_lighting_fs_source},
                    {"vis_point_prov.glslv", &vis_point_prov_vs_source},
                    {"vis_point_prov.glslf", &vis_point_prov_fs_source},
                    {"vis_surfel.glslv", &vis_surfel_vs_source},
                    {"vis_surfel.glslg", &vis_surfel_gs_source},
                    {"vis_surfel.glslf", &vis_surfel_fs_source},
                    {"vis_surfel_color.glslv", &vis_surfel_color_vs_source},
                    {"vis_surfel_color.glslg", &vis_surfel_color_gs_source},
                    {"vis_surfel_color.glslf", &vis_surfel_color_fs_source},
                    {"vis_surfel_color_lighting.glslv", &vis_surfel_color_lighting_vs_source},
                    {"vis_surfel_color_lighting.glslg", &vis_surfel_color_lighting_gs_source},
                    {"vis_surfel_color_lighting.glslf", &vis_surfel_color_lighting_fs_source},
                    {"vis_surfel_prov.glslv", &vis_surfel_prov_vs_source},
                    {"vis_surfel_prov.glslg", &vis_surfel_prov_gs_source},
                    {"vis_surfel_prov.glslf", &vis_surfel_prov_fs_source},
                    {"vis_line.glslv", &vis_line_vs_source},
                    {"vis_line.glslf", &vis_line_fs_source},
                    {"vis_surfel_pass1.glslv", &vis_surfel_pass1_vs_source},
                    {"vis_surfel_pass1.glslg", &vis_surfel_pass1_gs_source},
                    {"vis_surfel_pass1.glslf", &vis_surfel_pass1_fs_source},
                    {"vis_surfel_pass2.glslv", &vis_surfel_pass2_vs_source},
                    {"vis_surfel_pass2.glslg", &vis_surfel_pass2_gs_source},
                    {"vis_surfel_pass2.glslf", &vis_surfel_pass2_fs_source},
                    {"vis_surfel_pass3.glslv", &vis_surfel_pass3_vs_source},
                    {"vis_surfel_pass3.glslf", &vis_surfel_pass3_fs_source},
                    {"vis_debug.glslv", &vis_debug_vs_source},
                    {"vis_debug.glslf", &vis_debug_fs_source}
                }};
                std::array<std::string, 32> loaded_sources;

                for (size_t i = 0; i < shared_sources.size(); ++i)
                {
                    if (!readShader(shared_sources[i].first, loaded_sources[i]))
                    {
                        m_shader_sources_loaded = false;
                        std::cerr << "[Lamure][ERR] error reading shader files\n";
                        return false;
                    }
                }

                for (size_t i = 0; i < shared_sources.size(); ++i)
                {
                    *shared_sources[i].second = std::move(loaded_sources[i]);
                }
                m_shader_sources_loaded = true;
            }
            catch (std::exception &e) { 
                m_shader_sources_loaded = false;
                std::cerr << "[Lamure][ERR] Exception loading shaders: " << e.what() << "\n";
                return false;
            }
        }
    }

    // Compile shaders for this context from the shared sources
    res.sh_point.program                 = compileAndLinkShaders(vis_point_vs_source, vis_point_fs_source, res.ctx);
    res.sh_point_color.program           = compileAndLinkShaders(vis_point_color_vs_source, vis_point_color_fs_source, res.ctx);
    res.sh_point_color_lighting.program  = compileAndLinkShaders(vis_point_color_lighting_vs_source, vis_point_color_lighting_fs_source, res.ctx);
    res.sh_point_prov.program            = compileAndLinkShaders(vis_point_prov_vs_source, vis_point_prov_fs_source, res.ctx);
    res.sh_surfel.program                = compileAndLinkShaders(vis_surfel_vs_source, vis_surfel_gs_source, vis_surfel_fs_source, res.ctx);
    res.sh_surfel_color.program          = compileAndLinkShaders(vis_surfel_color_vs_source, vis_surfel_color_gs_source, vis_surfel_color_fs_source, res.ctx);
    res.sh_surfel_color_lighting.program = compileAndLinkShaders(vis_surfel_color_lighting_vs_source, vis_surfel_color_lighting_gs_source, vis_surfel_color_lighting_fs_source, res.ctx);
    res.sh_surfel_prov.program           = compileAndLinkShaders(vis_surfel_prov_vs_source, vis_surfel_prov_gs_source, vis_surfel_prov_fs_source, res.ctx);
    res.sh_line.program                  = compileAndLinkShaders(vis_line_vs_source, vis_line_fs_source, res.ctx);
    res.sh_surfel_pass1.program          = compileAndLinkShaders(vis_surfel_pass1_vs_source, vis_surfel_pass1_gs_source, vis_surfel_pass1_fs_source, res.ctx);
    res.sh_surfel_pass2.program          = compileAndLinkShaders(vis_surfel_pass2_vs_source, vis_surfel_pass2_gs_source, vis_surfel_pass2_fs_source, res.ctx);
    res.sh_surfel_pass3.program          = compileAndLinkShaders(vis_surfel_pass3_vs_source, vis_surfel_pass3_fs_source, res.ctx);
    {
        static const std::string coverageVs =
            "#version 330\n"
            "layout(location = 0) in vec3 in_position;\n"
            "void main(){ gl_Position = vec4(in_position, 1.0); }\n";
        static const std::string coverageFs =
            "#version 330\n"
            "out vec4 fragColor;\n"
            "void main(){ fragColor = vec4(1.0); }\n";
        res.sh_coverage_query.program = compileAndLinkShaders(coverageVs, coverageFs, res.ctx, "coverage_query");
    }

    res.shaders_initialized = true;
    return true;
}

void LamureRenderer::initSchismObjects(ContextResources& ctx)
{
    if (notifyOn()) { std::cout << "[Lamure] LamureRenderer::initSchismObjects()" << std::endl; }
    // Per-context device/context to avoid sharing across OSG contexts.
    if (!ctx.scm_device) {
        ctx.scm_device.reset(new scm::gl::render_device());
        if (!ctx.scm_device) {
            std::cerr << "[Lamure][ERR] error creating device\n";
            return;
        }
    }
    if (!ctx.scm_context) {
        ctx.scm_context = ctx.scm_device->main_context();
        if (!ctx.scm_context) {
            std::cerr << "[Lamure][ERR] error creating context\n";
        }
    }
}

bool LamureRenderer::initGpus(ContextResources& res)
{
    if (m_gpu_org_ready.load(std::memory_order_acquire)) {
        return true;
    }

    if (!res.gpu_info_logged) {
        LamureUtil::GpuInfo info = LamureUtil::queryGpuInfo();
        res.gpu_vendor = info.vendor;
        res.gpu_renderer = info.renderer;
        res.gpu_version = info.version;
        res.gpu_uuid = info.device_uuid;
        res.driver_uuid = info.driver_uuid;
        res.gpu_key = info.key.empty()
            ? res.gpu_vendor + "|" + res.gpu_renderer + "|" + res.gpu_version
            : info.key;

        res.gpu_info_logged = true;
    }

    auto* controller = lamure::ren::controller::get_instance();
    if (!controller) {
        return false;
    }

    std::vector<int> context_ids;
    if (auto* viewer = opencover::VRViewer::instance()) {
        osgViewer::ViewerBase::Cameras cameras;
        viewer->getCameras(cameras, true);
        std::unordered_set<int> unique_context_ids;
        for (auto* cam : cameras) {
            if (!cam) continue;
            osg::GraphicsContext* gc = cam->getGraphicsContext();
            if (!gc) continue;
            osg::State* gs = gc->getState();
            if (!gs) continue;
            const int ctx_id = static_cast<int>(gs->getContextID());
            if (ctx_id < 0) continue;
            unique_context_ids.insert(ctx_id);
        }
        context_ids.assign(unique_context_ids.begin(), unique_context_ids.end());
    }
    if (context_ids.empty()) {
        return false;
    }

    std::unordered_map<std::string, uint32_t> counts;
    std::vector<std::string> keys;
    keys.reserve(context_ids.size());
    {
        std::scoped_lock lock(m_renderMutex, m_ctx_mutex);
        if (m_ctx_res.size() < context_ids.size()) {
            return false;
        }
        for (int ctx_id : context_ids) {
            auto it = m_ctx_res.find(ctx_id);
            if (it == m_ctx_res.end()) {
                return false;
            }
            const auto& r = it->second;
            if (!r.gpu_info_logged || r.gpu_key.empty()) {
                return false;
            }
            keys.emplace_back(r.gpu_key);
            ++counts[r.gpu_key];
        }
    }

    for (size_t i = 0; i < context_ids.size(); ++i) {
        const int ctx_id = context_ids[i];
        const std::string& key = keys[i];
        auto it = counts.find(key);
        if (it == counts.end()) {
            continue;
        }
        lamure::context_t context_id = controller->deduce_context_id(ctx_id);
        controller->set_contexts(context_id, it->second);
    }

    std::cout << "[Lamure] GPU organization: gpus=" << counts.size()
              << " contexts=" << context_ids.size() << std::endl;

    std::scoped_lock lock(m_renderMutex, m_ctx_mutex);
    for (auto& kv : m_ctx_res) {
        kv.second.gpu_consistency_checked = true;
    }
    m_gpu_org_ready.store(true, std::memory_order_release);
    return true;
}

bool LamureRenderer::initCamera(ContextResources& res, osg::Camera* context_camera, int viewId)
{
    if (notifyOn()) { std::cout << "[Lamure] LamureRenderer::initCamera(" << viewId << ")" << std::endl; }
    if (!context_camera) {
        return false;
    }
    if (!m_osg_camera.valid()) {
        m_osg_camera = context_camera;
    }
    double look_dist = 1.0;
    double left, right, bottom, top, zNear, zFar;
    osg::Vec3d eye, center, up;
    context_camera->getProjectionMatrixAsFrustum(left, right, bottom, top, zNear, zFar);
    context_camera->getViewMatrixAsLookAt(eye, center, up, look_dist);

    osg::Matrix base = opencover::VRSceneGraph::instance()->getScaleTransform()->getMatrix();
    osg::Matrix trans = opencover::VRSceneGraph::instance()->getTransform()->getMatrix();
    base.postMult(trans);

    osg::Matrixd viewMat = context_camera->getViewMatrix();
    osg::Matrixd projMat = context_camera->getProjectionMatrix();

    res.view_ids[context_camera] = viewId;
    res.scm_cameras[context_camera] = std::make_shared<lamure::ren::camera>(
        static_cast<lamure::view_t>(viewId),
        zNear,
        zFar,
        LamureUtil::matConv4D(viewMat * base),
        LamureUtil::matConv4D(projMat));
    // Workaround for single-context/multi-view path:
    // keep camera mapping strictly data-only and avoid HUD camera manipulation during draw callback.
    return true;
}

unsigned int LamureRenderer::compileShader(unsigned int type, const std::string &source, uint8_t ctx_id, std::string desc)
{
    osg::GLExtensions* gl_api = new osg::GLExtensions(ctx_id);
    unsigned int id = gl_api->glCreateShader(type);
    const char* src = source.c_str();
    gl_api->glShaderSource(id, 1, &src, nullptr);
    gl_api->glCompileShader(id);
    int result;
    gl_api->glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == false)
    {
        int length;
        gl_api->glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        gl_api->glGetShaderInfoLog(id, length, &length, message);
        std::cerr << "[Lamure][ERR] Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader " << desc << "!\n";
        std::cerr << "[Lamure][ERR] " << message << "\n";
        gl_api->glDeleteShader(id);
        return 0;
    };
    return id;
}

GLuint LamureRenderer::compileAndLinkShaders(std::string vs_source, std::string fs_source, uint8_t ctx_id, std::string desc)
{
    GLuint program = glCreateProgram();
    GLuint vs = compileShader(GL_VERTEX_SHADER, vs_source, ctx_id, desc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fs_source, ctx_id, desc);
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

GLuint LamureRenderer::compileAndLinkShaders(std::string vs_source, std::string gs_source, std::string fs_source, uint8_t ctx_id, std::string desc)
{
    GLuint program = glCreateProgram();
    GLuint vs = compileShader(GL_VERTEX_SHADER, vs_source, ctx_id, desc);
    GLuint gs = compileShader(GL_GEOMETRY_SHADER, gs_source, ctx_id, desc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fs_source, ctx_id, desc);
    glAttachShader(program, vs);
    glAttachShader(program, gs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);
    glDeleteShader(vs);
    glDeleteShader(gs);
    glDeleteShader(fs);
    return program;
}


void LamureRenderer::initFrustumResources(ContextResources& res) {
    std::fill(res.frustum_vertices.begin(), res.frustum_vertices.end(), 0.0f);

    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint ibo = 0;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 res.frustum_idx.size() * sizeof(unsigned short),
                 res.frustum_idx.data(),
                 GL_STATIC_DRAW);

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float) * res.frustum_vertices.size(),
                 res.frustum_vertices.data(),
                 GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    res.geo_frustum.vao = vao;
    res.geo_frustum.vbo = vbo;
    res.geo_frustum.ibo = ibo;

    glBindVertexArray(0);
}

void LamureRenderer::updateSharedBoxData() {
    if (notifyOn()) { std::cout << "[Lamure] updateSharedBoxData (CPU precalc)\n"; }

    std::lock_guard<std::mutex> lock(m_sceneMutex); // Protect shared structures

    m_shared_box_vertices.clear();
    m_bvh_node_vertex_offsets.clear();

    if (!m_plugin || m_plugin->getSettings().models.empty())
        return;

    auto* ctrl = lamure::ren::controller::get_instance();
    auto* db   = lamure::ren::model_database::get_instance();

    const auto modelCount = static_cast<uint32_t>(m_plugin->getSettings().models.size());
    
    // Prepare aggregation for scene AABB
    const float fmax = std::numeric_limits<float>::max();
    scm::math::vec3f global_min(fmax, fmax, fmax);
    scm::math::vec3f global_max(-fmax, -fmax, -fmax);

    for (uint32_t model_idx = 0; model_idx < modelCount; ++model_idx) {
        const lamure::model_t m_id = ctrl->deduce_model_id(std::to_string(model_idx));
        const auto* bvh = db->get_model(m_id)->get_bvh();
        const auto& boxes = bvh->get_bounding_boxes();

        std::vector<uint32_t> current_offsets;
        current_offsets.reserve(boxes.size());

        for (uint64_t node_id = 0; node_id < boxes.size(); ++node_id) {
            // Offset in UNITS OF VERTICES (3 floats per vertex)
            current_offsets.push_back(static_cast<uint32_t>(m_shared_box_vertices.size() / 3));
            
            std::vector<float> corners = LamureUtil::getBoxCorners(boxes[node_id]);
            m_shared_box_vertices.insert(m_shared_box_vertices.end(), corners.begin(), corners.end());
        }
        m_bvh_node_vertex_offsets[m_id] = std::move(current_offsets);

        // Compute per-model root AABB in world (exact axis-aligned AABB via 8 corners)
        if (!boxes.empty()) {
            const auto& scene_nodes = m_plugin->getSceneNodes();
            if (model_idx < scene_nodes.size() && scene_nodes[model_idx].model_transform.valid()) {
                const osg::Matrixd model_osg = scene_nodes[model_idx].model_transform->getMatrix();
                const scm::math::mat4f M = scm::math::mat4f(LamureUtil::matConv4D(model_osg));

                const auto corners = LamureUtil::getBoxCorners(boxes[0]); // 8 corners * 3 floats
                const float inf = std::numeric_limits<float>::infinity();
                scm::math::vec3f bbw_min(+inf, +inf, +inf);
                scm::math::vec3f bbw_max(-inf, -inf, -inf);
                for (size_t i = 0; i < 8; ++i) {
                    const size_t k = i * 3;
                    scm::math::vec4f v(corners[k+0], corners[k+1], corners[k+2], 1.f);
                    const scm::math::vec4f tv = M * v;
                    bbw_min.x = std::min(bbw_min.x, tv.x);
                    bbw_min.y = std::min(bbw_min.y, tv.y);
                    bbw_min.z = std::min(bbw_min.z, tv.z);
                    bbw_max.x = std::max(bbw_max.x, tv.x);
                    bbw_max.y = std::max(bbw_max.y, tv.y);
                    bbw_max.z = std::max(bbw_max.z, tv.z);
                }
                scm::math::vec3f Cw = (bbw_min + bbw_max) * 0.5f;

                // Store into plugin meta (keeps vectors aligned with models)
                auto &mi = m_plugin->getModelInfo();
                if (mi.root_bb_min.size() <= m_id) {
                    mi.root_bb_min.resize(m_id+1);
                    mi.root_bb_max.resize(m_id+1);
                    mi.root_center.resize(m_id+1);
                }
                mi.root_bb_min[m_id] = bbw_min;
                mi.root_bb_max[m_id] = bbw_max;
                mi.root_center[m_id]  = Cw;

                global_min = scm::math::vec3f(std::min(global_min.x, bbw_min.x), std::min(global_min.y, bbw_min.y), std::min(global_min.z, bbw_min.z));
                global_max = scm::math::vec3f(std::max(global_max.x, bbw_max.x), std::max(global_max.y, bbw_max.y), std::max(global_max.z, bbw_max.z));
            }
        }
    }

    // Write aggregated scene AABB
    m_plugin->getModelInfo().models_min = global_min;
    m_plugin->getModelInfo().models_max = global_max;
    m_plugin->getModelInfo().models_center = scm::math::vec3d( (global_min.x + global_max.x) * 0.5,
                                                              (global_min.y + global_max.y) * 0.5,
                                                              (global_min.z + global_max.z) * 0.5 );
}

void LamureRenderer::initBoxResources(ContextResources& res) {
    if (notifyOn()) { std::cout << "[Lamure] init_box_resources() for ctx " << (int)res.ctx << "\n"; }

    // No CPU calculation here anymore! Just upload the shared data.
    res.geo_box.destroy();
    res.box_vaos.clear();
    
    GLuint tmpVao = 0, vbo = 0, ibo = 0;
    glGenVertexArrays(1, &tmpVao);
    glBindVertexArray(tmpVao);

    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, res.box_idx.size() * sizeof(GLushort), res.box_idx.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // Upload precomputed data
    if (!m_shared_box_vertices.empty()) {
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(m_shared_box_vertices.size() * sizeof(float)),
                     m_shared_box_vertices.data(),
                     GL_DYNAMIC_DRAW);
    } else {
        std::array<float, 8 * 3> emptyBoxVertices{};
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(emptyBoxVertices.size() * sizeof(float)),
                     emptyBoxVertices.data(),
                     GL_DYNAMIC_DRAW);
    }
    res.geo_box.vao = 0;
    res.geo_box.vbo = vbo;
    res.geo_box.ibo = ibo;

    glBindVertexArray(0);
    if (tmpVao != 0) {
        glDeleteVertexArrays(1, &tmpVao);
    }
}

void LamureRenderer::initPclResources(ContextResources& res){
    if (notifyOn()) std::cout << "[Lamure] initPclResources()\n";

    for (const auto& kv : res.point_vaos) {
        const GLuint vao = kv.second;
        if (vao != 0 && glIsVertexArray(vao) == GL_TRUE) {
            glDeleteVertexArrays(1, &vao);
        }
    }
    res.point_vaos.clear();
    for (const auto& kv : res.screen_quad_vaos) {
        const GLuint vao = kv.second;
        if (vao != 0 && glIsVertexArray(vao) == GL_TRUE) {
            glDeleteVertexArrays(1, &vao);
        }
    }
    res.screen_quad_vaos.clear();

    if (res.geo_screen_quad.vao != 0 && glIsVertexArray(res.geo_screen_quad.vao) == GL_TRUE) {
        glDeleteVertexArrays(1, &res.geo_screen_quad.vao);
    }
    res.geo_screen_quad.vao = 0;
    if (res.geo_screen_quad.vbo != 0 && glIsBuffer(res.geo_screen_quad.vbo) == GL_TRUE) {
        glDeleteBuffers(1, &res.geo_screen_quad.vbo);
    }
    res.geo_screen_quad.vbo = 0;

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    if (vbo != 0) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * res.screen_quad_vertex.size(), res.screen_quad_vertex.data(), GL_STATIC_DRAW);
        res.geo_screen_quad.vbo = vbo;
    }
}

void LamureRenderer::destroyMultipassTarget(MultipassTarget& target){
    if(target.texture_color){
        glDeleteTextures(1,&target.texture_color);
        target.texture_color = 0;
    }
    if(target.texture_normal){
        glDeleteTextures(1,&target.texture_normal);
        target.texture_normal = 0;
    }
    if(target.texture_position){
        glDeleteTextures(1,&target.texture_position);
        target.texture_position = 0;
    }
    if(target.depth_texture){
        glDeleteTextures(1,&target.depth_texture);
        target.depth_texture = 0;
    }
    if(target.fbo){
        glDeleteFramebuffers(1,&target.fbo);
        target.fbo = 0;
    }
    target.width = 0;
    target.height = 0;
}

void LamureRenderer::initializeMultipassTarget(MultipassTarget& target, int width, int height){
    GLint prev_fbo = 0;
    GLint prev_draw_buffer = 0;
    GLint prev_read_buffer = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
    glGetIntegerv(GL_DRAW_BUFFER, &prev_draw_buffer);
    glGetIntegerv(GL_READ_BUFFER, &prev_read_buffer);

    destroyMultipassTarget(target);

    glGenFramebuffers(1,&target.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER,target.fbo);

    glGenTextures(1,&target.texture_color);
    glBindTexture(GL_TEXTURE_2D,target.texture_color);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,width,height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,target.texture_color,0);

    glGenTextures(1,&target.texture_normal);
    glBindTexture(GL_TEXTURE_2D,target.texture_normal);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB16F,width,height,0,GL_RGB,GL_HALF_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,target.texture_normal,0);

    glGenTextures(1,&target.texture_position);
    glBindTexture(GL_TEXTURE_2D,target.texture_position);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB16F,width,height,0,GL_RGB,GL_HALF_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D,target.texture_position,0);

    glGenTextures(1,&target.depth_texture);
    glBindTexture(GL_TEXTURE_2D,target.depth_texture);
    glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,width,height,0,GL_DEPTH_COMPONENT,GL_UNSIGNED_INT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_COMPARE_MODE,GL_NONE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,target.depth_texture,0);

    const GLenum bufs[3]={GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3,bufs);

    const GLenum status=glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status!=GL_FRAMEBUFFER_COMPLETE){
        std::cerr << "[Lamure][ERR] Framebuffer incomplete (" << std::hex << status << std::dec << ") "
                  << width << "x" << height << "\n";
    }

    glBindTexture(GL_TEXTURE_2D,0);
    glBindFramebuffer(GL_FRAMEBUFFER,prev_fbo);
    glDrawBuffer(prev_draw_buffer);
    glReadBuffer(prev_read_buffer);

    target.width = width;
    target.height = height;
}

LamureRenderer::MultipassTarget& LamureRenderer::acquireMultipassTarget(int ctxId, int viewId, int width, int height){
    width = std::max(width, 1);
    height = std::max(height, 1);

    auto& res = getResources(ctxId);
    auto [it, inserted] = res.multipass_targets.try_emplace(viewId);
    if(inserted){
        initializeMultipassTarget(it->second, width, height);
    } else if(it->second.width != width || it->second.height != height){
        initializeMultipassTarget(it->second, width, height);
    }
    return it->second;
}

void LamureRenderer::releaseMultipassTargets(){
    std::scoped_lock lock(m_renderMutex, m_ctx_mutex);
    for (auto& [ctxId, ctx] : m_ctx_res) {
        for(auto& entry : ctx.multipass_targets){
            destroyMultipassTarget(entry.second);
        }
        ctx.multipass_targets.clear();
    }
}

void LamureRenderer::getMatricesFromRenderInfo(osg::RenderInfo& renderInfo, osg::Matrixd& outView, osg::Matrixd& outProj) {
    if (auto* state = renderInfo.getState()) {
        osg::Matrixd state_mv = state->getModelViewMatrix();
        osg::Matrixd state_proj = state->getProjectionMatrix();
        const osg::Camera* currentCamera = renderInfo.getCurrentCamera();

        if (opencover::cover && currentCamera && opencover::coVRConfig::instance()->getEnvMapMode() != opencover::coVRConfig::NONE)
        {
             osg::Matrixd cam_mv = currentCamera->getViewMatrix();
             osg::Matrixd rotonly = cam_mv;
             rotonly(3, 0) = 0.0; rotonly(3, 1) = 0.0; rotonly(3, 2) = 0.0; rotonly(3, 3) = 1.0;
             
             osg::Matrixd invRot;
             invRot.invert(rotonly);

             outView = (state_mv * opencover::cover->envCorrectMat) * rotonly;
             outProj = invRot * opencover::cover->invEnvCorrectMat * state_proj;
        }
        else
        {
            outView = state_mv;
            outProj = state_proj;
        }
    }
}

bool LamureRenderer::getModelViewProjectionFromRenderInfo(osg::RenderInfo& renderInfo, const osg::Node* node,
    osg::Matrixd& outModel, osg::Matrixd& outView, osg::Matrixd& outProj) const
{
    if (!getModelMatrix(node, outModel))
        return false;

    osg::State* state = renderInfo.getState();
    if (!state)
        return false;

    osg::Matrixd state_mv = state->getModelViewMatrix();
    osg::Matrixd state_proj = state->getProjectionMatrix();
    osg::Matrixd inv_model;
    if (!inv_model.invert(outModel))
        return false;
    const osg::Matrixd view_base = inv_model * state_mv;

    const osg::Camera* currentCamera = renderInfo.getCurrentCamera();
    if (opencover::cover && currentCamera && opencover::coVRConfig::instance()->getEnvMapMode() != opencover::coVRConfig::NONE)
    {
        osg::Matrixd cam_mv = currentCamera->getViewMatrix();
        osg::Matrixd rotonly = cam_mv;
        rotonly(3, 0) = 0.0; rotonly(3, 1) = 0.0; rotonly(3, 2) = 0.0; rotonly(3, 3) = 1.0;

        osg::Matrixd invRot;
        invRot.invert(rotonly);

        outView = (view_base * opencover::cover->envCorrectMat) * rotonly;
        outProj = invRot * opencover::cover->invEnvCorrectMat * state_proj;
    }
    else
    {
        outView = view_base;
        outProj = state_proj;
    }
    return true;
}

bool LamureRenderer::getModelMatrix(const osg::Node* node, osg::Matrixd& out) const
{
    out.makeIdentity();
    if (!node)
        return false;

    const osg::Node* objects_root = opencover::cover ? opencover::cover->getObjectsRoot() : nullptr;
    auto is_objects_root = [objects_root](const osg::Node* n) {
        return n && ((objects_root && n == objects_root) || n->getName() == "OBJECTS_ROOT");
    };

    const osg::NodePathList paths = node->getParentalNodePaths();
    if (paths.empty())
        return false;

    const osg::NodePath* chosen_path = nullptr;
    size_t root_index = 0;
    for (const auto& path : paths) {
        for (size_t i = 0; i < path.size(); ++i) {
            if (is_objects_root(path[i])) {
                chosen_path = &path;
                root_index = i;
                break;
            }
        }
        if (chosen_path)
            break;
    }

    if (chosen_path) {
        osg::NodePath subpath(chosen_path->begin() + static_cast<std::ptrdiff_t>(root_index), chosen_path->end());
        out = osg::computeLocalToWorld(subpath);
        return true;
    }

    out = osg::computeLocalToWorld(paths.front());
    return true;
}
