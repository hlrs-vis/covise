#include "LamureRenderer.h"
#include "Lamure.h"
#include "LamureUtil.h"

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
#include <osgViewer/Viewer>
#include <osgViewer/Renderer>
#include <osgText/Text>
#include <osg/NodeVisitor>

#include <lamure/ren/model_database.h>
#include <lamure/ren/cut_database.h>
#include <lamure/ren/controller.h>
#include <lamure/pvs/pvs_database.h>
#include <lamure/ren/policy.h>

#include <chrono>
#include <algorithm>
#include <limits>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <thread>
#include <stdexcept>
#include <unordered_set>
#include <iostream>
#include <gl_state.h>
#include <config/CoviseConfig.h>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

namespace {
    inline bool notifyOn(Lamure* p) { return p && p->getSettings().show_notify; }

    inline bool decideUseAniso(const scm::math::mat4 &projection_matrix, int anisoMode, float threshold)
    {
        // 0=off, 1=auto, 2=on
        if (anisoMode == 2) return true;
        if (anisoMode == 0) return false;
        // Off-axis heuristic: treat small offsets (e.g., stereo eye tiny shifts) as isotropic for performance.
        // Extract the row/column tied to off-axis terms via M * ez (works with our math conversion).
        // Consider anisotropic only if magnitude exceeds a practical threshold.
        const scm::math::vec4 ez(0.0f, 0.0f, 1.0f, 0.0f);
        const scm::math::vec4 v = projection_matrix * ez;
        const float mag = std::max(std::fabs(v[0]), std::fabs(v[1]));
        return mag > std::max(0.0f, threshold);
    }

    struct MeasCtx {
        bool active = false;
        bool sampleThisFrame = false;
        bool fullMode = false;
        bool liteMode = false;
    };

    inline MeasCtx makeMeasCtx(Lamure* plugin) {
        MeasCtx m;
        auto* meas = plugin ? plugin->getMeasurement() : nullptr;
        m.active = (meas && meas->isActive());
        m.sampleThisFrame = m.active ? meas->isSampleFrame() : false;
        auto& s = plugin->getSettings();
        m.fullMode = m.active && s.measure_full;
        m.liteMode = m.active && s.measure_light;
        return m;
    }

    struct SDStats {
        double sum_area_px = 0.0;
        double screen_px   = 0.0;
        float  scale_proj  = 0.0f;
        double k_orient    = 0.70;
    };

    inline SDStats makeSDStats(const MeasCtx& meas,
        const scm::math::vec2& viewport,
        const scm::math::mat4& projection_matrix,
        const Lamure::Settings& s) {
        SDStats sd;
        if (meas.sampleThisFrame) {
            sd.scale_proj = opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5];
            sd.screen_px  = static_cast<double>(viewport.x) * static_cast<double>(viewport.y);
            sd.k_orient   = s.point ? 1.0 : (s.surfel || s.splatting) ? 0.70 : 0.70;
        }
        return sd;
    }

    inline void accumulate_node_area_px(SDStats& sd,
        const lamure::ren::bvh* bvh,
        uint32_t node_id,
        const scm::math::mat4& mvp,
        const Lamure::Settings& s,
        size_t surfels_per_node,
        bool sampleThisFrame)
    {
        if (!sampleThisFrame) return;

        constexpr float EPS = 1e-6f;

        const auto& centroids = bvh->get_centroids();
        const scm::math::vec4f c_ws4(centroids[node_id], 1.0f);
        const scm::math::vec4f clip  = mvp * c_ws4;
        const float w_abs = std::max(EPS, std::fabs(clip.w));

        float r_raw = std::max(0.0f, bvh->get_avg_primitive_extent(node_id));
        if (r_raw <= EPS) return;

        float cut_scale = 1.0f;
        if (s.max_radius_cut > 0.0f && r_raw > s.max_radius_cut) {
            cut_scale = s.max_radius_cut / std::max(1e-6f, r_raw);
            r_raw     = s.max_radius_cut;
        }

        const float gamma = (s.scale_radius_gamma > 0.0f) ? s.scale_radius_gamma : 1.0f;
        float r_ws;
        if      (gamma == 1.0f) r_ws = s.scale_radius * s.scale_element * r_raw;
        else if (gamma == 2.0f) r_ws = s.scale_radius * s.scale_element * r_raw * r_raw;
        else                    r_ws = s.scale_radius * s.scale_element * std::pow(r_raw, gamma);

        r_ws = std::clamp(r_ws, s.min_radius, s.max_radius);
        if (r_ws <= EPS) return;

        float d_px = (2.0f * r_ws * sd.scale_proj) / std::max(EPS, w_abs);
        d_px = std::clamp(d_px, s.min_screen_size, s.max_screen_size);
        if (d_px <= EPS) return;

        const float area_one = (float(M_PI) * 0.25f * d_px * d_px * float(sd.k_orient)) * (cut_scale * cut_scale);
        sd.sum_area_px += static_cast<double>(area_one) * static_cast<double>(surfels_per_node);
    }

    inline void write_sd_metrics_if_sampled(const MeasCtx& meas,
        const SDStats& sd,
        uint64_t rendered_primitives,
        Lamure::RenderInfo& out)
    {
        if (!meas.sampleThisFrame) return;

        const double eps = 1e-9;

        double density_raw    = 0.0;
        double coverage_raw   = 0.0;
        double coverage_px_raw= 0.0;
        double overdraw_raw   = 0.0;

        if (sd.screen_px > eps) {
            density_raw     = sd.sum_area_px / sd.screen_px;
            coverage_raw    = 1.0 - std::exp(-density_raw);
            coverage_px_raw = coverage_raw * sd.screen_px;
            if (coverage_px_raw > eps)
                overdraw_raw = sd.sum_area_px / coverage_px_raw;
        }

        const double cap = std::max(0.0, 50.0);
        const double density_cap   = (cap > eps) ? std::min(density_raw, cap) : density_raw;
        const double coverage_cap  = 1.0 - std::exp(-density_cap);
        const double coverage_px_cap = coverage_cap * sd.screen_px;
        double overdraw_cap = 0.0;
        if (coverage_px_cap > eps)
            overdraw_cap = sd.sum_area_px / coverage_px_cap;

        out.est_screen_px      = float(sd.screen_px);
        out.est_sum_area_px    = float(sd.sum_area_px);
        out.est_density        = float(density_cap);
        out.est_coverage       = float(coverage_cap);
        out.est_coverage_px    = float(coverage_px_cap);
        out.est_overdraw       = float(overdraw_cap);
        out.est_density_raw     = float(density_raw);
        out.est_coverage_raw    = float(coverage_raw);
        out.est_coverage_px_raw = float(coverage_px_raw);
        out.est_overdraw_raw    = float(overdraw_raw);

        out.avg_area_px_per_prim = (rendered_primitives > 0)
            ? float(sd.sum_area_px / double(rendered_primitives)) : 0.0f;
    }

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

} // namespace


LamureRenderer::LamureRenderer(Lamure *plugin)
: m_plugin(plugin)
, m_renderer(this)
, m_rendering(false)
{
}

LamureRenderer::~LamureRenderer()
{
    releaseMultipassTargets();
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
    const int desired = std::min(m_clip_plane_count, kMaxClipPlanes);
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
    const GLint planeCount = std::min(m_clip_plane_count, kMaxClipPlanes);
    if (countLocation >= 0)
        glUniform1i(countLocation, planeCount);
    if (dataLocation >= 0 && planeCount > 0)
        glUniform4fv(dataLocation, planeCount, reinterpret_cast<const GLfloat*>(m_clip_planes.data()));
}

struct InitDrawCallback : public osg::Drawable::DrawCallback
{
    explicit InitDrawCallback(Lamure* plugin)
    : _plugin(plugin)
    , _renderer(plugin ? plugin->getRenderer() : nullptr)
    {
    }

    void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override
    {
        osg::Matrix mv_matrix = opencover::cover->getBaseMat() * _renderer->getOsgCamera()->getViewMatrix();
        scm::math::mat4d modelview_matrix = LamureUtil::matConv4D(mv_matrix);
        scm::math::mat4d projection_matrix = LamureUtil::matConv4D(_renderer->getOsgCamera()->getProjectionMatrix());

        _renderer->setModelViewMatrix(modelview_matrix);
        _renderer->setProjectionMatrix(projection_matrix);

        _renderer->getScmCamera()->set_projection_matrix(projection_matrix);
        if (_plugin->getUI()->getSyncButton()->state()) {
            _renderer->getScmCamera()->set_view_matrix(modelview_matrix);
        }

        if (_renderer && !_initialized) {
            GLState before = GLState::capture();
            _renderer->initSchismObjects();
            _renderer->initFrustumResources();
            _renderer->initBoxResources();
            _renderer->initPclResources();
            _renderer->initLamureShader();
            _renderer->initUniforms();
            _renderer->getPointcloudGeode()->setNodeMask(_plugin->getUI()->getPointcloudButton()->state() ? 0xFFFFFFFF : 0);
            _renderer->getBoundingboxGeode()->setNodeMask(_plugin->getUI()->getBoundingboxButton()->state() ? 0xFFFFFFFF : 0);
            _renderer->getFrustumGeode()->setNodeMask(_plugin->getUI()->getFrustumButton()->state() ? 0xFFFFFFFF : 0);
            _renderer->getTextGeode()->setNodeMask(_plugin->getUI()->getTextButton()->state() ? 0xFFFFFFFF : 0);
            before.restore();
            _initialized = true;
        }

        if (drawable)
            drawable->drawImplementation(renderInfo);
    }

private:
    Lamure* _plugin{nullptr};
    LamureRenderer* _renderer{nullptr};
    mutable bool _initialized{false};
};

struct InitGeometry : public osg::Geometry {
    InitGeometry(Lamure* plugin) : _plugin(plugin) {
        if (notifyOn(_plugin)) { std::cout << "[Lamure] InitGeometry()" << std::endl; }
        setUseDisplayList(false);
        setUseVertexBufferObjects(true);
        setUseVertexArrayObject(false);
        setDrawCallback(new InitDrawCallback(plugin));
        if (auto *stateSet = getOrCreateStateSet()) {
            stateSet->setRenderBinDetails(-10, "RenderBin");
        }
    }
    Lamure* _plugin;
};

struct TextDrawCallback : public osg::Drawable::DrawCallback
{
    TextDrawCallback(Lamure *plugin, osgText::Text *values, Lamure::RenderInfo *render_info)
    : _plugin(plugin)
    , _values(values)
    , _render_info(render_info)
    , _renderer(plugin ? plugin->getRenderer() : nullptr)
    , _lastUpdateTime(std::chrono::steady_clock::now())
    , _minInterval(std::chrono::milliseconds(100))
    {
    }

    void drawImplementation(osg::RenderInfo &renderInfo, const osg::Drawable *drawable) const override
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - _lastUpdateTime >= _minInterval)
        {
                    osg::Matrix baseMatrix = opencover::VRSceneGraph::instance()->getScaleTransform()->getMatrix();
                    osg::Matrix transformMatrix = opencover::VRSceneGraph::instance()->getTransform()->getMatrix();
                    baseMatrix.postMult(transformMatrix);

            if (_renderer)
            {
                osg::Matrixd view_osg;
                osg::Matrixd proj_osg;
                bool haveState = false;

                if (auto *state = renderInfo.getState())
                {
                    view_osg = state->getModelViewMatrix();
                    proj_osg = state->getProjectionMatrix();
                    haveState = true;
                }
                else if (auto osgCam = _renderer->getOsgCamera())
                {
                    view_osg = osgCam->getViewMatrix();
                    proj_osg = osgCam->getProjectionMatrix();
                    haveState = true;
                }

                if (haveState && _renderer->getScmCamera() && _values.valid() && _render_info)
                {
                    const scm::math::vec3d pos = _renderer->getScmCamera()->get_cam_pos();
                    const auto base = LamureUtil::matConv4D(baseMatrix);
                    const auto view = LamureUtil::matConv4D(view_osg);
                    const auto projection = LamureUtil::matConv4D(proj_osg);

                    std::stringstream value_ss;
                    std::stringstream modelview_ss;
                    std::stringstream projection_ss;
                    std::stringstream mvp_ss;

                    modelview_ss << view * base;
                    projection_ss << projection;
                    mvp_ss << projection * view * base;

                    double fpsAvg = 0.0;
                    if (auto *vs = opencover::VRViewer::instance()->getViewerStats())
                    {
                        (void)vs->getAveragedAttribute("Frame rate", fpsAvg);
                    }
                    if (fpsAvg <= 0.0)
                    {
                        const double fd = std::max(1e-6, opencover::cover->frameDuration());
                        fpsAvg = 1.0 / fd;
                    }

                    const double primMio = static_cast<double>(_render_info->rendered_primitives) / 1e6;
                    value_ss << "\n"
                             << std::fixed << std::setprecision(2)
                             << fpsAvg << "\n"
                             << _render_info->rendered_nodes << "\n"
                             << primMio << "\n"
                             << _render_info->rendered_bounding_boxes << "\n\n\n"
                             << pos.x << "\n"
                             << pos.y << "\n"
                             << pos.z << "\n\n\n\n"
                             << modelview_ss.str() << "\n\n\n"
                             << projection_ss.str() << "\n\n\n"
                             << mvp_ss.str() << "\n\n\n";

                    _values->setText(value_ss.str(), osgText::String::ENCODING_UTF8);
                    _lastUpdateTime = now;
                }
                else if (notifyOn(_plugin))
                {
                    std::cerr << "[Lamure] TextDrawCallback: missing renderer state, skip\n";
                }
            }
            else if (notifyOn(_plugin))
            {
                std::cerr << "[Lamure] TextDrawCallback: renderer unavailable\n";
            }
        }
        if (drawable) { drawable->drawImplementation(renderInfo); }
    }

private:
    Lamure *_plugin{nullptr};
    osg::ref_ptr<osgText::Text> _values;
    Lamure::RenderInfo *_render_info{nullptr};
    LamureRenderer *_renderer{nullptr};
    mutable std::chrono::steady_clock::time_point _lastUpdateTime;
    std::chrono::milliseconds _minInterval;
};

struct TextGeode : public osg::Geode
{
    TextGeode(Lamure* plugin)
    {
        if (notifyOn(plugin)) { std::cout << "[Lamure] TextGeode()" << std::endl; }
        osg::Vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
        std::string font = opencover::coVRFileManager::instance()->getFontFile(NULL);
        float characterSize = 18.0f;
        const osg::GraphicsContext::Traits* traits = opencover::coVRConfig::instance()->windows[0].context->getTraits();
        const float marginX = 12.f, marginY = 12.f;
        const float labelColumnOffset = 120.f;
        osg::Vec3 pos_label(traits->width - marginX - labelColumnOffset, traits->height - marginY, 0.0f);
        osg::Vec3 pos_value(traits->width - marginX,                      traits->height - marginY, 0.0f);
        osg::ref_ptr<osgText::Text> label = new osgText::Text();
        label->setColor(color);
        label->setFont(font);
        label->setCharacterSizeMode(osgText::TextBase::SCREEN_COORDS);
        label->setCharacterSize(characterSize);
        label->setAlignment(osgText::TextBase::RIGHT_TOP);
        label->setAutoRotateToScreen(false);
        label->setPosition(pos_label);
        std::stringstream label_ss;
        label_ss << "FPS:" << "\n"
            << "Nodes:" << "\n"
            << "Primitives (Mio):" << "\n"
            << "Boxes:" << "\n\n"
            << "Frustum Position" << "\n"
            << "X:" << "\n"
            << "Y:" << "\n"
            << "Z:" << "\n\n\n"
            << "ModelView" << "\n\n\n\n\n\n"
            << "Projection" << "\n\n\n\n\n\n"
            << "MVP" << "\n\n\n\n\n\n";
        label->setText(label_ss.str(), osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> value = new osgText::Text();
        value->setColor(color);
        value->setFont(font);
        value->setCharacterSizeMode(osgText::TextBase::SCREEN_COORDS);
        value->setCharacterSize(characterSize);
        value->setAlignment(osgText::TextBase::RIGHT_TOP);
        value->setAutoRotateToScreen(false);
        value->setPosition(pos_value);
        std::stringstream value_ss;
        value_ss << "\n"
            << "0.00:" << "\n"
            << "0.00" << "\n"
            << "0.00" << "\n"
            << "0.00:" << "\n\n\n"
            << "0.00" << "\n"
            << "0.00" << "\n"
            << "0.00" << "\n\n\n\n\n"
            << "0.00" << "\n\n\n\n";
        value->setText(value_ss.str(), osgText::String::ENCODING_UTF8);
        this->addDrawable(label.get());
        this->addDrawable(value.get());
        value->setDrawCallback(new TextDrawCallback(plugin, value.get(), &plugin->getRenderInfo()));
    }
};

struct FrustumDrawCallback : public osg::Drawable::DrawCallback
{
    FrustumDrawCallback(Lamure* plugin)
        : _plugin(plugin)
    {
        if (notifyOn(_plugin)) { std::cout << "[Lamure] FrustumDrawCallback()" << std::endl; }
        _renderer = _plugin->getRenderer();
    }

    virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override
    {
        GLState before = GLState::capture();

        scm::math::mat4f mvp_matrix = scm::math::mat4f(_renderer->getProjectionMatrix() * _renderer->getModelViewMatrix());
        std::vector<scm::math::vec3d> corner_values = _renderer->getScmCamera()->get_frustum_corners();

        for (size_t i = 0; i < corner_values.size(); ++i) {
            auto vv = scm::math::vec3f(corner_values[i]);
            _renderer->getFrustumResource().vertices[i * 3 + 0] = vv.x;
            _renderer->getFrustumResource().vertices[i * 3 + 1] = vv.y;
            _renderer->getFrustumResource().vertices[i * 3 + 2] = vv.z;
        }

        //glEnable(GL_DEPTH_CLAMP);
        //glDisable(GL_DEPTH_TEST);
        glLineWidth(1);
        glBindVertexArray(_renderer->getFrustumResource().vao);
        glBindBuffer(GL_ARRAY_BUFFER, _renderer->getFrustumResource().vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * _renderer->getFrustumResource().vertices.size(), _renderer->getFrustumResource().vertices.data());
        glUseProgram(_renderer->getLineShader().program);
        glUniformMatrix4fv(_renderer->getLineShader().mvp_matrix_location, 1, GL_FALSE, mvp_matrix.data_array);
        glUniform4f(_renderer->getLineShader().in_color_location, _plugin->getSettings().frustum_color[0], _plugin->getSettings().frustum_color[1], _plugin->getSettings().frustum_color[2], _plugin->getSettings().frustum_color[3]);
        glDrawElements(GL_LINES, _renderer->getFrustumResource().idx.size(), GL_UNSIGNED_SHORT, nullptr);

        before.restore();
    }
    Lamure* _plugin;
    LamureRenderer* _renderer;
};

struct FrustumGeometry : public osg::Geometry
{
    FrustumGeometry(Lamure* _plugin)
    {
        if (notifyOn(_plugin)) { std::cout << "[Lamure] FrustumGeometry()" << std::endl; }
        setUseDisplayList(false);
        setUseVertexBufferObjects(true);
        setUseVertexArrayObject(false);
        setDrawCallback(new FrustumDrawCallback(_plugin));
    }
};

struct BoundingBoxDrawCallback : public virtual osg::Drawable::DrawCallback
{
    BoundingBoxDrawCallback(Lamure* plugin)
        : _plugin(plugin), _initialized(false)
    {
        if (notifyOn(_plugin)) { std::cout << "[Lamure] BoundingBoxDrawCallback()" << std::endl; }
        _renderer = _plugin->getRenderer();
    }

    virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const 
    {
        GLState before = GLState::capture();

        GLint prevVAO = 0, prevProg = 0;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
        glGetIntegerv(GL_CURRENT_PROGRAM,      &prevProg);

        osg::State* state = renderInfo.getState();
        osg::Matrixd osg_view_matrix = state->getModelViewMatrix();
        osg::Matrixd osg_projection_matrix = state->getProjectionMatrix();
        const scm::math::mat4 view_matrix = LamureUtil::matConv4F(osg_view_matrix);
        const scm::math::mat4 projection_matrix = LamureUtil::matConv4F(osg_projection_matrix);

        lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
        lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
        lamure::ren::controller* controller = lamure::ren::controller::get_instance();
        lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();

        // Use the current render state's context for multi-camera setups
        lamure::context_t context_id = controller->deduce_context_id(renderInfo.getState()->getContextID());
        for (lamure::model_t m_id = 0; m_id < _plugin->getSettings().models.size(); ++m_id) {
            lamure::model_t model_id = controller->deduce_model_id(std::to_string(m_id));
            cuts->send_transform(context_id, model_id, scm::math::mat4(_plugin->getModelInfo().model_transformations[model_id]));
            cuts->send_threshold(context_id, model_id, _plugin->getSettings().lod_error);
            cuts->send_rendered(context_id, model_id);
            database->get_model(model_id)->set_transform(scm::math::mat4(_plugin->getModelInfo().model_transformations[model_id]));
        }

        lamure::view_t view_id = controller->deduce_view_id(context_id, _renderer->getScmCamera()->view_id());
        cuts->send_camera(context_id, view_id, *_renderer->getScmCamera());
        std::vector<scm::math::vec3d> corner_values = _renderer->getScmCamera()->get_frustum_corners();
        double top_minus_bottom = scm::math::length((corner_values[2]) - (corner_values[0]));
        float height_divided_by_top_minus_bottom = opencover::coVRConfig::instance()->windows[0].context->getTraits()->height / top_minus_bottom;
        cuts->send_height_divided_by_top_minus_bottom(context_id, view_id, height_divided_by_top_minus_bottom);

        if (_plugin->getSettings().use_pvs) {
            scm::math::vec3d cam_pos = _renderer->getScmCamera()->get_cam_pos();
            pvs->set_viewer_position(cam_pos);
        }

        if (_plugin->getSettings().lod_update && !_plugin->isResetInProgress()) {
            try {
                if (lamure::ren::policy::get_instance()->size_of_provenance() > 0)
                { controller->dispatch(context_id, _renderer->getDevice(), _plugin->getDataProvenance()); }
                else { controller->dispatch(context_id, _renderer->getDevice()); }
            } catch (const std::exception &e) {
                if (notifyOn(_plugin)) std::cerr << "[Lamure] dispatch skipped: " << e.what() << "\n";
            }
        }

        glBindVertexArray(_renderer->getBoxResource().vao);
        glUseProgram(_renderer->getLineShader().program);
        glUniform4f(_renderer->getLineShader().in_color_location, _plugin->getSettings().bvh_color[0], _plugin->getSettings().bvh_color[1], _plugin->getSettings().bvh_color[2], _plugin->getSettings().bvh_color[3]);

        uint64_t rendered_bounding_boxes = 0;
        for (uint16_t m_id = 0; m_id < _plugin->getSettings().models.size(); ++m_id) {
            if (!_renderer->isModelVisible(m_id))
                continue;
            const lamure::model_t model_id = controller->deduce_model_id(std::to_string(m_id));
            lamure::ren::cut& cut = cuts->get_cut(context_id, renderInfo.getState()->getContextID(), model_id);

            const auto& renderable = cut.complete_set();
            if (renderable.empty())
                continue;
            const lamure::ren::bvh* bvh = database->get_model(model_id)->get_bvh();
            const std::vector<scm::gl::boxf>& bbv = bvh->get_bounding_boxes();

            const scm::math::mat4 model_matrix = scm::math::mat4(_plugin->getModelInfo().model_transformations[model_id]);
            const scm::math::mat4 model_view_matrix = view_matrix * model_matrix;
            const scm::math::mat4 mvp_matrix = projection_matrix * model_view_matrix;
            const scm::gl::frustum frustum     = _renderer->getScmCamera()->get_frustum_by_model(model_matrix);

            glUniformMatrix4fv(_renderer->getLineShader().mvp_matrix_location, 1, GL_FALSE, mvp_matrix.data_array);
            const auto it = _renderer->m_bvh_node_vertex_offsets.find(model_id);
            if (it == _renderer->m_bvh_node_vertex_offsets.end())
                continue;

            const std::vector<uint32_t>& node_offsets = it->second;

            for (const auto& node_slot_aggregate : renderable) {
                const uint32_t node_id = node_slot_aggregate.node_id_;
                if (node_id >= bbv.size() || node_id >= node_offsets.size())
                    continue;

                const uint32_t cull = _renderer->getScmCamera()->cull_against_frustum(frustum, bbv[node_id]);
                if (cull != 1) {
                    glDrawElementsBaseVertex(
                        GL_LINES,
                        static_cast<GLsizei>(_renderer->getBoxResource().idx.size()),
                        GL_UNSIGNED_SHORT,
                        nullptr,
                        static_cast<GLint>(node_offsets[node_id])
                    );
                    ++rendered_bounding_boxes;
                }
            }
        }

        glUseProgram(prevProg);
        glBindVertexArray(prevVAO);
        _plugin->getRenderInfo().rendered_bounding_boxes = rendered_bounding_boxes;
        before.restore();

        if (notifyOn(_plugin)) {
            GLState after = GLState::capture();
            GLState::compare(before, after, "[Lamure] BoundingBoxDrawCallback::drawImplementation()");
        }

    };
    Lamure* _plugin;
    LamureRenderer* _renderer;
    mutable bool _initialized;
};

struct BoundingBoxGeometry : public osg::Geometry
{
    BoundingBoxGeometry(Lamure* plugin)
    {
        if (notifyOn(plugin)) { std::cout << "[Lamure] BoundingBoxGeometry()" << std::endl; }
        setUseDisplayList(false);
        setUseVertexBufferObjects(true);
        setUseVertexArrayObject(false);
        setDrawCallback(new BoundingBoxDrawCallback(plugin));
    }
};

struct PointsDrawCallback : public virtual osg::Drawable::DrawCallback
{
    PointsDrawCallback(Lamure* plugin)
        : _plugin(plugin), _initialized(false)
    {
        if (notifyOn(_plugin)) { std::cout << "[Lamure] PointsDrawCallback()" << std::endl; } 
        _renderer = _plugin->getRenderer();
    }

    virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
    {
        if (!_renderer->beginFrame()) { return; }
        if (_plugin->getSettings().models.empty()) { _renderer->endFrame(); return; }

        const MeasCtx meas = makeMeasCtx(_plugin);
        const auto& settings = _plugin->getSettings();

        GLState before = GLState::capture();
        glDisable(GL_CULL_FACE);

        _renderer->updateActiveClipPlanes();
        const bool useClipPlanes = (_renderer->clipPlaneCount() > 0);
        ClipDistanceScope clipScope(_renderer, useClipPlanes);

        osg::State* state = renderInfo.getState();
        state->setCheckForGLErrors(osg::State::CheckForGLErrors::NEVER_CHECK_GL_ERRORS);

        const osg::Matrixd osg_view_matrix = state->getModelViewMatrix();
        const osg::Matrixd osg_projection_matrix = state->getProjectionMatrix();
        const osg::Camera* currentCamera = renderInfo.getCurrentCamera();

        scm::math::mat4 view_matrix = LamureUtil::matConv4F(osg_view_matrix);
        scm::math::mat4 projection_matrix = LamureUtil::matConv4F(osg_projection_matrix);
        lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
        lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
        lamure::ren::controller* controller = lamure::ren::controller::get_instance();
        lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();

        lamure::context_t context_id = controller->deduce_context_id(renderInfo.getState()->getContextID());
        lamure::view_t    view_id = controller->deduce_view_id(context_id, _renderer->getScmCamera()->view_id());
        size_t surfels_per_node = database->get_primitives_per_node();

        if (database->num_models() == 0) { _renderer->endFrame(); before.restore(); return; }
        for (lamure::model_t model_id = 0; model_id < settings.models.size(); ++model_id) {
            if (!_renderer->isModelVisible(static_cast<std::size_t>(model_id)))
                continue;
            lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));
            const auto &trafo = _plugin->getModelInfo().model_transformations[m_id];
            cuts->send_transform(context_id, m_id, scm::math::mat4(trafo));
            cuts->send_threshold(context_id, m_id, _plugin->getSettings().lod_error);
            cuts->send_rendered(context_id, m_id);
            database->get_model(m_id)->set_transform(scm::math::mat4(trafo));
        }
        cuts->send_camera(context_id, view_id, *(_renderer->getScmCamera()));
        std::vector<scm::math::vec3d> corner_values = _renderer->getScmCamera()->get_frustum_corners();
        double top_minus_bottom = scm::math::length((corner_values[2]) - (corner_values[0]));
        float height_divided_by_top_minus_bottom = opencover::coVRConfig::instance()->windows[0].context->getTraits()->height / top_minus_bottom;
        cuts->send_height_divided_by_top_minus_bottom(context_id, view_id, height_divided_by_top_minus_bottom);

        if (_plugin->getSettings().use_pvs) {
            scm::math::vec3d cam_pos = _renderer->getScmCamera()->get_cam_pos();
            pvs->set_viewer_position(cam_pos);
        }
        if (_plugin->getSettings().lod_update) {
            if (lamure::ren::policy::get_instance()->size_of_provenance() > 0)
            { controller->dispatch(context_id, _renderer->getDevice(), _plugin->getDataProvenance()); }
            else { controller->dispatch(context_id, _renderer->getDevice()); }
        }
        if (_initialized) { glBindVertexArray(_renderer->getPclResource().vao); }
        _renderer->getContext()->apply_vertex_input();

        if (lamure::ren::policy::get_instance()->size_of_provenance() > 0) {
            _renderer->getContext()->bind_vertex_array(controller->get_context_memory(context_id, lamure::ren::bvh::primitive_type::POINTCLOUD, _renderer->getDevice(), _plugin->getDataProvenance()));
        }
        else { _renderer->getContext()->bind_vertex_array(controller->get_context_memory(context_id, lamure::ren::bvh::primitive_type::POINTCLOUD, _renderer->getDevice())); }
        
        // Use active camera viewport dimensions
        const osg::Viewport* osg_viewport = currentCamera->getViewport();
        const auto* traits = (currentCamera->getGraphicsContext()) ? currentCamera->getGraphicsContext()->getTraits() : nullptr;
        const double vpW = osg_viewport ? osg_viewport->width() : (traits ? traits->width : 1.0);
        const double vpH = osg_viewport ? osg_viewport->height() : (traits ? traits->height : 1.0);
        const scm::math::mat4d viewport_scale = scm::math::make_scale(vpW * 0.5, vpH * 0.5, 0.5);
        const scm::math::mat4d viewport_translate = scm::math::make_translation(1.0, 1.0, 1.0);
        scm::math::vec2 viewport((float)vpW, (float)vpH);
        const bool useAnisoThisPass = decideUseAniso(projection_matrix, settings.anisotropic_surfel_scaling, settings.anisotropic_auto_threshold);
        SDStats sd = makeSDStats(meas, viewport, projection_matrix, settings);
        uint64_t rendered_primitives = 0;
        uint64_t rendered_nodes = 0;

        if (_plugin->getSettings().shader_type == LamureRenderer::ShaderType::SurfelMultipass && _initialized) {
            // ================= MULTI-PASS RENDER PATH =================
            auto&       shared = _renderer->getPclResource();
            const auto& s   = _plugin->getSettings();
            const bool enableColorDebug = s.coloring;
            const bool showNormalsDebug = enableColorDebug && s.show_normals;
            const bool showAccuracyDebug = enableColorDebug && s.show_accuracy;
            const bool showRadiusDeviationDebug = enableColorDebug && s.show_radius_deviation;
            const bool showOutputSensitivityDebug = enableColorDebug && s.show_output_sensitivity;

            const int vpWidth  = static_cast<int>(vpW);
            const int vpHeight = static_cast<int>(vpH);
            lamure::context_t context_id = controller->deduce_context_id(renderInfo.getState()->getContextID());
            auto& target = _renderer->acquireMultipassTarget(context_id, currentCamera, vpWidth, vpHeight);
            GLint prev_fbo = 0;
            GLint prev_draw_buffer = 0;
            GLint prev_read_buffer = 0;
            GLint prev_viewport[4] = {0,0,0,0};
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
            glGetIntegerv(GL_DRAW_BUFFER, &prev_draw_buffer);
            glGetIntegerv(GL_READ_BUFFER, &prev_read_buffer);
            glGetIntegerv(GL_VIEWPORT, prev_viewport);

            const float scale_radius_combined = s.scale_radius * s.scale_element;
            const float scale_proj_pass = opencover::cover->getScale() * static_cast<float>(vpHeight) * 0.5f * projection_matrix.data_array[5];

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

            // --- PASS 1: Depth pre-pass (depth-only, kreisfoermiges discard im FS)
            glBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
            glViewport(0, 0, vpWidth, vpHeight);
            glDrawBuffer(GL_NONE);
            glReadBuffer(GL_NONE);
            glClear(GL_DEPTH_BUFFER_BIT);
            glDisable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LEQUAL);

            glUseProgram(_renderer->getSurfelPass1Shader().program);

            // Globale Uniforms (nur setzen, wenn vorhanden)
            if (_renderer->getSurfelPass1Shader().viewport_loc          >= 0) glUniform2f(_renderer->getSurfelPass1Shader().viewport_loc, static_cast<float>(vpWidth), static_cast<float>(vpHeight));
            if (_renderer->getSurfelPass1Shader().max_radius_loc        >= 0) glUniform1f(_renderer->getSurfelPass1Shader().max_radius_loc,   s.max_radius);
            if (_renderer->getSurfelPass1Shader().min_radius_loc        >= 0) glUniform1f(_renderer->getSurfelPass1Shader().min_radius_loc,   s.min_radius);
            if (_renderer->getSurfelPass1Shader().min_screen_size_loc   >= 0) glUniform1f(_renderer->getSurfelPass1Shader().min_screen_size_loc, s.min_screen_size);
            if (_renderer->getSurfelPass1Shader().max_screen_size_loc   >= 0) glUniform1f(_renderer->getSurfelPass1Shader().max_screen_size_loc, s.max_screen_size);
            if (_renderer->getSurfelPass1Shader().scale_radius_loc      >= 0) glUniform1f(_renderer->getSurfelPass1Shader().scale_radius_loc, scale_radius_combined);
            if (_renderer->getSurfelPass1Shader().scale_radius_gamma_loc  >= 0) glUniform1f(_renderer->getSurfelPass1Shader().scale_radius_gamma_loc,   s.scale_radius_gamma);
            if (_renderer->getSurfelPass1Shader().max_radius_cut_loc      >= 0) glUniform1f(_renderer->getSurfelPass1Shader().max_radius_cut_loc,   s.max_radius_cut);
            if (_renderer->getSurfelPass1Shader().scale_projection_loc    >= 0) glUniform1f(_renderer->getSurfelPass1Shader().scale_projection_loc, scale_proj_pass);
            if (_renderer->getSurfelPass1Shader().projection_matrix_loc >= 0) glUniformMatrix4fv(_renderer->getSurfelPass1Shader().projection_matrix_loc, 1, GL_FALSE, projection_matrix.data_array);
            if (_renderer->getSurfelPass1Shader().use_aniso_loc          >= 0) glUniform1i(_renderer->getSurfelPass1Shader().use_aniso_loc, useAnisoThisPass ? 1 : 0);
            for (uint16_t m_id = 0; m_id < s.models.size(); ++m_id) {
                if (!_renderer->isModelVisible(m_id))
                    continue;

                const lamure::model_t model_id = controller->deduce_model_id(std::to_string(m_id));
                
                lamure::ren::cut &cut = cuts->get_cut(context_id, renderInfo.getState()->getContextID(), model_id);
                const auto& renderable = cut.complete_set();
                if (renderable.empty())
                    continue;
                
                const lamure::ren::bvh *bvh = database->get_model(model_id)->get_bvh();
                const auto &bbox = bvh->get_bounding_boxes();
                scm::math::mat4 model_matrix      = scm::math::mat4(_plugin->getModelInfo().model_transformations[m_id]);
                scm::math::mat4 model_view_matrix = view_matrix * model_matrix;
                scm::math::mat3 normal_matrix     = scm::math::transpose(scm::math::inverse(LamureUtil::matConv4to3F(model_view_matrix)));
                scm::gl::frustum frustum          = _renderer->getScmCamera()->get_frustum_by_model(model_matrix);

                if (_renderer->getSurfelPass1Shader().model_view_matrix_loc >= 0)
                    glUniformMatrix4fv(_renderer->getSurfelPass1Shader().model_view_matrix_loc, 1, GL_FALSE, model_view_matrix.data_array);
                for (auto const &node_slot_aggregate : renderable) {
                    if (_renderer->getScmCamera()->cull_against_frustum(frustum, bbox[node_slot_aggregate.node_id_]) != 1) {
                        accumulate_node_area_px(sd, bvh, node_slot_aggregate.node_id_, projection_matrix*model_view_matrix, settings, surfels_per_node, meas.sampleThisFrame);
                        glDrawArrays(scm::gl::PRIMITIVE_POINT_LIST, (node_slot_aggregate.slot_id_) * (GLsizei)surfels_per_node, surfels_per_node);
                        rendered_primitives += surfels_per_node;
                        ++rendered_nodes;
                    }
                }
            }

            // --- PASS 2: Accumulation (additiv), FS macht NDC-Depth-Vergleich + Randmaske
            GLenum accumBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
            glDrawBuffers(3, accumBuffers);
            glClearColor(0.f, 0.f, 0.f, 0.f);
            glClear(GL_COLOR_BUFFER_BIT);

            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDepthFunc(GL_ALWAYS);

            glUseProgram(_renderer->getSurfelPass2Shader().program);
            glViewport(0, 0, vpWidth, vpHeight);

            // Tiefe aus Pass 1
            bindTexture2DToUnit(0, target.depth_texture);

            // Globale Uniforms fuer VS/GS/FS
            if (_renderer->getSurfelPass2Shader().viewport_loc            >= 0) glUniform2f(_renderer->getSurfelPass2Shader().viewport_loc, viewport.x, viewport.y);
            if (_renderer->getSurfelPass2Shader().max_radius_loc          >= 0) glUniform1f(_renderer->getSurfelPass2Shader().max_radius_loc,   s.max_radius);
            if (_renderer->getSurfelPass2Shader().min_radius_loc          >= 0) glUniform1f(_renderer->getSurfelPass2Shader().min_radius_loc,   s.min_radius);
            if (_renderer->getSurfelPass2Shader().scale_radius_loc        >= 0) glUniform1f(_renderer->getSurfelPass2Shader().scale_radius_loc, scale_radius_combined);
            if (_renderer->getSurfelPass2Shader().scale_radius_gamma_loc  >= 0) glUniform1f(_renderer->getSurfelPass2Shader().scale_radius_gamma_loc,   s.scale_radius_gamma);
            if (_renderer->getSurfelPass2Shader().max_radius_cut_loc      >= 0) glUniform1f(_renderer->getSurfelPass2Shader().max_radius_cut_loc,  s.max_radius_cut);
            if (_renderer->getSurfelPass2Shader().coloring_loc            >= 0) glUniform1i(_renderer->getSurfelPass2Shader().coloring_loc, enableColorDebug ? 1 : 0);     
            if (_renderer->getSurfelPass2Shader().show_normals_loc        >= 0) glUniform1i(_renderer->getSurfelPass2Shader().show_normals_loc,         showNormalsDebug ? 1 : 0);
            if (_renderer->getSurfelPass2Shader().show_output_sens_loc    >= 0) glUniform1i(_renderer->getSurfelPass2Shader().show_output_sens_loc,     showOutputSensitivityDebug ? 1 : 0);
            if (_renderer->getSurfelPass2Shader().show_radius_dev_loc     >= 0) glUniform1i(_renderer->getSurfelPass2Shader().show_radius_dev_loc,      showRadiusDeviationDebug ? 1 : 0);
            if (_renderer->getSurfelPass2Shader().show_accuracy_loc       >= 0) glUniform1i(_renderer->getSurfelPass2Shader().show_accuracy_loc,        showAccuracyDebug ? 1 : 0);
            if (_renderer->getSurfelPass2Shader().projection_matrix_loc   >= 0) glUniformMatrix4fv(_renderer->getSurfelPass2Shader().projection_matrix_loc, 1, GL_FALSE, projection_matrix.data_array);
            if (_renderer->getSurfelPass2Shader().min_screen_size_loc    >= 0) glUniform1f(_renderer->getSurfelPass2Shader().min_screen_size_loc, s.min_screen_size);
            if (_renderer->getSurfelPass2Shader().max_screen_size_loc    >= 0) glUniform1f(_renderer->getSurfelPass2Shader().max_screen_size_loc, s.max_screen_size);
            if (_renderer->getSurfelPass2Shader().scale_projection_loc >= 0) glUniform1f(_renderer->getSurfelPass2Shader().scale_projection_loc, scale_proj_pass);
            if (_renderer->getSurfelPass2Shader().use_aniso_loc          >= 0) glUniform1i(_renderer->getSurfelPass2Shader().use_aniso_loc, useAnisoThisPass ? 1 : 0);

            // Blending-Uniforms
            if (_renderer->getSurfelPass2Shader().depth_range_loc >= 0) glUniform1f(_renderer->getSurfelPass2Shader().depth_range_loc, s.depth_range);
            if (_renderer->getSurfelPass2Shader().flank_lift_loc  >= 0) glUniform1f(_renderer->getSurfelPass2Shader().flank_lift_loc, s.flank_lift);

            const bool needNodeUniforms = (showRadiusDeviationDebug || showAccuracyDebug);

            for (uint16_t m_id = 0; m_id < s.models.size(); ++m_id) {
                if (!_renderer->isModelVisible(m_id))
                    continue;
                const lamure::model_t model_id = controller->deduce_model_id(std::to_string(m_id));
                lamure::ren::cut &cut = cuts->get_cut(context_id, renderInfo.getState()->getContextID(), model_id);
                const auto& renderable = cut.complete_set();
                if (renderable.empty())
                    continue;
                const lamure::ren::bvh *bvh = database->get_model(model_id)->get_bvh();
                const auto &bbox = bvh->get_bounding_boxes();

                scm::math::mat4 model_matrix      = scm::math::mat4(_plugin->getModelInfo().model_transformations[m_id]);
                scm::math::mat4 model_view_matrix = view_matrix * model_matrix;
                scm::math::mat3 normal_matrix     = scm::math::transpose(scm::math::inverse(LamureUtil::matConv4to3F(model_view_matrix)));
                scm::gl::frustum frustum          = _renderer->getScmCamera()->get_frustum_by_model(model_matrix);

                if (_renderer->getSurfelPass2Shader().model_view_matrix_loc >= 0)
                    glUniformMatrix4fv(_renderer->getSurfelPass2Shader().model_view_matrix_loc, 1, GL_FALSE, model_view_matrix.data_array);
                if (_renderer->getSurfelPass2Shader().normal_matrix_loc >= 0)
                    glUniformMatrix3fv(_renderer->getSurfelPass2Shader().normal_matrix_loc, 1, GL_FALSE, normal_matrix.data_array);

                for (auto const &node_slot_aggregate : renderable) {
                    if (_renderer->getScmCamera()->cull_against_frustum(frustum, bbox[node_slot_aggregate.node_id_]) != 1) {
                        if (needNodeUniforms) { _renderer->setNodeUniforms(bvh, node_slot_aggregate.node_id_); }
                        glDrawArrays(scm::gl::PRIMITIVE_POINT_LIST,
                            (node_slot_aggregate.slot_id_) * (GLsizei)surfels_per_node,
                            surfels_per_node);
                    }
                }
            }

            // --- PASS 3: Resolve / Lighting (premultiplied coverage)

            glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
            glDrawBuffer(prev_draw_buffer);
            glReadBuffer(prev_read_buffer);
            glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);

            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LEQUAL);
            glDisable(GL_BLEND);
            //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA); // premultiplied resolve

            glUseProgram(_renderer->getSurfelPass3Shader().program);

            // G-Buffer
            bindTexture2DToUnit(0, target.texture_color);
            bindTexture2DToUnit(1, target.texture_normal);
            bindTexture2DToUnit(2, target.texture_position);
            bindTexture2DToUnit(3, target.depth_texture);

            // View-space lighting Setup
            scm::math::mat4 viewMat = view_matrix;
            scm::math::vec4 light_ws(s.point_light_pos[0], s.point_light_pos[1], s.point_light_pos[2], 1.0f);
            scm::math::vec4 light_vs4 = viewMat * light_ws;
            scm::math::vec3 light_vs(light_vs4[0], light_vs4[1], light_vs4[2]);

            if (_renderer->getSurfelPass3Shader().point_light_pos_vs_loc      >= 0) glUniform3fv(_renderer->getSurfelPass3Shader().point_light_pos_vs_loc, 1, light_vs.data_array);
            if (_renderer->getSurfelPass3Shader().point_light_intensity_loc   >= 0) glUniform1f(_renderer->getSurfelPass3Shader().point_light_intensity_loc, s.point_light_intensity);
            if (_renderer->getSurfelPass3Shader().ambient_intensity_loc       >= 0) glUniform1f(_renderer->getSurfelPass3Shader().ambient_intensity_loc,     s.ambient_intensity);
            if (_renderer->getSurfelPass3Shader().specular_intensity_loc      >= 0) glUniform1f(_renderer->getSurfelPass3Shader().specular_intensity_loc,    s.specular_intensity);
            if (_renderer->getSurfelPass3Shader().shininess_loc               >= 0) glUniform1f(_renderer->getSurfelPass3Shader().shininess_loc,             s.shininess);
            if (_renderer->getSurfelPass3Shader().gamma_loc                   >= 0) glUniform1f(_renderer->getSurfelPass3Shader().gamma_loc,                 s.gamma);
            if (_renderer->getSurfelPass3Shader().use_tone_mapping_loc        >= 0) glUniform1i(_renderer->getSurfelPass3Shader().use_tone_mapping_loc,      s.use_tone_mapping ? 1 : 0);
            if (_renderer->getSurfelPass3Shader().lighting_loc                >= 0) glUniform1f(_renderer->getSurfelPass3Shader().lighting_loc, s.lighting);  

            // Fullscreen-Quad
            glBindVertexArray(shared.screen_quad_vao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);
            glActiveTexture(GL_TEXTURE0);
            glDepthMask(GL_FALSE);
        }
        else {
            // ================= SINGLE-PASS RENDER PATH =================
            _renderer->setFrameUniforms(projection_matrix, viewport);
            for (uint16_t model_id = 0; model_id < _plugin->getSettings().models.size(); ++model_id) {
                if (!_renderer->isModelVisible(model_id))
                    continue;
                const lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));
                lamure::ren::cut& cut = cuts->get_cut(context_id, renderInfo.getState()->getContextID(), m_id);
                const auto& renderable = cut.complete_set();
                if (renderable.empty())
                    continue;

                const lamure::ren::bvh* bvh = database->get_model(m_id)->get_bvh();
                std::vector<scm::gl::boxf>const& bounding_box_vector = bvh->get_bounding_boxes();

                scm::math::mat4 model_matrix = scm::math::mat4(_plugin->getModelInfo().model_transformations[m_id]);
                scm::math::mat4 model_view_matrix = view_matrix * model_matrix;
                scm::math::mat4 mvp_matrix = projection_matrix * model_view_matrix;
                scm::gl::frustum frustum = _renderer->getScmCamera()->get_frustum_by_model(model_matrix);

                _renderer->setModelUniforms(mvp_matrix, model_matrix);
                for (auto const& node_slot_aggregate : renderable) {
                    
                    if (_renderer->getScmCamera()->cull_against_frustum(frustum, bounding_box_vector[node_slot_aggregate.node_id_]) != 1) {
                        _renderer->setNodeUniforms(bvh, node_slot_aggregate.node_id_);
                        accumulate_node_area_px(sd, bvh, node_slot_aggregate.node_id_, projection_matrix*model_view_matrix, settings, surfels_per_node, meas.sampleThisFrame);
                        glDrawArrays(scm::gl::PRIMITIVE_POINT_LIST, (node_slot_aggregate.slot_id_) * (GLsizei)surfels_per_node, surfels_per_node);
                        rendered_primitives += surfels_per_node;
                        ++rendered_nodes;
                    }
                }
            }
        }

        _plugin->getRenderInfo().rendered_primitives = rendered_primitives;
        _plugin->getRenderInfo().rendered_nodes = rendered_nodes;
        write_sd_metrics_if_sampled(meas, sd, rendered_primitives, _plugin->getRenderInfo());

        _renderer->endFrame();

        if (!_initialized) {
            GLState after = GLState::capture();
            if (after.getVertexArrayBinding() != before.getVertexArrayBinding())
            {
                _plugin->getRenderer()->getPclResource().vao = after.getVertexArrayBinding();
                _initialized = true;
            }
        }

        before.restore();
        if (notifyOn(_plugin)) {
            GLState after = GLState::capture();
            GLState::compare(before, after, "[Lamure] PointsDrawCallback::drawImplementation()");
        }
    }
    Lamure* _plugin;
    LamureRenderer* _renderer;
    mutable bool _initialized;
};


struct PointsGeometry : public osg::Geometry
{
    PointsGeometry(Lamure* plugin) : _plugin(plugin)
    {
        if (notifyOn(_plugin)) { std::cout << "[Lamure] PointsGeometry()" << std::endl; }
        setUseDisplayList(false);
        setUseVertexBufferObjects(true);
        setUseVertexArrayObject(false);
        setDrawCallback(new PointsDrawCallback(_plugin));
    }
    Lamure* _plugin;
    osg::BoundingSphere _bsphere;
    osg::BoundingBox _bbox;
};

void LamureRenderer::syncEditBrushGeometry()
{
    if (!m_edit_brush_geode)
        return;

    m_edit_brush_geode->removeDrawables(0, m_edit_brush_geode->getNumDrawables());

    osg::ref_ptr<osg::Shape> shape;
    shape = new osg::Sphere(osg::Vec3(), 1.0f);

    osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(shape.get());
    drawable->setName("LamureEditBrushDrawable");
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
        m_edit_brush_transform->setName("LamureEditBrush");
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
    if (notifyOn(m_plugin)) { std::cout << "[Lamure] LamureRenderer::init()" << std::endl; }
    std::lock_guard<std::mutex> sceneLock(m_sceneMutex);
    initCamera();

    if (auto* vs = opencover::VRViewer::instance()->getViewerStats()) {
        vs->collectStats("frame_rate", true);
    }

    m_init_geode         = new osg::Geode();
    m_pointcloud_geode   = new osg::Geode();
    m_boundingbox_geode  = new osg::Geode();
    m_frustum_geode      = new osg::Geode();
    m_text_geode         = new TextGeode(m_plugin);
    m_edit_brush_transform = new osg::MatrixTransform();
    m_edit_brush_transform->setName("LamureEditBrush");
    m_edit_brush_geode = new osg::Geode();
    m_edit_brush_transform->addChild(m_edit_brush_geode.get());

    m_init_stateset = new osg::StateSet();
    m_pointcloud_stateset = new osg::StateSet();
    m_boundingbox_stateset = new osg::StateSet();
    m_frustum_stateset = new osg::StateSet();
    m_text_stateset = new osg::StateSet();

    m_text_stateset->setRenderBinDetails(10, "RenderBin");
    auto ui = m_plugin->getUI();

    ui->getPointcloudButton()->setState(   m_plugin->getSettings().show_pointcloud );
    ui->getBoundingboxButton()->setState(  m_plugin->getSettings().show_boundingbox );
    ui->getFrustumButton()->setState(      m_plugin->getSettings().show_frustum );
    ui->getTextButton()->setState(         m_plugin->getSettings().show_text );
    ui->getSyncButton()->setState(         m_plugin->getSettings().show_sync );
    ui->getNotifyButton()->setState(       m_plugin->getSettings().show_notify );

    m_pointcloud_geode->setNodeMask(0);
    m_boundingbox_geode->setNodeMask(0);
    m_frustum_geode->setNodeMask(0);
    m_text_geode->setNodeMask(0);

    ui->getPointcloudButton()->setCallback([this](bool state) { 
        m_pointcloud_geode->setNodeMask(state ? 0xFFFFFFFF : 0x0); 
        if (!state) m_plugin->getRenderInfo().rendered_primitives = 0;
        if (!state) m_plugin->getRenderInfo().rendered_nodes = 0;
        });
    ui->getBoundingboxButton()->setCallback([this](bool state) { 
        m_boundingbox_geode->setNodeMask(state ? 0xFFFFFFFF : 0x0); 
        if (!state) m_plugin->getRenderInfo().rendered_bounding_boxes = 0;
        });
    ui->getFrustumButton()->setCallback([this](bool state) { m_frustum_geode->setNodeMask(state ? 0xFFFFFFFF : 0x0); });
    ui->getTextButton()->setCallback([this](bool state) { m_text_geode->setNodeMask(state ? 0xFFFFFFFF : 0x0); });
    ui->getDumpButton()->setCallback([this](bool state) {  });

    m_init_geode->setStateSet(m_init_stateset.get());
    m_pointcloud_geode->setStateSet(m_pointcloud_stateset.get());
    m_boundingbox_geode->setStateSet(m_boundingbox_stateset.get());
    m_frustum_geode->setStateSet(m_frustum_stateset.get());
    m_text_geode->setStateSet(m_text_stateset.get());

    m_plugin->getGroup()->addChild(m_frustum_geode);
    m_plugin->getGroup()->addChild(m_pointcloud_geode);
    m_plugin->getGroup()->addChild(m_boundingbox_geode);
    m_plugin->getGroup()->addChild(m_init_geode);
    m_hud_camera->addChild(m_text_geode.get());
    m_plugin->getGroup()->addChild(m_edit_brush_transform.get());

    // Configure HUD camera for screen-space text (ortho 2D in pixels)
    if (m_hud_camera.valid())
    {
        m_hud_camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
        m_hud_camera->setViewMatrix(osg::Matrix::identity());
        int W = opencover::coVRConfig::instance()->windows[0].context->getTraits()->width;
        int H = opencover::coVRConfig::instance()->windows[0].context->getTraits()->height;
        m_hud_camera->setProjectionMatrix(osg::Matrix::ortho2D(0.0, double(W), 0.0, double(H)));
    }

    m_init_geometry = new InitGeometry(m_plugin);
    m_pointcloud_geometry = new PointsGeometry(m_plugin);
    m_boundingbox_geometry = new BoundingBoxGeometry(m_plugin);
    m_frustum_geometry = new FrustumGeometry(m_plugin);

    m_init_geode->addDrawable(m_init_geometry);
    m_pointcloud_geode->addDrawable(m_pointcloud_geometry);
    m_boundingbox_geode->addDrawable(m_boundingbox_geometry);
    m_frustum_geode->addDrawable(m_frustum_geometry);

    // Avoid OSG frustum-culling for lamure point cloud; lamure does its own culling
    if (m_pointcloud_geode.valid())
        m_pointcloud_geode->setCullingActive(false);
}

void LamureRenderer::detachCallbacks()
{
    if (m_pointcloud_geometry.valid()) m_pointcloud_geometry->setDrawCallback(nullptr);
    if (m_boundingbox_geometry.valid()) m_boundingbox_geometry->setDrawCallback(nullptr);
    if (m_frustum_geometry.valid())     m_frustum_geometry->setDrawCallback(nullptr);
    if (m_init_geometry.valid())        m_init_geometry->setDrawCallback(nullptr);
}

void LamureRenderer::shutdown()
{
    // Wait for pools to idle before shutdown to avoid races
    if (auto* ctrl = lamure::ren::controller::get_instance()) {
        if (m_osg_camera.valid() && m_osg_camera->getGraphicsContext() && m_osg_camera->getGraphicsContext()->getState()) {
            lamure::context_t ctx = m_osg_camera->getGraphicsContext()->getState()->getContextID();
            ctrl->wait_for_idle(ctx);
        }
        ctrl->shutdown_pools();
    }
    if (auto* cache = lamure::ren::ooc_cache::get_instance()) {
        cache->wait_for_idle();
        cache->shutdown_pool();
    }

    releaseMultipassTargets();


    if (m_plugin && m_plugin->getGroup()) {
        if (m_frustum_geode.valid())    m_plugin->getGroup()->removeChild(m_frustum_geode);
        if (m_boundingbox_geode.valid())m_plugin->getGroup()->removeChild(m_boundingbox_geode);
        if (m_pointcloud_geode.valid()) m_plugin->getGroup()->removeChild(m_pointcloud_geode);
        if (m_init_geode.valid())       m_plugin->getGroup()->removeChild(m_init_geode);
    }

    if (m_osg_camera.valid() && m_hud_camera.valid())
        m_osg_camera->removeChild(m_hud_camera.get());

    if (m_hud_camera.valid())
        m_hud_camera->removeChildren(0, m_hud_camera->getNumChildren());

    m_init_geode = nullptr;
    m_pointcloud_geode = nullptr;
    m_boundingbox_geode = nullptr;
    m_frustum_geode = nullptr;
    m_text_geode = nullptr;
    m_init_stateset = nullptr;
    m_pointcloud_stateset = nullptr;
    m_boundingbox_stateset = nullptr;
    m_frustum_stateset = nullptr;
    m_text_stateset = nullptr;
    m_init_geometry = nullptr;
    m_pointcloud_geometry = nullptr;
    m_boundingbox_geometry = nullptr;
    m_frustum_geometry = nullptr;
    m_osg_camera = nullptr;
    m_hud_camera = nullptr;
    m_scm_camera = nullptr;

    //m_device.reset();
    //m_context.reset();

    lamure::ren::controller::destroy_instance();
    lamure::ren::ooc_cache::destroy_instance();
    lamure::ren::cut_database::destroy_instance();
    lamure::ren::model_database::destroy_instance();
}

bool LamureRenderer::beginFrame()
{
    std::unique_lock<std::mutex> lock(m_renderMutex);
    if ((!m_renderingAllowed && !m_pauseRequested) || m_rendering)
        return false;
    if (m_pauseRequested && m_framesPendingDrain == 0)
        return false;
    m_rendering = true;
    return true;
}

void LamureRenderer::endFrame()
{
    bool performFinish = false;
    {
        std::lock_guard<std::mutex> lock(m_renderMutex);
        m_rendering = false;

        if (m_pauseRequested)
        {
            if (m_framesPendingDrain > 0)
            {
                --m_framesPendingDrain;
                performFinish = true;
            }

            if (m_framesPendingDrain == 0)
            {
                m_renderingAllowed = false;
                m_pauseRequested = false;
            }
        }
    }
    if (performFinish) { glFinish(); }
    m_renderCondition.notify_all();
}

bool LamureRenderer::pauseAndWaitForIdle(uint32_t extraDrainFrames)
{
    std::unique_lock<std::mutex> lock(m_renderMutex);

    if (!m_renderingAllowed && !m_pauseRequested)
    {
        m_renderCondition.wait(lock, [this]() { return !m_rendering; });
        return false;
    }

    if (m_pauseRequested)
    {
        const uint32_t framesToDrain = std::max<uint32_t>(extraDrainFrames, 1u);
        if (m_framesPendingDrain < framesToDrain)
            m_framesPendingDrain = framesToDrain;
        return true;
    }

    m_pauseRequested = true;
    uint32_t framesToDrain = std::max<uint32_t>(extraDrainFrames, 1u);
    if (m_framesPendingDrain < framesToDrain)
        m_framesPendingDrain = framesToDrain;

    if (!m_rendering)
    {
        m_framesPendingDrain = 0;
        m_renderingAllowed = false;
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

    m_renderCondition.wait(lock, [this]() { return !m_rendering && !m_pauseRequested; });

    return true;
}

void LamureRenderer::resumeRendering()
{
    {
        std::lock_guard<std::mutex> lock(m_renderMutex);
        m_framesPendingDrain = 0;
        m_pauseRequested = false;
        m_renderingAllowed = true;
    }
    m_renderCondition.notify_all();
}

bool LamureRenderer::isRendering() const
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    return m_rendering;
}

void LamureRenderer::initUniforms() 
{
    if (notifyOn(m_plugin)) { std::cout << "[Lamure] LamureRenderer::initUniforms()" << std::endl; }
    glUseProgram(m_point_shader.program);
    m_point_shader.mvp_matrix_loc   = glGetUniformLocation(m_point_shader.program, "mvp_matrix");
    m_point_shader.model_matrix_loc = glGetUniformLocation(m_point_shader.program, "model_matrix");
    m_point_shader.clip_plane_count_loc = glGetUniformLocation(m_point_shader.program, "clip_plane_count");
    m_point_shader.clip_plane_data_loc  = glGetUniformLocation(m_point_shader.program, "clip_planes");
    m_point_shader.max_radius_loc   = glGetUniformLocation(m_point_shader.program, "max_radius");
    m_point_shader.min_radius_loc   = glGetUniformLocation(m_point_shader.program, "min_radius");
    m_point_shader.max_screen_size_loc   = glGetUniformLocation(m_point_shader.program, "max_screen_size");
    m_point_shader.min_screen_size_loc   = glGetUniformLocation(m_point_shader.program, "min_screen_size");
    m_point_shader.scale_radius_loc = glGetUniformLocation(m_point_shader.program, "scale_radius");
    m_point_shader.scale_projection_loc   = glGetUniformLocation(m_point_shader.program, "scale_projection");
    m_point_shader.max_radius_cut_loc = glGetUniformLocation(m_point_shader.program, "max_radius_cut");
    m_point_shader.scale_radius_gamma_loc   = glGetUniformLocation(m_point_shader.program, "scale_radius_gamma");
    m_point_shader.proj_col0_loc    = glGetUniformLocation(m_point_shader.program, "Pcol0");
    m_point_shader.proj_col1_loc    = glGetUniformLocation(m_point_shader.program, "Pcol1");
    m_point_shader.viewport_half_y_loc = glGetUniformLocation(m_point_shader.program, "viewport_half_y");
    m_point_shader.use_aniso_loc    = glGetUniformLocation(m_point_shader.program, "use_aniso");
    m_point_shader.aniso_normalize_loc = glGetUniformLocation(m_point_shader.program, "aniso_normalize");
    glUseProgram(0);

    glUseProgram(m_point_color_shader.program);
    m_point_color_shader.mvp_matrix_loc   = glGetUniformLocation(m_point_color_shader.program, "mvp_matrix");
    m_point_color_shader.model_matrix_loc = glGetUniformLocation(m_point_color_shader.program, "model_matrix");
    m_point_color_shader.clip_plane_count_loc = glGetUniformLocation(m_point_color_shader.program, "clip_plane_count");
    m_point_color_shader.clip_plane_data_loc  = glGetUniformLocation(m_point_color_shader.program, "clip_planes");
    m_point_color_shader.view_matrix_loc         = glGetUniformLocation(m_point_color_shader.program, "view_matrix");
    m_point_color_shader.normal_matrix_loc     = glGetUniformLocation(m_point_color_shader.program, "normal_matrix");
    m_point_color_shader.max_radius_loc   = glGetUniformLocation(m_point_color_shader.program, "max_radius");
    m_point_color_shader.min_radius_loc   = glGetUniformLocation(m_point_color_shader.program, "min_radius");
    m_point_color_shader.max_screen_size_loc   = glGetUniformLocation(m_point_color_shader.program, "max_screen_size");
    m_point_color_shader.min_screen_size_loc   = glGetUniformLocation(m_point_color_shader.program, "min_screen_size");
    m_point_color_shader.max_radius_cut_loc = glGetUniformLocation(m_point_color_shader.program, "max_radius_cut");
    m_point_color_shader.scale_radius_gamma_loc   = glGetUniformLocation(m_point_color_shader.program, "scale_radius_gamma");
    m_point_color_shader.scale_radius_loc = glGetUniformLocation(m_point_color_shader.program, "scale_radius");
    m_point_color_shader.scale_projection_loc   = glGetUniformLocation(m_point_color_shader.program, "scale_projection");
    m_point_color_shader.proj_col0_loc    = glGetUniformLocation(m_point_color_shader.program, "Pcol0");
    m_point_color_shader.proj_col1_loc    = glGetUniformLocation(m_point_color_shader.program, "Pcol1");
    m_point_color_shader.viewport_half_y_loc = glGetUniformLocation(m_point_color_shader.program, "viewport_half_y");
    m_point_color_shader.use_aniso_loc    = glGetUniformLocation(m_point_color_shader.program, "use_aniso");
    m_point_color_shader.aniso_normalize_loc = glGetUniformLocation(m_point_color_shader.program, "aniso_normalize");
    m_point_color_shader.show_normals_loc         = glGetUniformLocation(m_point_color_shader.program, "show_normals");
    m_point_color_shader.show_accuracy_loc        = glGetUniformLocation(m_point_color_shader.program, "show_accuracy");
    m_point_color_shader.show_radius_dev_loc      = glGetUniformLocation(m_point_color_shader.program, "show_radius_deviation");
    m_point_color_shader.show_output_sens_loc     = glGetUniformLocation(m_point_color_shader.program, "show_output_sensitivity");
    m_point_color_shader.accuracy_loc             = glGetUniformLocation(m_point_color_shader.program, "accuracy");
    m_point_color_shader.average_radius_loc       = glGetUniformLocation(m_point_color_shader.program, "average_radius");
    glUseProgram(0);

    glUseProgram(m_point_color_lighting_shader.program);
    m_point_color_lighting_shader.mvp_matrix_loc          = glGetUniformLocation(m_point_color_lighting_shader.program, "mvp_matrix");
    m_point_color_lighting_shader.model_matrix_loc        = glGetUniformLocation(m_point_color_lighting_shader.program, "model_matrix");
    m_point_color_lighting_shader.clip_plane_count_loc    = glGetUniformLocation(m_point_color_lighting_shader.program, "clip_plane_count");
    m_point_color_lighting_shader.clip_plane_data_loc     = glGetUniformLocation(m_point_color_lighting_shader.program, "clip_planes");
    m_point_color_lighting_shader.view_matrix_loc         = glGetUniformLocation(m_point_color_lighting_shader.program, "view_matrix");
    m_point_color_lighting_shader.normal_matrix_loc     = glGetUniformLocation(m_point_color_lighting_shader.program, "normal_matrix");
    m_point_color_lighting_shader.max_radius_loc          = glGetUniformLocation(m_point_color_lighting_shader.program, "max_radius");
    m_point_color_lighting_shader.min_radius_loc          = glGetUniformLocation(m_point_color_lighting_shader.program, "min_radius");
    m_point_color_lighting_shader.max_screen_size_loc          = glGetUniformLocation(m_point_color_lighting_shader.program, "max_screen_size");
    m_point_color_lighting_shader.min_screen_size_loc          = glGetUniformLocation(m_point_color_lighting_shader.program, "min_screen_size");
    m_point_color_lighting_shader.scale_radius_loc        = glGetUniformLocation(m_point_color_lighting_shader.program, "scale_radius");
    m_point_color_lighting_shader.scale_projection_loc   = glGetUniformLocation(m_point_color_lighting_shader.program, "scale_projection");
    m_point_color_lighting_shader.proj_col0_loc    = glGetUniformLocation(m_point_color_lighting_shader.program, "Pcol0");
    m_point_color_lighting_shader.proj_col1_loc    = glGetUniformLocation(m_point_color_lighting_shader.program, "Pcol1");
    m_point_color_lighting_shader.viewport_half_y_loc = glGetUniformLocation(m_point_color_lighting_shader.program, "viewport_half_y");
    m_point_color_lighting_shader.use_aniso_loc    = glGetUniformLocation(m_point_color_lighting_shader.program, "use_aniso");
    m_point_color_lighting_shader.aniso_normalize_loc = glGetUniformLocation(m_point_color_lighting_shader.program, "aniso_normalize");
    m_point_color_lighting_shader.max_radius_cut_loc = glGetUniformLocation(m_point_color_lighting_shader.program, "max_radius_cut");
    m_point_color_lighting_shader.scale_radius_gamma_loc    = glGetUniformLocation(m_point_color_lighting_shader.program, "scale_radius_gamma");
    m_point_color_lighting_shader.view_matrix_loc         = glGetUniformLocation(m_point_color_lighting_shader.program, "view_matrix");
    m_point_color_lighting_shader.normal_matrix_loc     = glGetUniformLocation(m_point_color_lighting_shader.program, "normal_matrix");
    m_point_color_lighting_shader.show_normals_loc        = glGetUniformLocation(m_point_color_lighting_shader.program, "show_normals");
    m_point_color_lighting_shader.show_accuracy_loc       = glGetUniformLocation(m_point_color_lighting_shader.program, "show_accuracy");
    m_point_color_lighting_shader.show_radius_dev_loc     = glGetUniformLocation(m_point_color_lighting_shader.program, "show_radius_deviation");
    m_point_color_lighting_shader.show_output_sens_loc    = glGetUniformLocation(m_point_color_lighting_shader.program, "show_output_sensitivity");
    m_point_color_lighting_shader.accuracy_loc            = glGetUniformLocation(m_point_color_lighting_shader.program, "accuracy");
    m_point_color_lighting_shader.average_radius_loc      = glGetUniformLocation(m_point_color_lighting_shader.program, "average_radius");
    m_point_color_lighting_shader.point_light_pos_vs_loc      = glGetUniformLocation(m_point_color_lighting_shader.program, "point_light_pos_vs");
    m_point_color_lighting_shader.point_light_intensity_loc   = glGetUniformLocation(m_point_color_lighting_shader.program, "point_light_intensity");
    m_point_color_lighting_shader.ambient_intensity_loc       = glGetUniformLocation(m_point_color_lighting_shader.program, "ambient_intensity");
    m_point_color_lighting_shader.specular_intensity_loc      = glGetUniformLocation(m_point_color_lighting_shader.program, "specular_intensity");
    m_point_color_lighting_shader.shininess_loc               = glGetUniformLocation(m_point_color_lighting_shader.program, "shininess");
    m_point_color_lighting_shader.gamma_loc                   = glGetUniformLocation(m_point_color_lighting_shader.program, "gamma");
    m_point_color_lighting_shader.use_tone_mapping_loc        = glGetUniformLocation(m_point_color_lighting_shader.program, "use_tone_mapping");
    glUseProgram(0);

    glUseProgram(m_point_prov_shader.program);
    m_point_prov_shader.mvp_matrix_loc   = glGetUniformLocation(m_point_prov_shader.program, "mvp_matrix");
    m_point_prov_shader.model_matrix_loc = glGetUniformLocation(m_point_prov_shader.program, "model_matrix");
    m_point_prov_shader.clip_plane_count_loc = glGetUniformLocation(m_point_prov_shader.program, "clip_plane_count");
    m_point_prov_shader.clip_plane_data_loc  = glGetUniformLocation(m_point_prov_shader.program, "clip_planes");
    m_point_prov_shader.max_radius_loc   = glGetUniformLocation(m_point_prov_shader.program, "max_radius");
    m_point_prov_shader.min_radius_loc   = glGetUniformLocation(m_point_prov_shader.program, "min_radius");
    m_point_prov_shader.max_screen_size_loc   = glGetUniformLocation(m_point_prov_shader.program, "max_screen_size");
    m_point_prov_shader.min_screen_size_loc   = glGetUniformLocation(m_point_prov_shader.program, "min_screen_size");
    m_point_prov_shader.scale_radius_loc = glGetUniformLocation(m_point_prov_shader.program, "scale_radius");
    m_point_prov_shader.max_radius_cut_loc = glGetUniformLocation(m_point_prov_shader.program, "max_radius_cut");
    m_point_prov_shader.scale_radius_gamma_loc    = glGetUniformLocation(m_point_prov_shader.program, "scale_radius_gamma");
    m_point_prov_shader.scale_projection_loc   = glGetUniformLocation(m_point_prov_shader.program, "scale_projection");
    m_point_prov_shader.show_normals_loc         = glGetUniformLocation(m_point_prov_shader.program, "show_normals");
    m_point_prov_shader.show_accuracy_loc        = glGetUniformLocation(m_point_prov_shader.program, "show_accuracy");
    m_point_prov_shader.show_radius_dev_loc      = glGetUniformLocation(m_point_prov_shader.program, "show_radius_deviation");
    m_point_prov_shader.show_output_sens_loc     = glGetUniformLocation(m_point_prov_shader.program, "show_output_sensitivity");
    m_point_prov_shader.accuracy_loc             = glGetUniformLocation(m_point_prov_shader.program, "accuracy");
    m_point_prov_shader.average_radius_loc       = glGetUniformLocation(m_point_prov_shader.program, "average_radius");
    m_point_prov_shader.channel_loc              = glGetUniformLocation(m_point_prov_shader.program, "channel");
    m_point_prov_shader.heatmap_loc              = glGetUniformLocation(m_point_prov_shader.program, "heatmap");
    m_point_prov_shader.heatmap_min_loc          = glGetUniformLocation(m_point_prov_shader.program, "heatmap_min");
    m_point_prov_shader.heatmap_max_loc          = glGetUniformLocation(m_point_prov_shader.program, "heatmap_max");
    m_point_prov_shader.heatmap_min_color_loc    = glGetUniformLocation(m_point_prov_shader.program, "heatmap_min_color");
    m_point_prov_shader.heatmap_max_color_loc    = glGetUniformLocation(m_point_prov_shader.program, "heatmap_max_color");
    glUseProgram(0);

    glUseProgram(m_surfel_shader.program);
    m_surfel_shader.mvp_matrix_loc          = glGetUniformLocation(m_surfel_shader.program, "mvp_matrix");
    m_surfel_shader.model_matrix_loc        = glGetUniformLocation(m_surfel_shader.program, "model_matrix");
    m_surfel_shader.clip_plane_count_loc    = glGetUniformLocation(m_surfel_shader.program, "clip_plane_count");
    m_surfel_shader.clip_plane_data_loc     = glGetUniformLocation(m_surfel_shader.program, "clip_planes");
    m_surfel_shader.max_radius_loc          = glGetUniformLocation(m_surfel_shader.program, "max_radius");
    m_surfel_shader.min_radius_loc          = glGetUniformLocation(m_surfel_shader.program, "min_radius");
    m_surfel_shader.max_screen_size_loc          = glGetUniformLocation(m_surfel_shader.program, "max_screen_size"); 
    m_surfel_shader.min_screen_size_loc          = glGetUniformLocation(m_surfel_shader.program, "min_screen_size");
    m_surfel_shader.scale_radius_loc        = glGetUniformLocation(m_surfel_shader.program, "scale_radius");
    m_surfel_shader.scale_projection_loc   = glGetUniformLocation(m_surfel_shader.program, "scale_projection");
    m_surfel_shader.max_radius_cut_loc = glGetUniformLocation(m_surfel_shader.program, "max_radius_cut");
    m_surfel_shader.scale_radius_gamma_loc   = glGetUniformLocation(m_surfel_shader.program, "scale_radius_gamma");
    m_surfel_shader.viewport_loc             = glGetUniformLocation(m_surfel_shader.program, "viewport");
    m_surfel_shader.use_aniso_loc            = glGetUniformLocation(m_surfel_shader.program, "use_aniso");
    glUseProgram(0);

    glUseProgram(m_surfel_color_shader.program);
    m_surfel_color_shader.mvp_matrix_loc    = glGetUniformLocation(m_surfel_color_shader.program, "mvp_matrix");
    m_surfel_color_shader.model_matrix_loc  = glGetUniformLocation(m_surfel_color_shader.program, "model_matrix");
    m_surfel_color_shader.clip_plane_count_loc = glGetUniformLocation(m_surfel_color_shader.program, "clip_plane_count");
    m_surfel_color_shader.clip_plane_data_loc  = glGetUniformLocation(m_surfel_color_shader.program, "clip_planes");
    m_surfel_color_shader.view_matrix_loc         = glGetUniformLocation(m_surfel_color_shader.program, "view_matrix");
    m_surfel_color_shader.normal_matrix_loc       = glGetUniformLocation(m_surfel_color_shader.program, "normal_matrix");
    m_surfel_color_shader.max_radius_loc    = glGetUniformLocation(m_surfel_color_shader.program, "max_radius");
    m_surfel_color_shader.min_radius_loc    = glGetUniformLocation(m_surfel_color_shader.program, "min_radius");
    m_surfel_color_shader.max_screen_size_loc = glGetUniformLocation(m_surfel_color_shader.program, "max_screen_size"); 
    m_surfel_color_shader.min_screen_size_loc = glGetUniformLocation(m_surfel_color_shader.program, "min_screen_size");
    m_surfel_color_shader.scale_radius_loc  = glGetUniformLocation(m_surfel_color_shader.program, "scale_radius");
    m_surfel_color_shader.max_radius_cut_loc = glGetUniformLocation(m_surfel_color_shader.program, "max_radius_cut");
    m_surfel_color_shader.scale_radius_gamma_loc   = glGetUniformLocation(m_surfel_color_shader.program, "scale_radius_gamma");
    m_surfel_color_shader.viewport_loc      = glGetUniformLocation(m_surfel_color_shader.program, "viewport");
    m_surfel_color_shader.scale_projection_loc = glGetUniformLocation(m_surfel_color_shader.program, "scale_projection");
    m_surfel_color_shader.use_aniso_loc     = glGetUniformLocation(m_surfel_color_shader.program, "use_aniso");
    m_surfel_color_shader.show_normals_loc      = glGetUniformLocation(m_surfel_color_shader.program, "show_normals");
    m_surfel_color_shader.show_accuracy_loc     = glGetUniformLocation(m_surfel_color_shader.program, "show_accuracy");
    m_surfel_color_shader.show_radius_dev_loc   = glGetUniformLocation(m_surfel_color_shader.program, "show_radius_deviation");
    m_surfel_color_shader.show_output_sens_loc  = glGetUniformLocation(m_surfel_color_shader.program, "show_output_sensitivity");
    m_surfel_color_shader.accuracy_loc          = glGetUniformLocation(m_surfel_color_shader.program, "accuracy");
    m_surfel_color_shader.average_radius_loc    = glGetUniformLocation(m_surfel_color_shader.program, "average_radius");
    glUseProgram(0);

    glUseProgram(m_surfel_color_lighting_shader.program);
    m_surfel_color_lighting_shader.mvp_matrix_loc          = glGetUniformLocation(m_surfel_color_lighting_shader.program, "mvp_matrix");
    m_surfel_color_lighting_shader.model_matrix_loc        = glGetUniformLocation(m_surfel_color_lighting_shader.program, "model_matrix");
    m_surfel_color_lighting_shader.clip_plane_count_loc    = glGetUniformLocation(m_surfel_color_lighting_shader.program, "clip_plane_count");
    m_surfel_color_lighting_shader.clip_plane_data_loc     = glGetUniformLocation(m_surfel_color_lighting_shader.program, "clip_planes");
    m_surfel_color_lighting_shader.view_matrix_loc         = glGetUniformLocation(m_surfel_color_lighting_shader.program, "view_matrix");
    m_surfel_color_lighting_shader.normal_matrix_loc       = glGetUniformLocation(m_surfel_color_lighting_shader.program, "normal_matrix");
    m_surfel_color_lighting_shader.max_radius_loc          = glGetUniformLocation(m_surfel_color_lighting_shader.program, "max_radius");
    m_surfel_color_lighting_shader.min_radius_loc          = glGetUniformLocation(m_surfel_color_lighting_shader.program, "min_radius");
    m_surfel_color_lighting_shader.max_screen_size_loc          = glGetUniformLocation(m_surfel_color_lighting_shader.program, "max_screen_size");
    m_surfel_color_lighting_shader.min_screen_size_loc          = glGetUniformLocation(m_surfel_color_lighting_shader.program, "min_screen_size");
    m_surfel_color_lighting_shader.scale_radius_loc        = glGetUniformLocation(m_surfel_color_lighting_shader.program, "scale_radius");
    m_surfel_color_lighting_shader.max_radius_cut_loc = glGetUniformLocation(m_surfel_color_lighting_shader.program, "max_radius_cut");
    m_surfel_color_lighting_shader.scale_radius_gamma_loc   = glGetUniformLocation(m_surfel_color_lighting_shader.program, "scale_radius_gamma");
    m_surfel_color_lighting_shader.viewport_loc            = glGetUniformLocation(m_surfel_color_lighting_shader.program, "viewport");
    m_surfel_color_lighting_shader.scale_projection_loc   = glGetUniformLocation(m_surfel_color_lighting_shader.program, "scale_projection");
    m_surfel_color_lighting_shader.use_aniso_loc          = glGetUniformLocation(m_surfel_color_lighting_shader.program, "use_aniso");
    m_surfel_color_lighting_shader.show_normals_loc        = glGetUniformLocation(m_surfel_color_lighting_shader.program, "show_normals");
    m_surfel_color_lighting_shader.show_accuracy_loc       = glGetUniformLocation(m_surfel_color_lighting_shader.program, "show_accuracy");
    m_surfel_color_lighting_shader.show_radius_dev_loc     = glGetUniformLocation(m_surfel_color_lighting_shader.program, "show_radius_deviation");
    m_surfel_color_lighting_shader.show_output_sens_loc    = glGetUniformLocation(m_surfel_color_lighting_shader.program, "show_output_sensitivity");
    m_surfel_color_lighting_shader.accuracy_loc            = glGetUniformLocation(m_surfel_color_lighting_shader.program, "accuracy");
    m_surfel_color_lighting_shader.average_radius_loc      = glGetUniformLocation(m_surfel_color_lighting_shader.program, "average_radius");
    m_surfel_color_lighting_shader.point_light_pos_vs_loc      = glGetUniformLocation(m_surfel_color_lighting_shader.program, "point_light_pos_vs");
    m_surfel_color_lighting_shader.point_light_intensity_loc   = glGetUniformLocation(m_surfel_color_lighting_shader.program, "point_light_intensity");
    m_surfel_color_lighting_shader.ambient_intensity_loc       = glGetUniformLocation(m_surfel_color_lighting_shader.program, "ambient_intensity");
    m_surfel_color_lighting_shader.specular_intensity_loc      = glGetUniformLocation(m_surfel_color_lighting_shader.program, "specular_intensity");
    m_surfel_color_lighting_shader.shininess_loc               = glGetUniformLocation(m_surfel_color_lighting_shader.program, "shininess");
    m_surfel_color_lighting_shader.gamma_loc                   = glGetUniformLocation(m_surfel_color_lighting_shader.program, "gamma");
    m_surfel_color_lighting_shader.use_tone_mapping_loc        = glGetUniformLocation(m_surfel_color_lighting_shader.program, "use_tone_mapping");
    glUseProgram(0);

    glUseProgram(m_surfel_prov_shader.program);
    m_surfel_prov_shader.mvp_matrix_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "mvp_matrix");
    m_surfel_prov_shader.model_matrix_loc  = glGetUniformLocation(m_surfel_prov_shader.program, "model_matrix");
    m_surfel_prov_shader.clip_plane_count_loc = glGetUniformLocation(m_surfel_prov_shader.program, "clip_plane_count");
    m_surfel_prov_shader.clip_plane_data_loc  = glGetUniformLocation(m_surfel_prov_shader.program, "clip_planes");
    m_surfel_prov_shader.max_radius_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "max_radius");
    m_surfel_prov_shader.min_radius_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "min_radius");
    m_surfel_prov_shader.min_screen_size_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "min_screen_size");
    m_surfel_prov_shader.max_screen_size_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "max_screen_size");
    m_surfel_prov_shader.scale_radius_loc  = glGetUniformLocation(m_surfel_prov_shader.program, "scale_radius");
    m_surfel_prov_shader.scale_radius_gamma_loc = glGetUniformLocation(m_surfel_prov_shader.program, "scale_radius_gamma");
    m_surfel_prov_shader.max_radius_cut_loc     = glGetUniformLocation(m_surfel_prov_shader.program, "max_radius_cut");
    m_surfel_prov_shader.viewport_loc           = glGetUniformLocation(m_surfel_prov_shader.program, "viewport");
    m_surfel_prov_shader.scale_projection_loc   = glGetUniformLocation(m_surfel_prov_shader.program, "scale_projection");
    m_surfel_prov_shader.show_normals_loc         = glGetUniformLocation(m_surfel_prov_shader.program, "show_normals");
    m_surfel_prov_shader.show_accuracy_loc        = glGetUniformLocation(m_surfel_prov_shader.program, "show_accuracy");
    m_surfel_prov_shader.show_radius_dev_loc      = glGetUniformLocation(m_surfel_prov_shader.program, "show_radius_deviation");
    m_surfel_prov_shader.show_output_sens_loc     = glGetUniformLocation(m_surfel_prov_shader.program, "show_output_sensitivity");
    m_surfel_prov_shader.accuracy_loc             = glGetUniformLocation(m_surfel_prov_shader.program, "accuracy");
    m_surfel_prov_shader.average_radius_loc       = glGetUniformLocation(m_surfel_prov_shader.program, "average_radius");
    m_surfel_prov_shader.channel_loc              = glGetUniformLocation(m_surfel_prov_shader.program, "channel");
    m_surfel_prov_shader.heatmap_loc              = glGetUniformLocation(m_surfel_prov_shader.program, "heatmap");
    m_surfel_prov_shader.heatmap_min_loc          = glGetUniformLocation(m_surfel_prov_shader.program, "heatmap_min");
    m_surfel_prov_shader.heatmap_max_loc          = glGetUniformLocation(m_surfel_prov_shader.program, "heatmap_max");
    m_surfel_prov_shader.heatmap_min_color_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "heatmap_min_color");
    m_surfel_prov_shader.heatmap_max_color_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "heatmap_max_color");
    glUseProgram(0);

    glUseProgram(m_line_shader.program);
    m_line_shader.mvp_matrix_location = glGetUniformLocation(m_line_shader.program, "mvp_matrix");
    m_line_shader.in_color_location = glGetUniformLocation(m_line_shader.program, "in_color");
    glUseProgram(0);

    // --- PASS 1 ---
    glUseProgram(m_surfel_pass1_shader.program);
    m_surfel_pass1_shader.mvp_matrix_loc         = glGetUniformLocation(m_surfel_pass1_shader.program, "mvp_matrix");
    m_surfel_pass1_shader.projection_matrix_loc  = glGetUniformLocation(m_surfel_pass1_shader.program, "projection_matrix");
    m_surfel_pass1_shader.model_view_matrix_loc  = glGetUniformLocation(m_surfel_pass1_shader.program, "model_view_matrix");
    m_surfel_pass1_shader.model_matrix_loc       = glGetUniformLocation(m_surfel_pass1_shader.program, "model_matrix");
    m_surfel_pass1_shader.viewport_loc           = glGetUniformLocation(m_surfel_pass1_shader.program, "viewport");
    m_surfel_pass1_shader.max_radius_loc         = glGetUniformLocation(m_surfel_pass1_shader.program, "max_radius");
    m_surfel_pass1_shader.min_radius_loc         = glGetUniformLocation(m_surfel_pass1_shader.program, "min_radius");
    m_surfel_pass1_shader.min_screen_size_loc         = glGetUniformLocation(m_surfel_pass1_shader.program, "min_screen_size");
    m_surfel_pass1_shader.max_screen_size_loc         = glGetUniformLocation(m_surfel_pass1_shader.program, "max_screen_size");
    m_surfel_pass1_shader.scale_radius_loc       = glGetUniformLocation(m_surfel_pass1_shader.program, "scale_radius");
    m_surfel_pass1_shader.scale_projection_loc   = glGetUniformLocation(m_surfel_pass1_shader.program, "scale_projection");
    m_surfel_pass1_shader.max_radius_cut_loc = glGetUniformLocation(m_surfel_pass1_shader.program, "max_radius_cut");
    m_surfel_pass1_shader.scale_radius_gamma_loc   = glGetUniformLocation(m_surfel_pass1_shader.program, "scale_radius_gamma");
    m_surfel_pass1_shader.use_aniso_loc            = glGetUniformLocation(m_surfel_pass1_shader.program, "use_aniso");
    glUseProgram(0);

    // --- PASS 2 ---
    glUseProgram(m_surfel_pass2_shader.program);
    m_surfel_pass2_shader.model_view_matrix_loc = glGetUniformLocation(m_surfel_pass2_shader.program, "model_view_matrix");
    m_surfel_pass2_shader.projection_matrix_loc = glGetUniformLocation(m_surfel_pass2_shader.program, "projection_matrix");
    m_surfel_pass2_shader.normal_matrix_loc     = glGetUniformLocation(m_surfel_pass2_shader.program, "normal_matrix");
    m_surfel_pass2_shader.depth_texture_loc     = glGetUniformLocation(m_surfel_pass2_shader.program, "depth_texture");
    m_surfel_pass2_shader.viewport_loc          = glGetUniformLocation(m_surfel_pass2_shader.program, "viewport");
    m_surfel_pass2_shader.scale_projection_loc  = glGetUniformLocation(m_surfel_pass2_shader.program, "scale_projection");
    m_surfel_pass2_shader.max_radius_loc        = glGetUniformLocation(m_surfel_pass2_shader.program, "max_radius");
    m_surfel_pass2_shader.min_radius_loc        = glGetUniformLocation(m_surfel_pass2_shader.program, "min_radius");
    m_surfel_pass2_shader.max_screen_size_loc   = glGetUniformLocation(m_surfel_pass2_shader.program, "max_screen_size");
    m_surfel_pass2_shader.min_screen_size_loc   = glGetUniformLocation(m_surfel_pass2_shader.program, "min_screen_size");
    m_surfel_pass2_shader.scale_radius_loc      = glGetUniformLocation(m_surfel_pass2_shader.program, "scale_radius");
    m_surfel_pass2_shader.scale_radius_gamma_loc = glGetUniformLocation(m_surfel_pass2_shader.program, "scale_radius_gamma");
    m_surfel_pass2_shader.max_radius_cut_loc    = glGetUniformLocation(m_surfel_pass2_shader.program, "max_radius_cut");
    m_surfel_pass2_shader.use_aniso_loc         = glGetUniformLocation(m_surfel_pass2_shader.program, "use_aniso");
    m_surfel_pass2_shader.show_normals_loc      = glGetUniformLocation(m_surfel_pass2_shader.program, "show_normals");
    m_surfel_pass2_shader.show_accuracy_loc     = glGetUniformLocation(m_surfel_pass2_shader.program, "show_accuracy");
    m_surfel_pass2_shader.show_radius_dev_loc   = glGetUniformLocation(m_surfel_pass2_shader.program, "show_radius_deviation");
    m_surfel_pass2_shader.show_output_sens_loc  = glGetUniformLocation(m_surfel_pass2_shader.program, "show_output_sensitivity");
    m_surfel_pass2_shader.accuracy_loc          = glGetUniformLocation(m_surfel_pass2_shader.program, "accuracy");
    m_surfel_pass2_shader.average_radius_loc    = glGetUniformLocation(m_surfel_pass2_shader.program, "average_radius");
    m_surfel_pass2_shader.depth_range_loc       = glGetUniformLocation(m_surfel_pass2_shader.program, "depth_range");
    m_surfel_pass2_shader.flank_lift_loc        = glGetUniformLocation(m_surfel_pass2_shader.program, "flank_lift");
    m_surfel_pass2_shader.coloring_loc          = glGetUniformLocation(m_surfel_pass2_shader.program, "coloring");

    if (m_surfel_pass2_shader.depth_texture_loc >= 0)
        glUniform1i(m_surfel_pass2_shader.depth_texture_loc, 0);
    glUseProgram(0);

    // --- PASS 3 ---
    glUseProgram(m_surfel_pass3_shader.program);
    m_surfel_pass3_shader.in_color_texture_loc        = glGetUniformLocation(m_surfel_pass3_shader.program, "in_color_texture");
    m_surfel_pass3_shader.in_normal_texture_loc       = glGetUniformLocation(m_surfel_pass3_shader.program, "in_normal_texture");
    m_surfel_pass3_shader.in_vs_position_texture_loc  = glGetUniformLocation(m_surfel_pass3_shader.program, "in_vs_position_texture");
    m_surfel_pass3_shader.in_depth_texture_loc        = glGetUniformLocation(m_surfel_pass3_shader.program, "in_depth_texture");
    m_surfel_pass3_shader.point_light_pos_vs_loc      = glGetUniformLocation(m_surfel_pass3_shader.program, "point_light_pos_vs");
    m_surfel_pass3_shader.point_light_intensity_loc   = glGetUniformLocation(m_surfel_pass3_shader.program, "point_light_intensity");
    m_surfel_pass3_shader.ambient_intensity_loc       = glGetUniformLocation(m_surfel_pass3_shader.program, "ambient_intensity");
    m_surfel_pass3_shader.specular_intensity_loc      = glGetUniformLocation(m_surfel_pass3_shader.program, "specular_intensity");
    m_surfel_pass3_shader.shininess_loc               = glGetUniformLocation(m_surfel_pass3_shader.program, "shininess");
    m_surfel_pass3_shader.gamma_loc                   = glGetUniformLocation(m_surfel_pass3_shader.program, "gamma");
    m_surfel_pass3_shader.use_tone_mapping_loc        = glGetUniformLocation(m_surfel_pass3_shader.program, "use_tone_mapping");
    m_surfel_pass3_shader.lighting_loc               = glGetUniformLocation(m_surfel_pass3_shader.program, "lighting");

    if (m_surfel_pass3_shader.in_color_texture_loc >= 0)       glUniform1i(m_surfel_pass3_shader.in_color_texture_loc, 0);
    if (m_surfel_pass3_shader.in_normal_texture_loc >= 0)      glUniform1i(m_surfel_pass3_shader.in_normal_texture_loc, 1);
    if (m_surfel_pass3_shader.in_vs_position_texture_loc >= 0) glUniform1i(m_surfel_pass3_shader.in_vs_position_texture_loc, 2);
    if (m_surfel_pass3_shader.in_depth_texture_loc >= 0)       glUniform1i(m_surfel_pass3_shader.in_depth_texture_loc, 3);
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

    const auto& model_path = settings.models[modelIndex];
    auto it_node = m_plugin->m_model_nodes.find(model_path);
    if (it_node == m_plugin->m_model_nodes.end())
        return false;

    osg::Node* node = it_node->second.get();
    if (!node || node->getNodeMask() == 0)
        return false;

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

    if (!hasPathToSceneRoot(node))
        return false;
    return true;
}

void LamureRenderer::setFrameUniforms(const scm::math::mat4& projection_matrix, const scm::math::vec2& viewport) {
    const auto &s = m_plugin->getSettings();

    // Decide anisotropic usage based on current projection and mode (0=off,1=auto,2=on)
    const bool useAnisoThisPass = decideUseAniso(projection_matrix, s.anisotropic_surfel_scaling, s.anisotropic_auto_threshold);

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
        scm::math::mat4d currentViewD = m_renderer->getModelViewMatrix();
        osg::Matrixd viewOsg = LamureUtil::matConv4D(currentViewD);
        ctx.viewMat = LamureUtil::matConv4F(viewOsg);
        auto viewMat3 = LamureUtil::matConv4to3F(currentViewD);
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
        auto& prog = m_point_shader;
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
        auto& prog = m_point_color_shader;
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
        auto& prog = m_point_color_lighting_shader;
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
        auto& prog = m_point_prov_shader;
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
        auto& prog = m_surfel_shader;
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
        auto& prog = m_surfel_color_shader;
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
        auto& prog = m_surfel_color_lighting_shader;
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
        auto& prog = m_surfel_prov_shader;
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

void LamureRenderer::setModelUniforms(const scm::math::mat4& mvp_matrix, const scm::math::mat4& model_matrix) {
    switch (m_plugin->getSettings().shader_type) {
    case ShaderType::Point:
        glUniformMatrix4fv(m_point_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (m_point_shader.model_matrix_loc >= 0)
            glUniformMatrix4fv(m_point_shader.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::PointColor:
        glUniformMatrix4fv(m_point_color_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (m_point_color_shader.model_matrix_loc >= 0)
            glUniformMatrix4fv(m_point_color_shader.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::PointColorLighting:
        glUniformMatrix4fv(m_point_color_lighting_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (m_point_color_lighting_shader.model_matrix_loc >= 0)
            glUniformMatrix4fv(m_point_color_lighting_shader.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::PointProv:
        glUniformMatrix4fv(m_point_prov_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (m_point_prov_shader.model_matrix_loc >= 0)
            glUniformMatrix4fv(m_point_prov_shader.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::Surfel:
        glUniformMatrix4fv(m_surfel_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (m_surfel_shader.model_matrix_loc >= 0)
            glUniformMatrix4fv(m_surfel_shader.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::SurfelColor:
        glUniformMatrix4fv(m_surfel_color_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (m_surfel_color_shader.model_matrix_loc >= 0)
            glUniformMatrix4fv(m_surfel_color_shader.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::SurfelColorLighting:
        glUniformMatrix4fv(m_surfel_color_lighting_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (m_surfel_color_lighting_shader.model_matrix_loc >= 0)
            glUniformMatrix4fv(m_surfel_color_lighting_shader.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::SurfelProv:
        glUniformMatrix4fv(m_surfel_prov_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array);
        if (m_surfel_prov_shader.model_matrix_loc >= 0)
            glUniformMatrix4fv(m_surfel_prov_shader.model_matrix_loc, 1, GL_FALSE, model_matrix.data_array);
        break;
    case ShaderType::SurfelMultipass:
        break;
    }
}

void LamureRenderer::setNodeUniforms(const lamure::ren::bvh* bvh, uint32_t node_id) {
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
            glUniform1f(m_point_color_shader.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius);
            glUniform1f(m_point_color_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::PointColorLighting: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_point_color_lighting_shader.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius);
            glUniform1f(m_point_color_lighting_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::PointProv: {
        if (s.show_accuracy) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_point_prov_shader.accuracy_loc, accuracy);
        }
        if (s.show_radius_deviation) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius);
            glUniform1f(m_point_prov_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelColor: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_surfel_color_shader.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius * s.scale_element);
            glUniform1f(m_surfel_color_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelProv: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_surfel_prov_shader.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius * s.scale_element);
            glUniform1f(m_surfel_prov_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelColorLighting: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_surfel_color_lighting_shader.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius * s.scale_element);
            glUniform1f(m_surfel_color_lighting_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelMultipass: {
        if (showAcc) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_surfel_pass2_shader.accuracy_loc, accuracy);
        }
        if (showRadiusDev) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float avg_ws  = calc_avg_ws(avg_raw, s.scale_radius * s.scale_element);
            glUniform1f(m_surfel_pass2_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    default: break;
    }
}


bool LamureRenderer::readShader(std::string const &path_string, std::string &shader_string, bool keep_optional_shader_code = false)
{
    if (!boost::filesystem::exists(path_string))
    {
        std::cout << "WARNING: File " << path_string << "does not exist." << std::endl;
        return false;
    }
    std::ifstream shader_source(path_string, std::ios::in);
    std::string line_buffer;
    std::string include_prefix("INCLUDE");
    std::string optional_begin_prefix("OPTIONAL_BEGIN");
    std::string optional_end_prefix("OPTIONAL_END");
    std::size_t slash_position = path_string.find_last_of("/\\");
    std::string const base_path = path_string.substr(0, slash_position + 1);

    bool disregard_code = false;
    while (std::getline(shader_source, line_buffer))
    {
        line_buffer = LamureUtil::stripWhitespace(line_buffer);
        if (LamureUtil::parsePrefix(line_buffer, include_prefix))
        {
            if (!disregard_code || keep_optional_shader_code)
            {
                std::string filename_string = line_buffer;
                readShader(base_path + filename_string, shader_string);
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


void LamureRenderer::initLamureShader()
{
    if (notifyOn(m_plugin)) { std::cout << "[Lamure] LamureRenderer::initLamureShader()" << std::endl; }
    vis_point_vs_source.clear();
    vis_point_fs_source.clear();
    vis_point_color_vs_source.clear();
    vis_point_color_fs_source.clear();
    vis_point_color_lighting_vs_source.clear();
    vis_point_color_lighting_fs_source.clear();
    vis_point_prov_vs_source.clear();
    vis_point_prov_fs_source.clear();
    vis_surfel_vs_source.clear();
    vis_surfel_fs_source.clear();
    vis_surfel_gs_source.clear();
    vis_surfel_color_vs_source.clear();
    vis_surfel_color_gs_source.clear();
    vis_surfel_color_fs_source.clear();
    vis_surfel_color_lighting_vs_source.clear();
    vis_surfel_color_lighting_gs_source.clear();
    vis_surfel_color_lighting_fs_source.clear();
    vis_surfel_prov_vs_source.clear();
    vis_surfel_prov_gs_source.clear();
    vis_surfel_prov_fs_source.clear();
    vis_line_vs_source.clear();
    vis_line_fs_source.clear();
    vis_surfel_pass1_vs_source.clear();
    vis_surfel_pass1_gs_source.clear();
    vis_surfel_pass1_fs_source.clear();
    vis_surfel_pass2_vs_source.clear();
    vis_surfel_pass2_gs_source.clear();
    vis_surfel_pass2_fs_source.clear();
    vis_surfel_pass3_vs_source.clear();
    vis_surfel_pass3_fs_source.clear();
    vis_debug_vs_source.clear();
    vis_debug_fs_source.clear();

    try
    {
        char * val;
        val = getenv( "COVISEDIR" );
        std::string shader_root_path=val;
        shader_root_path=shader_root_path+"/src/OpenCOVER/plugins/hlrs/LamurePointCloud/shaders";
        shader_root_path=std::string(val)+"/share/covise/shaders";
        if (!readShader(shader_root_path + "/vis/vis_point.glslv", vis_point_vs_source) ||
            !readShader(shader_root_path + "/vis/vis_point.glslf", vis_point_fs_source) ||
            !readShader(shader_root_path + "/vis/vis_point_color.glslv", vis_point_color_vs_source) ||
            !readShader(shader_root_path + "/vis/vis_point_color.glslf", vis_point_color_fs_source) ||
            !readShader(shader_root_path + "/vis/vis_point_color_lighting.glslv", vis_point_color_lighting_vs_source) ||
            !readShader(shader_root_path + "/vis/vis_point_color_lighting.glslf", vis_point_color_lighting_fs_source) ||
            !readShader(shader_root_path + "/vis/vis_point_prov.glslv", vis_point_prov_vs_source) ||
            !readShader(shader_root_path + "/vis/vis_point_prov.glslf", vis_point_prov_fs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel.glslv", vis_surfel_vs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel.glslg", vis_surfel_gs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel.glslf", vis_surfel_fs_source) || 
            !readShader(shader_root_path + "/vis/vis_surfel_color.glslv", vis_surfel_color_vs_source) || 
            !readShader(shader_root_path + "/vis/vis_surfel_color.glslg", vis_surfel_color_gs_source) || 
            !readShader(shader_root_path + "/vis/vis_surfel_color.glslf", vis_surfel_color_fs_source) || 
            !readShader(shader_root_path + "/vis/vis_surfel_color_lighting.glslv", vis_surfel_color_lighting_vs_source) || 
            !readShader(shader_root_path + "/vis/vis_surfel_color_lighting.glslg", vis_surfel_color_lighting_gs_source) || 
            !readShader(shader_root_path + "/vis/vis_surfel_color_lighting.glslf", vis_surfel_color_lighting_fs_source) || 
            !readShader(shader_root_path + "/vis/vis_surfel_prov.glslv", vis_surfel_prov_vs_source) || 
            !readShader(shader_root_path + "/vis/vis_surfel_prov.glslg", vis_surfel_prov_gs_source) || 
            !readShader(shader_root_path + "/vis/vis_surfel_prov.glslf", vis_surfel_prov_fs_source) || 
            !readShader(shader_root_path + "/vis/vis_line.glslv", vis_line_vs_source) || 
            !readShader(shader_root_path + "/vis/vis_line.glslf", vis_line_fs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel_pass1.glslv", vis_surfel_pass1_vs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel_pass1.glslg", vis_surfel_pass1_gs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel_pass1.glslf", vis_surfel_pass1_fs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel_pass2.glslv", vis_surfel_pass2_vs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel_pass2.glslg", vis_surfel_pass2_gs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel_pass2.glslf", vis_surfel_pass2_fs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel_pass3.glslv", vis_surfel_pass3_vs_source) ||
            !readShader(shader_root_path + "/vis/vis_surfel_pass3.glslf", vis_surfel_pass3_fs_source) ||
            !readShader(shader_root_path + "/vis/vis_debug.glslv", vis_debug_vs_source) ||
            !readShader(shader_root_path + "/vis/vis_debug.glslf", vis_debug_fs_source))
        {
            std::cout << "error reading shader files" << std::endl;
            exit(1);
        }

        m_point_shader.program = m_renderer->compileAndLinkShaders(vis_point_vs_source, vis_point_fs_source);
        m_point_color_shader.program = m_renderer->compileAndLinkShaders(vis_point_color_vs_source, vis_point_color_fs_source);
        m_point_color_lighting_shader.program = m_renderer->compileAndLinkShaders(vis_point_color_lighting_vs_source, vis_point_color_lighting_fs_source);
        m_point_prov_shader.program = m_renderer->compileAndLinkShaders(vis_point_prov_vs_source, vis_point_prov_fs_source);
        m_surfel_shader.program = m_renderer->compileAndLinkShaders(vis_surfel_vs_source, vis_surfel_gs_source, vis_surfel_fs_source);
        m_surfel_color_shader.program = m_renderer->compileAndLinkShaders(vis_surfel_color_vs_source, vis_surfel_color_gs_source, vis_surfel_color_fs_source);
        m_surfel_color_lighting_shader.program = m_renderer->compileAndLinkShaders(vis_surfel_color_lighting_vs_source, vis_surfel_color_lighting_gs_source, vis_surfel_color_lighting_fs_source);
        m_surfel_prov_shader.program = m_renderer->compileAndLinkShaders(vis_surfel_prov_vs_source, vis_surfel_prov_gs_source, vis_surfel_prov_fs_source);
        m_line_shader.program = m_renderer->compileAndLinkShaders(vis_line_vs_source, vis_line_fs_source);
        m_surfel_pass1_shader.program = m_renderer->compileAndLinkShaders(vis_surfel_pass1_vs_source, vis_surfel_pass1_gs_source, vis_surfel_pass1_fs_source);
        m_surfel_pass2_shader.program = m_renderer->compileAndLinkShaders(vis_surfel_pass2_vs_source, vis_surfel_pass2_gs_source, vis_surfel_pass2_fs_source);
        m_surfel_pass3_shader.program = m_renderer->compileAndLinkShaders(vis_surfel_pass3_vs_source, vis_surfel_pass3_fs_source);
    }
    catch (std::exception &e) { std::cout << e.what() << std::endl; }
}

void LamureRenderer::initSchismObjects()
{
    if (notifyOn(m_plugin)) { std::cout << "[Lamure] LamureRenderer::initSchismObjects()" << std::endl; }
    if (!m_device)
    {
        m_device.reset(new scm::gl::render_device());
        if (!m_device)
        {
            std::cout << "error creating device" << std::endl;
        }
    }
    if (!m_context)
    {
        m_context = m_device->main_context();
        if (!m_context)
        {
            std::cout << "error creating context" << std::endl;
        }
    }
}

void LamureRenderer::initCamera()
{
    if (notifyOn(m_plugin)) { std::cout << "[Lamure] LamureRenderer::initCamera()" << std::endl; }
    m_osg_camera = opencover::VRViewer::instance()->getCamera();

    lamure::context_t lmr_ctx = m_osg_camera->getGraphicsContext()->getState()->getContextID();
    double look_dist = 1.0;
    double left, right, bottom, top, zNear, zFar;
    osg::Vec3d eye, center, up;
    m_osg_camera->getProjectionMatrixAsFrustum(left, right, bottom, top, zNear, zFar);
    m_osg_camera->getViewMatrixAsLookAt(eye, center, up, look_dist);

    osg::Matrix base = opencover::VRSceneGraph::instance()->getScaleTransform()->getMatrix();
    osg::Matrix trans = opencover::VRSceneGraph::instance()->getTransform()->getMatrix();
    base.postMult(trans);

    osg::Matrixd view = m_osg_camera->getViewMatrix();
    osg::Matrixd proj = m_osg_camera->getProjectionMatrix();

    m_scm_camera = new lamure::ren::camera((lamure::view_t)lmr_ctx, zNear, zFar, LamureUtil::matConv4D(view * base), LamureUtil::matConv4D(proj));

    osgViewer::Viewer::Windows windows;
    opencover::VRViewer::instance()->getWindows(windows);
    osgViewer::GraphicsWindow* window = windows.front();
    m_hud_camera = new osg::Camera();
    m_hud_camera->setName("hud_camera");
    m_hud_camera->setGraphicsContext(window);
    m_hud_camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    m_hud_camera->setProjectionResizePolicy(osg::Camera::FIXED);
    m_hud_camera->setViewMatrix(m_osg_camera->getViewMatrix());
    m_hud_camera->setProjectionMatrix(m_osg_camera->getProjectionMatrix());
    m_hud_camera->setViewport(0, 0, window->getTraits()->width, window->getTraits()->height);
    m_hud_camera->setRenderOrder(osg::Camera::POST_RENDER, 2);
    m_hud_camera->setRenderOrder(osg::Camera::POST_RENDER, 10);
    m_hud_camera->setClearMask(0);
    m_hud_camera->setRenderer(new osgViewer::Renderer(m_hud_camera.get()));
    m_osg_camera->addChild(m_hud_camera.get());

    scm::math::vec3f temp_center = scm::math::vec3f::zero();
    scm::math::vec3f root_min_temp = scm::math::vec3f::zero();
    scm::math::vec3f root_max_temp = scm::math::vec3f::zero();
}

unsigned int LamureRenderer::compileShader(unsigned int type, const std::string& source, uint8_t ctx_id)
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
        std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader!" << std::endl;
        std::cout << message << std::endl;
        gl_api->glDeleteShader(id);
        return 0;
    };
    return id;
}

GLuint LamureRenderer::compileAndLinkShaders(std::string vs_source, std::string fs_source)
{
    GLuint program = glCreateProgram();
    GLuint vs = compileShader(GL_VERTEX_SHADER, vs_source, 0);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fs_source, 0);
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

GLuint LamureRenderer::compileAndLinkShaders(std::string vs_source, std::string gs_source, std::string fs_source)
{
    GLuint program = glCreateProgram();
    GLuint vs = compileShader(GL_VERTEX_SHADER, vs_source, 0);
    GLuint gs = compileShader(GL_GEOMETRY_SHADER, gs_source, 0);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fs_source, 0);
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


void LamureRenderer::initFrustumResources() {
    if (notifyOn(m_plugin)) { std::cout << "[Lamure] create_frustum_resources() " << std::endl; }
    std::vector<scm::math::vec3d> corner_values = m_renderer->getScmCamera()->get_frustum_corners();
    for (size_t i = 0; i < corner_values.size(); ++i) {
        auto vv = scm::math::vec3f(corner_values[i]);
        m_frustum_resource.vertices[i * 3 + 0] = vv.x;
        m_frustum_resource.vertices[i * 3 + 1] = vv.y;
        m_frustum_resource.vertices[i * 3 + 2] = vv.z;
    }
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_frustum_resource.idx.size() * sizeof(unsigned short), m_frustum_resource.idx.data(), GL_STATIC_DRAW);
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * m_frustum_resource.vertices.size(), m_frustum_resource.vertices.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    m_frustum_resource.vao = vao;
    m_frustum_resource.vbo = vbo;
    m_frustum_resource.ibo = ibo;
    glBindVertexArray(0);
}

void LamureRenderer::initBoxResources() {
    if (notifyOn(m_plugin)) { std::cout << "[Lamure] init_box_resources()\n"; }

    if (!m_plugin->getSettings().models.size())
        return;

    std::vector<float> all_box_vertices;
    m_bvh_node_vertex_offsets.clear();

    auto* ctrl = lamure::ren::controller::get_instance();
    auto* db   = lamure::ren::model_database::get_instance();

    const auto modelCount = static_cast<uint32_t>(m_plugin->getSettings().models.size());
    // Prepare aggregation for scene AABB (optional meta)
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
            current_offsets.push_back(static_cast<uint32_t>(all_box_vertices.size() / 3));
            std::vector<float> corners = LamureUtil::getBoxCorners(boxes[node_id]);
            all_box_vertices.insert(all_box_vertices.end(), corners.begin(), corners.end());
        }
        m_bvh_node_vertex_offsets[m_id] = std::move(current_offsets);

        // Compute per-model root AABB in world (exact axis-aligned AABB via 8 corners)
        if (!boxes.empty()) {
            const auto &M4d = m_plugin->getModelInfo().model_transformations[m_id];
            const scm::math::mat4f M = scm::math::mat4f(M4d);

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

    GLuint vao = 0, vbo = 0, ibo = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
        m_box_resource.idx.size() * sizeof(GLushort),
        m_box_resource.idx.data(),
        GL_STATIC_DRAW);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
        all_box_vertices.size() * sizeof(float),
        all_box_vertices.data(),
        GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    m_box_resource.vao = vao;
    m_box_resource.vbo = vbo;
    m_box_resource.ibo = ibo;

    glBindVertexArray(0);

    // Write aggregated scene AABB
    m_plugin->getModelInfo().models_min = global_min;
    m_plugin->getModelInfo().models_max = global_max;
    m_plugin->getModelInfo().models_center = scm::math::vec3d( (global_min.x + global_max.x) * 0.5,
                                                              (global_min.y + global_max.y) * 0.5,
                                                              (global_min.z + global_max.z) * 0.5 );
}

void LamureRenderer::initPclResources(){
    if(notifyOn(m_plugin)) std::cout<<"[Lamure] initPclResources()\n";

    // Screen-Quad einmalig anlegen
    if(m_pcl_resource.screen_quad_vao==0){
        GLuint vao=0,vbo=0;
        glGenVertexArrays(1,&vao);
        glBindVertexArray(vao);
        glGenBuffers(1,&vbo);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,sizeof(float)*m_pcl_resource.screen_quad_vertex.size(),m_pcl_resource.screen_quad_vertex.data(),GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
        m_pcl_resource.screen_quad_vao=vao; m_pcl_resource.screen_quad_vbo=vbo;
        glBindVertexArray(0);
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
        std::cerr<<"ERROR::FRAMEBUFFER incomplete ("<<std::hex<<status<<std::dec<<") "<<width<<"x"<<height<<"\n";
    }else if(notifyOn(m_plugin)){
        std::cout<<"[Lamure] FBO ready "<<width<<"x"<<height<<"\n";
    }

    glBindTexture(GL_TEXTURE_2D,0);
    glBindFramebuffer(GL_FRAMEBUFFER,prev_fbo);
    glDrawBuffer(prev_draw_buffer);
    glReadBuffer(prev_read_buffer);

    target.width = width;
    target.height = height;
}

LamureRenderer::MultipassTarget& LamureRenderer::acquireMultipassTarget(lamure::context_t contextID, const osg::Camera* camera, int width, int height){
    if(width <= 0 || height <= 0){
        const osg::GraphicsContext::Traits* traits = nullptr;
        if(camera && camera->getGraphicsContext()){
            traits = camera->getGraphicsContext()->getTraits();
        }
        if(traits){
            width = traits->width;
            height = traits->height;
        }else{
            width = std::max(width, 1);
            height = std::max(height, 1);
        }
    }

    MultipassTargetKey key{contextID, camera};
    auto [it, inserted] = m_multipass_targets.try_emplace(key);
    if(inserted){
        initializeMultipassTarget(it->second, width, height);
    }else if(it->second.width != width || it->second.height != height){
        if(notifyOn(m_plugin)){
            std::cout<<"[Lamure] FBO resize "<<it->second.width<<"x"<<it->second.height<<" -> "<<width<<"x"<<height<<"\n";
        }
        initializeMultipassTarget(it->second, width, height);
    }
    return it->second;
}

void LamureRenderer::releaseMultipassTargets(){
    for(auto& entry : m_multipass_targets){
        destroyMultipassTarget(entry.second);
    }
    m_multipass_targets.clear();
}

