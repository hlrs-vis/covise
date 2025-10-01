#include "LamureRenderer.h"
#include "Lamure.h"
#include "LamureUtil.h"

#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>

#include <osg/Geometry>
#include <osg/Vec3>
#include <osg/State>
#include <osgViewer/Viewer>
#include <osgViewer/Renderer>
#include <osgText/Text>

#include <lamure/ren/model_database.h>
#include <lamure/ren/cut_database.h>
#include <lamure/ren/controller.h>
#include <lamure/pvs/pvs_database.h>
#include <lamure/ren/policy.h>

#include <iostream>
#include <gl_state.h>
#include <config/CoviseConfig.h>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <cover/coVRNavigationManager.h>

std::string shader_root_path = LAMURE_SHADERS_DIR;
std::string font_root_path = LAMURE_FONTS_DIR;



LamureRenderer::LamureRenderer(Lamure *plugin) : 
    m_plugin(plugin)
{
    m_renderer = this;
    m_rendering = false;
    m_group = new osg::Group();
    m_group->setName("LamureRendererGroup");
}

LamureRenderer::~LamureRenderer()
{
}


struct InitCullCallback : public osg::Drawable::CullCallback {
    InitCullCallback(Lamure* plugin) : _plugin(plugin), _initialized(false)
    {
        if (_plugin->getUI()->getNotifyButton()->state()) { std::cout << "[Notify] InitDrawCallback()" << std::endl; }
        _renderer = _plugin->getRenderer();
    }

    virtual bool cull(osg::NodeVisitor* nv, osg::Drawable* drawable, osg::RenderInfo* renderInfo) const override {

        osg::Matrix mv_matrix = opencover::cover->getBaseMat() * _renderer->getOsgCamera()->getViewMatrix();
        scm::math::mat4d modelview_matrix = LamureUtil::matConv4D(mv_matrix);
        scm::math::mat4d projection_matrix = LamureUtil::matConv4D(_renderer->getOsgCamera()->getProjectionMatrix());

        _renderer->setModelViewMatrix(modelview_matrix);
        _renderer->setProjectionMatrix(projection_matrix);

        //osg::Matrix base = opencover::VRSceneGraph::instance()->getScaleTransform()->getMatrix();
        //osg::Matrix trans = opencover::VRSceneGraph::instance()->getTransform()->getMatrix();
        //base.postMult(trans);

        //osg::Matrixd view = _renderer->getOsgCamera()->getViewMatrix();
        //osg::Matrixd proj = _renderer->getOsgCamera()->getProjectionMatrix();

        //scm::math::mat4d modelview_matrix = LamureUtil::matConv4D(osg::Matrixd(renderInfo.getState()->getModelViewMatrix()));
        //scm::math::mat4d projection_matrix = LamureUtil::matConv4D(osg::Matrixd(renderInfo.getState()->getProjectionMatrix()));


        _renderer->ensurePclFboSizeUpToDate();

        _renderer->getScmCamera()->set_projection_matrix(projection_matrix);
        if (_plugin->getUI()->getSyncButton()->state() == 1) {
            _renderer->getScmCamera()->set_view_matrix(modelview_matrix);
        }

        if (!_initialized) {
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
            _initialized = true;
            before.restore();
        }
        return false;
    }
    Lamure* _plugin;
    LamureRenderer* _renderer;
    mutable bool _initialized;
};

struct InitGeometry : public osg::Geometry {
    InitGeometry(Lamure* plugin) : _plugin(plugin) {
        if (_plugin->getUI()->getNotifyButton()->state()) { std::cout << "[Notify] InitGeometry()" << std::endl; }
        setUseDisplayList(false);
        setUseVertexBufferObjects(true);
        setUseVertexArrayObject(false);
        setCullCallback(new InitCullCallback(plugin));
    }
    Lamure* _plugin;
};

struct TextCullCallback : public osg::Drawable::CullCallback
{
    TextCullCallback(Lamure* plugin, osgText::Text* values, Lamure::RenderInfo* render_info)
        : _plugin(plugin),
        _values(values),
        _render_info(render_info)
    {
        _lastUpdateTime = std::chrono::steady_clock::now();
        _minInterval = std::chrono::milliseconds(100);
        _renderer = _plugin->getRenderer();
    }

    virtual bool cull(osg::NodeVisitor* nv, osg::Drawable* drawable, osg::RenderInfo* renderInfo) const override
    {
        auto now = std::chrono::steady_clock::now();
        if (now - _lastUpdateTime >= _minInterval)
        {
            osg::Matrix baseMatrix = opencover::VRSceneGraph::instance()->getScaleTransform()->getMatrix();
            osg::Matrix transformMatrix = opencover::VRSceneGraph::instance()->getTransform()->getMatrix();
            baseMatrix.postMult(transformMatrix);

            scm::math::vec3d pos = _renderer->getScmCamera()->get_cam_pos();
            scm::math::mat4d base = LamureUtil::matConv4D(baseMatrix);
            scm::math::mat4d view = LamureUtil::matConv4D(_renderer->getOsgCamera()->getViewMatrix());
            scm::math::mat4d projection = LamureUtil::matConv4D(_renderer->getOsgCamera()->getProjectionMatrix());

            std::stringstream value_ss;
            std::stringstream modelview_ss;
            std::stringstream projection_ss;
            std::stringstream mvp_ss;

            modelview_ss << view * base;
            projection_ss << projection;
            mvp_ss << projection * view * base;

            double fpsAvg = 0.0;
            if (auto* vs = opencover::VRViewer::instance()->getViewerStats()) {
                (void)vs->getAveragedAttribute("Frame rate", fpsAvg); // geglättet
            }
            if (fpsAvg <= 0.0) {
                // Fallback, falls (noch) keine Stats vorhanden sind
                const double fd = std::max(1e-6, opencover::cover->frameDuration());
                fpsAvg = 1.0 / fd;
            }

            value_ss << "\n"
                << std::fixed << std::setprecision(2) 
                << fpsAvg << "\n"
                << _render_info->rendered_nodes << "\n"                
                << _render_info->rendered_primitives << "\n"                
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
        return false;
    }
    Lamure* _plugin;
    LamureRenderer* _renderer;
    osg::ref_ptr<osgText::Text> _values;
    Lamure::RenderInfo* _render_info;
    mutable std::chrono::steady_clock::time_point _lastUpdateTime;
    std::chrono::milliseconds _minInterval;
};

struct TextGeode : public osg::Geode
{
    TextGeode(Lamure* plugin)
    {
        if (plugin->getUI()->getNotifyButton()->state()) { std::cout << "[Notify] TextGeode()" << std::endl; }
        osg::Quat rotation(osg::DegreesToRadians(90.0f), osg::Vec3(1.0f, 0.0f, 0.0f));
        osg::Vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
        std::string font = opencover::coVRFileManager::instance()->getFontFile(NULL);
        float characterSize = 20.0f;
        const osg::GraphicsContext::Traits* traits = opencover::coVRConfig::instance()->windows[0].context->getTraits();
        osg::Vec3 pos_label(+traits->width * 0.5 * 0.475f, 0.0f, traits->height * 0.5 * 0.7f);
        osg::Vec3 pos_value = pos_label + osg::Vec3(100.0f, 0.0f, 0.0f);
        osg::ref_ptr<osgText::Text> label = new osgText::Text();
        label->setRotation(rotation);
        label->setColor(color);
        label->setFont(font);
        label->setCharacterSize(characterSize);
        label->setPosition(pos_label);
        std::stringstream label_ss;
        label_ss << "Rendering" << "\n"
            << "FPS:" << "\n"
            << "Nodes:" << "\n"
            << "Primitive:" << "\n"
            << "Boxes:" << "\n\n"
            << "Frustum Position" << "\n"
            << "X:" << "\n"
            << "Y:" << "\n"
            << "Z:" << "\n\n\n"
            << "ModelView:" << "\n\n\n\n\n\n"
            << "Projection:" << "\n\n\n\n\n\n"
            << "MVP:" << "\n\n\n\n\n\n";
        label->setText(label_ss.str(), osgText::String::ENCODING_UTF8);

        osg::ref_ptr<osgText::Text> value = new osgText::Text();
        value->setRotation(rotation);
        value->setColor(color);
        value->setFont(font);
        value->setCharacterSize(characterSize);
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
        value->setCullCallback(new TextCullCallback(plugin, value.get(), &plugin->getRenderInfo()));
    }
};

struct FrustumDrawCallback : public osg::Drawable::DrawCallback
{
    FrustumDrawCallback(Lamure* plugin)
        : _plugin(plugin)
    {
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

        glEnable(GL_DEPTH_CLAMP);
        glDisable(GL_DEPTH_TEST);
        glLineWidth(2);
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
        if (_plugin->getUI()->getNotifyButton()->state()) { std::cout << "[Notify] FrustumGeometryGL()" << std::endl; }
        setUseDisplayList(false);
        setUseVertexBufferObjects(true);
        setUseVertexArrayObject(false);
        setDrawCallback(new FrustumDrawCallback(_plugin));
    }
};

struct BoundingBoxDrawCallback : public virtual osg::Drawable::DrawCallback
{
    BoundingBoxDrawCallback(Lamure* plugin)
        : _plugin(plugin)
    {
        if (_plugin->getUI()->getNotifyButton()->state()) { std::cout << "[Notify] BoundingBoxDrawCallback()" << std::endl; }
        _renderer = _plugin->getRenderer();
    }

    virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const 
    {
        GLState before = GLState::capture();
        osg::State* state = renderInfo.getState();
        scm::math::mat4 view_matrix = LamureUtil::matConv4F(state->getModelViewMatrix());
        scm::math::mat4 projection_matrix = LamureUtil::matConv4F(state->getProjectionMatrix());
        scm::math::mat4 osg_scale = LamureUtil::matConv4F(opencover::cover->getObjectsScale()->getMatrix());

        lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
        lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
        lamure::ren::controller* controller = lamure::ren::controller::get_instance();
        lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();

        if (lamure::ren::policy::get_instance()->size_of_provenance() > 0) { controller->reset_system(_plugin->getDataProvenance()); }
        else { controller->reset_system(); }

        lamure::context_t context_id = controller->deduce_context_id(_renderer->getOsgCamera()->getGraphicsContext()->getState()->getContextID());
        for (lamure::model_t model_id = 0; model_id < _plugin->getSettings().models.size(); ++model_id) {
            lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));
            cuts->send_transform(context_id, m_id, scm::math::mat4(_plugin->getModelInfo().model_transformations[m_id]));
            cuts->send_threshold(context_id, m_id, _plugin->getSettings().lod_error);
            cuts->send_rendered(context_id, m_id);
            database->get_model(m_id)->set_transform(scm::math::mat4(_plugin->getModelInfo().model_transformations[m_id]));
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

        if (_plugin->getSettings().lod_update) {
            if (lamure::ren::policy::get_instance()->size_of_provenance() > 0)
            { controller->dispatch(context_id, _renderer->getDevice(), _plugin->getDataProvenance()); }
            else { controller->dispatch(context_id, _renderer->getDevice()); }
        }

        glBindVertexArray(_renderer->getBoxResource().vao);
        glUseProgram(_renderer->getLineShader().program);
        glUniform4f(_renderer->getLineShader().in_color_location, _plugin->getSettings().bvh_color[0], _plugin->getSettings().bvh_color[1], _plugin->getSettings().bvh_color[2], _plugin->getSettings().bvh_color[3]);

        uint64_t rendered_bounding_boxes = 0;
        for (uint16_t model_id = 0; model_id < _plugin->getSettings().models.size(); ++model_id) {
            if (!_plugin->isModelVisible(model_id)) 
                continue;
            const lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));
            lamure::ren::cut& cut = cuts->get_cut(context_id, _renderer->getOsgCamera()->getGraphicsContext()->getState()->getContextID(), m_id);

            const auto renderable = cut.complete_set();
            const lamure::ren::bvh* bvh = database->get_model(m_id)->get_bvh();
            const std::vector<scm::gl::boxf>& bbv = bvh->get_bounding_boxes();
            const scm::math::mat4 model_matrix = scm::math::mat4(_plugin->getModelInfo().model_transformations[m_id]);
            const scm::math::mat4 mvp_matrix   = projection_matrix * view_matrix * model_matrix;
            const scm::gl::frustum frustum     = _renderer->getScmCamera()->get_frustum_by_model(model_matrix);

            glUniformMatrix4fv(_renderer->getLineShader().mvp_matrix_location, 1, GL_FALSE, mvp_matrix.data_array);
            const auto it = _renderer->m_bvh_node_vertex_offsets.find(m_id);
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
        _plugin->getRenderInfo().rendered_bounding_boxes = rendered_bounding_boxes;
        before.restore();
    };
    Lamure* _plugin;
    LamureRenderer* _renderer;
};

struct BoundingBoxGeometry : public osg::Geometry
{
    BoundingBoxGeometry(Lamure* plugin)
    {
        if (plugin->getUI()->getNotifyButton()->state()) { std::cout << "[Notify] BoundingBoxGeometry()" << std::endl; }
        setUseDisplayList(false);
        setUseVertexBufferObjects(true);
        setUseVertexArrayObject(false);
        setDrawCallback(new BoundingBoxDrawCallback(plugin));
    }
};

void print_mat4_flat(const float* data, const std::string& name)
{
    std::cout << name << " (flat 0-15):\n";
    for (int i = 0; i < 16; ++i) {
        std::cout << "[" << std::setw(2) << i << "] "
            << std::setprecision(6) << std::fixed
            << data[i];
        // nach jedem vierten Element Zeilenumbruch
        if ((i % 4) == 3) std::cout << "\n";
        else             std::cout << "  ";
    }
    std::cout << "\n";
}

namespace {
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
        double sum_area_px = 0.0;     // akkumulierte Fläche aller Splats (px^2)
        double screen_px   = 0.0;     // Breite*Höhe (px)
        float  scale_proj  = 0.0f;    // opencover*H*0.5*P[1][1]
        double k_orient    = 0.70;    // Orientierungsfaktor (Point=1.0, Surfel/Splat=0.7)
    };

    // Initialisiert SDStats nur, wenn in diesem Frame gesampelt wird
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

    // Sammelt Flächenbeitrag eines Node (nur wenn gesampelt)
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

    // Finalisiert und schreibt die SD-Metriken (nur wenn gesampelt)
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

} // namespace


struct PointsDrawCallback : public virtual osg::Drawable::DrawCallback
{
    PointsDrawCallback(Lamure* plugin)
        : _plugin(plugin), _initialized(false)
    {
        if (_plugin->getUI()->getNotifyButton()->state()) { std::cout << "[Notify] PointsDrawCallback()" << std::endl; } 
        _renderer = _plugin->getRenderer();
    }

    virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
    {
        if (_renderer->getRendering()) { return; }
        _renderer->setRendering(true);
        const MeasCtx meas = makeMeasCtx(_plugin);
        const auto& settings = _plugin->getSettings();

        GLState before = GLState::capture();
        glDisable(GL_CULL_FACE);

        osg::State* state = renderInfo.getState();
        state->setCheckForGLErrors(osg::State::CheckForGLErrors::ONCE_PER_ATTRIBUTE);

        //scm::math::mat4 view_matrix = scm::math::mat4(_renderer->getModelViewMatrix());
        //scm::math::mat4 projection_matrix = scm::math::mat4(_renderer->getProjectionMatrix());
        scm::math::mat4 view_matrix = LamureUtil::matConv4F(state->getModelViewMatrix());
        scm::math::mat4 projection_matrix = LamureUtil::matConv4F(state->getProjectionMatrix());
        lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
        lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
        lamure::ren::controller* controller = lamure::ren::controller::get_instance();
        lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();

        if (lamure::ren::policy::get_instance()->size_of_provenance() > 0) { controller->reset_system(_plugin->getDataProvenance()); }
        else { controller->reset_system(); }

        lamure::context_t context_id = controller->deduce_context_id(_renderer->getOsgCamera()->getGraphicsContext()->getState()->getContextID());
        lamure::view_t    view_id = controller->deduce_view_id(context_id, _renderer->getScmCamera()->view_id());
        size_t surfels_per_node = database->get_primitives_per_node();

        for (lamure::model_t model_id = 0; model_id < settings.models.size(); ++model_id) {
            lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));
            cuts->send_transform(context_id, m_id, scm::math::mat4(_plugin->getModelInfo().model_transformations[m_id]));
            cuts->send_threshold(context_id, m_id, _plugin->getSettings().lod_error);
            cuts->send_rendered(context_id, m_id);
            database->get_model(m_id)->set_transform(scm::math::mat4(_plugin->getModelInfo().model_transformations[m_id]));
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
        
        const scm::math::mat4d viewport_scale = scm::math::make_scale(opencover::coVRConfig::instance()->windows[0].context->getTraits()->width * 0.5, opencover::coVRConfig::instance()->windows[0].context->getTraits()->height * 0.5, 0.5);
        const scm::math::mat4d viewport_translate = scm::math::make_translation(1.0, 1.0, 1.0);

        scm::math::vec2 viewport = scm::math::vec2f(opencover::coVRConfig::instance()->windows[0].context->getTraits()->width, opencover::coVRConfig::instance()->windows[0].context->getTraits()->height);
        SDStats sd = makeSDStats(meas, viewport, projection_matrix, settings);
        uint64_t rendered_primitives = 0;
        uint64_t rendered_nodes = 0;

        if (_plugin->getSettings().shader_type == LamureRenderer::ShaderType::SurfelMultipass && _initialized) {
            // ================= MULTI-PASS RENDER PATH =================
            auto&       res = _renderer->getPclResource();
            const auto& s   = _plugin->getSettings();

            GLint prev_fbo = 0;
            GLint prev_viewport[4] = {0,0,0,0};
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
            glGetIntegerv(GL_VIEWPORT, prev_viewport);

            const int height = opencover::coVRConfig::instance()->windows[0].context->getTraits()->height;
            const int width  = opencover::coVRConfig::instance()->windows[0].context->getTraits()->width;

            // --- PASS 1: Depth pre-pass (depth-only, kreisförmiges discard im FS)
            glBindFramebuffer(GL_FRAMEBUFFER, res.fbo);
            glViewport(0, 0, width, height);
            glDrawBuffer(GL_NONE);
            glReadBuffer(GL_NONE);
            glClear(GL_DEPTH_BUFFER_BIT);

            glDisable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LEQUAL);

            glUseProgram(_renderer->getSurfelPass1Shader().program);

            // Globale Uniforms (nur setzen, wenn vorhanden)
            if (_renderer->getSurfelPass1Shader().viewport_loc          >= 0) glUniform2f(_renderer->getSurfelPass1Shader().viewport_loc, width, height);
            if (_renderer->getSurfelPass1Shader().near_plane_loc        >= 0) glUniform1f(_renderer->getSurfelPass1Shader().near_plane_loc, _renderer->getScmCamera()->near_plane_value());
            if (_renderer->getSurfelPass1Shader().far_plane_loc         >= 0) glUniform1f(_renderer->getSurfelPass1Shader().far_plane_loc,  _renderer->getScmCamera()->far_plane_value());
            if (_renderer->getSurfelPass1Shader().max_radius_loc        >= 0) glUniform1f(_renderer->getSurfelPass1Shader().max_radius_loc,   s.max_radius);
            if (_renderer->getSurfelPass1Shader().min_radius_loc        >= 0) glUniform1f(_renderer->getSurfelPass1Shader().min_radius_loc,   s.min_radius);
            if (_renderer->getSurfelPass1Shader().scale_radius_loc      >= 0) glUniform1f(_renderer->getSurfelPass1Shader().scale_radius_loc, s.scale_radius * s.scale_element);
            if (_renderer->getSurfelPass1Shader().scale_radius_gamma_loc  >= 0) glUniform1f(_renderer->getSurfelPass1Shader().scale_radius_gamma_loc,   s.scale_radius_gamma);
            if (_renderer->getSurfelPass1Shader().max_radius_cut_loc      >= 0) glUniform1f(_renderer->getSurfelPass1Shader().max_radius_cut_loc,   s.max_radius_cut);
            if (_renderer->getSurfelPass1Shader().projection_matrix_loc >= 0) glUniformMatrix4fv(_renderer->getSurfelPass1Shader().projection_matrix_loc, 1, GL_FALSE, projection_matrix.data_array);
            if (_renderer->getSurfelPass1Shader().min_screen_size_loc    >= 0) glUniform1f(_renderer->getSurfelPass1Shader().min_screen_size_loc, s.min_screen_size);
            if (_renderer->getSurfelPass1Shader().max_screen_size_loc    >= 0) glUniform1f(_renderer->getSurfelPass1Shader().max_screen_size_loc, s.max_screen_size);
            if (_renderer->getSurfelPass1Shader().scale_projection_loc >= 0) glUniform1f(_renderer->getSurfelPass1Shader().scale_projection_loc, opencover::cover->getScale() * height * 0.5f * projection_matrix.data_array[5]);

            for (uint16_t model_id = 0; model_id < s.models.size(); ++model_id) {
                if (!_plugin->isModelVisible(model_id)) continue;

                const lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));
                
                lamure::ren::cut &cut = cuts->get_cut(context_id, _renderer->getOsgCamera()->getGraphicsContext()->getState()->getContextID(), m_id);
                auto renderable = cut.complete_set();
                
                const lamure::ren::bvh *bvh = database->get_model(m_id)->get_bvh();
                const auto &bbox = bvh->get_bounding_boxes();

                scm::math::mat4 model_matrix      = scm::math::mat4(_plugin->getModelInfo().model_transformations[m_id]);
                scm::math::mat4 model_view_matrix = view_matrix * model_matrix;
                scm::math::mat3 normal_matrix     = scm::math::transpose(scm::math::inverse(LamureUtil::matConv4to3F(model_view_matrix)));
                scm::gl::frustum frustum          = _renderer->getScmCamera()->get_frustum_by_model(model_matrix);

                if (_renderer->getSurfelPass1Shader().model_view_matrix_loc >= 0)
                    glUniformMatrix4fv(_renderer->getSurfelPass1Shader().model_view_matrix_loc, 1, GL_FALSE, model_view_matrix.data_array);
                if (_renderer->getSurfelPass1Shader().normal_matrix_loc >= 0)
                    glUniformMatrix3fv(_renderer->getSurfelPass1Shader().normal_matrix_loc, 1, GL_FALSE, normal_matrix.data_array);

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
            glBlendFunc(GL_ONE, GL_ONE);       // Summenbildung (premultiplied sums)
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDepthFunc(GL_ALWAYS);            // Tiefe entscheidet rein im FS

            glUseProgram(_renderer->getSurfelPass2Shader().program);

            // Tiefe aus Pass 1
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, res.depth_texture);
            if (_renderer->getSurfelPass2Shader().depth_texture_loc >= 0)
                glUniform1i(_renderer->getSurfelPass2Shader().depth_texture_loc, 0);

            // Globale Uniforms für VS/GS/FS
            if (_renderer->getSurfelPass2Shader().viewport_loc            >= 0) glUniform2f(_renderer->getSurfelPass2Shader().viewport_loc, viewport.x, viewport.y);
            if (_renderer->getSurfelPass2Shader().max_radius_loc          >= 0) glUniform1f(_renderer->getSurfelPass2Shader().max_radius_loc,   s.max_radius);
            if (_renderer->getSurfelPass2Shader().min_radius_loc          >= 0) glUniform1f(_renderer->getSurfelPass2Shader().min_radius_loc,   s.min_radius);
            if (_renderer->getSurfelPass2Shader().scale_radius_loc        >= 0) glUniform1f(_renderer->getSurfelPass2Shader().scale_radius_loc, s.scale_radius * s.scale_element);
            if (_renderer->getSurfelPass2Shader().scale_radius_gamma_loc  >= 0) glUniform1f(_renderer->getSurfelPass2Shader().scale_radius_gamma_loc,   s.scale_radius_gamma);
            if (_renderer->getSurfelPass2Shader().max_radius_cut_loc      >= 0) glUniform1f(_renderer->getSurfelPass2Shader().max_radius_cut_loc,  s.max_radius_cut);
            if (_renderer->getSurfelPass2Shader().coloring_loc            >= 0) glUniform1f(_renderer->getSurfelPass2Shader().coloring_loc, s.coloring);     
            if (_renderer->getSurfelPass2Shader().show_normals_loc        >= 0) glUniform1i(_renderer->getSurfelPass2Shader().show_normals_loc,         s.show_normals);
            if (_renderer->getSurfelPass2Shader().show_output_sens_loc    >= 0) glUniform1i(_renderer->getSurfelPass2Shader().show_output_sens_loc,     s.show_output_sensitivity);
            if (_renderer->getSurfelPass2Shader().show_radius_dev_loc     >= 0) glUniform1i(_renderer->getSurfelPass2Shader().show_radius_dev_loc,      s.show_radius_deviation);
            if (_renderer->getSurfelPass2Shader().show_accuracy_loc       >= 0) glUniform1i(_renderer->getSurfelPass2Shader().show_accuracy_loc,        s.show_accuracy);
            if (_renderer->getSurfelPass2Shader().projection_matrix_loc   >= 0) glUniformMatrix4fv(_renderer->getSurfelPass2Shader().projection_matrix_loc, 1, GL_FALSE, projection_matrix.data_array);
            if (_renderer->getSurfelPass2Shader().min_screen_size_loc    >= 0) glUniform1f(_renderer->getSurfelPass2Shader().min_screen_size_loc, s.min_screen_size);
            if (_renderer->getSurfelPass2Shader().max_screen_size_loc    >= 0) glUniform1f(_renderer->getSurfelPass2Shader().max_screen_size_loc, s.max_screen_size);
            if (_renderer->getSurfelPass2Shader().scale_projection_loc >= 0) glUniform1f(_renderer->getSurfelPass2Shader().scale_projection_loc, opencover::cover->getScale() * height * 0.5f * projection_matrix.data_array[5]);

            // Blending-Uniforms
            if (_renderer->getSurfelPass2Shader().depth_range_loc >= 0) glUniform1f(_renderer->getSurfelPass2Shader().depth_range_loc, s.depth_range);
            if (_renderer->getSurfelPass2Shader().flank_lift_loc             >= 0) glUniform1f(_renderer->getSurfelPass2Shader().flank_lift_loc,            s.flank_lift);

            const bool needNodeUniforms = (s.show_radius_deviation || s.show_accuracy);

            for (uint16_t model_id = 0; model_id < s.models.size(); ++model_id) {
                //if (!_plugin->isModelVisible(model_id)) continue;
                const lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));
                lamure::ren::cut &cut = cuts->get_cut(context_id,
                    _renderer->getOsgCamera()->getGraphicsContext()->getState()->getContextID(), m_id);
                auto renderable = cut.complete_set();
                const lamure::ren::bvh *bvh = database->get_model(m_id)->get_bvh();
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
            glBindFramebuffer(GL_READ_FRAMEBUFFER, res.fbo);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev_fbo);
            glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

            glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
            glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);

            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);

            glDisable(GL_BLEND);
            //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA); // premultiplied resolve

            glUseProgram(_renderer->getSurfelPass3Shader().program);

            // G-Buffer
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, res.texture_color);
            if (_renderer->getSurfelPass3Shader().in_color_texture_loc >= 0)
                glUniform1i(_renderer->getSurfelPass3Shader().in_color_texture_loc, 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, res.texture_normal);
            if (_renderer->getSurfelPass3Shader().in_normal_texture_loc >= 0)
                glUniform1i(_renderer->getSurfelPass3Shader().in_normal_texture_loc, 1);

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, res.texture_position);
            if (_renderer->getSurfelPass3Shader().in_vs_position_texture_loc >= 0)
                glUniform1i(_renderer->getSurfelPass3Shader().in_vs_position_texture_loc, 2);

            // View-space lighting Setup
            osg::Matrix base = opencover::VRSceneGraph::instance()->getScaleTransform()->getMatrix();
            base.postMult(opencover::VRSceneGraph::instance()->getTransform()->getMatrix());
            scm::math::mat4 viewMat = LamureUtil::matConv4F(_renderer->getOsgCamera()->getViewMatrix()) * LamureUtil::matConv4F(base);
            scm::math::vec3 background_color = LamureUtil::vecConv3F(_renderer->getOsgCamera()->getClearColor());

            scm::math::vec4 light_ws(s.point_light_pos[0], s.point_light_pos[1], s.point_light_pos[2], 1.0f);
            scm::math::vec4 light_vs4 = viewMat * light_ws;
            scm::math::vec3 light_vs(light_vs4[0], light_vs4[1], light_vs4[2]);

            if (_renderer->getSurfelPass3Shader().background_color_loc        >= 0) glUniform3fv(_renderer->getSurfelPass3Shader().background_color_loc, 1, background_color.data_array);
            if (_renderer->getSurfelPass3Shader().view_matrix_loc             >= 0) glUniformMatrix4fv(_renderer->getSurfelPass3Shader().view_matrix_loc,  1, GL_FALSE, viewMat.data_array);
            if (_renderer->getSurfelPass3Shader().point_light_pos_vs_loc      >= 0) glUniform3fv(_renderer->getSurfelPass3Shader().point_light_pos_vs_loc, 1, light_vs.data_array);
            if (_renderer->getSurfelPass3Shader().point_light_intensity_loc   >= 0) glUniform1f(_renderer->getSurfelPass3Shader().point_light_intensity_loc, s.point_light_intensity);
            if (_renderer->getSurfelPass3Shader().ambient_intensity_loc       >= 0) glUniform1f(_renderer->getSurfelPass3Shader().ambient_intensity_loc,     s.ambient_intensity);
            if (_renderer->getSurfelPass3Shader().specular_intensity_loc      >= 0) glUniform1f(_renderer->getSurfelPass3Shader().specular_intensity_loc,    s.specular_intensity);
            if (_renderer->getSurfelPass3Shader().shininess_loc               >= 0) glUniform1f(_renderer->getSurfelPass3Shader().shininess_loc,             s.shininess);
            if (_renderer->getSurfelPass3Shader().gamma_loc                   >= 0) glUniform1f(_renderer->getSurfelPass3Shader().gamma_loc,                 s.gamma);
            if (_renderer->getSurfelPass3Shader().use_tone_mapping_loc        >= 0) glUniform1i(_renderer->getSurfelPass3Shader().use_tone_mapping_loc,      s.use_tone_mapping ? 1 : 0);
            if (_renderer->getSurfelPass3Shader().lighting_loc                >= 0) glUniform1f(_renderer->getSurfelPass3Shader().lighting_loc, s.lighting);  

            // Fullscreen-Quad
            glBindVertexArray(res.screen_quad_vao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);

            glActiveTexture(GL_TEXTURE0);
        }
        else {
            // ================= SINGLE-PASS RENDER PATH =================
            _renderer->setFrameUniforms(projection_matrix, viewport);
            for (uint16_t model_id = 0; model_id < _plugin->getSettings().models.size(); ++model_id) {
                if (!_plugin->isModelVisible(model_id)) { continue; }
                const lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));
                lamure::ren::cut& cut = cuts->get_cut(context_id, _renderer->getOsgCamera()->getGraphicsContext()->getState()->getContextID(), m_id);
                std::vector<lamure::ren::cut::node_slot_aggregate> renderable = cut.complete_set();
                
                const lamure::ren::bvh* bvh = database->get_model(m_id)->get_bvh();
                std::vector<scm::gl::boxf>const& bounding_box_vector = bvh->get_bounding_boxes();

                scm::math::mat4 model_matrix = scm::math::mat4(_plugin->getModelInfo().model_transformations[m_id]);
                scm::math::mat4 model_view_matrix = view_matrix * model_matrix;
                scm::math::mat4 mvp_matrix = projection_matrix * model_view_matrix;
                scm::gl::frustum frustum = _renderer->getScmCamera()->get_frustum_by_model(model_matrix);
                
                _renderer->setModelUniforms(mvp_matrix);
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
        _renderer->setRendering(false);

        if (!_initialized) {
            GLState after = GLState::capture();
            if (after.getVertexArrayBinding() != before.getVertexArrayBinding())
            {
                _plugin->getRenderer()->getPclResource().vao = after.getVertexArrayBinding();
                _initialized = true;
            }
        }

        before.restore();
        if (_plugin->getUI()->getNotifyButton()->state()) {
            GLState after = GLState::capture();
            GLState::compare(before, after, "[Notify] PointsDrawCallback::drawImplementation()");
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
        if (_plugin->getUI()->getNotifyButton()->state()) { std::cout << "[Notify] PointsGeometry()" << std::endl; }
        setUseDisplayList(false);
        setUseVertexBufferObjects(true);
        setUseVertexArrayObject(false);
        setDrawCallback(new PointsDrawCallback(_plugin));

        //osg::Vec3 minPt = LamureUtil::vecConv3F(_plugin->getModelInfo().models_min);
        //osg::Vec3 maxPt = LamureUtil::vecConv3F(_plugin->getModelInfo().models_max);
        //osg::Vec3 halfExtents(std::max(fabs(minPt.x()), fabs(maxPt.x())),
        //    std::max(fabs(minPt.y()), fabs(maxPt.y())),
        //    std::max(fabs(minPt.z()), fabs(maxPt.z())));
        //_bbox = osg::BoundingBox(-halfExtents, halfExtents);
        //_bsphere = osg::BoundingSphere(_bbox.center(), _bbox.radius());
        //setInitialBound(_bbox);
    }
    Lamure* _plugin;
    osg::BoundingSphere _bsphere;
    osg::BoundingBox _bbox;
};


void LamureRenderer::init()
{
    std::cout << "LamureRenderer::init()" << std::endl;
    initCamera();

    if (auto* vs = opencover::VRViewer::instance()->getViewerStats()) {
        vs->collectStats("frame_rate", true);
    }

    m_init_geode         = new osg::Geode();
    m_pointcloud_geode   = new osg::Geode();
    m_boundingbox_geode  = new osg::Geode();
    m_frustum_geode      = new osg::Geode();
    m_text_geode         = new TextGeode(m_plugin);

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

    osg::ref_ptr<osg::MatrixTransform> frustumTransform = new osg::MatrixTransform;
    auto updateFrustum = [frustumTransform](const osg::Vec3 &pos) { frustumTransform->setMatrix(osg::Matrix::translate(pos)); };
    updateFrustum(osg::Vec3(m_scm_camera->get_cam_pos()[0], m_scm_camera->get_cam_pos()[1], m_scm_camera->get_cam_pos()[2]));
    m_group->addChild(frustumTransform);
    m_group->addChild(m_frustum_geode);
    m_group->addChild(m_boundingbox_geode);
    m_group->addChild(m_pointcloud_geode);
    m_group->addChild(m_init_geode);
    m_hud_camera->addChild(m_text_geode.get());

    m_init_geometry = new InitGeometry(m_plugin);
    m_pointcloud_geometry = new PointsGeometry(m_plugin);
    m_boundingbox_geometry = new BoundingBoxGeometry(m_plugin);
    m_frustum_geometry = new FrustumGeometry(m_plugin);

    m_init_geode->addDrawable(m_init_geometry);
    m_pointcloud_geode->addDrawable(m_pointcloud_geometry);
    m_boundingbox_geode->addDrawable(m_boundingbox_geometry);
    m_frustum_geode->addDrawable(m_frustum_geometry);
}


void LamureRenderer::initUniforms() 
{
    cout << "[Notify] initUniforms()" << endl;

    glUseProgram(m_point_shader.program);
    m_point_shader.mvp_matrix_loc   = glGetUniformLocation(m_point_shader.program, "mvp_matrix");
    m_point_shader.max_radius_loc   = glGetUniformLocation(m_point_shader.program, "max_radius");
    m_point_shader.min_radius_loc   = glGetUniformLocation(m_point_shader.program, "min_radius");
    m_point_shader.max_screen_size_loc   = glGetUniformLocation(m_point_shader.program, "max_screen_size");
    m_point_shader.min_screen_size_loc   = glGetUniformLocation(m_point_shader.program, "min_screen_size");
    m_point_shader.scale_radius_loc = glGetUniformLocation(m_point_shader.program, "scale_radius");
    m_point_shader.scale_projection_loc   = glGetUniformLocation(m_point_shader.program, "scale_projection");
    m_point_shader.max_radius_cut_loc = glGetUniformLocation(m_point_shader.program, "max_radius_cut");
    m_point_shader.scale_radius_gamma_loc   = glGetUniformLocation(m_point_shader.program, "scale_radius_gamma");
    glUseProgram(0);

    glUseProgram(m_point_color_shader.program);
    m_point_color_shader.mvp_matrix_loc   = glGetUniformLocation(m_point_color_shader.program, "mvp_matrix");
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
    m_point_color_shader.show_normals_loc         = glGetUniformLocation(m_point_color_shader.program, "show_normals");
    m_point_color_shader.show_accuracy_loc        = glGetUniformLocation(m_point_color_shader.program, "show_accuracy");
    m_point_color_shader.show_radius_dev_loc      = glGetUniformLocation(m_point_color_shader.program, "show_radius_deviation");
    m_point_color_shader.show_output_sens_loc     = glGetUniformLocation(m_point_color_shader.program, "show_output_sensitivity");
    m_point_color_shader.accuracy_loc             = glGetUniformLocation(m_point_color_shader.program, "accuracy");
    m_point_color_shader.average_radius_loc       = glGetUniformLocation(m_point_color_shader.program, "average_radius");
    glUseProgram(0);

    glUseProgram(m_point_color_lighting_shader.program);
    m_point_color_lighting_shader.mvp_matrix_loc          = glGetUniformLocation(m_point_color_lighting_shader.program, "mvp_matrix");
    m_point_color_lighting_shader.view_matrix_loc         = glGetUniformLocation(m_point_color_lighting_shader.program, "view_matrix");
    m_point_color_lighting_shader.normal_matrix_loc     = glGetUniformLocation(m_point_color_lighting_shader.program, "normal_matrix");
    m_point_color_lighting_shader.max_radius_loc          = glGetUniformLocation(m_point_color_lighting_shader.program, "max_radius");
    m_point_color_lighting_shader.min_radius_loc          = glGetUniformLocation(m_point_color_lighting_shader.program, "min_radius");
    m_point_color_lighting_shader.max_screen_size_loc          = glGetUniformLocation(m_point_color_lighting_shader.program, "max_screen_size");
    m_point_color_lighting_shader.min_screen_size_loc          = glGetUniformLocation(m_point_color_lighting_shader.program, "min_screen_size");
    m_point_color_lighting_shader.scale_radius_loc        = glGetUniformLocation(m_point_color_lighting_shader.program, "scale_radius");
    m_point_color_lighting_shader.scale_projection_loc   = glGetUniformLocation(m_point_color_lighting_shader.program, "scale_projection");
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
    m_surfel_shader.max_radius_loc          = glGetUniformLocation(m_surfel_shader.program, "max_radius");
    m_surfel_shader.min_radius_loc          = glGetUniformLocation(m_surfel_shader.program, "min_radius");
    m_surfel_shader.max_screen_size_loc          = glGetUniformLocation(m_surfel_shader.program, "max_screen_size");
    m_surfel_shader.min_screen_size_loc          = glGetUniformLocation(m_surfel_shader.program, "min_screen_size");
    m_surfel_shader.scale_radius_loc        = glGetUniformLocation(m_surfel_shader.program, "scale_radius");
    m_surfel_shader.scale_projection_loc   = glGetUniformLocation(m_surfel_shader.program, "scale_projection");
    m_surfel_shader.max_radius_cut_loc = glGetUniformLocation(m_surfel_shader.program, "max_radius_cut");
    m_surfel_shader.scale_radius_gamma_loc   = glGetUniformLocation(m_surfel_shader.program, "scale_radius_gamma");
    glUseProgram(0);

    glUseProgram(m_surfel_color_shader.program);
    m_surfel_color_shader.mvp_matrix_loc    = glGetUniformLocation(m_surfel_color_shader.program, "mvp_matrix");
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
    m_surfel_color_shader.show_normals_loc      = glGetUniformLocation(m_surfel_color_shader.program, "show_normals");
    m_surfel_color_shader.show_accuracy_loc     = glGetUniformLocation(m_surfel_color_shader.program, "show_accuracy");
    m_surfel_color_shader.show_radius_dev_loc   = glGetUniformLocation(m_surfel_color_shader.program, "show_radius_deviation");
    m_surfel_color_shader.show_output_sens_loc  = glGetUniformLocation(m_surfel_color_shader.program, "show_output_sensitivity");
    m_surfel_color_shader.accuracy_loc          = glGetUniformLocation(m_surfel_color_shader.program, "accuracy");
    m_surfel_color_shader.average_radius_loc    = glGetUniformLocation(m_surfel_color_shader.program, "average_radius");
    glUseProgram(0);

    glUseProgram(m_surfel_color_lighting_shader.program);
    m_surfel_color_lighting_shader.mvp_matrix_loc          = glGetUniformLocation(m_surfel_color_lighting_shader.program, "mvp_matrix");
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
    m_surfel_prov_shader.max_radius_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "max_radius");
    m_surfel_prov_shader.min_radius_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "min_radius");
    m_surfel_prov_shader.min_screen_size_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "min_screen_size");
    m_surfel_prov_shader.max_screen_size_loc    = glGetUniformLocation(m_surfel_prov_shader.program, "max_screen_size");
    m_surfel_prov_shader.scale_radius_loc  = glGetUniformLocation(m_surfel_prov_shader.program, "scale_radius");
    m_surfel_prov_shader.viewport_loc      = glGetUniformLocation(m_surfel_prov_shader.program, "viewport");
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
    m_surfel_pass1_shader.normal_matrix_loc      = glGetUniformLocation(m_surfel_pass1_shader.program, "normal_matrix");
    m_surfel_pass1_shader.projection_matrix_loc  = glGetUniformLocation(m_surfel_pass1_shader.program, "projection_matrix");
    m_surfel_pass1_shader.model_view_matrix_loc  = glGetUniformLocation(m_surfel_pass1_shader.program, "model_view_matrix");
    m_surfel_pass1_shader.model_matrix_loc       = glGetUniformLocation(m_surfel_pass1_shader.program, "model_matrix");
    m_surfel_pass1_shader.near_plane_loc         = glGetUniformLocation(m_surfel_pass1_shader.program, "near_plane");
    m_surfel_pass1_shader.far_plane_loc          = glGetUniformLocation(m_surfel_pass1_shader.program, "far_plane");
    m_surfel_pass1_shader.viewport_loc           = glGetUniformLocation(m_surfel_pass1_shader.program, "viewport");
    m_surfel_pass1_shader.max_radius_loc         = glGetUniformLocation(m_surfel_pass1_shader.program, "max_radius");
    m_surfel_pass1_shader.min_radius_loc         = glGetUniformLocation(m_surfel_pass1_shader.program, "min_radius");
    m_surfel_pass1_shader.min_screen_size_loc         = glGetUniformLocation(m_surfel_pass1_shader.program, "min_screen_size");
    m_surfel_pass1_shader.max_screen_size_loc         = glGetUniformLocation(m_surfel_pass1_shader.program, "max_screen_size");
    m_surfel_pass1_shader.scale_radius_loc       = glGetUniformLocation(m_surfel_pass1_shader.program, "scale_radius");
    m_surfel_pass1_shader.scale_projection_loc   = glGetUniformLocation(m_surfel_pass1_shader.program, "scale_projection");
    m_surfel_pass1_shader.max_radius_cut_loc = glGetUniformLocation(m_surfel_pass1_shader.program, "max_radius_cut");
    m_surfel_pass1_shader.scale_radius_gamma_loc   = glGetUniformLocation(m_surfel_pass1_shader.program, "scale_radius_gamma");
    glUseProgram(0);

    // --- PASS 2 ---
    glUseProgram(m_surfel_pass2_shader.program);
    m_surfel_pass2_shader.mvp_matrix_loc             = glGetUniformLocation(m_surfel_pass2_shader.program, "mvp_matrix");
    m_surfel_pass2_shader.model_view_matrix_loc      = glGetUniformLocation(m_surfel_pass2_shader.program, "model_view_matrix");
    m_surfel_pass2_shader.projection_matrix_loc      = glGetUniformLocation(m_surfel_pass2_shader.program, "projection_matrix");
    m_surfel_pass2_shader.normal_matrix_loc          = glGetUniformLocation(m_surfel_pass2_shader.program, "normal_matrix");
    m_surfel_pass2_shader.model_matrix_loc           = glGetUniformLocation(m_surfel_pass2_shader.program, "model_matrix");
    m_surfel_pass2_shader.inv_mv_matrix_loc          = glGetUniformLocation(m_surfel_pass2_shader.program, "inv_mv_matrix");
    m_surfel_pass2_shader.model_to_screen_matrix_loc = glGetUniformLocation(m_surfel_pass2_shader.program, "model_to_screen_matrix");

    m_surfel_pass2_shader.depth_epsilon_vs_loc          = glGetUniformLocation(m_surfel_pass2_shader.program, "depth_epsilon");
    m_surfel_pass2_shader.depth_texture_loc          = glGetUniformLocation(m_surfel_pass2_shader.program, "depth_texture");
    m_surfel_pass2_shader.near_plane_loc             = glGetUniformLocation(m_surfel_pass2_shader.program, "near_plane");
    m_surfel_pass2_shader.far_plane_loc              = glGetUniformLocation(m_surfel_pass2_shader.program, "far_plane");
    m_surfel_pass2_shader.viewport_loc               = glGetUniformLocation(m_surfel_pass2_shader.program, "viewport");
    m_surfel_pass2_shader.scale_projection_loc   = glGetUniformLocation(m_surfel_pass2_shader.program, "scale_projection");

    m_surfel_pass2_shader.max_radius_loc             = glGetUniformLocation(m_surfel_pass2_shader.program, "max_radius");
    m_surfel_pass2_shader.min_radius_loc             = glGetUniformLocation(m_surfel_pass2_shader.program, "min_radius");
    m_surfel_pass2_shader.max_screen_size_loc             = glGetUniformLocation(m_surfel_pass2_shader.program, "max_screen_size");
    m_surfel_pass2_shader.min_screen_size_loc             = glGetUniformLocation(m_surfel_pass2_shader.program, "min_screen_size");
    m_surfel_pass2_shader.scale_radius_loc           = glGetUniformLocation(m_surfel_pass2_shader.program, "scale_radius");
    m_surfel_pass2_shader.max_radius_cut_loc = glGetUniformLocation(m_surfel_pass2_shader.program, "max_radius_cut");
    m_surfel_pass2_shader.scale_radius_gamma_loc   = glGetUniformLocation(m_surfel_pass2_shader.program, "scale_radius_gamma");

    m_surfel_pass2_shader.show_normals_loc           = glGetUniformLocation(m_surfel_pass2_shader.program, "show_normals");
    m_surfel_pass2_shader.show_accuracy_loc          = glGetUniformLocation(m_surfel_pass2_shader.program, "show_accuracy");
    m_surfel_pass2_shader.show_radius_dev_loc        = glGetUniformLocation(m_surfel_pass2_shader.program, "show_radius_deviation");
    m_surfel_pass2_shader.show_output_sens_loc       = glGetUniformLocation(m_surfel_pass2_shader.program, "show_output_sensitivity");

    m_surfel_pass2_shader.accuracy_loc               = glGetUniformLocation(m_surfel_pass2_shader.program, "accuracy");
    m_surfel_pass2_shader.average_radius_loc         = glGetUniformLocation(m_surfel_pass2_shader.program, "average_radius");

    m_surfel_pass2_shader.channel_loc                = glGetUniformLocation(m_surfel_pass2_shader.program, "channel");
    m_surfel_pass2_shader.heatmap_loc                = glGetUniformLocation(m_surfel_pass2_shader.program, "heatmap");
    m_surfel_pass2_shader.heatmap_min_loc            = glGetUniformLocation(m_surfel_pass2_shader.program, "heatmap_min");
    m_surfel_pass2_shader.heatmap_max_loc            = glGetUniformLocation(m_surfel_pass2_shader.program, "heatmap_max");
    m_surfel_pass2_shader.heatmap_min_color_loc      = glGetUniformLocation(m_surfel_pass2_shader.program, "heatmap_min_color");
    m_surfel_pass2_shader.heatmap_max_color_loc      = glGetUniformLocation(m_surfel_pass2_shader.program, "heatmap_max_color");

    m_surfel_pass2_shader.edge_profile_loc           = glGetUniformLocation(m_surfel_pass2_shader.program, "edge_profile");

    m_surfel_pass2_shader.depth_range_loc            = glGetUniformLocation(m_surfel_pass2_shader.program, "depth_range");
    m_surfel_pass2_shader.flank_lift_loc             = glGetUniformLocation(m_surfel_pass2_shader.program, "flank_lift");

    m_surfel_pass2_shader.coloring_loc               = glGetUniformLocation(m_surfel_pass2_shader.program, "coloring");

    // Feste Samplerbindung (einmalig, spart pro-Frame-Setups)
    if (m_surfel_pass2_shader.depth_texture_loc >= 0)
        glUniform1i(m_surfel_pass2_shader.depth_texture_loc, 0);
    glUseProgram(0);

    // --- PASS 3 ---
    glUseProgram(m_surfel_pass3_shader.program);
    m_surfel_pass3_shader.in_color_texture_loc        = glGetUniformLocation(m_surfel_pass3_shader.program, "in_color_texture");
    m_surfel_pass3_shader.in_normal_texture_loc       = glGetUniformLocation(m_surfel_pass3_shader.program, "in_normal_texture");
    m_surfel_pass3_shader.in_vs_position_texture_loc  = glGetUniformLocation(m_surfel_pass3_shader.program, "in_vs_position_texture");

    m_surfel_pass3_shader.background_color_loc        = glGetUniformLocation(m_surfel_pass3_shader.program, "background_color");

    m_surfel_pass3_shader.view_matrix_loc             = glGetUniformLocation(m_surfel_pass3_shader.program, "view_matrix");
    m_surfel_pass3_shader.normal_matrix_loc           = glGetUniformLocation(m_surfel_pass3_shader.program, "normal_matrix");

    m_surfel_pass3_shader.point_light_pos_vs_loc      = glGetUniformLocation(m_surfel_pass3_shader.program, "point_light_pos_vs");
    m_surfel_pass3_shader.point_light_intensity_loc   = glGetUniformLocation(m_surfel_pass3_shader.program, "point_light_intensity");
    m_surfel_pass3_shader.ambient_intensity_loc       = glGetUniformLocation(m_surfel_pass3_shader.program, "ambient_intensity");
    m_surfel_pass3_shader.specular_intensity_loc      = glGetUniformLocation(m_surfel_pass3_shader.program, "specular_intensity");
    m_surfel_pass3_shader.shininess_loc               = glGetUniformLocation(m_surfel_pass3_shader.program, "shininess");
    m_surfel_pass3_shader.gamma_loc                   = glGetUniformLocation(m_surfel_pass3_shader.program, "gamma");
    m_surfel_pass3_shader.use_tone_mapping_loc        = glGetUniformLocation(m_surfel_pass3_shader.program, "use_tone_mapping");

    m_surfel_pass3_shader.show_normals_loc            = glGetUniformLocation(m_surfel_pass3_shader.program, "show_normals");
    m_surfel_pass3_shader.show_radius_dev_loc         = glGetUniformLocation(m_surfel_pass3_shader.program, "show_radius_deviation");
    m_surfel_pass3_shader.show_output_sens_loc        = glGetUniformLocation(m_surfel_pass3_shader.program, "show_output_sensitivity");
    m_surfel_pass3_shader.show_accuracy_loc           = glGetUniformLocation(m_surfel_pass3_shader.program, "show_accuracy");
    m_surfel_pass3_shader.accuracy_loc                = glGetUniformLocation(m_surfel_pass3_shader.program, "accuracy");
    m_surfel_pass3_shader.average_radius_loc          = glGetUniformLocation(m_surfel_pass3_shader.program, "average_radius");

    m_surfel_pass3_shader.lighting_loc               = glGetUniformLocation(m_surfel_pass3_shader.program, "lighting");

    if (m_surfel_pass3_shader.in_color_texture_loc >= 0)       glUniform1i(m_surfel_pass3_shader.in_color_texture_loc, 0);
    if (m_surfel_pass3_shader.in_normal_texture_loc >= 0)      glUniform1i(m_surfel_pass3_shader.in_normal_texture_loc, 1);
    if (m_surfel_pass3_shader.in_vs_position_texture_loc >= 0) glUniform1i(m_surfel_pass3_shader.in_vs_position_texture_loc, 2);

    glUseProgram(0);
}


void LamureRenderer::setFrameUniforms(const scm::math::mat4& projection_matrix, const scm::math::vec2& viewport) {
    const auto &s = m_plugin->getSettings();
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
        glUniform1f(prog.scale_projection_loc, opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5]);
        break;
    }
    case ShaderType::PointColor: {
        osg::Matrix base = opencover::VRSceneGraph::instance()->getScaleTransform()->getMatrix();
        base.postMult(opencover::VRSceneGraph::instance()->getTransform()->getMatrix());
        scm::math::mat4 viewMat = LamureUtil::matConv4F(m_renderer->getOsgCamera()->getViewMatrix()) * LamureUtil::matConv4F(base);
        scm::math::mat3 viewMat3 = LamureUtil::matConv4to3F(viewMat);
        scm::math::mat3 normalMat = scm::math::transpose(scm::math::inverse(viewMat3));
        scm::math::vec4 light_ws(s.point_light_pos[0], s.point_light_pos[1], s.point_light_pos[2], 1.0f);
        scm::math::vec4 light_vs4 = viewMat * light_ws;
        scm::math::vec3 light_vs(light_vs4[0], light_vs4[1], light_vs4[2]);
        auto& prog = m_point_color_shader;
        glUseProgram(prog.program);
        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUniformMatrix4fv(prog.view_matrix_loc,        1, GL_FALSE, viewMat.data_array);
        glUniformMatrix3fv(prog.normal_matrix_loc, 1, GL_FALSE, normalMat.data_array);
        glUniform1f(prog.min_radius_loc, s.min_radius);
        glUniform1f(prog.max_radius_loc, s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, s.scale_radius);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform1f(prog.scale_projection_loc, opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5]);
        glUniform1i(prog.show_normals_loc, s.show_normals);
        glUniform1i(prog.show_radius_dev_loc, s.show_radius_deviation);
        glUniform1i(prog.show_output_sens_loc, s.show_output_sensitivity);
        glUniform1i(prog.show_accuracy_loc, s.show_accuracy);
        break;
    }
    case ShaderType::PointColorLighting: {
        osg::Matrix base = opencover::VRSceneGraph::instance()->getScaleTransform()->getMatrix();
        base.postMult(opencover::VRSceneGraph::instance()->getTransform()->getMatrix());
        scm::math::mat4 viewMat = LamureUtil::matConv4F(m_renderer->getOsgCamera()->getViewMatrix()) * LamureUtil::matConv4F(base);
        scm::math::mat3 viewMat3 = LamureUtil::matConv4to3F(viewMat);
        scm::math::mat3 normalMat = scm::math::transpose(scm::math::inverse(viewMat3));
        scm::math::vec4 light_ws(s.point_light_pos[0], s.point_light_pos[1], s.point_light_pos[2], 1.0f);
        scm::math::vec4 light_vs4 = viewMat * light_ws;
        scm::math::vec3 light_vs(light_vs4[0], light_vs4[1], light_vs4[2]);
        auto& prog = m_point_color_lighting_shader;
        glUseProgram(prog.program);
        glUniformMatrix4fv(prog.view_matrix_loc, 1, GL_FALSE, viewMat.data_array);
        glUniformMatrix3fv(prog.normal_matrix_loc, 1, GL_FALSE, normalMat.data_array);
        glUniform1f(prog.min_radius_loc, s.min_radius);
        glUniform1f(prog.max_radius_loc, s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, s.scale_radius);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform1f(prog.scale_projection_loc, opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5]);
        glUniform1i(prog.show_normals_loc, s.show_normals);
        glUniform1i(prog.show_radius_dev_loc, s.show_radius_deviation);
        glUniform1i(prog.show_output_sens_loc, s.show_output_sensitivity);
        glUniform1i(prog.show_accuracy_loc, s.show_accuracy);
        glUniform3fv(prog.point_light_pos_vs_loc, 1, light_vs.data_array);
        glUniform1f(prog.point_light_intensity_loc, s.point_light_intensity);
        glUniform1f(prog.ambient_intensity_loc,     s.ambient_intensity);
        glUniform1f(prog.specular_intensity_loc,    s.specular_intensity);
        glUniform1f(prog.shininess_loc,             s.shininess);
        glUniform1f(prog.gamma_loc,                 s.gamma);
        glUniform1i(prog.use_tone_mapping_loc,      s.use_tone_mapping ? 1 : 0);
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
        glUniform1f(prog.scale_projection_loc, opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5]);
        glUniform1i(prog.show_normals_loc, s.show_normals);
        glUniform1i(prog.show_radius_dev_loc, s.show_radius_deviation);
        glUniform1i(prog.show_output_sens_loc, s.show_output_sensitivity);
        glUniform1i(prog.show_accuracy_loc, s.show_accuracy);
        glUniform1i(prog.channel_loc, s.channel);
        glUniform1i(prog.heatmap_loc, s.heatmap);
        glUniform1f(prog.heatmap_min_loc, s.heatmap_min);
        glUniform1f(prog.heatmap_max_loc, s.heatmap_max);
        glUniform3fv(prog.heatmap_min_color_loc, 1, s.heatmap_color_min.data_array);
        glUniform3fv(prog.heatmap_max_color_loc, 1, s.heatmap_color_max.data_array);
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
        glUniform1f(prog.scale_radius_loc, s.scale_radius * s.scale_element);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform1f(prog.scale_projection_loc, opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5]);
        break;
    }
    case ShaderType::SurfelColor: {
        osg::Matrix base = opencover::VRSceneGraph::instance()->getScaleTransform()->getMatrix();
        base.postMult(opencover::VRSceneGraph::instance()->getTransform()->getMatrix());
        scm::math::mat4 viewMat = LamureUtil::matConv4F(m_renderer->getOsgCamera()->getViewMatrix()) * LamureUtil::matConv4F(base);
        scm::math::mat3 viewMat3 = LamureUtil::matConv4to3F(viewMat);
        scm::math::mat3 normalMat = scm::math::transpose(scm::math::inverse(viewMat3));
        scm::math::vec4 light_ws(s.point_light_pos[0], s.point_light_pos[1], s.point_light_pos[2], 1.0f);
        scm::math::vec4 light_vs4 = viewMat * light_ws;
        scm::math::vec3 light_vs(light_vs4[0], light_vs4[1], light_vs4[2]);
        auto& prog = m_surfel_color_shader;
        //glEnable(GL_DEPTH_TEST);
        glUseProgram(prog.program);
        glUniformMatrix4fv(prog.view_matrix_loc,        1, GL_FALSE, viewMat.data_array);
        glUniformMatrix3fv(prog.normal_matrix_loc, 1, GL_FALSE, normalMat.data_array);
        glUniform1f(prog.min_radius_loc, s.min_radius);
        glUniform1f(prog.max_radius_loc, s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, s.scale_radius * s.scale_element);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform2fv(prog.viewport_loc, 1, viewport.data_array);
        glUniform1f(prog.scale_projection_loc, opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5]);
        glUniform1i(prog.show_normals_loc, s.show_normals);
        glUniform1i(prog.show_radius_dev_loc, s.show_radius_deviation);
        glUniform1i(prog.show_output_sens_loc, s.show_output_sensitivity);
        glUniform1i(prog.show_accuracy_loc, s.show_accuracy);
        break;
    }
    case ShaderType::SurfelColorLighting: {
        osg::Matrix base = opencover::VRSceneGraph::instance()->getScaleTransform()->getMatrix();
        base.postMult(opencover::VRSceneGraph::instance()->getTransform()->getMatrix());
        scm::math::mat4 viewMat = LamureUtil::matConv4F(m_renderer->getOsgCamera()->getViewMatrix()) * LamureUtil::matConv4F(base);
        scm::math::mat3 viewMat3 = LamureUtil::matConv4to3F(viewMat);
        scm::math::mat3 normalMat = scm::math::transpose(scm::math::inverse(viewMat3));
        scm::math::vec4 light_ws(s.point_light_pos[0], s.point_light_pos[1], s.point_light_pos[2], 1.0f);
        scm::math::vec4 light_vs4 = viewMat * light_ws;
        scm::math::vec3 light_vs(light_vs4[0], light_vs4[1], light_vs4[2]);
        auto& prog = m_surfel_color_lighting_shader;
        glEnable(GL_DEPTH_TEST);
        glUseProgram(prog.program);
        glUniformMatrix4fv(prog.view_matrix_loc,        1, GL_FALSE, viewMat.data_array);
        glUniformMatrix3fv(prog.normal_matrix_loc, 1, GL_FALSE, normalMat.data_array);
        glUniform1f(prog.min_radius_loc,   s.min_radius);
        glUniform1f(prog.max_radius_loc,   s.max_radius);
        glUniform1f(prog.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog.scale_radius_loc, s.scale_radius * s.scale_element);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform2fv(prog.viewport_loc, 1, viewport.data_array);
        glUniform1f(prog.scale_projection_loc, opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5]);

        glUniform1i(prog.show_normals_loc,         s.show_normals ? 1 : 0);
        glUniform1i(prog.show_radius_dev_loc,      s.show_radius_deviation ? 1 : 0);
        glUniform1i(prog.show_output_sens_loc,     s.show_output_sensitivity ? 1 : 0);
        glUniform1i(prog.show_accuracy_loc,        s.show_accuracy ? 1 : 0);

        glUniform3fv(prog.point_light_pos_vs_loc, 1, light_vs.data_array);
        glUniform1f(prog.point_light_intensity_loc, s.point_light_intensity);
        glUniform1f(prog.ambient_intensity_loc,     s.ambient_intensity);
        glUniform1f(prog.specular_intensity_loc,    s.specular_intensity);
        glUniform1f(prog.shininess_loc,             s.shininess);
        glUniform1f(prog.gamma_loc,                 s.gamma);
        glUniform1i(prog.use_tone_mapping_loc,      s.use_tone_mapping ? 1 : 0);
        //print_active_uniforms(prog.program, "SurfelColorLighting");
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
        glUniform1f(prog.scale_radius_loc, s.scale_radius * s.scale_element);
        glUniform1f(prog.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform2fv(prog.viewport_loc, 1, viewport.data_array);
        glUniform1f(prog.scale_projection_loc, opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5]);
        glUniform1i(prog.show_normals_loc, s.show_normals);
        glUniform1i(prog.show_radius_dev_loc, s.show_radius_deviation);
        glUniform1i(prog.show_output_sens_loc, s.show_output_sensitivity);
        glUniform1i(prog.show_accuracy_loc, s.show_accuracy);
        glUniform1i(prog.channel_loc, s.channel);
        glUniform1i(prog.heatmap_loc, s.heatmap);
        glUniform1f(prog.heatmap_min_loc, s.heatmap_min);
        glUniform1f(prog.heatmap_max_loc, s.heatmap_max);
        glUniform3fv(prog.heatmap_min_color_loc, 1, s.heatmap_color_min.data_array);
        glUniform3fv(prog.heatmap_max_color_loc, 1, s.heatmap_color_max.data_array);
        break;
    }
    case ShaderType::SurfelMultipass: {
        auto& prog1 = m_surfel_pass1_shader;
        glUseProgram(prog1.program);
        glUniform1f(prog1.max_radius_loc, s.max_radius);
        glUniform1f(prog1.min_radius_loc, s.min_radius);
        glUniform1f(prog1.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog1.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog1.scale_radius_loc, s.scale_radius * s.scale_element);
        glUniform1f(prog1.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog1.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform1f(prog1.scale_projection_loc, opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5]);
        glUniform1f(prog1.far_plane_loc, m_plugin->getRenderer()->getScmCamera()->far_plane_value());

        auto& prog2 = m_surfel_pass2_shader;
        glUseProgram(prog2.program);
        //coco->nearClip(), coco->farClip()
        glUniform1f(prog2.near_plane_loc, m_plugin->getRenderer()->getScmCamera()->near_plane_value());
        glUniform1f(prog2.max_radius_loc, s.max_radius);
        glUniform1f(prog2.min_radius_loc, s.min_radius);
        glUniform1f(prog2.min_screen_size_loc, s.min_screen_size);
        glUniform1f(prog2.max_screen_size_loc, s.max_screen_size);
        glUniform1f(prog2.scale_radius_loc, s.scale_radius * s.scale_element);
        glUniform1f(prog2.max_radius_cut_loc, s.max_radius_cut);
        glUniform1f(prog2.scale_radius_gamma_loc, s.scale_radius_gamma);
        glUniform1f(prog2.scale_projection_loc, opencover::cover->getScale() * viewport.y * 0.5f * projection_matrix.data_array[5]);
        glUniform1i(prog2.show_normals_loc, s.show_normals);
        glUniform1i(prog2.show_radius_dev_loc, s.show_radius_deviation);
        glUniform1i(prog2.show_output_sens_loc, s.show_output_sensitivity);
        glUniform1i(prog2.show_accuracy_loc, s.show_accuracy);
        glUniform1i(prog2.channel_loc, s.channel);
        glUniform1i(prog2.heatmap_loc, s.heatmap);
        glUniform1f(prog2.heatmap_min_loc, s.heatmap_min);
        glUniform1f(prog2.heatmap_max_loc, s.heatmap_max);
        glUniform3fv(prog2.heatmap_min_color_loc, 1, s.heatmap_color_min.data_array);
        glUniform3fv(prog2.heatmap_max_color_loc, 1, s.heatmap_color_max.data_array);
        break;
    }
    }
}

void LamureRenderer::setModelUniforms(const scm::math::mat4& mvp_matrix) {
    switch (m_plugin->getSettings().shader_type) {
    case ShaderType::Point:       glUniformMatrix4fv(m_point_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array); break;
    case ShaderType::PointColor:  glUniformMatrix4fv(m_point_color_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array); break;
    case ShaderType::PointColorLighting: glUniformMatrix4fv(m_point_color_lighting_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array); break;
    case ShaderType::PointProv:   glUniformMatrix4fv(m_point_prov_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array); break;
    case ShaderType::Surfel:      glUniformMatrix4fv(m_surfel_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array); break;
    case ShaderType::SurfelColor: glUniformMatrix4fv(m_surfel_color_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array); break;
    case ShaderType::SurfelColorLighting: glUniformMatrix4fv(m_surfel_color_lighting_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array); break;
    case ShaderType::SurfelProv:  glUniformMatrix4fv(m_surfel_prov_shader.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix.data_array); break;
    case ShaderType::SurfelMultipass: break;
    }
}


void LamureRenderer::setNodeUniforms(const lamure::ren::bvh* bvh, uint32_t node_id) {
    const auto &s = m_plugin->getSettings();

    switch (s.shader_type) {
    case ShaderType::PointColor: {
        if (s.show_accuracy) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_point_color_shader.accuracy_loc, accuracy);
        }
        if (s.show_radius_deviation) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float gamma     = (s.scale_radius_gamma > 0.0f) ? s.scale_radius_gamma : 1.0f;
            const float avg_ws  = std::pow(std::max(0.0f, avg_raw), gamma) * s.scale_radius;
            glUniform1f(m_point_color_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::PointColorLighting: {
        if (s.show_accuracy) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_point_color_lighting_shader.accuracy_loc, accuracy);
        }
        if (s.show_radius_deviation) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float gamma     = (s.scale_radius_gamma > 0.0f) ? s.scale_radius_gamma : 1.0f;
            const float avg_ws  = std::pow(std::max(0.0f, avg_raw), gamma) * s.scale_radius;
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
            const float gamma     = (s.scale_radius_gamma > 0.0f) ? s.scale_radius_gamma : 1.0f;
            const float avg_ws  = std::pow(std::max(0.0f, avg_raw), gamma) * s.scale_radius;
            glUniform1f(m_point_prov_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelColor: {
        if (s.show_accuracy) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_surfel_color_shader.accuracy_loc, accuracy);
        }
        if (s.show_radius_deviation) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float gamma     = (s.scale_radius_gamma > 0.0f) ? s.scale_radius_gamma : 1.0f;
            const float avg_ws  = std::pow(std::max(0.0f, avg_raw), gamma) * s.scale_radius;
            glUniform1f(m_surfel_color_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelProv: {
        if (s.show_accuracy) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_surfel_prov_shader.accuracy_loc, accuracy);
        }
        if (s.show_radius_deviation) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float gamma     = (s.scale_radius_gamma > 0.0f) ? s.scale_radius_gamma : 1.0f;
            const float avg_ws  = std::pow(std::max(0.0f, avg_raw), gamma) * s.scale_radius;
            glUniform1f(m_surfel_prov_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelColorLighting: {
        if (s.show_accuracy) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_surfel_color_lighting_shader.accuracy_loc, accuracy);
        }
        if (s.show_radius_deviation) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float gamma     = (s.scale_radius_gamma > 0.0f) ? s.scale_radius_gamma : 1.0f;
            const float avg_ws  = std::pow(std::max(0.0f, avg_raw), gamma) * s.scale_radius;
            glUniform1f(m_surfel_color_lighting_shader.average_radius_loc, avg_ws);
        }
        break;
    }
    case ShaderType::SurfelMultipass: {
        if (s.show_accuracy) {
            float accuracy = 1.0f - float(bvh->get_depth_of_node(node_id)) / float(bvh->get_depth() - 1);
            glUniform1f(m_surfel_pass2_shader.accuracy_loc, accuracy);
        }
        if (s.show_radius_deviation) {
            const float avg_raw = bvh->get_avg_primitive_extent(node_id);
            const float gamma     = (s.scale_radius_gamma > 0.0f) ? s.scale_radius_gamma : 1.0f;
            const float avg_ws  = std::pow(std::max(0.0f, avg_raw), gamma) * s.scale_radius;
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
    try
    {
        char * val;
        val = getenv( "COVISEDIR" );
        shader_root_path=val;
        shader_root_path=shader_root_path+"/src/OpenCOVER/plugins/hlrs/LamurePointCloud/shaders";
        //shader_root_path=shader_root_path+"/share/covise/shaders";
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
    std::cout << "LamureRenderer::initCamera()" << std::endl;
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
    osgViewer::GraphicsWindow *window = windows.front();
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

    //for (lamure::model_t model_id = 0; model_id < m_plugin->getSettings().models.size(); ++model_id)
    //{
    //    lamure::model_t m_id = lamure::ren::controller::get_instance()->deduce_model_id(std::to_string(model_id));
    //    auto root_bb = lamure::ren::model_database::get_instance()->get_model(model_id)->get_bvh()->get_bounding_boxes()[0];
    //    m_plugin->getModelInfo().root_bb_min.push_back(scm::math::mat4f(m_plugin->getModelInfo().model_transformations[model_id]) * scm::math::vec4f(root_bb.min_vertex()[0], root_bb.min_vertex()[1], root_bb.min_vertex()[2], 1));
    //    m_plugin->getModelInfo().root_bb_max.push_back(scm::math::mat4f(m_plugin->getModelInfo().model_transformations[model_id]) * scm::math::vec4f(root_bb.max_vertex()[0], root_bb.max_vertex()[1], root_bb.max_vertex()[2], 1));
    //    m_plugin->getModelInfo().root_center.push_back(scm::math::mat4f(m_plugin->getModelInfo().model_transformations[model_id]) * scm::math::vec4f(root_bb.center()[0], root_bb.center()[1], root_bb.center()[2], 1));

    //    temp_center += m_plugin->getModelInfo().root_center.back();
    //    if (m_plugin->getModelInfo().root_bb_min[model_id][0] < root_min_temp[0])
    //    {
    //        root_min_temp[0] = m_plugin->getModelInfo().root_bb_min[model_id][0];
    //    }
    //    if (m_plugin->getModelInfo().root_bb_min[model_id][1] < root_min_temp[1])
    //    {
    //        root_min_temp[1] = m_plugin->getModelInfo().root_bb_min[model_id][1];
    //    }
    //    if (m_plugin->getModelInfo().root_bb_min[model_id][2] < root_min_temp[2])
    //    {
    //        root_min_temp[2] = m_plugin->getModelInfo().root_bb_min[model_id][2];
    //    }
    //    if (m_plugin->getModelInfo().root_bb_max[model_id][0] > root_max_temp[0])
    //    {
    //        root_max_temp[0] = m_plugin->getModelInfo().root_bb_max[model_id][0];
    //    }
    //    if (m_plugin->getModelInfo().root_bb_max[model_id][1] > root_max_temp[1])
    //    {
    //        root_max_temp[1] = m_plugin->getModelInfo().root_bb_max[model_id][1];
    //    }
    //    if (m_plugin->getModelInfo().root_bb_max[model_id][2] > root_max_temp[2])
    //    {
    //        root_max_temp[2] = m_plugin->getModelInfo().root_bb_max[model_id][2];
    //    }
    //}
    //m_plugin->getModelInfo().models_center = temp_center / m_plugin->getSettings().models.size();
    //m_plugin->getModelInfo().models_min = root_min_temp;
    //m_plugin->getModelInfo().models_max = root_max_temp;
}

//void LamureRenderer::initFramebuffer()
//{
//    if (m_plugin->getUI()->getNotifyButton()->state()) { std::cout << "[Notify] initFramebuffer() " << std::endl; }
//    fbo.reset();
//    fbo_color_buffer.reset();
//    fbo_depth_buffer.reset();
//    auto traits = opencover::coVRConfig::instance()->windows[0].context->getTraits();
//
//    fbo = m_device->create_frame_buffer();
//    fbo_color_buffer = m_device->create_texture_2d(scm::math::vec2ui(traits->width, traits->height), scm::gl::FORMAT_RGBA_32F, 1, 1, 1);
//    fbo_depth_buffer = m_device->create_texture_2d(scm::math::vec2ui(traits->width, traits->height), scm::gl::FORMAT_D24, 1, 1, 1);
//    fbo->attach_color_buffer(0, fbo_color_buffer);
//    fbo->attach_depth_stencil_buffer(fbo_depth_buffer);
//}


unsigned int LamureRenderer::createShader(const std::string& vertexShader, const std::string& fragmentShader, uint8_t ctx_id)
{
    osg::GLExtensions* gl_api = new osg::GLExtensions(ctx_id);
    unsigned int program = gl_api->glCreateProgram();
    unsigned int vs = compileShader(GL_VERTEX_SHADER, vertexShader, ctx_id);
    unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fragmentShader, ctx_id);
    gl_api->glAttachShader(program, vs);
    gl_api->glAttachShader(program, fs);
    gl_api->glLinkProgram(program);
    gl_api->glValidateProgram(program);
    gl_api->glDeleteShader(vs);
    gl_api->glDeleteShader(fs);
    return program;
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
    if (m_plugin->getUI()->getNotifyButton()->state()) {
        std::cout << "[Notify] create_frustum_resources() " << std::endl;
    }
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
    if (m_plugin->getUI()->getNotifyButton()->state()) { std::cout << "[Notify] init_box_resources() " << std::endl; }
    std::vector<float> all_box_vertices;
    m_bvh_node_vertex_offsets.clear(); 
    for (uint32_t model_id = 0; model_id < m_plugin->getSettings().models.size(); ++model_id) {
        const auto& bvh = lamure::ren::model_database::get_instance()->get_model(model_id)->get_bvh();
        const auto& bounding_boxes = bvh->get_bounding_boxes();

        std::vector<uint32_t> current_model_offsets;
        for (uint64_t node_id = 0; node_id < bounding_boxes.size(); ++node_id) {
            current_model_offsets.push_back(all_box_vertices.size() / 3);
            std::vector<float> corners = LamureUtil::getBoxCorners(bounding_boxes[node_id]);
            all_box_vertices.insert(all_box_vertices.end(), corners.begin(), corners.end());
        }
        m_bvh_node_vertex_offsets[model_id] = current_model_offsets;
    }

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_box_resource.idx.size() * sizeof(unsigned short), m_box_resource.idx.data(), GL_STATIC_DRAW);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, all_box_vertices.size() * sizeof(float), all_box_vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    m_box_resource.vao = vao;
    m_box_resource.vbo = vbo;
    m_box_resource.ibo = ibo;

    glBindVertexArray(0); // VAO unbinden
}



std::string LamureRenderer::glTypeToString(GLenum type) {
    switch (type) {
    case GL_FLOAT: return "float";
    case GL_FLOAT_VEC2: return "vec2";
    case GL_FLOAT_VEC3: return "vec3";
    case GL_FLOAT_VEC4: return "vec4";
    case GL_INT: return "int";
    case GL_BOOL: return "bool";
    case GL_FLOAT_MAT3: return "mat3";
    case GL_FLOAT_MAT4: return "mat4";
    case GL_SAMPLER_2D: return "sampler2D";
    default: return "other";
    }
}


void LamureRenderer::print_active_uniforms(GLuint programID, const std::string& shaderName) {

    if (programID == 0) {
        std::cout << "--- Uniforms for " << shaderName << ": INVALID PROGRAM ID (0) ---" << std::endl;
        return;
    }

    std::cout << "--- Active Uniforms for Shader: " << shaderName << " (Program ID: " << programID << ") ---" << std::endl;
    GLint uniformCount = 0;
    glGetProgramiv(programID, GL_ACTIVE_UNIFORMS, &uniformCount);

    if (uniformCount == 0) {
        std::cout << "  No active uniforms found." << std::endl;
        return;
    }

    GLint maxNameLen = 0;
    glGetProgramiv(programID, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxNameLen);
    std::vector<GLchar> nameBuffer(maxNameLen);

    std::cout << std::left << std::setw(10) << "Location" << std::setw(10) << "Type" << std::setw(8) << "Size" << "Name" << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;

    for (GLint i = 0; i < uniformCount; ++i) {
        GLsizei nameLen;
        GLint size;
        GLenum type;
        glGetActiveUniform(programID, i, maxNameLen, &nameLen, &size, &type, nameBuffer.data());
        std::string name(nameBuffer.data(), nameLen);
        GLint location = glGetUniformLocation(programID, name.c_str());
        std::cout << std::left << std::setw(10) << location << std::setw(10) << glTypeToString(type) << std::setw(8) << size << name << std::endl;
    }
}


void LamureRenderer::initPclResources(){
    if(m_plugin->getUI()->getNotifyButton()->state()) std::cout<<"[Notify] initPclResources()\n";

    int width=0,height=0;
    if(osg::ref_ptr<osg::Camera> cam=getOsgCamera()){
        if(osg::Viewport* vp=cam->getViewport()){ width=int(vp->width()); height=int(vp->height()); }
    }
    if(width<=0||height<=0){
        const auto* tr=opencover::coVRConfig::instance()->windows[0].context->getTraits();
        width=tr->width; height=tr->height;
    }

    // Alte Ressourcen freigeben (nur vorhandene Handles)
    if(m_pcl_resource.fbo){
        if(m_pcl_resource.texture_color)   glDeleteTextures(1,&m_pcl_resource.texture_color);
        if(m_pcl_resource.texture_normal)  glDeleteTextures(1,&m_pcl_resource.texture_normal);
        if(m_pcl_resource.texture_position)glDeleteTextures(1,&m_pcl_resource.texture_position);
        if(m_pcl_resource.depth_texture)   glDeleteTextures(1,&m_pcl_resource.depth_texture);
        glDeleteFramebuffers(1,&m_pcl_resource.fbo);
        m_pcl_resource.fbo=0; m_pcl_resource.texture_color=0; m_pcl_resource.texture_normal=0; m_pcl_resource.texture_position=0; m_pcl_resource.depth_texture=0;
    }

    // FBO anlegen
    glGenFramebuffers(1,&m_pcl_resource.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER,m_pcl_resource.fbo);

    // COLOR: RGBA16F
    glGenTextures(1,&m_pcl_resource.texture_color);
    glBindTexture(GL_TEXTURE_2D,m_pcl_resource.texture_color);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,width,height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,m_pcl_resource.texture_color,0);

    // NORMAL: RGB16F
    glGenTextures(1,&m_pcl_resource.texture_normal);
    glBindTexture(GL_TEXTURE_2D,m_pcl_resource.texture_normal);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB16F,width,height,0,GL_RGB,GL_HALF_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,m_pcl_resource.texture_normal,0);

    // POSITION: RGB16F
    glGenTextures(1,&m_pcl_resource.texture_position);
    glBindTexture(GL_TEXTURE_2D,m_pcl_resource.texture_position);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB16F,width,height,0,GL_RGB,GL_HALF_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D,m_pcl_resource.texture_position,0);

    // DEPTH: D24 (stabil und schnell)
    glGenTextures(1,&m_pcl_resource.depth_texture);
    glBindTexture(GL_TEXTURE_2D,m_pcl_resource.depth_texture);
    glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,width,height,0,GL_DEPTH_COMPONENT,GL_UNSIGNED_INT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_COMPARE_MODE,GL_NONE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,m_pcl_resource.depth_texture,0);

    // Draw-Buffers
    const GLenum bufs[3]={GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3,bufs);

    // Check
    const GLenum status=glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status!=GL_FRAMEBUFFER_COMPLETE){
        std::cerr<<"ERROR::FRAMEBUFFER incomplete ("<<std::hex<<status<<std::dec<<") "<<width<<"x"<<height<<"\n";
    }else if(m_plugin->getUI()->getNotifyButton()->state()){
        std::cout<<"[Notify] FBO ready "<<width<<"x"<<height<<"\n";
    }

    // Unbind sauber
    glBindTexture(GL_TEXTURE_2D,0);
    glBindFramebuffer(GL_FRAMEBUFFER,0);

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


bool LamureRenderer::ensurePclFboSizeUpToDate(){
    if(m_pcl_resource.fbo==0||m_pcl_resource.depth_texture==0) return false;
    const int curW=opencover::coVRConfig::instance()->windows[0].context->getTraits()->width;
    const int curH=opencover::coVRConfig::instance()->windows[0].context->getTraits()->height;
    GLint texW=0,texH=0; GLint prevTex=0; glGetIntegerv(GL_TEXTURE_BINDING_2D,&prevTex);
    glBindTexture(GL_TEXTURE_2D,m_pcl_resource.depth_texture);
    glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&texW);
    glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&texH);
    glBindTexture(GL_TEXTURE_2D,prevTex);
    if(texW!=curW||texH!=curH){ if(m_plugin->getUI()->getNotifyButton()->state()) std::cout<<"[Notify] FBO resize "<<texW<<"x"<<texH<<" -> "<<curW<<"x"<<curH<<"\n"; initPclResources(); return true; }
    return false;
}
