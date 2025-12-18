#include "LamureEditTool.h"
#include "Lamure.h"
#include "LamureUtil.h"
#include "LamureRenderer.h"

#include <cover/coVRPluginSupport.h>
#include <cover/input/input.h>
#include <cover/input/coMousePointer.h>
#include <cover/coVRConfig.h>
#include <cover/coVRNavigationManager.h>
#include <cover/OpenCOVER.h>
#include <PluginUtil/coVR3DTransInteractor.h>
#include <OpenVRUI/sginterface/vruiButtons.h>
#include <osg/GraphicsContext>

#include <algorithm>
#include <cmath>

#include <lamure/ren/dataset.h>
#include <lamure/ren/ooc_cache.h>
#include <lamure/ren/controller.h>
#include <lamure/ren/model_database.h>

namespace {

    scm::math::mat4f resolveModelMatrix(const Lamure* plugin, uint16_t modelIndex) {
        const auto& nodes = plugin->getSceneNodes();
        const osg::Matrixd model_osg = nodes[modelIndex].model_transform->getMatrix();
        return scm::math::mat4f(LamureUtil::matConv4D(model_osg));
    }

    constexpr double kBrushHullScale = 10.0;

    osg::Matrixd makeBrushLocal(const osg::Matrixd& invRoot, const osg::Vec3& viewOffset)
    {
        osg::Matrixd viewer = opencover::cover->getViewerMat();
        const float sceneScale = opencover::cover->getScale();
        osg::Vec3 posWorld = viewer.getTrans() + viewer.getRotate() * (viewOffset * sceneScale);

        osg::Matrixd m;
        m.makeIdentity();
        m.setTrans(posWorld);
        return m * invRoot;
    }

    osg::Vec3 makeDepthOffsetLocal(const osg::Matrixd& invRoot, const osg::Vec3& viewOffset)
    {
        osg::Quat viewRot = opencover::cover->getViewerMat().getRotate();
        double scale = opencover::cover->getScale();
        if (scale < 1e-6)
            scale = 1.0;
        osg::Vec3 offsetWorld = viewRot * (viewOffset * scale);
        return osg::Matrix::transform3x3(offsetWorld, invRoot);
    }

    osg::Vec3 viewOffsetFromWorld(const osg::Vec3& worldPos)
    {
        osg::Matrixd viewer = opencover::cover->getViewerMat();
        osg::Quat viewRot = viewer.getRotate();
        double scale = opencover::cover->getScale();
        if (scale < 1e-6)
            scale = 1.0;
        return (viewRot.inverse() * (worldPos - viewer.getTrans())) / scale;
    }

    class PlanarInteractor : public opencover::coVR3DTransInteractor {
    public:
        PlanarInteractor(const osg::Vec3& pos, float size, vrui::coInteraction::InteractionType type)
            : opencover::coVR3DTransInteractor(pos, size, type, "hand", "LamureBrush", vrui::coInteraction::Medium) 
        {
        }

        osg::Matrix fullMatrix() const {
            return moveTransform->getMatrix() * scaleTransform->getMatrix();
        }

    protected:
        void keepSize() override {
            // Keep constant world size instead of screen-size scaling.
            const float s = getInteractorSize();
            _scale = s;
            scaleTransform->setMatrix(osg::Matrix::scale(s, s, s));
        }

        void startInteraction() override {
            _startCenter = getMatrix().getTrans();
            const osg::Matrix invBase = opencover::cover->getInvBaseMat();
            osg::Quat viewRot = opencover::cover->getViewerMat().getRotate();
            osg::Vec3 fwdWorld = viewRot * osg::Vec3(0, 1, 0);
            _planeNormal = osg::Matrix::transform3x3(fwdWorld, invBase);
            if (!_planeNormal.length2())
                _planeNormal = osg::Vec3(0, 1, 0);
            _planeNormal.normalize();
            opencover::coVR3DTransInteractor::startInteraction();
            if (auto* nav = opencover::coVRNavigationManager::instance()) {
                nav->stopMouseNav();
            }
        }

        void doInteraction() override {
            if (getState() != vrui::coInteraction::Active) {
                opencover::coVR3DTransInteractor::doInteraction();
                return;
            }

            // Ray-plane intersection: intersect current pointer ray with plane through _startCenter, normal _planeNormal.
            osg::Matrix currentHandMat = getPointerMat();
            osg::Vec3 origin(0, 0, 0), yaxis(0, 1, 0);
            osg::Vec3 lp1 = origin * currentHandMat;
            osg::Vec3 lp2 = yaxis * currentHandMat;
            osg::Vec3 lp1_o = lp1 * opencover::cover->getInvBaseMat();
            osg::Vec3 lp2_o = lp2 * opencover::cover->getInvBaseMat();
            osg::Vec3 dir = lp2_o - lp1_o;
            if (!dir.length2())
                return;
            dir.normalize();
            const float denom = dir * _planeNormal;
            if (std::abs(denom) < 1e-5f)
                return;
            const float t = (_planeNormal * (_startCenter - lp1_o)) / denom;
            osg::Vec3 hit = lp1_o + dir * t;
            updateTransform(hit);
        }

    private:
        osg::Vec3 _startCenter{0.f, 0.f, 0.f};
        osg::Vec3 _planeNormal{0.f, 1.f, 0.f};
    };

    class DepthInteractor : public opencover::coVR3DTransInteractor {
    public:
        DepthInteractor(const osg::Vec3& pos, float size, vrui::coInteraction::InteractionType type)
            : opencover::coVR3DTransInteractor(pos, size, type, "hand", "LamureBrushDepth", vrui::coInteraction::Medium)
        {
        }

        osg::Matrix fullMatrix() const {
            return moveTransform->getMatrix() * scaleTransform->getMatrix();
        }

        void setOffset(const osg::Vec3& offset) { _offset = offset; }
        osg::Vec3 offset() const { return _offset; }

        void keepSize() override {
            const float s = getInteractorSize();
            _scale = s;
            scaleTransform->setMatrix(osg::Matrix::scale(s, s, s));
        }

    void preFrame() override {
        // Keep base preFrame for state/hover; movement is handled in doInteraction.
        opencover::coVR3DTransInteractor::preFrame();
    }

    void doInteraction() override
    {
        if (getState() != vrui::coInteraction::Active) {
            opencover::coVR3DTransInteractor::doInteraction();
            return;
        }

        auto* inp = opencover::Input::instance();
        if (!inp || !inp->mouse()) return;

        const float currY = static_cast<float>(inp->mouse()->y());
        float winH = std::max(1.0f, inp->mouse()->winHeight());
        const float dy = (currY - _startMouseY) / winH;

        const osg::Matrix invBase = opencover::cover->getInvBaseMat();
        osg::Vec3 fwdWorld = opencover::cover->getViewerMat().getRotate() * osg::Vec3(0, 1, 0);
        osg::Vec3 fwdLocal = osg::Matrix::transform3x3(fwdWorld, invBase); // rotate into object space, drop translation
        if (!fwdLocal.length2()) return;
        fwdLocal.normalize();

        double scale = opencover::cover->getScale();
        if (scale < 1e-6) scale = 1.0;

        const double speed = 0.1; // reduced sensitivity

        osg::Matrix m = _startMat;
        m.postMult(osg::Matrix::translate(fwdLocal * (dy * scale * speed)));
        osg::Vec3 basePos = m.getTrans();
        updateTransform(basePos + _offset);
    }

        void startInteraction() override {
            _startMouseY = opencover::Input::instance() && opencover::Input::instance()->mouse()
                ? static_cast<float>(opencover::Input::instance()->mouse()->y())
                : 0.0f;
            _startMat = getMatrix();
            _startMat.setTrans(_startMat.getTrans() - _offset);
            opencover::coVR3DTransInteractor::startInteraction();
            if (auto* nav = opencover::coVRNavigationManager::instance()) {
                nav->stopMouseNav();
            }
        }

    private:
        float _startMouseY = 0.0f;
        osg::Matrix _startMat;
        osg::Vec3 _offset{0.f, 0.f, 0.f};
    };

    class SizeInteractor : public opencover::coVR3DTransInteractor {
    public:
        SizeInteractor(const osg::Vec3& pos, float size, vrui::coInteraction::InteractionType type, LamureEditTool* owner)
            : opencover::coVR3DTransInteractor(pos, size, type, "hand", "LamureBrushSize", vrui::coInteraction::Medium)
            , _owner(owner)
        {
        }

        void setOffset(const osg::Vec3& offset) { _offset = offset; }
        osg::Vec3 offset() const { return _offset; }

        void keepSize() override {
            const float s = getInteractorSize();
            _scale = s;
            scaleTransform->setMatrix(osg::Matrix::scale(s, s, s));
        }

        void startInteraction() override {
            _startMouseX = opencover::Input::instance() && opencover::Input::instance()->mouse()
                ? static_cast<float>(opencover::Input::instance()->mouse()->x())
                : 0.0f;
            _startSize = (_owner) ? _owner->brushSize() : getInteractorSize();
            opencover::coVR3DTransInteractor::startInteraction();
            if (auto* nav = opencover::coVRNavigationManager::instance()) {
                nav->stopMouseNav();
            }
        }

        void doInteraction() override {
            if (getState() != vrui::coInteraction::Active) {
                opencover::coVR3DTransInteractor::doInteraction();
                return;
            }
            // Let base place the interactor under the mouse (pointer-relative).
            opencover::coVR3DTransInteractor::doInteraction();

            auto* inp = opencover::Input::instance();
            if (!inp || !inp->mouse() || !_owner)
                return;

            // Derive size from current distance center->handle (local space), keep center fixed.
            const osg::Vec3 pos = getMatrix().getTrans();
            const osg::Vec3 center = _owner->brushCenter();
            const float dist = (pos - center).length();
            const float hullScale = static_cast<float>(kBrushHullScale);
            const float newSize = std::max(0.001f, dist / hullScale);
            _owner->applySizeChange(newSize);
        }

    private:
        LamureEditTool* _owner{nullptr};
        float _startMouseX = 0.0f; // kept for potential future use
        float _startSize = 1.0f;
        osg::Vec3 _offset{0.f, 0.f, 0.f};
    };
}

LamureEditTool::LamureEditTool(Lamure* plugin)
: m_plugin(plugin)
, m_dragOffsetView(0.f, 3.f, 0.f)
{
    m_brushLocalMat.makeIdentity();
}

LamureEditTool::~LamureEditTool() {
    disable();
    if (m_planar_interactor) {
        delete m_planar_interactor;
        m_planar_interactor = nullptr;
    }
    if (m_depth_interactor) {
        delete m_depth_interactor;
        m_depth_interactor = nullptr;
    }
    if (m_size_interactor) {
        delete m_size_interactor;
        m_size_interactor = nullptr;
    }
}

void LamureEditTool::setBrushPose(const osg::Vec3& center, const osg::Vec3& scale) {
    m_brush.center = center;
    m_brush.scale = scale;
    m_brush.active = true;
}

void LamureEditTool::setBrushAction(BrushAction action) {
    m_action = action;
}

void LamureEditTool::clearBrush() {
    m_brush.active = false;
}

void LamureEditTool::applySizeChange(float newSize) {
    const float clamped = std::max(0.001f, newSize);
    m_brushSize = clamped;
    const float hullRadius = clamped * static_cast<float>(kBrushHullScale);
    m_depthOffsetView = osg::Vec3(0.0f, 0.0f, hullRadius); // place depth handle at top of the brush hull
    m_sizeOffsetView = osg::Vec3(hullRadius, 0.0f, 0.0f);
    // Keep visual/hit size constant; base size captured during creation.
    if (m_planar_interactor)
        m_planar_interactor->setInteractorSize(m_interactorBaseSize);
    if (m_depth_interactor)
        m_depth_interactor->setInteractorSize(m_interactorBaseSize);
    if (m_size_interactor)
        m_size_interactor->setInteractorSize(m_interactorBaseSize);
}

void LamureEditTool::updateSizeInteractorPose(const osg::Vec3& center) {
    if (!m_size_interactor)
        return;

    osg::Matrixd rootWorld;
    rootWorld.makeIdentity();
    if (auto g = m_plugin->getGroup()) {
        if (!g->getWorldMatrices().empty()) rootWorld = g->getWorldMatrices().front();
    }
    osg::Matrixd invRoot = osg::Matrixd::inverse(rootWorld);
    osg::Vec3 sizeOffset = makeDepthOffsetLocal(invRoot, m_sizeOffsetView);
    if (auto* resize = dynamic_cast<SizeInteractor*>(m_size_interactor)) {
        resize->setOffset(sizeOffset);
    }
    m_size_interactor->updateTransform(center + sizeOffset);
}

void LamureEditTool::applyBrushMask() {
    if (!m_plugin)
        return;
    auto* ren = m_plugin->getRenderer();
    if (!ren || !m_plugin->getGroup())
        return;
    if (auto* node = ren->ensureEditBrushNode(m_plugin->getGroup().get())) {
        const unsigned int visibleMask = ~opencover::Isect::Intersection & ~opencover::Isect::Pick;
        node->setNodeMask(m_brushMasked ? 0x0 : visibleMask);
    }
}

void LamureEditTool::enable() {
    if (m_enabled || !m_plugin) return;

    if (!m_planar_interactor) {
        float size = m_plugin->getSettings().scale_radius;
        auto* planar = new PlanarInteractor(osg::Vec3(0,0,0), size, vrui::coInteraction::ButtonA);
        planar->setShared(true);

        auto* depth = new DepthInteractor(osg::Vec3(0,0,0), size, vrui::coInteraction::ButtonA);
        depth->setShared(false); // separate visual so offset is visible

        auto* resize = new SizeInteractor(osg::Vec3(0,0,0), size, vrui::coInteraction::ButtonA, this);
        resize->setShared(false);

        m_planar_interactor = planar;
        m_depth_interactor = depth;
        m_size_interactor = resize;
        m_interactorBaseSize = size;
        applySizeChange(size);
    }

    osg::Matrixd rootWorld;
    rootWorld.makeIdentity();
    if (auto g = m_plugin->getGroup()) {
        if (!g->getWorldMatrices().empty()) rootWorld = g->getWorldMatrices().front();
    }
    osg::Matrixd invRoot = osg::Matrixd::inverse(rootWorld);

    osg::Vec3 targetCenterLocal;
    if (m_savedPoseValid) {
        m_brushLocalMat = makeBrushLocal(invRoot, m_dragOffsetView);
        targetCenterLocal = m_brushLocalMat.getTrans();
        m_savedPoseValid = false;
    } else {
        m_brushLocalMat = makeBrushLocal(invRoot, m_dragOffsetView);
        targetCenterLocal = m_brushLocalMat.getTrans();
    }
    m_brushLocalMat.makeIdentity();
    m_brushLocalMat.setTrans(targetCenterLocal);
    m_planar_interactor->updateTransform(targetCenterLocal);

    // Apply lateral offset only to the depth interactor visual.
    if (auto* depth = dynamic_cast<DepthInteractor*>(m_depth_interactor)) {
        const osg::Vec3 offsetLocal = makeDepthOffsetLocal(invRoot, m_depthOffsetView);
        depth->setOffset(offsetLocal);
        m_depth_interactor->updateTransform(targetCenterLocal + offsetLocal);
    } else {
        m_depth_interactor->updateTransform(targetCenterLocal);
    }
    if (auto* resize = dynamic_cast<SizeInteractor*>(m_size_interactor)) {
        const osg::Vec3 offsetLocal = makeDepthOffsetLocal(invRoot, m_sizeOffsetView);
        resize->setOffset(offsetLocal);
        m_size_interactor->updateTransform(targetCenterLocal + offsetLocal);
    } else if (m_size_interactor) {
        m_size_interactor->updateTransform(targetCenterLocal);
    }
    m_planar_interactor->enableIntersection();
    m_depth_interactor->enableIntersection();
    if (m_size_interactor)
        m_size_interactor->enableIntersection();
    m_planar_interactor->show();
    m_depth_interactor->show();
    if (m_size_interactor)
        m_size_interactor->show();
    m_brushMasked = false;
    applyBrushMask();
    m_enabled = true;

    if (auto* ren = m_plugin->getRenderer()) {
        if (m_plugin->getGroup()) ren->ensureEditBrushNode(m_plugin->getGroup().get());
        updateBrushInteraction(m_planar_interactor);
    }
}

void LamureEditTool::disable(bool resetPose, bool maskBrush) {
    if (m_enabled && !resetPose && m_planar_interactor) {
        osg::Matrixd rootWorld;
        rootWorld.makeIdentity();
        if (auto g = m_plugin->getGroup()) {
            if (!g->getWorldMatrices().empty()) rootWorld = g->getWorldMatrices().front();
        }
        m_savedBrushWorldPos = (m_planar_interactor->getMatrix() * rootWorld).getTrans();
        m_savedPoseValid = true;
    } else if (resetPose) {
        m_savedPoseValid = false;
    }

    if (!m_enabled) {
        m_brushMasked = m_brushMasked || maskBrush;
        applyBrushMask();
        return;
    }

    if (m_planar_interactor) {
        m_planar_interactor->disableIntersection();
        m_planar_interactor->stopInteraction();
        m_planar_interactor->hide();
    }
    if (m_depth_interactor) {
        m_depth_interactor->disableIntersection();
        m_depth_interactor->stopInteraction();
        m_depth_interactor->hide();
    }
    if (m_size_interactor) {
        m_size_interactor->disableIntersection();
        m_size_interactor->stopInteraction();
        m_size_interactor->hide();
    }

    if (maskBrush) {
        m_brushMasked = true;
        applyBrushMask();
    }

    // Reset position offset to initial view-based default for the next enable unless we preserve it.
    if (resetPose) {
        m_dragOffsetView = m_initialDragOffsetView;
        m_brushLocalMat.makeIdentity();
    }

    m_brushMasked = true;

    m_enabled = false;
}

void LamureEditTool::update() {
    if (m_plugin && m_plugin->isBrushFrozen())
        return;

    if (m_enabled && m_planar_interactor && m_depth_interactor) {
        const bool isDepthActive = m_depth_interactor->getState() == vrui::coInteraction::Active
            || m_depth_interactor->getState() == vrui::coInteraction::PendingActive;
        const bool isPlanarActive = m_planar_interactor->getState() == vrui::coInteraction::Active
            || m_planar_interactor->getState() == vrui::coInteraction::PendingActive;
        const bool isSizeActive = m_size_interactor
            && (m_size_interactor->getState() == vrui::coInteraction::Active
                || m_size_interactor->getState() == vrui::coInteraction::PendingActive);

        // If nothing is grabbed, keep the interactor pose in front of the camera.
        if (!isDepthActive && !isPlanarActive && !isSizeActive) {
            osg::Matrixd rootWorld;
            rootWorld.makeIdentity();
            if (auto g = m_plugin->getGroup()) {
                if (!g->getWorldMatrices().empty()) rootWorld = g->getWorldMatrices().front();
            }
            osg::Matrixd invRoot = osg::Matrixd::inverse(rootWorld);

            m_brushLocalMat = makeBrushLocal(invRoot, m_dragOffsetView);

            osg::Vec3 depthOffset(0.f, 0.f, 0.f);
            if (auto* depth = dynamic_cast<DepthInteractor*>(m_depth_interactor)) {
                depthOffset = makeDepthOffsetLocal(invRoot, m_depthOffsetView);
                depth->setOffset(depthOffset);
            }
            osg::Vec3 sizeOffset(0.f, 0.f, 0.f);
            if (auto* resize = dynamic_cast<SizeInteractor*>(m_size_interactor)) {
                sizeOffset = makeDepthOffsetLocal(invRoot, m_sizeOffsetView);
                resize->setOffset(sizeOffset);
            }

            m_planar_interactor->updateTransform(m_brushLocalMat.getTrans());
            m_depth_interactor->updateTransform(m_brushLocalMat.getTrans() + depthOffset);
            if (m_size_interactor)
                m_size_interactor->updateTransform(m_brushLocalMat.getTrans() + sizeOffset);
        }

        // Prefer the depth interactor if active, otherwise planar.
        opencover::coVR3DTransInteractor* active = nullptr;
        if (isDepthActive) {
            active = m_depth_interactor;
        } else if (isSizeActive && m_size_interactor) {
            active = m_size_interactor;
        } else {
            active = m_planar_interactor;
        }
        updateBrushInteraction(active);
    }
}

void LamureEditTool::updateBrushInteraction(opencover::coVR3DTransInteractor* interactor) {
    if (!m_plugin || !interactor) return;

    interactor->preFrame();

    osg::Matrixd mat = interactor->getMatrix(); // local/object space
    osg::Vec3 centerPos = mat.getTrans();

    osg::Matrixd rootWorld;
    rootWorld.makeIdentity();
    if (auto g = m_plugin->getGroup()) {
        if (!g->getWorldMatrices().empty()) rootWorld = g->getWorldMatrices().front();
    }
    osg::Matrixd invRoot = osg::Matrixd::inverse(rootWorld);

    osg::Vec3 depthOffset(0.f, 0.f, 0.f);
    if (auto* depth = dynamic_cast<DepthInteractor*>(m_depth_interactor)) {
        depthOffset = makeDepthOffsetLocal(invRoot, m_depthOffsetView);
        depth->setOffset(depthOffset);
    }
    osg::Vec3 sizeOffset(0.f, 0.f, 0.f);
    if (auto* resize = dynamic_cast<SizeInteractor*>(m_size_interactor)) {
        sizeOffset = makeDepthOffsetLocal(invRoot, m_sizeOffsetView);
        resize->setOffset(sizeOffset);
    }

    if (interactor == m_size_interactor) {
        // Keep the center fixed while scaling; ignore the interactor's own translation.
        centerPos = m_brush.center;
        mat.setTrans(centerPos);
    }

    // If depth interactor is active, remove its visual offset for the brush transform.
    if (interactor == m_depth_interactor) {
        centerPos -= depthOffset;
        mat.setTrans(centerPos);
    }

    // Update drag offset in view space only while interacting, so released brushes keep their last user-applied offset when the camera moves.
    if (interactor->getState() == vrui::coInteraction::Active
        || interactor->getState() == vrui::coInteraction::PendingActive) {
        osg::Matrixd baseLocal = makeBrushLocal(invRoot, osg::Vec3(0.f, 0.f, 0.f));
        osg::Vec3 baseWorld = (baseLocal * rootWorld).getTrans();
        osg::Vec3 curWorld  = (mat * rootWorld).getTrans();
        osg::Quat viewRot = opencover::cover->getViewerMat().getRotate();
        
        double scale = opencover::cover->getScale();
        if (scale < 1e-6) scale = 1.0;
        m_dragOffsetView = (viewRot.inverse() * (curWorld - baseWorld)) / scale; // World -> View (unscaled)
    }

    if (auto* ren = m_plugin->getRenderer()) {
        if (auto* node = ren->ensureEditBrushNode(m_plugin->getGroup().get())) {
            const unsigned int mask = ~opencover::Isect::Intersection & ~opencover::Isect::Pick;
            node->setNodeMask(m_brushMasked ? 0x0 : mask);
        }
        double hullScale = m_brushSize * kBrushHullScale;
        ren->updateEditBrushFromInteractor(mat, invRoot, hullScale);
    }

    osg::Vec3 pos = mat.getTrans();
    float sz = m_brushSize;

    // Keep visuals aligned: planar at center, depth at center + depthOffset (if defined).
    if (m_planar_interactor && m_depth_interactor) {
        if (auto* depth = dynamic_cast<DepthInteractor*>(m_depth_interactor)) {
            depth->setOffset(depthOffset);
        }
        const bool sizeActive = m_size_interactor && (m_size_interactor->getState() == vrui::coInteraction::Active
            || m_size_interactor->getState() == vrui::coInteraction::PendingActive);
        m_planar_interactor->updateTransform(pos);
        m_depth_interactor->updateTransform(pos + depthOffset);
        if (!sizeActive) {
            if (auto* resize = dynamic_cast<SizeInteractor*>(m_size_interactor)) {
                resize->setOffset(sizeOffset);
                m_size_interactor->updateTransform(pos + sizeOffset);
            } else if (m_size_interactor) {
                m_size_interactor->updateTransform(pos);
            }
        }
    }

    setBrushPose(pos, osg::Vec3(sz, sz, sz));

    if (m_action != BrushAction::None) {
        applyBrush(m_action, true);
    }
}

std::vector<LamureEditTool::NodeHit> LamureEditTool::collectNodesInBrush(bool requireVisible) const {
    std::vector<NodeHit> hits;
    if (!m_plugin || !m_brush.active) return hits;

    auto* controller = lamure::ren::controller::get_instance();
    auto* database = lamure::ren::model_database::get_instance();
    if (!controller || !database) return hits;

    const auto& settings = m_plugin->getSettings();
    const size_t modelCount = settings.models.size();

    hits.reserve(32);

    for (uint16_t m_idx = 0; m_idx < modelCount; ++m_idx) {
        if (requireVisible && !m_plugin->isModelVisible(m_idx)) continue;

        const lamure::model_t model_id = controller->deduce_model_id(std::to_string(m_idx));
        lamure::ren::dataset* dataset = nullptr;
        try { dataset = database->get_model(model_id); } catch (...) { continue; }
        if (!dataset) continue;

        const auto* bvh = dataset->get_bvh();
        if (!bvh) continue;

        const auto& boxes = bvh->get_bounding_boxes();
        if (boxes.empty()) continue;

        const auto modelMatrix = resolveModelMatrix(m_plugin, m_idx);

        for (lamure::node_t node_id = 0; node_id < static_cast<lamure::node_t>(boxes.size()); ++node_id) {
            hits.push_back(NodeHit{model_id, node_id, m_idx});
        }
    }
    return hits;
}

std::vector<LamureEditTool::SurfelKey> LamureEditTool::collectSurfelsInBrush(BrushAction action, bool requireVisible) const {
    std::vector<SurfelKey> keys;
    if (!m_plugin || !m_brush.active || action == BrushAction::None)
        return keys;

    auto* controller = lamure::ren::controller::get_instance();
    auto* database = lamure::ren::model_database::get_instance();
    auto* cache = lamure::ren::ooc_cache::get_instance();
    if (!controller || !database || !cache)
        return keys;

    const auto nodes = collectNodesInBrush(requireVisible);
    if (nodes.empty())
        return keys;

    const uint32_t surfelsPerNode = database->get_primitives_per_node();
    keys.reserve(nodes.size() * 16);

    cache->lock_pool();
    cache->refresh();

    for (const auto& hit : nodes) {
        if (!cache->is_node_resident_and_aquired(hit.model_id, hit.node_id)) {
            cache->register_node(hit.model_id, hit.node_id, 0);
            cache->refresh();
            cache->wait_for_idle();
            if (!cache->is_node_resident_and_aquired(hit.model_id, hit.node_id))
                continue;
        }

        char* raw = cache->node_data(hit.model_id, hit.node_id);
        if (!raw)
            continue;

        auto* surfels = reinterpret_cast<lamure::ren::dataset::serialized_surfel*>(raw);
        const auto modelMatrix = resolveModelMatrix(m_plugin, hit.model_index);

        for (uint32_t i = 0; i < surfelsPerNode; ++i) {
            const auto& s = surfels[i];
            const scm::math::vec4f local(s.x, s.y, s.z, 1.0f);
            const scm::math::vec4f world = modelMatrix * local;
            const osg::Vec3 worldPos(world.x, world.y, world.z);
            const float surfelRadius = s.size;
            if (!containsSurfel(worldPos, surfelRadius))
                continue;
            keys.push_back(SurfelKey{hit.model_id, hit.node_id, i});
        }
    }

    cache->unlock_pool();
    return keys;
}

std::size_t LamureEditTool::applyBrush(BrushAction action, bool requireVisible) {
    const auto keys = collectSurfelsInBrush(action, requireVisible);
    if (keys.empty())
        return 0;

    if (action == BrushAction::Erase) {
        for (const auto& k : keys)
            markErased(k);
    } else if (action == BrushAction::Restore) {
        for (const auto& k : keys)
            markRestored(k);
    }
    return keys.size();
}

void LamureEditTool::markErased(const SurfelKey& key) {
    if (key.model == lamure::invalid_model_t || key.node == lamure::invalid_node_t)
        return;
    if (m_erased.insert(key).second) {
        NodeKey nk{key.model, key.node};
        auto &vec = m_erased_by_node[nk];
        vec.push_back(key.surfel);
    }
}

void LamureEditTool::markRestored(const SurfelKey& key) {
    if (key.model == lamure::invalid_model_t || key.node == lamure::invalid_node_t)
        return;
    const auto itSet = m_erased.find(key);
    if (itSet == m_erased.end())
        return;
    m_erased.erase(itSet);

    NodeKey nk{key.model, key.node};
    auto it = m_erased_by_node.find(nk);
    if (it != m_erased_by_node.end()) {
        auto &vec = it->second;
        vec.erase(std::remove(vec.begin(), vec.end(), key.surfel), vec.end());
        if (vec.empty()) {
            m_erased_by_node.erase(it);
        }
    }
}

bool LamureEditTool::isErased(const SurfelKey& key) const {
    return m_erased.find(key) != m_erased.end();
}

void LamureEditTool::clearEdits() {
    m_erased.clear();
    m_erased_by_node.clear();
}

bool LamureEditTool::hasNodeEdits(lamure::model_t model, lamure::node_t node) const {
    return m_erased_by_node.find(NodeKey{model, node}) != m_erased_by_node.end();
}

const std::vector<uint32_t>* LamureEditTool::erasedInNode(lamure::model_t model, lamure::node_t node) const {
    auto it = m_erased_by_node.find(NodeKey{model, node});
    if (it == m_erased_by_node.end())
        return nullptr;
    return &it->second;
}

bool LamureEditTool::containsSurfel(const osg::Vec3& worldPos, float surfelRadius) const {
    if (!m_brush.active)
        return false;

    const float r = std::max({m_brush.scale.x(), m_brush.scale.y(), m_brush.scale.z()});
    const float rr = (r + surfelRadius) * (r + surfelRadius);
    return (worldPos - m_brush.center).length2() <= rr;
}
