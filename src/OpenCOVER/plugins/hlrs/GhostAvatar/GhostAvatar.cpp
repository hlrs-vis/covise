#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>

#include "controls/PlanarAvatarControls.h"
#include "controls/TestAvatarControls.h"
#include "texture/StripesTerroirTexture.h"
#include "texture/SplotchTerroirTexture.h"

#include "GhostAvatar.h"

using namespace opencover;

#include <osg/Quat>
#include <osg/Vec3d>
#include <cmath>
#include <algorithm>

static inline double determinant3x3(const osg::Matrix &m)
{
    return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
        - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
        + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
}

static osg::Matrix extractRotation(const osg::Matrix &m)
{
    auto noTrans = m;
    noTrans.setTrans(osg::Vec3(0, 0, 0));

    {
        auto noTransposed = noTrans;
        noTransposed.transpose(noTrans);
        auto ata = noTransposed * noTrans;

        double maxDev = 0.0;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                maxDev = std::max(maxDev, std::fabs(ata(r, c) - (r == c ? 1.0 : 0.0)));

        if (maxDev < 1e-6)
            return noTrans;
    }

    auto x = noTrans;
    constexpr int maxIter = 12;
    for (int iter = 0; iter < maxIter; ++iter)
    {
        auto invX = osg::Matrix::inverse(x);
        auto tInv = invX;
        tInv.transpose(invX);

        double maxDiff = 0.0;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
            {
                const double next = 0.5 * (x(r, c) + tInv(r, c));
                maxDiff = std::max(maxDiff, std::fabs(next - x(r, c)));
                x(r, c) = next;
            }

        if (maxDiff < 1e-9)
            break;
    }

    const double d = determinant3x3(x);
    if (d < 0.0)
    {
        for (int c = 0; c < 3; ++c)
            x(0, c) = -x(0, c);
    }

    return x;
}

static osg::Matrix sanitizeRigid(const osg::Matrix &m)
{
    auto rot = extractRotation(m);
    rot.setTrans(m.getTrans());
    return rot;
}

GhostAvatar::GhostAvatar()
    : coVRPlugin(COVER_PLUGIN_NAME)
    //, m_avatarControls(std::make_unique<TestAvatarControls>("/data/STARTS-ECHO/Avatars/shaderTests/ghost_cave_minimal_fix.fbx", "RightArm", ""))
    , m_avatarControls(std::make_unique<PlanarAvatarControls>("/data/STARTS-ECHO/Avatars/planarAvatar/PLANEE6_fix.fbx", "Arm", "Head"))
    //, m_avatarTexture(std::make_unique<SplotchTerroirTexture>(100))
    , m_avatarTexture(std::make_unique<StripesTerroirTexture>(100))
    , m_avatarControlsUI(GhostAvatarControlsUI(COVER_PLUGIN_NAME, *m_avatarControls))
{
    m_avatarTexture->setCameraForwardDir(m_avatarControls->getForwardDirection());
    m_avatarTexture->setCameraUpDir(m_avatarControls->getUpDirection());
}

bool GhostAvatar::update()
{
    static bool first = true;
    if (first)
    {
        first = false;
        m_avatarControls->loadAvatar();
        m_avatarTexture->applyTexture(m_avatarControls->getAvatarNode());
        m_avatarControlsUI.initialize();

        if (m_useInteractors)
        {
            createInteractors();
        }
        else
        {
            initializeTransforms();
        }

        return true;
    }

    return false;
}

void GhostAvatar::preFrame()
{
    if (m_useInteractors)
    {
        updateInteractors();

        if (!m_interactorFloor || !m_interactorHand || !m_interactorHead)
            return;

        m_avatarControls->updateBones(m_interactorFloor->getMatrix(), m_interactorHand->getMatrix(), m_interactorHead->getMatrix());
        m_avatarTexture->updateTexture(m_avatarControls->getEyeOffset());
        m_avatarControlsUI.update(m_interactorFloor->getMatrix(), m_interactorHand->getMatrix(), m_interactorHead->getMatrix());
    }
    else
    {
        if (!m_floorTransform || !m_handTransform || !m_headTransform)
            return;

        initializeTransforms();

        m_avatarControls->updateBones(m_floorTransform->getMatrix(), m_handTransform->getMatrix(), m_headTransform->getMatrix());
        m_avatarTexture->updateTexture(m_avatarControls->getEyeOffset());
        m_avatarControlsUI.update(m_floorTransform->getMatrix(), m_handTransform->getMatrix(), m_headTransform->getMatrix());
    }
}

void GhostAvatar::createInteractors()
{
    osg::Matrix m;
    auto interSize = 10;
    m.setTrans(0, 0, 0);
    m_interactorFloor.reset(new coVR3DTransformInteractor(interSize, vrui::coInteraction::InteractionType::ButtonA, "floor", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactorFloor->updateTransform(m);
    m_interactorFloor->enableIntersection();
    m_interactorFloor->show();

    m.setTrans(120, 0, 80);
    m_interactorHand.reset(new coVR3DTransformInteractor(interSize, vrui::coInteraction::InteractionType::ButtonA, "hand", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactorHand->updateTransform(m);
    m_interactorHand->enableIntersection();
    m_interactorHand->show();

    m.setTrans(0, 0, 160);
    m_interactorHead.reset(new coVR3DTransformInteractor(interSize, vrui::coInteraction::InteractionType::ButtonA, "head", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactorHead->updateTransform(m);
    m_interactorHead->enableIntersection();
    m_interactorHead->show();
}

void GhostAvatar::updateInteractors()
{
    m_interactorFloor->preFrame();
    m_interactorHand->preFrame();
    m_interactorHead->preFrame();
}

// TODO: use CAVE transform for feet and viewerMat for moving head
void GhostAvatar::initializeTransforms()
{
    osg::Matrix invbase = cover->getInvBaseMat();
    osg::Matrix handmat = cover->getPointerMat();
    handmat *= invbase;
    osg::Matrix headmat = cover->getViewerMat();
    osg::Vec3 toFeet;
    toFeet = headmat.getTrans();
    toFeet[2] = VRSceneGraph::instance()->floorHeight();
    osg::Matrix feetmat = headmat;
    feetmat.setTrans(toFeet);

    headmat *= invbase;
    feetmat *= invbase;

    // offset for testing in the CAVE
    double offset = 5.0f;

    auto headMat = headmat;
    auto headTrans = headMat.getTrans();
    headTrans.y() += offset;
    headMat.setTrans(headTrans);

    auto handMat = handmat;
    auto handTrans = handMat.getTrans();
    handTrans.y() += offset;
    handMat.setTrans(handTrans);

    auto floorMat = feetmat;
    auto floorTrans = floorMat.getTrans();
    floorTrans.y() += offset;
    floorMat.setTrans(floorTrans);

    // Match interactor behavior: keep translation, strip scale/shear from rotation basis.
    handMat = sanitizeRigid(handMat);
    headMat = sanitizeRigid(headMat);
    floorMat = sanitizeRigid(floorMat);
    // offset for testing in the CAVE

    if (!m_handTransform)
        m_handTransform = new osg::MatrixTransform;
    if (!m_headTransform)
        m_headTransform = new osg::MatrixTransform;
    if (!m_floorTransform)
        m_floorTransform = new osg::MatrixTransform;

    m_handTransform->setMatrix(handMat);
    m_headTransform->setMatrix(headMat);
    m_floorTransform->setMatrix(floorMat);

    // TODO: delete, this is just for debugging
    if (!m_interactorFloor)
    {
        auto interSize = 0.1;
        m_interactorFloor.reset(new coVR3DTransformInteractor(interSize, vrui::coInteraction::InteractionType::ButtonA, "floor", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
        m_interactorFloor->enableIntersection();
        m_interactorFloor->show();
    }

    m_interactorFloor->updateTransform(m_floorTransform->getMatrix());
}