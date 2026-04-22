#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>

#include "controls/PlanarAvatarControls.h"
#include "controls/TestAvatarControls.h"
#include "texture/StripesTerroirTexture.h"
#include "texture/SplotchTerroirTexture.h"

#include "util/SanitizeRigidTransform.h"

#include "GhostAvatar.h"

using namespace opencover;

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
            updateMatrices();
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
        updateMatrices();

        m_avatarControls->updateBones(m_floorMatrix, m_handMatrix, m_headMatrix);
        m_avatarTexture->updateTexture(m_avatarControls->getEyeOffset());
        m_avatarControlsUI.update(m_floorMatrix, m_handMatrix, m_headMatrix);
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
void GhostAvatar::updateMatrices()
{
    osg::Matrix invbase = cover->getInvBaseMat();

    m_handMatrix = cover->getPointerMat();
    m_handMatrix *= invbase;

    m_headMatrix = cover->getViewerMat();

    osg::Vec3 toFeet;
    toFeet = m_headMatrix.getTrans();
    toFeet[2] = VRSceneGraph::instance()->floorHeight();
    m_floorMatrix.makeTranslate(toFeet[0], toFeet[1], toFeet[2]);
    m_floorMatrix *= invbase;

    m_headMatrix *= invbase;

    // offset for testing in the CAVE
    double offset = 5.0f;

    auto headTrans = m_headMatrix.getTrans();
    headTrans.y() += offset;
    m_headMatrix.setTrans(headTrans);

    auto handTrans = m_handMatrix.getTrans();
    handTrans.y() += offset;
    m_handMatrix.setTrans(handTrans);

    m_floorMatrix = m_floorMatrix;
    auto floorTrans = m_floorMatrix.getTrans();
    floorTrans.y() += offset;
    m_floorMatrix.setTrans(floorTrans);
    // offset for testing in the CAVE

    // match interactor behavior: keep translation, strip scale/shear from rotation basis
    // otherwise rotation does not match rotation of the glasses/3D controller
    m_handMatrix = sanitizeRigidTransform(m_handMatrix);
    m_headMatrix = sanitizeRigidTransform(m_headMatrix);
}