#include "GhostAvatar.h"
#include "PlanarAvatarControls.h"
#include "TestAvatarControls.h"

using namespace covise;
using namespace opencover;
using namespace ui;

GhostAvatar::GhostAvatar()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , avatarControls(std::make_unique<TestAvatarControls>("/data/STARTS-ECHO/Avatars/shaderTests/ghost_cave_uniform.fbx", "LeftArm", ""))
    //, avatarControls(std::make_unique<PlanarAvatarControls>("/data/STARTS-ECHO/Avatars/planarAvatar/PLANEE6.fbx", "Arm", "Head"))
    , avatarControlsUI(GhostAvatarControlsUI(COVER_PLUGIN_NAME, *avatarControls))
{
}

bool GhostAvatar::update()
{
    static bool first = true;
    if (first)
    {
        first = false;
        avatarControls->loadAvatar();

        createInteractors();

        avatarControlsUI.initialize();

        return true;
    }

    return false;
}

void GhostAvatar::preFrame()
{
    updateInteractors();

    if (!m_interactorFloor || !m_interactorHead || !m_interactorHand)
        return;

    avatarControls->updateBones(m_interactorFloor->getMatrix(), m_interactorHand->getMatrix(), m_interactorHead->getMatrix());
    avatarControlsUI.update(m_interactorFloor->getMatrix(), m_interactorHand->getMatrix(), m_interactorHead->getMatrix());
}

void GhostAvatar::createInteractors()
{
    // TODO: try to set them to the default position first
    osg::Matrix m;
    auto interSize = 10;
    m.setTrans(0, 0.0, 0.0);
    m.setRotate(osg::Quat(0, 0, 0.707107, 0.707107));
    m_interactorFloor.reset(new coVR3DTransRotInteractor(m, interSize, vrui::coInteraction::InteractionType::ButtonA, "floor", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactorFloor->enableIntersection();
    m_interactorFloor->show();

    m.setTrans(0.0, 900.0, -1100.0);
    m_interactorHand.reset(new coVR3DTransRotInteractor(m, interSize, vrui::coInteraction::InteractionType::ButtonA, "hand", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactorHand->enableIntersection();
    m_interactorHand->show();

    m.setTrans(0.0, 0.0, -1900.0);
    m_interactorHead.reset(new coVR3DTransRotInteractor(m, interSize, vrui::coInteraction::InteractionType::ButtonA, "head", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactorHead->enableIntersection();
    m_interactorHead->show();
}

void GhostAvatar::updateInteractors()
{
    m_interactorFloor->preFrame();
    m_interactorHand->preFrame();
    m_interactorHead->preFrame();
}