#include "controls/PlanarAvatarControls.h"
#include "controls/TestAvatarControls.h"
#include "texture/StripesTerroirTexture.h"
#include "texture/SplotchTerroirTexture.h"

#include "GhostAvatar.h"

using namespace covise;
using namespace opencover;
using namespace ui;

GhostAvatar::GhostAvatar()
    : coVRPlugin(COVER_PLUGIN_NAME)
    //, m_avatarControls(std::make_unique<TestAvatarControls>("/data/STARTS-ECHO/Avatars/shaderTests/ghost_cave_uniform.fbx", "LeftArm", ""))
    , m_avatarControls(std::make_unique<PlanarAvatarControls>("/data/STARTS-ECHO/Avatars/planarAvatar/PLANEE6.fbx", "Arm", "Head"))
    , m_avatarTexture(std::make_unique<SplotchTerroirTexture>(100))
    //, m_avatarTexture(std::make_unique<StripesTerroirTexture>(100))
    , m_avatarControlsUI(GhostAvatarControlsUI(COVER_PLUGIN_NAME, *m_avatarControls))
{
    m_avatarTexture->setCameraForwardDir({ 0, -1, 0 }); // planar
    m_avatarTexture->setCameraUpDir({ 0, 0, -1 }); // planar
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

        createInteractors();

        return true;
    }

    return false;
}

void GhostAvatar::preFrame()
{
    updateInteractors();

    if (!m_interactorFloor || !m_interactorHead || !m_interactorHand)
        return;

    m_avatarControls->updateBones(m_interactorFloor->getMatrix(), m_interactorHand->getMatrix(), m_interactorHead->getMatrix());
    m_avatarTexture->updateTexture();
    m_avatarControlsUI.update(m_interactorFloor->getMatrix(), m_interactorHand->getMatrix(), m_interactorHead->getMatrix());
}

void GhostAvatar::createInteractors()
{
    // TODO: try to set them to the default position first
    osg::Matrix m;
    auto interSize = 10;
    m.setTrans(0, 0, 0);
    // m.setRotate(osg::Quat(0, 0, 0.707107, 0.707107)); // ghost
    m.setRotate(osg::Quat(1, 0, 0, 0)); // planar
    m_interactorFloor.reset(new coVR3DTransRotInteractor(m, interSize, vrui::coInteraction::InteractionType::ButtonA, "floor", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactorFloor->enableIntersection();
    m_interactorFloor->show();

    // m.setTrans(-120, 0, 80); // ghost
    m.setTrans(900, 0, 1100); // planar
    m_interactorHand.reset(new coVR3DTransRotInteractor(m, interSize, vrui::coInteraction::InteractionType::ButtonA, "hand", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactorHand->enableIntersection();
    m_interactorHand->show();

    // m.setTrans(0, 0, 160); //ghost
    m.setTrans(0, 0, 1900); // planar
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