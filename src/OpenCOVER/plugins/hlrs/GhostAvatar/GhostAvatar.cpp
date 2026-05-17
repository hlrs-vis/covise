#include <iostream>

#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>

#include "controls/GhostAvatarControlsFactory.h"
#include "texture/TerroirTextureFactory.h"

#include "util/SanitizeRigidTransform.h"

#include "GhostAvatar.h"

using namespace opencover;

GhostAvatar::GhostAvatar()
    : coVRPlugin(COVER_PLUGIN_NAME)
{
    m_avatarControls = GhostAvatarControlsFactory::getAvatarByName(covise::coCoviseConfig::getEntry("avatarType", "COVER.Plugin.GhostAvatar", "planar"));
    m_avatarControlsUI = std::make_unique<GhostAvatarControlsUI>(COVER_PLUGIN_NAME, *m_avatarControls);

    m_avatarTexture = TerroirTextureFactory::getTextureByName(covise::coCoviseConfig::getEntry("textureType", "COVER.Plugin.GhostAvatar", "splotches"), covise::coCoviseConfig::getFloat("distanceThreshold", "COVER.Plugin.GhostAvatar", 5.f));
    m_avatarTexture->setCameraForwardDir(m_avatarControls->getForwardDirection());
    m_avatarTexture->setCameraUpDir(m_avatarControls->getUpDirection());

    m_useInteractors = covise::coCoviseConfig::getEntry("useInteractors", "COVER.Plugin.GhostAvatar", "false") == "true";
    m_mirrorsForScene = covise::coCoviseConfig::getInt("mirrorsForScene", "COVER.Plugin.GhostAvatar", 0);
}

bool GhostAvatar::init()
{
    m_floorHeight = VRSceneGraph::instance()->floorHeight();

    m_avatarControls->loadAvatar();
    m_avatarTexture->applyTexture(m_avatarControls->getAvatarNode());
    m_avatarControlsUI->initialize();

    addMirrorsToScene();

    if (m_useInteractors)
    {
        m_avatarControls->addAvatarToScene();
        createInteractors();
    }

    return true;
}

void GhostAvatar::preFrame()
{
    moveAvatar();
    m_avatarTexture->updateTexture(m_avatarControls->getEyeOffset());
    if (m_mirrorsInScene)
        updateMirrorViews();
}

void GhostAvatar::moveAvatar()
{
    if (m_useInteractors)
        moveAvatarWithInteractors();
    else
        moveAvatarWithTrackedPoses();
}

void GhostAvatar::moveAvatarWithInteractors()
{
    updateInteractors();

    m_avatarControls->updateBones(m_interactorFloor->getMatrix(), m_interactorHand->getMatrix(), m_interactorHead->getMatrix());
    m_avatarControlsUI->update(m_interactorFloor->getMatrix(), m_interactorHand->getMatrix(), m_interactorHead->getMatrix());
}

void GhostAvatar::moveAvatarWithTrackedPoses()
{
    updateTrackedPoses();

    m_avatarControls->updateBones(m_trackedFloor, m_trackedHand, m_trackedHead);
    m_avatarControlsUI->update(m_trackedFloor, m_trackedHand, m_trackedHead);
}

// TODO: use CAVE transform for feet and viewerMat for moving head
void GhostAvatar::updateTrackedPoses()
{
    m_trackedHand = cover->getPointerMat();
    m_trackedHead = cover->getViewerMat();
    m_trackedFloor.makeTranslate(m_trackedHead.getTrans().x(), m_trackedHead.getTrans().y(), m_floorHeight);

    // transform from world to object coordinates
    auto invbase = cover->getInvBaseMat();
    m_trackedHand *= invbase;
    m_trackedHead *= invbase;
    m_trackedFloor *= invbase;

    // offset for testing in the CAVE
    // offsetTrackedPoses({0, 2, 0});

    // match interactor behavior: keep translation, strip scale/shear from rotation basis
    // otherwise rotation does not match rotation of the glasses/3D controller
    m_trackedHand = sanitizeRigidTransform(m_trackedHand);
    m_trackedHead = sanitizeRigidTransform(m_trackedHead);
}

void GhostAvatar::addTranslationalOffset(osg::Matrix &matrix, const osg::Vec3 &offset)
{
    auto translation = matrix.getTrans();
    matrix.setTrans(translation.x() + offset.x(), translation.y() + offset.y(), translation.z() + offset.z());
}

void GhostAvatar::offsetTrackedPoses(const osg::Vec3 &offset)
{
    addTranslationalOffset(m_trackedFloor, offset);
    addTranslationalOffset(m_trackedHand, offset);
    addTranslationalOffset(m_trackedHead, offset);
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

    m.setTrans(0.8, 0.8, 0.8);
    m_interactorHand.reset(new coVR3DTransformInteractor(interSize, vrui::coInteraction::InteractionType::ButtonA, "hand", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactorHand->updateTransform(m);
    m_interactorHand->enableIntersection();
    m_interactorHand->show();

    m.setTrans(0, 0, 1);
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

void GhostAvatar::addMirrorsToScene()
{
    if (m_mirrorsForScene == 1)
    {
        m_mirrors.reserve(3);
        m_mirrors.emplace_back(osg::Vec3(-1.14, -2.97, 0), 1, 2, osg::Quat(0, 0, 0.4226183, 0.9063078));
        m_mirrors.emplace_back(osg::Vec3(3.93, 6.57, 0), 1, 3, osg::Quat(0, 0, -0.3007058, 0.953717));
        m_mirrors.emplace_back(osg::Vec3(-6.92, 7.02, 0), 1, 1.9, osg::Quat(0, 0, 0.7071068, 0.7071068));
    }
    else if (m_mirrorsForScene == 2)
    {
        m_mirrors.reserve(4);
        m_mirrors.emplace_back(osg::Vec3(-6.4326, -5.45797, 0.0431504), 1.925, 1.925, osg::Quat(0, 0, -0.4617486, 0.8870108));
        m_mirrors.emplace_back(osg::Vec3(-4.33584, 3.18557, 0.03881), 1.925, 1.925, osg::Quat(0, 0, 0.3173047, 0.9483237));
        m_mirrors.emplace_back(osg::Vec3(7.32158, 0.284769, 0.0310617), 1.925, 1.925, osg::Quat(0, 0, 0.9483237, -0.3173047));
        m_mirrors.emplace_back(osg::Vec3(11.9669, 12.6238, 0.0433261), 1.925, 1.925, osg::Quat(0, 0, 0.4617486, -0.8870108));
    }
    else
    {
        return;
    }

    m_mirrorsInScene = true;

    if (!m_useInteractors)
        for (auto &mirror : m_mirrors)
            mirror.setReflectedNode(m_avatarControls->getAvatarNode());
}

void GhostAvatar::updateMirrorViews()
{
    for (auto &mirror : m_mirrors)
        mirror.updateView();
}