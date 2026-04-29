#ifndef COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H
#define COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H

#include <memory>
#include <vector>

#include <osg/Matrix>

#include <cover/coVRPlugin.h>
#include <PluginUtil/coVR3DTransformInteractor.h>

#include "controls/GhostAvatarControls.h"
#include "scene/Mirror.h"
#include "texture/TerroirTexture.h"
#include "ui/GhostAvatarControlsUI.h"

//TODO: get info from config file instead of changing constructors
class GhostAvatar : public opencover::coVRPlugin
{
public:
    GhostAvatar();

    bool init() override;
    void preFrame() override;

private:
    std::unique_ptr<GhostAvatarControls> m_avatarControls;
    std::unique_ptr<TerroirTexture> m_avatarTexture;
    GhostAvatarControlsUI m_avatarControlsUI;

    const bool m_useInteractors = true;
    void moveAvatar();
    void moveAvatarWithInteractors();
    void moveAvatarWithTrackedPoses();

    float m_floorHeight = 0.f;
    osg::Matrix m_trackedFloor, m_trackedHand, m_trackedHead;
    void updateTrackedPoses();
    void addTranslationalOffset(osg::Matrix &matrix, const osg::Vec3 &offset);
    void offsetTrackedPoses(const osg::Vec3 &offset);

    std::unique_ptr<opencover::coVR3DTransformInteractor> m_interactorFloor, m_interactorHand, m_interactorHead;
    void createInteractors();
    void updateInteractors();

    std::vector<Mirror> m_mirrors;
    void addMirrorsToScene();
    void updateMirrorViews();
};

COVERPLUGIN(GhostAvatar)

#endif // COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H