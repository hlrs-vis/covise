#ifndef COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H
#define COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H

#include <memory>

#include <osg/Matrix>

#include <cover/coVRPlugin.h>
#include <PluginUtil/coVR3DTransformInteractor.h>

#include "controls/GhostAvatarControls.h"
#include "scene/Mirror.h"
#include "texture/TerroirTexture.h"
#include "ui/GhostAvatarControlsUI.h"

class GhostAvatar : public opencover::coVRPlugin
{
public:
    GhostAvatar();

    bool update() override;
    void preFrame() override;

private:
    const bool m_useInteractors = false;
    bool m_initialized = false;

    std::unique_ptr<GhostAvatarControls> m_avatarControls;
    std::unique_ptr<TerroirTexture> m_avatarTexture;
    GhostAvatarControlsUI m_avatarControlsUI;

    Mirror m_mirror;

    // interactors
    std::unique_ptr<opencover::coVR3DTransformInteractor> m_interactorFloor, m_interactorHand, m_interactorHead;
    void createInteractors();
    void updateInteractors();

    float m_floorHeight = 0.f;
    osg::Matrix m_trackedFloor, m_trackedHand, m_trackedHead;
    void updateMatrices();
};

COVERPLUGIN(GhostAvatar)

#endif // COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H