#ifndef COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H
#define COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H

#include <memory>

#include <cover/coVRPluginSupport.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>

#include "controls/GhostAvatarControls.h"
#include "texture/StripesTerroirTexture.h"
#include "texture/SplotchTerroirTexture.h"
#include "ui/GhostAvatarControlsUI.h"

// TODO:
//   - think about how to deal with avatar scale (checkout VR avatar for that first)?

class GhostAvatar : public opencover::coVRPlugin
{
public:
    GhostAvatar();

    bool update() override;
    void preFrame() override;

private:
    std::unique_ptr<GhostAvatarControls> m_avatarControls;
    SplotchTerroirTexture m_avatarTexture;
    GhostAvatarControlsUI m_avatarControlsUI;

    // interactors
    std::unique_ptr<opencover::coVR3DTransRotInteractor> m_interactorHead, m_interactorFloor, m_interactorHand;
    void createInteractors();
    void updateInteractors();
};

COVERPLUGIN(GhostAvatar)

#endif // COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H