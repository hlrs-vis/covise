#ifndef COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H
#define COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H

#include <memory>
#include <string>
#include <vector>

#include <osg/Matrix>

#include <cover/coVRPlugin.h>
#include <PluginUtil/coVR3DTransformInteractor.h>

#include "controls/GhostAvatarControls.h"
#include "scene/Mirror.h"
#include "texture/TerroirTexture.h"
#include "ui/GhostAvatarControlsUI.h"

/*
    This plugin creates a ghost avatar that was developed for the STARTS project with
    the artist Bernat Cuni (VR Terroir). The avatar is controlled by the VR glasses and
    controller in the CAVE. As the avatar roams through the virtual world screenshots
    of the environment from the user's point of view are added to the avatar's texture.

    For the test scene during the GATE festival, mirrors can also be loaded into the scene
    s.t. the user can inspect the avatar (which represents themselves). In the future the
    avatar will be embedded into COVER's collaborative mode.

    The user can change the following settings in the config file (first option is default):
    - `avatarType="planar"` or `"ghost"` - chooses the model of the avatar
    - `textureType="splotches"` or `"stripes"` - chooses the shape of the screenshots
                                                 that are added to the avatar's texture
    - `distanceThreshold="5"` - the distance the user must travel (in world units)
                                until the texture is updated
    - `useInteractors="false"` - (for debugging) if true, the avatar is visible in the scene
                                  and can be controlled with three pick interactors which mimick
                                  the output from the MoCap system (floor, glasses, 3D controller)
    - `mirrorsForScene=0` - if not 0, mirrors will be placed into scene at pre-defined positions defined
                            in `addMirrorsToScene` (for the scenes used during the GATE festival)

*/
class GhostAvatar : public opencover::coVRPlugin
{
public:
    GhostAvatar();

    bool init() override;
    void preFrame() override;

private:
    // Loads the avatar model (FBX + rig) and prepares it so the plugin can move it based on the MoCap input.
    std::unique_ptr<GhostAvatarControls> m_avatarControls;

    // Places an invisible camera at the avatar's eye position and takes screenshots every time the user has
    // travelled `m_distanceThreshold` units in the virtual world.
    std::unique_ptr<TerroirTexture> m_avatarTexture;

    // Sets up some control and debbuging settings in the TabletUI.
    std::unique_ptr<GhostAvatarControlsUI> m_avatarControlsUI;

    /*
        If true, the avatar is placed into the scene and controlled by three pick interactors simulating
        the MoCap input (useful for debugging the controls). If false, the avatar is not visible in the scene
        (except in mirrors) and controlled by the MoCap output.
    */
    bool m_useInteractors;
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

    /*
        For the GATE festival scene: Places mirrors into the scene so the user's can see the avatar
        and especially its texture that changes over time.
    */
    int m_mirrorsForScene = 0;
    bool m_mirrorsInScene = false;

    std::vector<Mirror> m_mirrors;
    void addMirrorsToScene();
    void updateMirrorViews();
};

COVERPLUGIN(GhostAvatar)

#endif // COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H