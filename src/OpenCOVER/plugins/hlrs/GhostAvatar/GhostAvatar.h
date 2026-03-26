#ifndef COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H
#define COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H

#include <array>
#include <memory>

#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Vec3>
#include <osg/ref_ptr>

#include <cover/coVRPluginSupport.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/VectorEditField.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>

#include "BoneParser.h"
#include "PlanarAvatarControls.h"

class GhostAvatar : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    GhostAvatar();

    // positions avatar at m_interactorFloor and makes the avatar's arm follow m_interactorHand
    bool update() override;
    void preFrame() override;

private:
    PlanarAvatarControls avatarControls = PlanarAvatarControls("/data/STARTS-ECHO/Avatars/planarAvatar/PLANEE6.fbx", "Arm", "Head");

    // interactors
    std::unique_ptr<opencover::coVR3DTransRotInteractor> m_interactorHead, m_interactorFloor, m_interactorHand;
    void createInteractors();
    void updateInteractors();

    // debugging
    osg::ref_ptr<osg::MatrixTransform> m_armTargetLine;
    osg::ref_ptr<osg::MatrixTransform> m_headTargetLine;

    osg::ref_ptr<osg::MatrixTransform> m_globalFrame;
    osg::ref_ptr<osg::MatrixTransform> m_armLocalFrame;
    osg::ref_ptr<osg::MatrixTransform> m_headLocalFrame;

    void drawLine(const osg::Vec3 &origin, const osg::Vec3 &end, osg::ref_ptr<osg::MatrixTransform> &linePtr);
    void drawFrame(const osg::Vec3 &origin, const osg::Matrix &orientation, float length, const std::string &name, osg::ref_ptr<osg::MatrixTransform> &framePtr);
    void cleanUpDebugLines();

    // UI elements
    opencover::ui::Menu *m_mainMenu = nullptr;

    opencover::ui::Menu *m_settingsMenu = nullptr;
    opencover::ui::Action *m_tabletUINote = nullptr;
    opencover::ui::Menu *m_armBaseVecMenu = nullptr;
    opencover::ui::VectorEditField *m_armBaseVecField = nullptr;
    opencover::ui::Menu *m_adjustMatrixMenu = nullptr;
    std::array<opencover::ui::VectorEditField *, 3> m_adjustMatrixVecFields;
    opencover::ui::Menu *m_adjustMatrixHeadMenu = nullptr;
    std::array<opencover::ui::VectorEditField *, 3> m_adjustMatrixHeadVecFields;

    opencover::ui::Menu *m_debugMenu = nullptr;
    opencover::ui::Button *m_showFrames = nullptr;
    opencover::ui::Button *m_showTargetLines = nullptr;
    opencover::ui::Action *m_axisNote = nullptr;

    // methods to create UI elements
    void createSettingsMenu();
    void createArmBaseVectorMenu();
    void createAdjustMatrixMenu();
    void createAdjustMatrixHeadMenu();
    void createDebugMenu();
};

COVERPLUGIN(GhostAvatar)

#endif // COVER_PLUGIN_GHOSTAVATAR_GhostAvatar_H