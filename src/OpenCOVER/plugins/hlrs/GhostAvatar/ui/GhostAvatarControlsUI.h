#ifndef COVER_PLUGIN_GHOSTAVATAR_UI_GhostAvatarControlsUI_H
#define COVER_PLUGIN_GHOSTAVATAR_UI_GhostAvatarControlsUI_H

#include <array>
#include <string>

#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/ref_ptr>
#include <osg/Vec3>

#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/VectorEditField.h>

#include "../GhostAvatarControls.h"

class GhostAvatarControlsUI : public opencover::ui::Owner
{
public:

    GhostAvatarControlsUI(const std::string& pluginName, GhostAvatarControls& avatarControls);
    
    void update(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix);

private:

    GhostAvatarControls& m_avatarControls;

    // debugging
    osg::ref_ptr<osg::MatrixTransform> m_armTargetLine = nullptr;
    osg::ref_ptr<osg::MatrixTransform> m_headTargetLine = nullptr;

    osg::ref_ptr<osg::MatrixTransform> m_globalFrame = nullptr;
    osg::ref_ptr<osg::MatrixTransform> m_armLocalFrame = nullptr;
    osg::ref_ptr<osg::MatrixTransform> m_headLocalFrame = nullptr;

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

#endif // COVER_PLUGIN_GHOSTAVATAR_UI_GhostAvatarControlsUI_H