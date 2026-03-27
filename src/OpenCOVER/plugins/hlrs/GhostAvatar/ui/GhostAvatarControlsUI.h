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

#include "CoverDrawObject.h"
#include "../GhostAvatarControls.h"

class GhostAvatarControlsUI : public opencover::ui::Owner
{
public:
    GhostAvatarControlsUI(const std::string &pluginName, GhostAvatarControls &avatarControls);

    void update(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix);

private:
    GhostAvatarControls &m_avatarControls;

    CoverLine m_armTargetLine;
    CoverLine m_headTargetLine;
    osg::Vec4 m_targetLineColor = { 0.31, 0.31, 0.294, 1 };
    float m_targetLineWidth = 3;

    CoverFrame m_globalFrame;
    CoverFrame m_armLocalFrame;
    CoverFrame m_headLocalFrame;
    float m_frameLineLength = 500;
    float m_frameLineWidth = 3;

    void cleanUpDebugLines();

    opencover::ui::Menu *m_mainMenu;
    opencover::ui::Action* m_tabletUINote;

    opencover::ui::Menu *m_settingsMenu;
    opencover::ui::Menu *m_armSettingsMenu;
    opencover::ui::VectorEditField* m_armBaseVectorField;
    opencover::ui::Menu* m_armAdjustMatrixMenu;
    std::array<opencover::ui::VectorEditField *, 3>  m_armAdjustMatrixFields;

    opencover::ui::Menu *m_headSettingsMenu;
    opencover::ui::VectorEditField* m_headBaseVectorField;
    opencover::ui::Menu* m_headAdjustMatrixMenu;
    std::array<opencover::ui::VectorEditField *, 3>  m_headAdjustMatrixFields;

    opencover::ui::Menu *m_debugMenu;
    opencover::ui::Button *m_showFrames;
    opencover::ui::Button *m_showTargetLines;
    opencover::ui::Action *m_axisNote;

    // methods to create UI elements
    void createSettingsMenu();
    void createArmSettingsMenu();
    void createHeadSettingsMenu();

    void createDebugMenu();
};

#endif // COVER_PLUGIN_GHOSTAVATAR_UI_GhostAvatarControlsUI_H