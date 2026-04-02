#include <osg/Geode>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>

#include <cover/coVRPluginSupport.h> // for cover

#include "GhostAvatarControlsUI.h"

using namespace opencover;
using namespace ui;

GhostAvatarControlsUI::GhostAvatarControlsUI(const std::string &pluginName, GhostAvatarControls &avatarControls)
    : Owner(pluginName, cover->ui)
    , m_mainMenu(new ui::Menu("GhostAvatar", this))
    , m_avatarControls(avatarControls)
{
    m_armTargetLine = CoverLine(m_targetLineColor, m_targetLineWidth, "ArmTargetLine");
    m_headTargetLine = CoverLine(m_targetLineColor, m_targetLineWidth, "HeadTargetLine");

    m_globalFrame = CoverFrame(m_frameLineLength, m_frameLineWidth, "GlobalFrame");
    m_armLocalFrame = CoverFrame(m_frameLineLength, m_frameLineWidth, "ArmLocalFrame");
    m_headLocalFrame = CoverFrame(m_frameLineLength, m_frameLineWidth, "HeadLocalFrame");
}

void GhostAvatarControlsUI::update(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix)
{
    if (m_showFrames && m_showFrames->state())
    {
        m_globalFrame.draw(floorMatrix.getTrans(), osg::Matrix::identity());

        if (m_armAvailable)
            m_armLocalFrame.draw(m_avatarControls.getInitialArmPosition(), m_avatarControls.getArmLocalToWorldMatrix());

        if (m_headAvailable)
            m_headLocalFrame.draw(m_avatarControls.getInitialHeadPosition(), m_avatarControls.getHeadLocalToWorldMatrix());
    }

    if (m_showTargetLines && m_showTargetLines->state())
    {
        if (m_armAvailable)
            m_armTargetLine.draw(m_avatarControls.getInitialArmPosition(), handMatrix.getTrans());

        if (m_headAvailable)
            m_headTargetLine.draw(m_avatarControls.getInitialHeadPosition(), headMatrix.getTrans());
    }
}

void GhostAvatarControlsUI::initialize()
{
    m_armAvailable = m_avatarControls.hasArm();
    m_headAvailable = m_avatarControls.hasHead();

    createSettingsMenu();
    createDebugMenu();
}

void GhostAvatarControlsUI::cleanUpDebugLines()
{
    if ((!m_showFrames || !m_showFrames->state()))
    {
        m_globalFrame.clean();
        m_armLocalFrame.clean();
        m_headLocalFrame.clean();
    }

    if ((!m_showTargetLines || !m_showTargetLines->state()))
    {
        m_armTargetLine.clean();
        m_headTargetLine.clean();
    }
}

void GhostAvatarControlsUI::createSettingsMenu()
{
    m_settingsMenu = new ui::Menu(m_mainMenu, "Settings");
    m_tabletUINote = new ui::Action(m_settingsMenu, "tabletUINote");
    m_tabletUINote->setText("Changes can only be made in the TabletUI!");

    if (m_armAvailable)
        createArmSettingsMenu();
    if (m_headAvailable)
        createHeadSettingsMenu();
}

void GhostAvatarControlsUI::createArmSettingsMenu()
{
    m_armSettingsMenu = new ui::Menu(m_settingsMenu, "Arm");

    m_armBaseVectorField = new ui::VectorEditField(m_armSettingsMenu, "armBaseVectorField");
    m_armBaseVectorField->setText("Base Vector");
    m_armBaseVectorField->setValue(m_avatarControls.getArmBaseVector());
    m_armBaseVectorField->setCallback([this](const osg::Vec3 &dir)
        { m_avatarControls.setArmBaseVector(dir); });

    m_armAdjustMatrixMenu = new ui::Menu(m_armSettingsMenu, "armAdjustMatrixMenu");
    m_armAdjustMatrixMenu->setText("Adjust Matrix");
    for (int row = 0; row < 3; ++row)
    {
        auto adjustMatrixArm = m_avatarControls.getArmAdjustMatrix();
        osg::Vec3 rowVec(adjustMatrixArm(row, 0), adjustMatrixArm(row, 1), adjustMatrixArm(row, 2));
        m_armAdjustMatrixFields[row] = new ui::VectorEditField(m_armAdjustMatrixMenu, "armAdjustMatrixFieldsRow" + std::to_string(row));
        m_armAdjustMatrixFields[row]->setText("Row " + std::to_string(row));
        m_armAdjustMatrixFields[row]->setValue(rowVec);
        m_armAdjustMatrixFields[row]->setCallback([this, row](const osg::Vec3 &v)
            { m_avatarControls.setArmAdjustMatrix(row, v); });
    }
}

void GhostAvatarControlsUI::createHeadSettingsMenu()
{
    m_headSettingsMenu = new ui::Menu(m_settingsMenu, "Head");

    m_headBaseVectorField = new ui::VectorEditField(m_headSettingsMenu, "headBaseVectorField");
    m_headBaseVectorField->setText("Base Vector");
    m_headBaseVectorField->setValue(m_avatarControls.getHeadBaseVector());
    m_headBaseVectorField->setCallback([this](const osg::Vec3 &dir)
        { m_avatarControls.setHeadBaseVector(dir); });

    m_headAdjustMatrixMenu = new ui::Menu(m_headSettingsMenu, "headAdjustMatrixMenu");
    m_headAdjustMatrixMenu->setText("Adjust Matrix");
    for (int row = 0; row < 3; ++row)
    {
        auto adjustMatrixHead = m_avatarControls.getHeadAdjustMatrix();
        osg::Vec3 rowVec(adjustMatrixHead(row, 0), adjustMatrixHead(row, 1), adjustMatrixHead(row, 2));
        m_headAdjustMatrixFields[row] = new ui::VectorEditField(m_headAdjustMatrixMenu, "headAdjustMatrixFieldsRow" + std::to_string(row));
        m_headAdjustMatrixFields[row]->setText("Row " + std::to_string(row));
        m_headAdjustMatrixFields[row]->setValue(rowVec);
        m_headAdjustMatrixFields[row]->setCallback([this, row](const osg::Vec3 &v)
            { m_avatarControls.setHeadAdjustMatrix(row, v); });
    }
}

void GhostAvatarControlsUI::createDebugMenu()
{
    m_debugMenu = new ui::Menu(m_mainMenu, "Debugging");

    m_showTargetLines = new ui::Button(m_debugMenu, "showTargetLines");
    m_showTargetLines->setText("Show Target Lines");
    m_showTargetLines->setState(false);
    m_showTargetLines->setCallback([this](bool state)
        { m_showTargetLines->setState(state); 
                                       cleanUpDebugLines(); });

    m_showFrames = new ui::Button(m_debugMenu, "showFrames");
    m_showFrames->setText("Show Frames");
    m_showFrames->setState(false);
    m_showFrames->setCallback([this](bool state)
        { m_showFrames->setState(state);
                                    cleanUpDebugLines(); });

    m_axisNote = new ui::Action(m_debugMenu, "axisNote");
    m_axisNote->setText("x - red, y - green, z - blue");
    m_axisNote->setEnabled(false);
}
