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
    createSettingsMenu();
    createDebugMenu();
}

void GhostAvatarControlsUI::update(const osg::Matrix &floorMatrix, const osg::Matrix &handMatrix, const osg::Matrix &headMatrix)
{
    if (m_showFrames && m_showFrames->state())
    {
        drawFrame(floorMatrix.getTrans(), osg::Matrix::identity(), 500.0f, "GlobalFrame", m_globalFrame);
        drawFrame(m_avatarControls.getInitialArmPosition(), m_avatarControls.getArmLocalToWorldMatrix(), 500.0f, "ArmLocalFrame", m_armLocalFrame);
        drawFrame(m_avatarControls.getInitialHeadPosition(), m_avatarControls.getHeadLocalToWorldMatrix(), 500.0f, "HeadLocalFrame", m_headLocalFrame);
    }

    if (m_showTargetLines && m_showTargetLines->state())
    {
        drawLine(m_avatarControls.getInitialArmPosition(), handMatrix.getTrans(), m_armTargetLine);
        drawLine(m_avatarControls.getInitialHeadPosition(), headMatrix.getTrans(), m_headTargetLine);
    }
}

void GhostAvatarControlsUI::drawLine(const osg::Vec3 &origin, const osg::Vec3 &end, osg::ref_ptr<osg::MatrixTransform> &linePtr)
{
    // Remove previous line if exists
    if (linePtr.valid())
    {
        cover->getObjectsRoot()->removeChild(linePtr);
        linePtr = nullptr;
    }

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    vertices->push_back(origin);
    vertices->push_back(end);

    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
    colors->push_back(osg::Vec4(0.31, 0.31, 0.294, 1));
    colors->push_back(osg::Vec4(0.31, 0.31, 0.294, 1));

    geom->setVertexArray(vertices);
    geom->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2));

    osg::ref_ptr<osg::LineWidth> linewidth = new osg::LineWidth(3.0f);
    geom->getOrCreateStateSet()->setAttributeAndModes(linewidth, osg::StateAttribute::ON);
    geom->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);

    linePtr = new osg::MatrixTransform();
    linePtr->addChild(geode);

    cover->getObjectsRoot()->addChild(linePtr);
}

void GhostAvatarControlsUI::drawFrame(const osg::Vec3 &origin, const osg::Matrix &orientation, float length, const std::string &name, osg::ref_ptr<osg::MatrixTransform> &framePtr)
{
    if (framePtr.valid())
    {
        cover->getObjectsRoot()->removeChild(framePtr);
        framePtr = nullptr;
    }

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();

    // Extract rotation part of orientation matrix
    osg::Matrix rotMat = orientation;
    rotMat.setTrans(0, 0, 0);

    // X axis (red)
    osg::Vec3 xAxis = osg::Vec3(1, 0, 0) * rotMat;
    xAxis.normalize();
    vertices->push_back(origin);
    vertices->push_back(origin + xAxis * length);
    colors->push_back(osg::Vec4(1, 0, 0, 1));
    colors->push_back(osg::Vec4(1, 0, 0, 1));

    // Y axis (green)
    osg::Vec3 yAxis = osg::Vec3(0, 1, 0) * rotMat;
    yAxis.normalize();
    vertices->push_back(origin);
    vertices->push_back(origin + yAxis * length);
    colors->push_back(osg::Vec4(0, 1, 0, 1));
    colors->push_back(osg::Vec4(0, 1, 0, 1));

    // Z axis (blue)
    osg::Vec3 zAxis = osg::Vec3(0, 0, 1) * rotMat;
    zAxis.normalize();
    vertices->push_back(origin);
    vertices->push_back(origin + zAxis * length);
    colors->push_back(osg::Vec4(0, 0, 1, 1));
    colors->push_back(osg::Vec4(0, 0, 1, 1));

    geom->setVertexArray(vertices);
    geom->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 6));
    osg::ref_ptr<osg::LineWidth> linewidth = new osg::LineWidth(4.0f);
    geom->getOrCreateStateSet()->setAttributeAndModes(linewidth, osg::StateAttribute::ON);
    geom->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);

    framePtr = new osg::MatrixTransform();
    framePtr->setName(name);
    framePtr->addChild(geode);

    cover->getObjectsRoot()->addChild(framePtr);
}

void GhostAvatarControlsUI::cleanUpDebugLines()
{
    if ((!m_showFrames || !m_showFrames->state()) && m_globalFrame.valid())
    {
        cover->getObjectsRoot()->removeChild(m_globalFrame);
        m_globalFrame = nullptr;
    }
    if ((!m_showFrames || !m_showFrames->state()) && m_armLocalFrame.valid())
    {
        cover->getObjectsRoot()->removeChild(m_armLocalFrame);
        m_armLocalFrame = nullptr;
    }
    if ((!m_showFrames || !m_showFrames->state()) && m_headLocalFrame.valid())
    {
        cover->getObjectsRoot()->removeChild(m_headLocalFrame);
        m_headLocalFrame = nullptr;
    }
    if ((!m_showTargetLines || !m_showTargetLines->state()) && m_armTargetLine.valid())
    {
        cover->getObjectsRoot()->removeChild(m_armTargetLine);
        m_armTargetLine = nullptr;
    }
    if ((!m_showTargetLines || !m_showTargetLines->state()) && m_headTargetLine.valid())
    {
        cover->getObjectsRoot()->removeChild(m_headTargetLine);
        m_headTargetLine = nullptr;
    }
}

void GhostAvatarControlsUI::createSettingsMenu()
{
    m_settingsMenu = new ui::Menu(m_mainMenu, "Settings");
    m_tabletUINote = new ui::Action(m_settingsMenu, "Changes can only be made in the TabletUI!");

    createArmBaseVectorMenu();
    createAdjustMatrixMenu();
    createAdjustMatrixHeadMenu();
}

void GhostAvatarControlsUI::createArmBaseVectorMenu()
{
    m_armBaseVecMenu = new ui::Menu(m_settingsMenu, "Arm Base Vector");
    m_armBaseVecField = new ui::VectorEditField(m_armBaseVecMenu, "Vector");
    m_armBaseVecField->setValue(m_avatarControls.getArmBaseVector());
    m_armBaseVecField->setCallback([this](const osg::Vec3 &dir)
        { m_avatarControls.setArmBaseVector(dir); });
}

void GhostAvatarControlsUI::createAdjustMatrixMenu()
{
    m_adjustMatrixMenu = new ui::Menu(m_settingsMenu, "Adjust Matrix");

    for (int row = 0; row < 3; ++row)
    {
        auto adjustMatrixArm = m_avatarControls.getArmAdjustMatrix();
        osg::Vec3 rowVec(adjustMatrixArm(row, 0), adjustMatrixArm(row, 1), adjustMatrixArm(row, 2));
        std::string label = "Row " + std::to_string(row);
        m_adjustMatrixVecFields[row] = new ui::VectorEditField(m_adjustMatrixMenu, label);
        m_adjustMatrixVecFields[row]->setValue(rowVec);
        m_adjustMatrixVecFields[row]->setCallback([this, row](const osg::Vec3 &v)
            { m_avatarControls.setArmAdjustMatrix(row, v); });
    }
}

void GhostAvatarControlsUI::createAdjustMatrixHeadMenu()
{
    m_adjustMatrixHeadMenu = new ui::Menu(m_settingsMenu, "Adjust Matrix Head");

    for (int row = 0; row < 3; ++row)
    {
        auto adjustMatrixHead = m_avatarControls.getHeadAdjustMatrix();
        osg::Vec3 rowVec(adjustMatrixHead(row, 0), adjustMatrixHead(row, 1), adjustMatrixHead(row, 2));
        std::string label = "Row " + std::to_string(row);
        m_adjustMatrixHeadVecFields[row] = new ui::VectorEditField(m_adjustMatrixHeadMenu, label);
        m_adjustMatrixHeadVecFields[row]->setValue(rowVec);
        m_adjustMatrixHeadVecFields[row]->setCallback([this, row](const osg::Vec3 &v)
            { m_avatarControls.setHeadAdjustMatrix(row, v); });
    }
}

void GhostAvatarControlsUI::createDebugMenu()
{
    m_debugMenu = new ui::Menu(m_mainMenu, "Debugging");

    m_showTargetLines = new ui::Button(m_debugMenu, "Show Target Lines");
    m_showTargetLines->setState(false);
    m_showTargetLines->setCallback([this](bool state)
        { m_showTargetLines->setState(state); 
                                       cleanUpDebugLines(); });

    m_showFrames = new ui::Button(m_debugMenu, "Show Frames");
    m_showFrames->setState(false);
    m_showFrames->setCallback([this](bool state)
        { m_showFrames->setState(state);
                                    cleanUpDebugLines(); });

    m_axisNote = new ui::Action(m_debugMenu, "x - red, y - green, z - blue");
    m_axisNote->setEnabled(false);
}
