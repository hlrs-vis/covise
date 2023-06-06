/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: ParticlePathPlugin                                         **
 **              for VR4Schule                                             **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "Const.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <net/message.h>

#include "cover/coTranslator.h"

#include "ParticlePathPlugin.h"

using namespace opencover;
using namespace vrui;
using covise::coCoviseConfig;

ParticlePathPlugin *ParticlePathPlugin::plugin = NULL;

//
// Constructor
//
ParticlePathPlugin::ParticlePathPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, GenericGuiObject("ParticlePath")
, boundingBox(NULL)
, electricFieldArrow(NULL)
, magneticFieldArrow(NULL)
, path(NULL)
, previousPath(NULL)
, target(NULL)
, sliderMoving(0)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nParticlePathPlugin::ParticlePathPlugin\n");
}

//
// Destructor
//
ParticlePathPlugin::~ParticlePathPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nParticlePathPlugin::~ParticlePathPlugin\n");

    if (path)
        delete path;
    if (previousPath)
        delete path;
    if (electricFieldArrow)
        delete electricFieldArrow;
    if (magneticFieldArrow)
        delete magneticFieldArrow;
    if (target)
        delete target;
    if (boundingBox)
        delete boundingBox;
}

//
// INIT
//
bool ParticlePathPlugin::init()
{
    if (plugin)
        return false;
    if (cover->debugLevel(3))
        fprintf(stderr, "\nParticlePathPlugin::ParticlePathPlugin\n");

    // set plugin
    ParticlePathPlugin::plugin = this;

    // add gui params
    p_visible = addGuiParamBool("Visible", false);
    p_mass = addGuiParamFloat("Config.Mass", 20.0f);
    p_charge = addGuiParamFloat("Config.Charge", 1.0f);
    p_velocity = addGuiParamFloat("Config.Velocity", 100.0f);
    p_voltage = addGuiParamFloat("Config.Voltage", 1.0f);
    p_angle = addGuiParamFloat("Config.Angle", 0.0f);
    p_electricField = addGuiParamFloat("Config.ElectricField", 0.0f);
    p_magneticField = addGuiParamFloat("Config.MagneticField", 0.0f);
    p_mass_visible = addGuiParamBool("Config.Mass.Visible", true);
    p_charge_visible = addGuiParamBool("Config.Charge.Visible", true);
    p_velocity_visible = addGuiParamBool("Config.Velocity.Visible", true);
    p_voltage_visible = addGuiParamBool("Config.Voltage.Visible", true);
    p_angle_visible = addGuiParamBool("Config.Angle.Visible", true);
    p_electricField_visible = addGuiParamBool("Config.ElectricField.Visible", true);
    p_magneticField_visible = addGuiParamBool("Config.MagneticField.Visible", true);

    // menu
    createMenu();

    pluginBaseNode = new osg::Group();

    // draw bounding box
    boundingBox = new BoundingBox(pluginBaseNode);

    // target
    target = new Target(pluginBaseNode);

    // field arrows
    electricFieldArrow = new Arrow(pluginBaseNode, ELECTRIC_FORCE_ARROW_COLOR);
    magneticFieldArrow = new Arrow(pluginBaseNode, MAGNETIC_FORCE_ARROW_COLOR);

    // path
    path = new Path(pluginBaseNode);
    updatePath();

    return true;
}

void ParticlePathPlugin::createMenu()
{
    menu = new coRowMenu("Parameter");
    menu->setVisible(true);
    menu->setAttachment(coUIElement::RIGHT);

    // position Menu
    OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;
    //position the menu
    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", -1000.0f);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", 0.0f);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 600.0f);

    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.ParticlePath.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.ParticlePath.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.ParticlePath.MenuPosition", pz);

    // default is Mathematic.MenuSize then COVER.Menu.Size then 1.0
    float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("value", "COVER.Plugin.ParticlePath.MenuSize", s);

    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0.0f, 90.0f, 0.0f);
    scaleMatrix.makeScale(s, s, s);

    matrix.makeIdentity();
    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    menu->setTransformMatrix(&matrix);
    menu->setScale(cover->getSceneSize() / 2500.0f);

    m_mass = new coSliderMenuItem(coTranslator::coTranslate("Masse in u"), 10.0f, 40.0f, p_mass->getValue());
    m_mass->setMenuListener(this);
    m_charge = new coSliderMenuItem(coTranslator::coTranslate("Ladung in e"), -2.0f, 2.0f, p_charge->getValue());
    m_charge->setMenuListener(this);
    m_velocity = new coSliderMenuItem(coTranslator::coTranslate("Geschwindigkeit in km/s"), -150.0f, 150.0f, p_velocity->getValue());
    m_velocity->setMenuListener(this);
    m_voltage = new coSliderMenuItem(coTranslator::coTranslate("Anodenspannung in kV"), 0.0f, 3.0f, p_voltage->getValue());
    m_voltage->setMenuListener(this);
    m_angle = new coSliderMenuItem(coTranslator::coTranslate("Winkel in Grad"), 0.0f, 90.0f, p_angle->getValue());
    m_angle->setMenuListener(this);
    m_electricField = new coSliderMenuItem(coTranslator::coTranslate("Elektrisches Feld in kV/m"), -15.0f, 15.0f, p_electricField->getValue());
    m_electricField->setMenuListener(this);
    m_magneticField = new coSliderMenuItem(coTranslator::coTranslate("Magnetisches Feld in mT"), -150.0f, 150.0f, p_magneticField->getValue());
    m_magneticField->setMenuListener(this);

    rebuildMenu();
}

void ParticlePathPlugin::rebuildMenu()
{
    menu->removeAll();
    // Dont hide the menu if p_visible is false, just remove all the elements.
    // Hiding/Showing conflicts with the CyberClassroom menu handling.
    if (p_visible->getValue())
    {
        if (p_mass_visible->getValue())
            menu->add(m_mass);
        if (p_charge_visible->getValue())
            menu->add(m_charge);
        if (p_velocity_visible->getValue())
            menu->add(m_velocity);
        if (p_voltage_visible->getValue())
            menu->add(m_voltage);
        if (p_angle_visible->getValue())
            menu->add(m_angle);
        if (p_electricField_visible->getValue())
            menu->add(m_electricField);
        if (p_magneticField_visible->getValue())
            menu->add(m_magneticField);
    }
}

void ParticlePathPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    GenericGuiObject::guiToRenderMsg(msg);
    if (target)
        target->guiToRenderMsg(msg);
}

void ParticlePathPlugin::preFrame()
{
    if (path != NULL)
        path->preFrame();
}

void ParticlePathPlugin::updatePreviousPath(bool visible)
{
    if (previousPath != NULL)
    {
        delete previousPath;
        previousPath = NULL;
    }
    if (visible)
    {
        previousPath = path;
        path = new Path(pluginBaseNode);
        previousPath->setInactive();
        path->tracer->config = previousPath->tracer->config;
    }
}

void ParticlePathPlugin::updatePath()
{
    path->tracer->config.mass = (double)p_mass->getValue() * GUI_SCALING_MASS;
    path->tracer->config.charge = (double)p_charge->getValue() * GUI_SCALING_CHARGE;
    if (p_voltage_visible->getValue())
    {
        // only use voltage if slider is visible
        // v = sqrt(2qU/m)
        path->tracer->config.velocity = sqrt((2.0 * fabs(path->tracer->config.charge) * (double)p_voltage->getValue() * GUI_SCALING_VOLTAGE) / path->tracer->config.mass);
    }
    else
    {
        path->tracer->config.velocity = (double)p_velocity->getValue() * GUI_SCALING_VELOCITY;
    }
    path->tracer->config.angle = (double)p_angle->getValue() * GUI_SCALING_ANGLE;
    path->tracer->config.electricField = (double)p_electricField->getValue() * GUI_SCALING_ELECTRIC_FIELD;
    path->tracer->config.magneticField = (double)p_magneticField->getValue() * GUI_SCALING_MAGNETIC_FIELD;
    path->calculateNewPath();

    electricFieldArrow->update(TRACE_CENTER, BASE_VECTOR_ELECTRIC * path->tracer->config.electricField * ELECTRIC_FIELD_ARROW_SCALING);
    magneticFieldArrow->update(TRACE_CENTER, BASE_VECTOR_MAGNETIC * path->tracer->config.magneticField * MAGNETIC_FIELD_ARROW_SCALING);
}

void ParticlePathPlugin::guiParamChanged(GuiParam *guiParam)
{
    if (guiParam == p_visible)
    {
        if (p_visible->getValue())
        {
            if (!cover->getObjectsRoot()->containsNode(pluginBaseNode.get()))
                cover->getObjectsRoot()->addChild(pluginBaseNode.get());
        }
        else
        {
            if (cover->getObjectsRoot()->containsNode(pluginBaseNode.get()))
                cover->getObjectsRoot()->removeChild(pluginBaseNode.get());
        }
    }

    if ((guiParam == p_mass)
        || (guiParam == p_charge)
        || (guiParam == p_velocity)
        || (guiParam == p_voltage)
        || (guiParam == p_voltage_visible) // voltage is only used when menu is visible
        || (guiParam == p_angle)
        || (guiParam == p_electricField)
        || (guiParam == p_magneticField))
    {
        updatePreviousPath(false);
        if (guiParam == p_mass)
            m_mass->setValue(p_mass->getValue());
        if (guiParam == p_charge)
            m_charge->setValue(p_charge->getValue());
        if (guiParam == p_velocity)
            m_velocity->setValue(p_velocity->getValue());
        if (guiParam == p_voltage)
            m_voltage->setValue(p_voltage->getValue());
        if (guiParam == p_angle)
            m_angle->setValue(p_angle->getValue());
        if (guiParam == p_electricField)
            m_electricField->setValue(p_electricField->getValue());
        if (guiParam == p_magneticField)
            m_magneticField->setValue(p_magneticField->getValue());
        updatePath();
    }

    if ((guiParam == p_visible)
        || (guiParam == p_mass_visible)
        || (guiParam == p_charge_visible)
        || (guiParam == p_velocity_visible)
        || (guiParam == p_voltage_visible)
        || (guiParam == p_angle_visible)
        || (guiParam == p_electricField_visible)
        || (guiParam == p_magneticField_visible))
    {
        rebuildMenu();
    }
}

void ParticlePathPlugin::menuEvent(coMenuItem *menuItem)
{
    // previous path
    if (sliderMoving == 0)
    {
        sliderMoving = 1;
        updatePreviousPath(true);
    }
    if (sliderMoving == 2)
    {
        sliderMoving = 0;
    }

    // update
    if (menuItem == m_mass)
        p_mass->setValue(m_mass->getValue());
    if (menuItem == m_charge)
        p_charge->setValue(m_charge->getValue());
    if (menuItem == m_velocity)
        p_velocity->setValue(m_velocity->getValue());
    if (menuItem == m_voltage)
        p_voltage->setValue(m_voltage->getValue());
    if (menuItem == m_angle)
        p_angle->setValue(m_angle->getValue());
    if (menuItem == m_electricField)
        p_electricField->setValue(m_electricField->getValue());
    if (menuItem == m_magneticField)
        p_magneticField->setValue(m_magneticField->getValue());

    updatePath();
}

void ParticlePathPlugin::menuReleaseEvent(coMenuItem *menuItem)
{
    // snap slider to 0.0 if close
    coSliderMenuItem *slider = dynamic_cast<coSliderMenuItem *>(menuItem);
    if ((slider != NULL) && (slider->getMin() < 0.0f) && (slider->getMax() > 0.0f))
    {
        if (fabs(slider->getValue()) < (slider->getMax() - slider->getMin()) * 0.05f)
        {
            slider->setValue(0.0f);
            menuEvent(menuItem);
        }
    }

    sliderMoving = 2;
}

COVERPLUGIN(ParticlePathPlugin)
