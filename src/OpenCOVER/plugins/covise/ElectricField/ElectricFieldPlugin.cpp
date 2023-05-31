/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: ElectricFieldPlugin                                       **
 **              for VR4Schule                                             **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "ElectricFieldPlugin.h"

#include <grmsg/coGRGenericParamRegisterMsg.h>
#include <grmsg/coGRGenericParamChangedMsg.h>

#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include "cover/coTranslator.h"

#include <config/CoviseConfig.h>
#include <cover/coVRNavigationManager.h>
#include <cover/VRSceneGraph.h>

#include <osg/LineWidth>

ElectricFieldPlugin *ElectricFieldPlugin::plugin = NULL;

//
// Constructor
//
ElectricFieldPlugin::ElectricFieldPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , GenericGuiObject("ElectricFieldPlugin")
    , probe(NULL)
    , tracer(NULL)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nElectricFieldPlugin::ElectricFieldPlugin\n");
}

//
// Destructor
//
ElectricFieldPlugin::~ElectricFieldPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nElectricFieldPlugin::~ElectricFieldPlugin\n");
}

//
// INIT
//
bool ElectricFieldPlugin::init()
{
    if (plugin)
        return false;
    if (cover->debugLevel(3))
        fprintf(stderr, "\nElectricFieldPlugin::ElectricFieldPlugin\n");

    presentationMode_ = coCoviseConfig::isOn("value", "COVER.Plugin.ElectricField.PresentationMode", true);

    // set plugin
    ElectricFieldPlugin::plugin = this;

    // add bounding box
    drawBoundingBox();

    // add menu
    createMenu();

    // add tracer
    tracer = new Tracer();

    // add probe
    probe = new Probe();

    // add charged objects and perform an plugin wide update via field calculation
    ChargedObjectHandler::Instance()->dirtyField();

    // vr-prepare (add param after menu was created)
    p_showMenu = addGuiParamBool("ShowMenu", true);

    return true;
}

void ElectricFieldPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    GenericGuiObject::guiToRenderMsg(msg);
    ChargedObjectHandler::Instance()->guiToRenderMsg(msg);
    probe->guiToRenderMsg(msg);
    tracer->guiToRenderMsg(msg);
}

void ElectricFieldPlugin::preFrame()
{
    ChargedObjectHandler::Instance()->preFrame();
    probe->preFrame();
    tracer->preFrame();
}

void ElectricFieldPlugin::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == objectsAddPointEntry)
    {
        ChargedObjectHandler::Instance()->addPoint();
    }
    if (menuItem == objectsAddPlateEntry)
    {
        ChargedObjectHandler::Instance()->addPlate();
    }
    if (menuItem == objectsClearEntry)
    {
        ChargedObjectHandler::Instance()->removeAllObjects();
    }
    if (menuItem == menuItemRadius)
    {
        ChargedObjectHandler::Instance()->setRadiusOfPlates(menuItemRadius->getValue() / 100.0);
    }
}

void ElectricFieldPlugin::menuReleaseEvent(coMenuItem *menuItem)
{
    if (menuItem == menuItemRadius)
    {
        ChargedObjectHandler::Instance()->dirtyField();
    }
}

void ElectricFieldPlugin::setRadiusOfPlates(float radius)
{
    menuItemRadius->setValue(radius * 100.0);
}

void ElectricFieldPlugin::guiParamChanged(GuiParam *guiParam)
{
    if (guiParam == p_showMenu)
    {
        objectsMenu->setVisible(p_showMenu->getValue());
        VRSceneGraph::instance()->applyMenuModeToMenus(); // apply menuMode state to menus just made visible
    }
}

void ElectricFieldPlugin::createMenu()
{
    objectsMenu = new coRowMenu(coTranslator::coTranslate("Geladene Objekte").c_str());
    objectsMenu->setVisible(true);
    objectsMenu->setAttachment(coUIElement::RIGHT);

    // position Menu
    OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;
    //position the menu
    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", -1000);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", 0);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 600);

    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.ElectricField.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.ElectricField.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.ElectricField.MenuPosition", pz);

    // default is Mathematic.MenuSize then COVER.Menu.Size then 1.0
    float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("value", "COVER.Plugin.ElectricField.MenuSize", s);

    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.makeScale(s, s, s);

    matrix.makeIdentity();
    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    objectsMenu->setTransformMatrix(&matrix);
    objectsMenu->setScale(cover->getSceneSize() / 2500);

    objectsAddPointEntry = new coButtonMenuItem(coTranslator::coTranslate("Neuer Punkt"));
    objectsAddPointEntry->setMenuListener(this);
    objectsMenu->add(objectsAddPointEntry);
    objectsAddPlateEntry = new coButtonMenuItem(coTranslator::coTranslate("Neue Platte"));
    objectsAddPlateEntry->setMenuListener(this);
    objectsMenu->add(objectsAddPlateEntry);
    //    objectsClearEntry = new coButtonMenuItem("Alle Objekte entfernen");
    //    objectsClearEntry->setMenuListener(this);
    //    objectsMenu->add(objectsClearEntry);

    // menu
    menuItemRadius = new coSliderMenuItem(coTranslator::coTranslate("Radius Platten in cm"), 10.0, 100.0, 50.0);
    menuItemRadius->setMenuListener(this);
    objectsMenu->add(menuItemRadius);

    objectsMenu->setVisible(false);
}

void ElectricFieldPlugin::drawBoundingBox()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "MathematicPlugin::drawBoundingBox\n");

    boxGeode = new osg::Geode();
    float min = ChargedObjectHandler::Instance()->getGridMin();
    float max = ChargedObjectHandler::Instance()->getGridMax();

    osg::Vec3 bpoints[8];
    bpoints[0].set(min, min, min);
    bpoints[1].set(max, min, min);
    bpoints[2].set(max, max, min);
    bpoints[3].set(min, max, min);
    bpoints[4].set(min, min, max);
    bpoints[5].set(max, min, max);
    bpoints[6].set(max, max, max);
    bpoints[7].set(min, max, max);

    osg::Geometry *lineGeometry[12];
    osg::Vec3Array *vArray[12];
    osg::DrawArrays *drawable[12];

    for (int i = 0; i < 12; i++)
    {
        lineGeometry[i] = new osg::Geometry();
        vArray[i] = new osg::Vec3Array();
        //fprintf(stderr,"MathematicPlugin::drawBoundingBox bpoints[0] %f %f %f\n", bpoints[0].x(), bpoints[0].y(), bpoints[0].z());
        //fprintf(stderr,"MathematicPlugin::drawBoundingBox bpoints[1] %f %f %f\n", bpoints[1].x(), bpoints[1].y(), bpoints[1].z());
        lineGeometry[i]->setVertexArray(vArray[i]);
        drawable[i] = new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2);
        lineGeometry[i]->addPrimitiveSet(drawable[i]);
        osg::LineWidth *linewidth = new osg::LineWidth();
        linewidth->setWidth(1.0);
        boxGeode->addDrawable(lineGeometry[i]);
    }

    // lines
    vArray[0]->push_back(bpoints[0]);
    vArray[0]->push_back(bpoints[1]);
    vArray[1]->push_back(bpoints[1]);
    vArray[1]->push_back(bpoints[2]);
    vArray[2]->push_back(bpoints[2]);
    vArray[2]->push_back(bpoints[3]);
    vArray[3]->push_back(bpoints[3]);
    vArray[3]->push_back(bpoints[0]);
    vArray[4]->push_back(bpoints[4]);
    vArray[4]->push_back(bpoints[5]);
    vArray[5]->push_back(bpoints[5]);
    vArray[5]->push_back(bpoints[6]);
    vArray[6]->push_back(bpoints[6]);
    vArray[6]->push_back(bpoints[7]);
    vArray[7]->push_back(bpoints[7]);
    vArray[7]->push_back(bpoints[4]);
    vArray[8]->push_back(bpoints[0]);
    vArray[8]->push_back(bpoints[4]);
    vArray[9]->push_back(bpoints[3]);
    vArray[9]->push_back(bpoints[7]);
    vArray[10]->push_back(bpoints[2]);
    vArray[10]->push_back(bpoints[6]);
    vArray[11]->push_back(bpoints[1]);
    vArray[11]->push_back(bpoints[5]);

    osg::Material *material = new osg::Material();
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
    osg::StateSet *stateSet = boxGeode->getOrCreateStateSet();
    stateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(material);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    boxGeode->setStateSet(stateSet);
    boxGeode->setNodeMask(boxGeode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    cover->getObjectsRoot()->addChild(boxGeode);
    setBoundingBoxVisible(false);
}

void ElectricFieldPlugin::setBoundingBoxVisible(bool visible)
{
    if (!boxGeode)
    {
        return;
    }
    if (visible)
    {
        boxGeode->setNodeMask(boxGeode->getNodeMask() | (Isect::Visible));
    }
    else
    {
        boxGeode->setNodeMask(boxGeode->getNodeMask() & (~Isect::Visible));
    }
}

void ElectricFieldPlugin::fieldChanged()
{
    if (probe)
    {
        probe->update();
    }
    if (tracer)
    {
        tracer->update();
    }
}

COVERPLUGIN(ElectricFieldPlugin)
