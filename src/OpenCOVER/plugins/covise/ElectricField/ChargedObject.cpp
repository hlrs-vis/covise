/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ChargedObject.h"

#include "cover/VRSceneGraph.h"

#include "cover/coTranslator.h"

#include "ElectricFieldPlugin.h"
#include "ChargedObjectHandler.h"

ChargedObject::ChargedObject(unsigned int type_id, std::string name, float initialCharge)
    : GenericGuiObject(name)
    , coVR3DTransInteractor(osg::Vec3(0.0, 0.0, 0.0), 1.0, coInteraction::ButtonA, "hand", name.c_str(), coInteraction::Medium)
    , active(false)
    , charge(initialCharge)
{
    this->type_id = type_id;

    this->hide();
    this->disableIntersection();

    // menu
    menuItemSeparator = new coLabelMenuItem("__________________________");
    menuItemCaption = new coLabelMenuItem(coTranslator::coTranslate(name));
    menuItemDelete = new coButtonMenuItem(coTranslator::coTranslate("Entfernen"));
    menuItemDelete->setMenuListener(this);
    menuItemCharge = new coSliderMenuItem(coTranslator::coTranslate("Ladung in V/m"), -500.0, 500.0, initialCharge);
    menuItemCharge->setMenuListener(this);

    // geometry
    objectMaterial = new osg::Material();
    objectMaterial->setSpecular(osg::Material::FRONT, osg::Vec4(1.0, 1.0, 1.0, 1.0));
    objectMaterial->setShininess(osg::Material::FRONT, 25.0);
    setColorAccordingToCharge();

    // label
    label = new coVRLabel(coTranslator::coTranslate(name).c_str(), 24, 100.0, osg::Vec4(0.5451, 0.7020, 0.2431, 1.0), osg::Vec4(0.0, 0.0, 0.0, 0.8));
    label->hide();

    // vr-prepare
    p_active = addGuiParamBool("Active", active);
    p_charge = addGuiParamFloat("Charge", charge);
    p_position = addGuiParamVec3("Position", osg::Vec3(0.0, 0.0, 0.0));

    changedFromUser_ = false;
}

ChargedObject::~ChargedObject()
{
}

void ChargedObject::preFrame()
{

    osg::Vec3 position = this->getPos(); // interactor

    if (this->isRunning()) // interactor
    {

        if (type_id == TYPE_PLATE)
        {
            // restrict movement of plate
            position[1] = 0.0;
            position[2] = 0.0;
        }

        float min = ChargedObjectHandler::Instance()->getGridMin() / 2.0;
        float max = ChargedObjectHandler::Instance()->getGridMax() / 2.0;
        // reposition interactor if outside
        if (position[0] > max)
            position[0] = max;
        else if (position[0] < min)
            position[0] = min;
        if (position[1] > max)
            position[1] = max;
        else if (position[1] < min)
            position[1] = min;
        if (position[2] > max)
            position[2] = max;
        else if (position[2] < min)
            position[2] = min;

        setPosition(position);
        changedFromUser_ = true;
    }

    label->setPosition(position * cover->getBaseMat()); // always do this -> needs update when camera changes

    if (this->wasStopped()) // interactor
    {
        adaptPositionToGrid();
        p_position->setValue(this->getPos()); // interactor
        ChargedObjectHandler::Instance()->dirtyField();
    }
}

void ChargedObject::setActive(bool active)
{
    bool changed = (this->active != active);
    this->active = active;

    if (changed)
    {
        // vr-prepare
        p_active->setValue(active);
        // geometry and menu
        if (active)
        {
            this->show(); // interactor
            this->enableIntersection();
            ElectricFieldPlugin::plugin->getObjectsMenu()->add(menuItemSeparator);
            ElectricFieldPlugin::plugin->getObjectsMenu()->add(menuItemCaption);
            ElectricFieldPlugin::plugin->getObjectsMenu()->add(menuItemDelete);
            ElectricFieldPlugin::plugin->getObjectsMenu()->add(menuItemCharge);
        }
        if (!active)
        {
            this->hide(); // interactor
            this->disableIntersection();
            ElectricFieldPlugin::plugin->getObjectsMenu()->remove(menuItemSeparator);
            ElectricFieldPlugin::plugin->getObjectsMenu()->remove(menuItemCaption);
            ElectricFieldPlugin::plugin->getObjectsMenu()->remove(menuItemDelete);
            ElectricFieldPlugin::plugin->getObjectsMenu()->remove(menuItemCharge);
        }
        ChargedObjectHandler::Instance()->objectsActiveStateChanged();
        activeStateChanged(); // let subclasses do relevant stuff
    }
}

void ChargedObject::adaptPositionToGrid()
{
    if (type_id == TYPE_PLATE)
    {
        float grid_max = ChargedObjectHandler::Instance()->getGridMax();
        float grid_min = ChargedObjectHandler::Instance()->getGridMin();
        int grid_steps = ChargedObjectHandler::Instance()->getGridSteps();

        float d = (grid_max - grid_min) / (float)grid_steps;
        int n = 0;
        n = (int)(0.5 + ((position)[0] - grid_min) / d);
        position[0] = grid_min + n * d;

        setPosition(position);
    }
}

void ChargedObject::setPosition(osg::Vec3 position)
{
    this->position = position;
    this->updateTransform(position); // interactor

    osg::Matrix m;
    m.makeTranslate(position);
}

void ChargedObject::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == menuItemDelete)
    {
        setActive(false);
        ChargedObjectHandler::Instance()->dirtyField();
        changedFromUser_ = false;
    }
    if (menuItem == menuItemCharge)
    {
        setCharge(menuItemCharge->getValue());
        changedFromUser_ = true;
    }
}

void ChargedObject::menuReleaseEvent(coMenuItem *menuItem)
{
    if (menuItem == menuItemCharge)
    {
        ChargedObjectHandler::Instance()->dirtyField();
    }
}

void ChargedObject::guiParamChanged(GuiParam *guiParam)
{
    if (guiParam == p_active)
    {
        setActive(p_active->getValue());
        ChargedObjectHandler::Instance()->dirtyField();
    }
    if ((guiParam == p_charge && ElectricFieldPlugin::plugin->presentationOn()) || (!changedFromUser_ && !ElectricFieldPlugin::plugin->presentationOn()))
    {
        setCharge(p_charge->getValue());
        ChargedObjectHandler::Instance()->dirtyField();
    }
    if ((guiParam == p_position && ElectricFieldPlugin::plugin->presentationOn()) || (!changedFromUser_ && !ElectricFieldPlugin::plugin->presentationOn()))
    {
        setPosition(p_position->getValue());
        ChargedObjectHandler::Instance()->dirtyField();
    }
}

void ChargedObject::setCharge(float charge)
{
    bool changed = (this->charge != charge);
    this->charge = charge;

    if (changed)
    {
        // vr-prepare
        p_charge->setValue(charge);
        // slider
        menuItemCharge->setValue(charge);
        // geometry
        setColorAccordingToCharge();
    }
}

void ChargedObject::setLabelVisibility(bool v)
{
    if (v)
    {
        label->show();
    }
    else
    {
        label->hide();
    }
}

void ChargedObject::setColorAccordingToCharge()
{
    if (charge >= 0)
    {
        objectMaterial->setDiffuse(osg::Material::FRONT, osg::Vec4(1.0, 0.0, 0.0, 1.0));
        objectMaterial->setAmbient(osg::Material::FRONT, osg::Vec4(0.5, 0.0, 0.0, 1.0));
    }
    else
    {
        objectMaterial->setDiffuse(osg::Material::FRONT, osg::Vec4(0.0, 0.0, 1.0, 1.0));
        objectMaterial->setAmbient(osg::Material::FRONT, osg::Vec4(0.0, 0.0, 0.5, 1.0));
    }
}
