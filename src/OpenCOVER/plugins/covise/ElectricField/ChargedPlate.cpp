/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ChargedPlate.h"
#include <iostream>
#include <algorithm>

#include "cover/coTranslator.h"

#include "ElectricFieldPlugin.h"
#include "ChargedObjectHandler.h"

using namespace std;

ChargedPlate::ChargedPlate(std::string name, float initialCharge)
    : ChargedObject(TYPE_PLATE, name, initialCharge)
    , otherPlate(NULL)
    , radius(0.5)
{
    // interactor
    createGeometry();

    // menu
    //   menuItemRadius = new coSliderMenuItem("Radius in cm", 10.0, 100.0, 50.0);
    //   menuItemRadius->setMenuListener(this);

    // vr-prepare
    p_radius = addGuiParamFloat(coTranslator::coTranslate("Radius"), radius);
}

ChargedPlate::~ChargedPlate()
{
}

void ChargedPlate::setOtherPlate(ChargedPlate *plate)
{
    this->otherPlate = plate;
}

void ChargedPlate::activeStateChanged()
{
    /*
   if (isActive())
   {
      ElectricFieldPlugin::plugin->getObjectsMenu()->add(menuItemRadius);
   } else {
      ElectricFieldPlugin::plugin->getObjectsMenu()->remove(menuItemRadius);
   }
	*/
}

void ChargedPlate::menuEvent(coMenuItem *menuItem)
{
    ChargedObject::menuEvent(menuItem);
    /*
   if (menuItem == menuItemRadius)
   {
      setRadius(menuItemRadius->getValue()/100.0);
   }*/
}

void ChargedPlate::menuReleaseEvent(coMenuItem *menuItem)
{
    ChargedObject::menuReleaseEvent(menuItem);
    /*
   if (menuItem == menuItemRadius)
   {
      ChargedObjectHandler::Instance()->dirtyField();
   }*/
}

void ChargedPlate::guiParamChanged(GuiParam *guiParam)
{
    ChargedObject::guiParamChanged(guiParam);
    if ((guiParam == p_radius && ElectricFieldPlugin::plugin->presentationOn()) || (!changedFromUser_ && !ElectricFieldPlugin::plugin->presentationOn()))
    {
        setRadius(p_radius->getValue());
        ChargedObjectHandler::Instance()->dirtyField();
    }
}

void ChargedPlate::setCharge(float charge)
{
    bool changed = (charge != this->charge); // avoid loops between the plates
    ChargedObject::setCharge(charge);
    if (changed)
        otherPlate->setCharge(-charge);
}

void ChargedPlate::setRadius(float radius, bool fromUser)
{
    bool changed = (radius != this->radius); // avoid loops between the plates
    this->radius = radius;

    if (changed)
    {
        // vr-prepare
        p_radius->setValue(radius);
        // slider
        ElectricFieldPlugin::plugin->setRadiusOfPlates(radius);
        // geometry
        plateCylinder->setRadius(radius);
        plateDrawable->dirtyDisplayList();
    }

    if (fromUser)
        changedFromUser_ = true;

    if (changed)
        otherPlate->setRadius(radius);
}

void ChargedPlate::createGeometry()
{
    // remove old geometry
    scaleTransform->removeChild(0, scaleTransform->getNumChildren());
    // create new goemtry
    geometryNode = new osg::Geode();
    plateCylinder = new osg::Cylinder(position, radius, 0.05);
    osg::Matrix m;
    m.makeRotate(PI * 0.5, osg::Vec3(0.0, 1.0, 0.0));
    plateCylinder->setRotation(m.getRotate());
    plateDrawable = new osg::ShapeDrawable(plateCylinder.get());
    ((osg::Geode *)this->geometryNode.get())->addDrawable(plateDrawable.get());
    ((osg::Geode *)this->geometryNode.get())->getOrCreateStateSet()->setAttribute(objectMaterial.get());
    // add
    scaleTransform->addChild(geometryNode.get());
}

///////////////////////////////////////////////////////////////////////////////

osg::Vec3 ChargedPlate::getFieldAt(osg::Vec3 point)
{
    float minLen = radius * 0.05;
    osg::Vec3 direction = osg::Vec3(0.0, position[1] - point[1], position[2] - point[2]);
    if (direction.length2() == 0.0)
        direction = osg::Vec3(0.0, 0.0, 1.0);
    else
        direction.normalize();
    direction *= radius / float(NUM_APPROXIMATION_POINTS / 2);
    osg::Vec3 field = osg::Vec3(0.0, 0.0, 0.0);
    for (int progress = -NUM_APPROXIMATION_POINTS / 2; progress < NUM_APPROXIMATION_POINTS / 2; ++progress)
    {
        osg::Vec3 tmp_position = position + direction * float(progress);
        osg::Vec3 vector = point - tmp_position;
        float len = vector.length();
        len = max(len, minLen);
        field += (vector / (len * len * len));
    }
    return field * (EPSILON_POINT * charge / float(NUM_APPROXIMATION_POINTS));
}

float ChargedPlate::getPotentialAt(osg::Vec3 point)
{
    float minLen = radius * 0.05;
    osg::Vec3 direction = osg::Vec3(0.0, position[1] - point[1], position[2] - point[2]);
    if (direction.length2() == 0.0)
        direction = osg::Vec3(0.0, 0.0, 1.0);
    else
        direction.normalize();
    direction *= radius / float(NUM_APPROXIMATION_POINTS / 2);
    float potential = 0.0;
    for (int progress = -NUM_APPROXIMATION_POINTS / 2; progress < NUM_APPROXIMATION_POINTS / 2; ++progress)
    {
        osg::Vec3 tmp_position = position + direction * float(progress);
        osg::Vec3 vector = point - tmp_position;
        float len = vector.length();
        len = max(len, minLen);
        potential += 1.0 / len;
    }
    //    if ((fabs(point[0]-position[0]) < 0.01) && ((osg::Vec3(position[0], point[1], point[2])-position).length() < radius))
    //    {
    //       potential = 0.0;
    //    }
    return potential * (EPSILON_POINT * charge / float(NUM_APPROXIMATION_POINTS));
}

osg::Vec4 ChargedPlate::getFieldAndPotentialAt(osg::Vec3 point)
{
    float minLen = radius * 0.05;
    osg::Vec3 direction = osg::Vec3(0.0, position[1] - point[1], position[2] - point[2]);
    if (direction.length2() == 0.0)
        direction = osg::Vec3(0.0, 0.0, 1.0);
    else
        direction.normalize();
    direction *= radius / float(NUM_APPROXIMATION_POINTS / 2);
    osg::Vec4 fieldAndPotential = osg::Vec4(0.0, 0.0, 0.0, 0.0);
    for (int progress = -NUM_APPROXIMATION_POINTS / 2; progress < NUM_APPROXIMATION_POINTS / 2; ++progress)
    {
        osg::Vec3 tmp_position = position + direction * float(progress);
        osg::Vec3 vector = point - tmp_position;
        float len = vector.length();
        len = max(len, minLen);
        fieldAndPotential += osg::Vec4(
            (vector / (len * len * len)),
            1.0 / len);
    }
    //    if ((fabs(point[0]-position[0]) < 0.01) && ((osg::Vec3(position[0], point[1], point[2])-position).length() < radius))
    //    {
    //       fieldAndPotential[0] = 0.0;
    //       fieldAndPotential[1] = 0.0;
    //       fieldAndPotential[2] = 0.0;
    //    }
    return fieldAndPotential * (EPSILON_POINT * charge / float(NUM_APPROXIMATION_POINTS));
}

void ChargedPlate::correctField(osg::Vec3 point, osg::Vec3 &field)
{
    // make sure we are between the plates
    if (((point[0] <= position[0]) && (point[0] <= otherPlate->position[0])) || ((point[0] >= position[0]) && (point[0] >= otherPlate->position[0])))
        return;
    // calculate correct field
    osg::Vec3 normal = osg::Vec3(1.0, 0.0, 0.0);
    if (position[0] > otherPlate->position[0])
        normal = osg::Vec3(-1.0, 0.0, 0.0);
    osg::Vec3 correctField = normal * (charge / (PI * EPSILON_0 * EPSILON_R * radius * radius));
    // blend values
    //   At the border of the plates, we only have the approximated field (percentCorrect=0).
    //   The influence of the correct field grows as we go towards the middle of the plates.
    //   The line from where on we only have the correct one (percentCorrect=1) is <distance of the plates> away from the border.
    float platesDistance = fabs(position[0] - otherPlate->position[0]);
    float borderDistance = radius - (osg::Vec3(position[0], point[1], point[2]) - position).length();
    float percentCorrect = max(0.0f, min(1.0f, borderDistance / max(platesDistance, 0.0001f)));
    field = field * (1 - percentCorrect) + correctField * percentCorrect;
}

void ChargedPlate::correctPotential(osg::Vec3 point, float &potential)
{
    // make sure we are between the plates
    if (((point[0] <= position[0]) && (point[0] <= otherPlate->position[0])) || ((point[0] >= position[0]) && (point[0] >= otherPlate->position[0])))
        return;
    // calculate correct potential
    float s = (position[0] + otherPlate->position[0]) / 2 - point[0]; // potential is 0 in the middle of the plates
    if (position[0] > otherPlate->position[0])
        s *= -1.0;
    float correctPotential = s * (charge / (PI * EPSILON_0 * EPSILON_R * radius * radius));
    // blend values
    //   At the border of the plates, we only have the approximated field (percentCorrect=0).
    //   The influence of the correct field grows as we go towards the middle of the plates.
    //   The line from where on we only have the correct one (percentCorrect=1) is <distance of the plates> away from the border.
    float platesDistance = fabs(position[0] - otherPlate->position[0]);
    float borderDistance = radius - (osg::Vec3(position[0], point[1], point[2]) - position).length();
    float percentCorrect = max(0.0f, min(1.0f, borderDistance / max(platesDistance, 0.0001f)));
    potential = potential * (1 - percentCorrect) + correctPotential * percentCorrect;
}

void ChargedPlate::correctFieldAndPotential(osg::Vec3 point, osg::Vec4 &fieldAndPotential)
{
    // make sure we are between the plates
    if (((point[0] <= position[0]) && (point[0] <= otherPlate->position[0])) || ((point[0] >= position[0]) && (point[0] >= otherPlate->position[0])))
        return;
    // calculate correct field and potential
    osg::Vec3 normal = osg::Vec3(1.0, 0.0, 0.0);
    float s = (position[0] + otherPlate->position[0]) / 2 - point[0]; // potential is 0 in the middle of the plates
    if (position[0] > otherPlate->position[0])
    {
        normal = osg::Vec3(-1.0, 0.0, 0.0);
        s *= -1.0;
    }
    osg::Vec4 correctFieldAndPotential = osg::Vec4(
        normal * (charge / (PI * EPSILON_0 * EPSILON_R * radius * radius)),
        s * (charge / (PI * EPSILON_0 * EPSILON_R * radius * radius)));
    // blend values
    //   At the border of the plates, we only have the approximated field (percentCorrect=0).
    //   The influence of the correct field grows as we go towards the middle of the plates.
    //   The line from where on we only have the correct one (percentCorrect=1) is <distance of the plates> away from the border.
    float platesDistance = fabs(position[0] - otherPlate->position[0]);
    float borderDistance = radius - (osg::Vec3(position[0], point[1], point[2]) - position).length();
    float percentCorrect = max(0.0f, min(1.0f, borderDistance / max(platesDistance, 0.0001f)));
    fieldAndPotential = fieldAndPotential * (1 - percentCorrect) + correctFieldAndPotential * percentCorrect;
}
