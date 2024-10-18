/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "solarSystemNode.h"

void VrmlNodeSolarSystem::initFields(VrmlNodeSolarSystem *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
        eventOutCallBack("venusRotation", node->d_venusRotation),
        eventOutCallBack("venusTranslation", node->d_venusTranslation),
        eventOutCallBack("marsRotation", node->d_marsRotation),
        eventOutCallBack("marsTranslation", node->d_marsTranslation),
        eventOutCallBack("earthRotation", node->d_earthRotation),
        eventOutCallBack("earthTranslation", node->d_earthTranslation),
        eventOutCallBack("saturnRotation", node->d_saturnRotation),
        eventOutCallBack("saturnTranslation", node->d_saturnTranslation),
        eventOutCallBack("jupiterRotation", node->d_jupiterRotation),
        eventOutCallBack("jupiterTranslation", node->d_jupiterTranslation),
        eventOutCallBack("planetScale", node->d_planetScale)
    );
}

const char *VrmlNodeSolarSystem::name()
{
    return "SolarSystem";
}

VrmlNodeSolarSystem::VrmlNodeSolarSystem(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_venusRotation(1, 0, 0, 0)
    , d_venusTranslation(0, 0, 0)
    , d_marsRotation(1, 0, 0, 0)
    , d_marsTranslation(0, 0, 0)
    , d_earthRotation(1, 0, 0, 0)
    , d_earthTranslation(0, 0, 0)
    , d_saturnRotation(1, 0, 0, 0)
    , d_saturnTranslation(0, 0, 0)
    , d_jupiterRotation(1, 0, 0, 0)
    , d_jupiterTranslation(0, 0, 0)
    , d_comet_CG_Translation(0, 0, 0)
    , d_rosettaTranslation(0, 0, 0)
    , d_planetScale(1, 1, 1)
{
    setModified();
    inst = this;
}

VrmlNodeSolarSystem *VrmlNodeSolarSystem::inst = NULL;
;

VrmlNodeSolarSystem::VrmlNodeSolarSystem(const VrmlNodeSolarSystem &n)
    : VrmlNodeChild(n)
    , d_venusRotation(n.d_venusRotation)
    , d_venusTranslation(n.d_venusTranslation)
    , d_marsRotation(n.d_marsRotation)
    , d_marsTranslation(n.d_marsTranslation)
    , d_earthRotation(n.d_earthRotation)
    , d_earthTranslation(n.d_earthTranslation)
    , d_saturnRotation(n.d_saturnRotation)
    , d_saturnTranslation(n.d_saturnTranslation)
    , d_jupiterRotation(n.d_jupiterRotation)
    , d_jupiterTranslation(n.d_jupiterTranslation)
    , d_comet_CG_Translation(n.d_comet_CG_Translation)
    , d_rosettaTranslation(n.d_rosettaTranslation)
    , d_planetScale(n.d_planetScale)
{
    setModified();
}

VrmlNodeSolarSystem *VrmlNodeSolarSystem::toSolarSystemWheel() const
{
    return (VrmlNodeSolarSystem *)this;
}

void VrmlNodeSolarSystem::eventIn(double timeStamp,
                                  const char *eventName,
                                  const VrmlField *fieldValue)
{

    VrmlNode::eventIn(timeStamp, eventName, fieldValue);

    setModified();
}

void VrmlNodeSolarSystem::setVenusPosition(double x, double y, double z)
{
    timeStamp = System::the->time();
    d_venusTranslation.set(x, y, z);
    eventOut(timeStamp, "venusTranslation", d_venusTranslation);
}

void VrmlNodeSolarSystem::setMarsPosition(double x, double y, double z)
{
    timeStamp = System::the->time();
    d_marsTranslation.set(x, y, z);
    eventOut(timeStamp, "marsTranslation", d_marsTranslation);
}
void VrmlNodeSolarSystem::setEarthPosition(double x, double y, double z)
{
    timeStamp = System::the->time();
    d_earthTranslation.set(x, y, z);
    eventOut(timeStamp, "earthTranslation", d_earthTranslation);
}
void VrmlNodeSolarSystem::setSaturnPosition(double x, double y, double z)
{
    timeStamp = System::the->time();
    d_saturnTranslation.set(x, y, z);
    eventOut(timeStamp, "saturnTranslation", d_saturnTranslation);
}
void VrmlNodeSolarSystem::setJupiterPosition(double x, double y, double z)
{
    timeStamp = System::the->time();
    d_jupiterTranslation.set(x, y, z);
    eventOut(timeStamp, "jupiterTranslation", d_jupiterTranslation);
}
void VrmlNodeSolarSystem::setComet_CG_Position(double x, double y, double z)
{
    timeStamp = System::the->time();
    d_comet_CG_Translation.set(x, y, z);
    eventOut(timeStamp, "comet_CG_Translation", d_comet_CG_Translation);
}
void VrmlNodeSolarSystem::setRosettaPosition(double x, double y, double z)
{
    timeStamp = System::the->time();
    d_rosettaTranslation.set(x, y, z);
    eventOut(timeStamp, "rosettaTranslation", d_rosettaTranslation);
}
void VrmlNodeSolarSystem::setPlanetScale(double s)
{
    timeStamp = System::the->time();
    d_planetScale.set(s, s, s);
    eventOut(timeStamp, "planetScale", d_planetScale);
}
