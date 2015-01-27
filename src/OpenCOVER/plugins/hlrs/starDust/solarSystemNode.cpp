/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "solarSystemNode.h"

static VrmlNode *creatorSolarSystem(VrmlScene *scene)
{
    if (VrmlNodeSolarSystem::instance())
        return VrmlNodeSolarSystem::instance();
    return new VrmlNodeSolarSystem(scene);
}

// Define the built in VrmlNodeType:: "SteeringWheel" fields

VrmlNodeType *VrmlNodeSolarSystem::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("SolarSystem", creatorSolarSystem);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addEventOut("venusRotation", VrmlField::SFROTATION);
    t->addEventOut("venusTranslation", VrmlField::SFVEC3F);
    t->addEventOut("marsRotation", VrmlField::SFROTATION);
    t->addEventOut("marsTranslation", VrmlField::SFVEC3F);
    t->addEventOut("earthRotation", VrmlField::SFROTATION);
    t->addEventOut("earthTranslation", VrmlField::SFVEC3F);
    t->addEventOut("saturnRotation", VrmlField::SFROTATION);
    t->addEventOut("saturnTranslation", VrmlField::SFVEC3F);
    t->addEventOut("jupiterRotation", VrmlField::SFROTATION);
    t->addEventOut("jupiterTranslation", VrmlField::SFVEC3F);
    t->addEventOut("planetScale", VrmlField::SFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodeSolarSystem::nodeType() const
{
    return defineType(0);
}

VrmlNodeSolarSystem::VrmlNodeSolarSystem(VrmlScene *scene)
    : VrmlNodeChild(scene)
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
    : VrmlNodeChild(n.d_scene)
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

VrmlNodeSolarSystem::~VrmlNodeSolarSystem()
{
}

VrmlNode *VrmlNodeSolarSystem::cloneMe() const
{
    return new VrmlNodeSolarSystem(*this);
}

VrmlNodeSolarSystem *VrmlNodeSolarSystem::toSolarSystemWheel() const
{
    return (VrmlNodeSolarSystem *)this;
}

ostream &VrmlNodeSolarSystem::printFields(ostream &os, int indent)
{
    if (!d_venusRotation.get())
        PRINT_FIELD(venusRotation);
    if (!d_venusTranslation.get())
        PRINT_FIELD(venusTranslation);
    if (!d_marsRotation.get())
        PRINT_FIELD(marsRotation);
    if (!d_marsTranslation.get())
        PRINT_FIELD(marsTranslation);
    if (!d_earthRotation.get())
        PRINT_FIELD(earthRotation);
    if (!d_earthTranslation.get())
        PRINT_FIELD(earthTranslation);
    if (!d_saturnRotation.get())
        PRINT_FIELD(saturnRotation);
    if (!d_saturnTranslation.get())
        PRINT_FIELD(saturnTranslation);
    if (!d_jupiterRotation.get())
        PRINT_FIELD(jupiterRotation);
    if (!d_jupiterTranslation.get())
        PRINT_FIELD(jupiterTranslation);
    if (!d_comet_CG_Translation.get())
        PRINT_FIELD(comet_CG_Translation);
    if (!d_rosettaTranslation.get())
        PRINT_FIELD(rosettaTranslation);

    if (!d_planetScale.get())
        PRINT_FIELD(planetScale);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeSolarSystem::setField(const char *fieldName,
                                   const VrmlField &fieldValue)
{
    if
        TRY_FIELD(venusRotation, SFRotation)
    else if
        TRY_FIELD(venusTranslation, SFVec3f)
    else if
        TRY_FIELD(marsRotation, SFRotation)
    else if
        TRY_FIELD(marsTranslation, SFVec3f)
    else if
        TRY_FIELD(earthRotation, SFRotation)
    else if
        TRY_FIELD(earthTranslation, SFVec3f)
    else if
        TRY_FIELD(saturnRotation, SFRotation)
    else if
        TRY_FIELD(saturnTranslation, SFVec3f)
    else if
        TRY_FIELD(jupiterRotation, SFRotation)
    else if
        TRY_FIELD(jupiterTranslation, SFVec3f)
    else if
        TRY_FIELD(planetScale, SFVec3f)
}

const VrmlField *VrmlNodeSolarSystem::getField(const char *fieldName)
{
    if (strcmp(fieldName, "venusRotation") == 0)
        return &d_venusRotation;
    else if (strcmp(fieldName, "venusTranslation") == 0)
        return &d_venusTranslation;
    else if (strcmp(fieldName, "marsRotation") == 0)
        return &d_marsRotation;
    else if (strcmp(fieldName, "marsTranslation") == 0)
        return &d_marsTranslation;
    else if (strcmp(fieldName, "earthRotation") == 0)
        return &d_earthRotation;
    else if (strcmp(fieldName, "earthTranslation") == 0)
        return &d_earthTranslation;
    else if (strcmp(fieldName, "saturnRotation") == 0)
        return &d_saturnRotation;
    else if (strcmp(fieldName, "saturnTranslation") == 0)
        return &d_saturnTranslation;
    else if (strcmp(fieldName, "jupiterRotation") == 0)
        return &d_jupiterRotation;
    else if (strcmp(fieldName, "planetScale") == 0)
        return &d_planetScale;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
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
