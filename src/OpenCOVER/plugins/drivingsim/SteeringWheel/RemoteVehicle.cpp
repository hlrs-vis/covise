/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <osgUtil/IntersectVisitor>

#include "RemoteVehicle.h"
#include "SteeringWheel.h"
#include <OpenVRUI/osg/mathUtils.h>

// Define the built in VrmlNodeType:: "SteeringWheel" fields

static VrmlNode *creatorRemoteVehicle(VrmlScene *scene)
{
    if (VrmlNodeRemoteVehicle::instance())
        return VrmlNodeRemoteVehicle::instance();
    return new VrmlNodeRemoteVehicle(scene);
}

VrmlNodeType *VrmlNodeRemoteVehicle::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("RemoteVehicle", creatorRemoteVehicle);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("carRotation", VrmlField::SFROTATION);
    t->addExposedField("carTranslation", VrmlField::SFVEC3F);
    t->addExposedField("carBodyRotation", VrmlField::SFROTATION);
    t->addExposedField("carBodyTranslation", VrmlField::SFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodeRemoteVehicle::nodeType() const
{
    return defineType(0);
}

VrmlNodeRemoteVehicle::VrmlNodeRemoteVehicle(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_carRotation(1, 0, 0, 0)
    , d_carTranslation(0, 0, 0)
    , d_carBodyRotation(1, 0, 0, 0)
    , d_carBodyTranslation(0, 0, 0)
{
    clearModified();
    singleton = this;
}

VrmlNodeRemoteVehicle::VrmlNodeRemoteVehicle(const VrmlNodeRemoteVehicle &n)
    : VrmlNodeChild(n.d_scene)
    , d_carRotation(n.d_carRotation)
    , d_carTranslation(n.d_carTranslation)
    , d_carBodyRotation(n.d_carBodyRotation)
    , d_carBodyTranslation(n.d_carBodyTranslation)
{
    clearModified();
    singleton = this;
}

VrmlNodeRemoteVehicle::~VrmlNodeRemoteVehicle()
{
}

VrmlNode *VrmlNodeRemoteVehicle::cloneMe() const
{
    return new VrmlNodeRemoteVehicle(*this);
}

VrmlNodeRemoteVehicle *VrmlNodeRemoteVehicle::toRemoteVehicle() const
{
    return (VrmlNodeRemoteVehicle *)this;
}

ostream &VrmlNodeRemoteVehicle::printFields(ostream &os, int indent)
{
    if (!d_carRotation.get())
        PRINT_FIELD(carRotation);
    if (!d_carTranslation.get())
        PRINT_FIELD(carTranslation);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeRemoteVehicle::setField(const char *fieldName,
                                     const VrmlField &fieldValue)
{
    if
        TRY_FIELD(carRotation, SFRotation)
    else if
        TRY_FIELD(carTranslation, SFVec3f)
    else if
        TRY_FIELD(carBodyRotation, SFRotation)
    else if
        TRY_FIELD(carBodyTranslation, SFVec3f)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeRemoteVehicle::getField(const char *fieldName)
{
    if (strcmp(fieldName, "carRotation") == 0)
        return &d_carRotation;
    else if (strcmp(fieldName, "carTranslation") == 0)
        return &d_carTranslation;
    else if (strcmp(fieldName, "carBodyRotation") == 0)
        return &d_carBodyRotation;
    else if (strcmp(fieldName, "carBodyTranslation") == 0)
        return &d_carBodyTranslation;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeRemoteVehicle::eventIn(double timeStamp,
                                    const char *eventName,
                                    const VrmlField *fieldValue)
{
    if (strcmp(eventName, "carRotation") == 0)
    {
    }
    else if (strcmp(eventName, "carTranslation") == 0)
    {
    }
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void VrmlNodeRemoteVehicle::render(Viewer *)
{
}

/*
void VrmlNodeRemoteVehicle::setVRMLVehicleBody(int body, const osg::Matrix &trans)
{
      double timeStamp = System::the->time();
   osg::Quat qBody;
   qBody.set(trans);
   osg::Quat::value_type orientBody[4];
   qBody.getRotate(orientBody[3], orientBody[0], orientBody[1], orientBody[2]);
   d_bodyRotation[body].set(orientBody[0],orientBody[1],orientBody[2],orientBody[3]);  

   d_bodyTranslation[body].set(trans(3,0), trans(3,1), trans(3,2));  

   eventOut(timeStamp, bodyTransName[body], d_bodyTranslation[body]);
   eventOut(timeStamp, bodyRotName[body], d_bodyRotation[body]);
}
*/

VrmlNodeRemoteVehicle *VrmlNodeRemoteVehicle::singleton = NULL;

VrmlNodeRemoteVehicle *VrmlNodeRemoteVehicle::instance()
{
    return singleton;
}
void VrmlNodeRemoteVehicle::setVRMLVehicle(const osg::Matrix &trans)
{
    double timeStamp = System::the->time();
    osg::Quat q;
    q.set(trans);
    osg::Quat::value_type orient[4];
    q.getRotate(orient[3], orient[0], orient[1], orient[2]);
    d_carTranslation.set(trans(3, 0), trans(3, 1), trans(3, 2));
    d_carRotation.set(orient[0], orient[1], orient[2], orient[3]);

    eventOut(timeStamp, "carTranslation", d_carTranslation);
    eventOut(timeStamp, "carRotation", d_carRotation);
}
