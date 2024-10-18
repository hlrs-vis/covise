/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include <osg/LineSegment>
#include <osg/MatrixTransform>

#include "RemoteVehicle.h"
#include "SteeringWheel.h"
#include <OpenVRUI/osg/mathUtils.h>


void VrmlNodeRemoteVehicle::initFields(VrmlNodeRemoteVehicle *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); 
    initFieldsHelper(node, t,
                    exposedField("carRotation", node->d_carRotation),
                    exposedField("carTranslation", node->d_carTranslation),
                    exposedField("carBodyRotation", node->d_carBodyRotation),
                    exposedField("carBodyTranslation", node->d_carBodyTranslation));
}

const char *VrmlNodeRemoteVehicle::name()
{
    return "RemoteVehicle";
}

VrmlNodeRemoteVehicle::VrmlNodeRemoteVehicle(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_carRotation(1, 0, 0, 0)
    , d_carTranslation(0, 0, 0)
    , d_carBodyRotation(1, 0, 0, 0)
    , d_carBodyTranslation(0, 0, 0)
{
    clearModified();
    assert(!singleton);
    singleton = this;
}

VrmlNodeRemoteVehicle::VrmlNodeRemoteVehicle(const VrmlNodeRemoteVehicle &n)
    : VrmlNodeChild(n)
    , d_carRotation(n.d_carRotation)
    , d_carTranslation(n.d_carTranslation)
    , d_carBodyRotation(n.d_carBodyRotation)
    , d_carBodyTranslation(n.d_carBodyTranslation)
{
    clearModified();
    singleton = this;
}

VrmlNodeRemoteVehicle *VrmlNodeRemoteVehicle::toRemoteVehicle() const
{
    return (VrmlNodeRemoteVehicle *)this;
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
