/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//

#include "Landing.h"
#include "Car.h"
#include "Elevator.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>

using namespace covise;

void VrmlNodeLanding::initFields(VrmlNodeLanding *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t, exposedField("LandingNumber", node->d_LandingNumber));
    if(t)
    {
        t->addEventOut("doorClose", VrmlField::SFTIME);
        t->addEventOut("doorOpen", VrmlField::SFTIME);
    }
}

const char *VrmlNodeLanding::name()
{
    return "Landing";
}

VrmlNodeLanding::VrmlNodeLanding(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
{
    state=Uninitialized;
    currentCar = NULL;
}

VrmlNodeLanding::VrmlNodeLanding(const VrmlNodeLanding &n)
    : VrmlNodeChild(n)
{
    state=Uninitialized;
    currentCar = NULL;
}

VrmlNodeLanding *VrmlNodeLanding::toLanding() const
{
    return (VrmlNodeLanding *)this;
}

void VrmlNodeLanding::eventIn(double timeStamp,
                          const char *eventName,
                          const VrmlField *fieldValue)
{
    //if (strcmp(eventName, "LandingNumber"))
    // {
    //}
    // Check exposedFields
    //else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

}

void VrmlNodeLanding::render(Viewer *)
{


}

void VrmlNodeLanding::update()
{
    

}

int VrmlNodeLanding::getCarNumber()
{
    if(currentCar!=NULL)
        return currentCar->getID();
    return -1;
}
void VrmlNodeLanding::setCar(VrmlNodeCar *c)
{
    if(c)
    {
        state = Occupied;
    }
    else
    {
        state = Idle;
    }
    currentCar = c;
}

void VrmlNodeLanding::setElevator(VrmlNodeElevator *e)
{
    elevator = e;
    state = Idle;

}

void VrmlNodeLanding::openDoor()
{
    d_doorOpen = System::the->time();
    eventOut(d_doorOpen.get(), "doorOpen", d_doorOpen);
}
void VrmlNodeLanding::closeDoor()
{

    d_doorClose = System::the->time();
    eventOut(d_doorClose.get(), "doorClose", d_doorClose);
}
