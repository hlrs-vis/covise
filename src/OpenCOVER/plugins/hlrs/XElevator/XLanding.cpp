/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//

#include "XLanding.h"
#include "XCar.h"
#include "XElevator.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>

using namespace covise;


void VrmlNodeXLanding::initFields(VrmlNodeXLanding *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
        exposedField("LandingNumber", node->d_LandingNumber));
    if(t)
    {
        t->addEventIn("callButton", VrmlField::SFTIME);
        t->addEventOut("doorClose", VrmlField::SFTIME);
        t->addEventOut("doorOpen", VrmlField::SFTIME);
    }

}

const char *VrmlNodeXLanding::name()
{
    return "XLanding";
}

VrmlNodeXLanding::VrmlNodeXLanding(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
{
    state=Uninitialized;
    currentCar = NULL;
}

VrmlNodeXLanding::VrmlNodeXLanding(const VrmlNodeXLanding &n)
    : VrmlNodeChild(n)
{
    state=Uninitialized;
    currentCar = NULL;
}

VrmlNodeXLanding *VrmlNodeXLanding::toXLanding() const
{
    return (VrmlNodeXLanding *)this;
}

void VrmlNodeXLanding::eventIn(double timeStamp,
                          const char *eventName,
                          const VrmlField *fieldValue)
{
    //if (strcmp(eventName, "LandingNumber"))
    // {
    //}
    // Check exposedFields
    //else
    if (strcmp(eventName, "callButton")==0)
    {
        Elevator->goTo(d_LandingNumber.get());
    }

    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

}

void VrmlNodeXLanding::render(Viewer *)
{


}

void VrmlNodeXLanding::update()
{
    

}

int VrmlNodeXLanding::getCarNumber()
{
    if(currentCar!=NULL)
        return currentCar->getID();
    return -1;
}
void VrmlNodeXLanding::setCar(VrmlNodeXCar *c)
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

void VrmlNodeXLanding::setElevator(VrmlNodeXElevator *e)
{
    Elevator = e;
    state = Idle;

}

void VrmlNodeXLanding::openDoor()
{
    d_doorOpen = System::the->time();
    eventOut(d_doorOpen.get(), "doorOpen", d_doorOpen);
}
void VrmlNodeXLanding::closeDoor()
{

    d_doorClose = System::the->time();
    eventOut(d_doorClose.get(), "doorClose", d_doorClose);
}
