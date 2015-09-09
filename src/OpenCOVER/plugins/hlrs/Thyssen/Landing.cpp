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


static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeLanding(scene);
}

// Define the built in VrmlNodeType:: "Landing" fields

VrmlNodeType *VrmlNodeLanding::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Landing", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    
    t->addExposedField("LandingNumber", VrmlField::SFINT32);
    t->addEventOut("doorClose", VrmlField::SFTIME);
    t->addEventOut("doorOpen", VrmlField::SFTIME);

    return t;
}

VrmlNodeType *VrmlNodeLanding::nodeType() const
{
    return defineType(0);
}

VrmlNodeLanding::VrmlNodeLanding(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    state=Uninitialized;
    currentCar = NULL;
}

VrmlNodeLanding::VrmlNodeLanding(const VrmlNodeLanding &n)
    : VrmlNodeChild(n.d_scene)
{
    state=Uninitialized;
    currentCar = NULL;
}

VrmlNodeLanding::~VrmlNodeLanding()
{
}

VrmlNode *VrmlNodeLanding::cloneMe() const
{
    return new VrmlNodeLanding(*this);
}

VrmlNodeLanding *VrmlNodeLanding::toLanding() const
{
    return (VrmlNodeLanding *)this;
}

ostream &VrmlNodeLanding::printFields(ostream &os, int indent)
{

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeLanding::setField(const char *fieldName,
                           const VrmlField &fieldValue)
{
    if
        TRY_FIELD(LandingNumber, SFInt)
    else
    VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeLanding::getField(const char *fieldName)
{
    if (strcmp(fieldName, "LandingNumber") == 0)
        return &d_LandingNumber;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
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
