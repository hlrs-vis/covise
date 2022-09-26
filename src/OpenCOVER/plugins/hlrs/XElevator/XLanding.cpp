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


static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeXLanding(scene);
}

// Define the built in VrmlNodeType:: "XLanding" fields

VrmlNodeType *VrmlNodeXLanding::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("XLanding", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    
    t->addExposedField("LandingNumber", VrmlField::SFINT32);
    t->addEventIn("callButton", VrmlField::SFTIME);
    t->addEventOut("doorClose", VrmlField::SFTIME);
    t->addEventOut("doorOpen", VrmlField::SFTIME);

    return t;
}

VrmlNodeType *VrmlNodeXLanding::nodeType() const
{
    return defineType(0);
}

VrmlNodeXLanding::VrmlNodeXLanding(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    state=Uninitialized;
    currentCar = NULL;
}

VrmlNodeXLanding::VrmlNodeXLanding(const VrmlNodeXLanding &n)
    : VrmlNodeChild(n.d_scene)
{
    state=Uninitialized;
    currentCar = NULL;
}

VrmlNodeXLanding::~VrmlNodeXLanding()
{
}

VrmlNode *VrmlNodeXLanding::cloneMe() const
{
    return new VrmlNodeXLanding(*this);
}

VrmlNodeXLanding *VrmlNodeXLanding::toXLanding() const
{
    return (VrmlNodeXLanding *)this;
}

ostream &VrmlNodeXLanding::printFields(ostream &os, int indent)
{

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeXLanding::setField(const char *fieldName,
                           const VrmlField &fieldValue)
{
    if
        TRY_FIELD(LandingNumber, SFInt)
    else
    VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeXLanding::getField(const char *fieldName)
{
    if (strcmp(fieldName, "LandingNumber") == 0)
        return &d_LandingNumber;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
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
