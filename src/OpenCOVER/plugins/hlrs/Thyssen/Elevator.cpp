/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "Elevator.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>

using namespace covise;


static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeElevator(scene);
}

// Define the built in VrmlNodeType:: "Elevator" fields

VrmlNodeType *VrmlNodeElevator::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Elevator", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addEventOut("car0Pos", VrmlField::SFVEC3F);
    t->addEventOut("car1Pos", VrmlField::SFVEC3F);
    t->addEventOut("car2Pos", VrmlField::SFVEC3F);
    t->addEventOut("ints_changed", VrmlField::MFINT32);
    t->addEventOut("floats_changed", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeElevator::nodeType() const
{
    return defineType(0);
}

VrmlNodeElevator::VrmlNodeElevator(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    setModified();
}

VrmlNodeElevator::VrmlNodeElevator(const VrmlNodeElevator &n)
    : VrmlNodeChild(n.d_scene)
{
    setModified();
}

VrmlNodeElevator::~VrmlNodeElevator()
{
}

VrmlNode *VrmlNodeElevator::cloneMe() const
{
    return new VrmlNodeElevator(*this);
}

VrmlNodeElevator *VrmlNodeElevator::toElevator() const
{
    return (VrmlNodeElevator *)this;
}

ostream &VrmlNodeElevator::printFields(ostream &os, int indent)
{

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeElevator::setField(const char *fieldName,
                               const VrmlField &fieldValue)
{
    //if
    //    TRY_FIELD(enabled, SFBool)
    //else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeElevator::getField(const char *fieldName)
{
    //if (strcmp(fieldName, "floats_changed") == 0)
    //    return &d_floats;
    //else if (strcmp(fieldName, "ints_changed") == 0)
    //    return &d_ints;
    //else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeElevator::eventIn(double timeStamp,
                              const char *eventName,
                              const VrmlField *fieldValue)
{

    // Check exposedFields
    //else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void VrmlNodeElevator::render(Viewer *)
{
    
 
}

