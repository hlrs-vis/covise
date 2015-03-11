/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//

#include "Elevator.h"
#include "Car.h"

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

    VrmlNodeGroup::defineType(t); // Parent class
    t->addExposedField("landingHeights", VrmlField::MFFLOAT);
    t->addExposedField("shaftPositions", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeElevator::nodeType() const
{
    return defineType(0);
}

VrmlNodeElevator::VrmlNodeElevator(VrmlScene *scene)
    : VrmlNodeGroup(scene)
{
    setModified();
}

VrmlNodeElevator::VrmlNodeElevator(const VrmlNodeElevator &n)
    : VrmlNodeGroup(n.d_scene)
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
    if
        TRY_FIELD(landingHeights, MFFloat)
    else if
    TRY_FIELD(shaftPositions, MFFloat)
    else
    VrmlNodeGroup::setField(fieldName, fieldValue);
    
    if(strcmp(fieldName,"children")==0)
    {
        for(int i=0;i<d_children.size();i++)
        {
            VrmlNodeCar *car = dynamic_cast<VrmlNodeCar *>(d_children[i]);
            car->setElevator(this);
        }
    }
}

const VrmlField *VrmlNodeElevator::getField(const char *fieldName)
{
    if (strcmp(fieldName, "landingHeights") == 0)
        return &d_landingHeights;
    else if (strcmp(fieldName, "shaftPositions") == 0)
        return &d_shaftPositions;
    else
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
    for(int i=0;i<d_children.size();i++)
    {
        VrmlNodeCar *car = dynamic_cast<VrmlNodeCar *>(d_children[i]);
        if(car->getState()==VrmlNodeCar::Idle)
        {
            // tell it to move to next stop
            if(d_shaftPositions.size()==2)
            {
                if(car->d_carPos.x()==d_shaftPositions[0]) // left shat, move up
                {
                    if(car->getLandingNumber() == d_landingHeights.size()-1) // we are on top
                        car->setDestination(car->getLandingNumber(),1);
                    else
                        car->setDestination(car->getLandingNumber()+1,0);
                }
                else // right shaft, move down
                {
                    if(car->getLandingNumber() == 0) // we are on the lowest level
                        car->setDestination(car->getLandingNumber(),0);
                    else
                        car->setDestination(car->getLandingNumber()-1,1);
                }
            }
        }
        car->update();
    }

}

