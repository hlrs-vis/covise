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
    bool allIdle = true;
    for(int i=0;i<d_children.size();i++)
    {
        VrmlNodeCar *car = dynamic_cast<VrmlNodeCar *>(d_children[i]);
        if(car!=NULL)
        {
            int lowerLanding=0;
            if(car->getID()%2)
                lowerLanding = 1;
            int upperLanding=d_landingHeights.size()-1;
            if(car->getID()%2)
                upperLanding--;
            if(car->getState()!=VrmlNodeCar::Idle)
            {
                allIdle = false;
            }
            if(car->d_carPos.x()==d_shaftPositions[0] && car->getLandingNumber() == upperLanding && car->getState()==VrmlNodeCar::DoorOpening) // we are on top and want to move right
            {
             
                if(car->getTravelDirection()!=VrmlNodeCar::MoveRight)
                {
                    car->setTravelDirection(VrmlNodeCar::MoveRight);
                }
            }
            if(car->d_carPos.x()==d_shaftPositions[1] && car->getLandingNumber() == lowerLanding && car->getState()==VrmlNodeCar::DoorOpening) // we are on top and want to move right
            {
                
                if(car->getTravelDirection()!=VrmlNodeCar::MoveLeft)
                {
                    car->setTravelDirection(VrmlNodeCar::MoveLeft);
                }
            }
        }
    }
    for(int i=0;i<d_children.size();i++)
    {
        VrmlNodeCar *car = dynamic_cast<VrmlNodeCar *>(d_children[i]);
        if(car!=NULL)
        {
            int lowerLanding=0;
            if(car->getID()%2)
                lowerLanding = 1;
            int upperLanding=d_landingHeights.size()-1;
            if(car->getID()%2)
                upperLanding--;
            if(allIdle)
            {
                // tell it to move to next stop
                car->moveToNext();
                /*
                if(d_shaftPositions.size()==2)
                {
                    if(car->d_carPos.x()==d_shaftPositions[0]) // left shat, move up
                    {
                        if(car->getLandingNumber() ==upperLanding) // we are on top
                        {
                            car->setDestination(car->getLandingNumber(),1);
                            car->setTravelDirection(VrmlNodeCar::MoveDown);
                        }
                        else
                            car->setDestination(car->getLandingNumber()+1,0);
                    }
                    else // right shaft, move down
                    {
                        if(car->getLandingNumber() == lowerLanding) // we are on the lowest level
                        {
                            car->setDestination(car->getLandingNumber(),0);
                            car->setTravelDirection(VrmlNodeCar::MoveUp);
                        }
                        else
                            car->setDestination(car->getLandingNumber()-1,1);
                    }
                }*/
            }
            car->update();
        }
    }

}

