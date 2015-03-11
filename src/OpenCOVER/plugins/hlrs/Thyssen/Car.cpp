/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//

#include "Car.h"
#include "Elevator.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>

using namespace covise;


static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeCar(scene);
}

// Define the built in VrmlNodeType:: "Car" fields

VrmlNodeType *VrmlNodeCar::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Car", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class


    t->addExposedField("carNumber", VrmlField::SFINT32);
    t->addExposedField("carPos", VrmlField::SFVEC3F);
    t->addEventOut("carDoorClose", VrmlField::SFTIME);
    t->addEventOut("carDoorOpen", VrmlField::SFTIME);
    t->addEventOut("carAngle", VrmlField::SFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeCar::nodeType() const
{
    return defineType(0);
}

VrmlNodeCar::VrmlNodeCar(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    state=Uninitialized;
    aMax = 1;
    vMax = 5;
    v=0;a=0;
}

VrmlNodeCar::VrmlNodeCar(const VrmlNodeCar &n)
    : VrmlNodeChild(n.d_scene)
{
    state=Uninitialized;
    aMax = 1;
    vMax = 5;
    v=0;a=0;
}

VrmlNodeCar::~VrmlNodeCar()
{
}

VrmlNode *VrmlNodeCar::cloneMe() const
{
    return new VrmlNodeCar(*this);
}

VrmlNodeCar *VrmlNodeCar::toCar() const
{
    return (VrmlNodeCar *)this;
}

ostream &VrmlNodeCar::printFields(ostream &os, int indent)
{

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeCar::setField(const char *fieldName,
                           const VrmlField &fieldValue)
{

    if
        TRY_FIELD(carNumber, SFInt)
    else if
    TRY_FIELD(carPos, SFVec3f)
    else if
    TRY_FIELD(carDoorClose, SFTime)
    else if
    TRY_FIELD(carDoorOpen, SFTime)
    else if
    TRY_FIELD(carAngle, SFFloat)
    else
    VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeCar::getField(const char *fieldName)
{
    if (strcmp(fieldName, "carNumber") == 0)
        return &d_carNumber;
    else if (strcmp(fieldName, "carPos") == 0)
        return &d_carPos;
    else if (strcmp(fieldName, "carDoorClose") == 0)
        return &d_carDoorClose;
    else if (strcmp(fieldName, "carDoorOpen") == 0)
        return &d_carDoorOpen;
    else if (strcmp(fieldName, "carAngle") == 0)
        return &d_carAngle;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeCar::eventIn(double timeStamp,
                          const char *eventName,
                          const VrmlField *fieldValue)
{


    VrmlSFInt   d_carNumber;
    VrmlSFVec3f d_carPos;
    VrmlSFTime  d_carDoorClose;
    VrmlSFTime  d_carDoorOpen;
    VrmlSFFloat d_carAngle;

    //if (strcmp(eventName, "carNumber"))
    // {
    //}
    // Check exposedFields
    //else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

}

void VrmlNodeCar::render(Viewer *)
{


}

void VrmlNodeCar::update()
{
    if(state == Moving)
    {
        float dt = cover->frameDuration();
        if(dt > 1000) // first frameDuration is off because last FrameTime is 0
            dt=0.00001;
        if(d_carPos.x() != destinationX) //moving horizontally
        {
            float direction;
            float diff = fabs(destinationX - d_carPos.x());
            if(d_carPos.x() < destinationX)
                direction = 1;
            else
                direction = -1;
            float v2 = v*v;
            if(diff > (v2/(2*aMax))*1.5)
            { // beschleunigen
                a+=0.5*dt;
                if(a > aMax)
                    a=aMax;
                v += a*dt;
                if(v > vMax)
                    v=vMax;
                d_carPos.get()[0] += direction*v*dt;
            }
            else
            { // verzögern
                if(diff > 0.0001)
                {
                    a = v2/(2*diff);
                    v -= a*dt;
                }
                else
                {
                    a=0;v=0;
                }
                d_carPos.get()[0] += direction*v*dt;
                if(v <= 0)
                {
                    d_carPos.get()[0]=destinationX;
                    v=0;
                }
            }
            double timeStamp = System::the->time();
            eventOut(timeStamp, "carPos", d_carPos);
        }
        else if(d_carPos.y() != destinationY) // moving vertically
        {
            float direction;
            float diff = fabs(destinationY - d_carPos.y());
            if(d_carPos.y() < destinationY)
                direction = 1;
            else
                direction = -1;
            float v2 = v*v;
            if(diff > (v2/(2*aMax))*1.5)
            { // beschleunigen
                a+=0.5*dt;
                if(a > aMax)
                    a=aMax;
                v += a*dt;
                if(v > vMax)
                    v=vMax;
                d_carPos.get()[1] += direction*v*dt;
            }
            else
            { // verzögern
                if(diff > 0.0001)
                {
                    a = v2/(2*diff);
                    v -= a*dt;
                }
                else
                {
                    a=0;v=0;
                }
                if(v <= 0)
                {
                    d_carPos.get()[1]=destinationY;
                    v=0;
                }
                else
                {
                    d_carPos.get()[1] += direction*v*dt;
                }
            }
            if(!(v>=0) || !(a>=0) || !(v<10) || !(a<4))
            {
                fprintf(stderr,"oops\n");
            }
            double timeStamp = System::the->time();
            eventOut(timeStamp, "carPos", d_carPos);
        }
        else // we are there
        {
            timeoutStart = cover->frameTime();
            state = DoorOpening;
            v=0;a=0;
        }
    }
    else if(state == DoorOpening)
    {
        if((cover->frameTime() - timeoutStart) > 1 )
        {
            timeoutStart = cover->frameTime();
            state = DoorOpen;
        }
    }
    else if(state == DoorOpen)
    {
        if((cover->frameTime() - timeoutStart) > d_doorTimeout.get() )
        {
            timeoutStart = cover->frameTime();
            state = DoorClosing;
        }
    }
    else if(state == DoorClosing)
    {
        if((cover->frameTime() - timeoutStart) > 1 )
        {
            timeoutStart = cover->frameTime();
            state = Idle;
        }
    }

}

void VrmlNodeCar::setElevator(VrmlNodeElevator *e)
{
    elevator = e;
    for(int i=0;i<elevator->d_landingHeights.size();i++)
    {
        if(d_carPos.y()==elevator->d_landingHeights[i])
        {
            landingNumber = i;
        }
    }
    for(int i=0;i<elevator->d_shaftPositions.size();i++)
    {
        if(d_carPos.x()==elevator->d_shaftPositions[i])
        {
            shaftNumber = i;
        }
    }
    state = Idle;

}
void VrmlNodeCar::setDestination(int landing, int shaft)
{
    landingNumber = landing;
    shaftNumber = shaft;
    state = Moving;
    
    destinationY = elevator->d_landingHeights[landing];
    destinationX = elevator->d_shaftPositions[shaft];
}
