/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//

#include "Exchanger.h"
#include "Car.h"
#include "Elevator.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>

using namespace covise;


static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeExchanger(scene);
}

// Define the built in VrmlNodeType:: "Exchanger" fields

VrmlNodeType *VrmlNodeExchanger::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Exchanger", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    
    t->addExposedField("LandingNumber", VrmlField::SFINT32);
    t->addEventOut("Fraction", VrmlField::SFFLOAT);
    t->addEventOut("Rotation", VrmlField::SFROTATION);

    return t;
}

VrmlNodeType *VrmlNodeExchanger::nodeType() const
{
    return defineType(0);
}

VrmlNodeExchanger::VrmlNodeExchanger(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    state=Uninitialized;
    aaMax = 0.3;
    avMax = 3;
    av=0;aa=0;
    angle=0;
    currentCar = NULL;
}

VrmlNodeExchanger::VrmlNodeExchanger(const VrmlNodeExchanger &n)
    : VrmlNodeChild(n.d_scene)
{
    state=Uninitialized;
    aaMax = 0.3;
    avMax = 3;
    av=0;aa=0;
    angle=0;
    currentCar = NULL;
}

VrmlNodeExchanger::~VrmlNodeExchanger()
{
}

VrmlNode *VrmlNodeExchanger::cloneMe() const
{
    return new VrmlNodeExchanger(*this);
}

VrmlNodeExchanger *VrmlNodeExchanger::toExchanger() const
{
    return (VrmlNodeExchanger *)this;
}

ostream &VrmlNodeExchanger::printFields(ostream &os, int indent)
{

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeExchanger::setField(const char *fieldName,
                           const VrmlField &fieldValue)
{
    if
        TRY_FIELD(LandingNumber, SFInt)
    else
    VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeExchanger::getField(const char *fieldName)
{
    if (strcmp(fieldName, "LandingNumber") == 0)
        return &d_LandingNumber;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeExchanger::eventIn(double timeStamp,
                          const char *eventName,
                          const VrmlField *fieldValue)
{
    //if (strcmp(eventName, "ExchangerNumber"))
    // {
    //}
    // Check exposedFields
    //else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

}

void VrmlNodeExchanger::render(Viewer *)
{


}

void VrmlNodeExchanger::update()
{
    /*if(state == RotatingLeft || state == RotatingRight)
    {
        float dt = cover->frameDuration();
        if(dt > 1000) // first frameDuration is off because last FrameTime is 0
            dt=0.00001;
        float direction;
        float diff;
        float destinationAngle;
        if(state == RotatingRight)
        {
            direction = 1;
            diff = M_PI_2 - angle;
            destinationAngle = M_PI_2;
            if(angle == M_PI_2) // we are there
            {
                timeoutStart = cover->frameTime();
                state = Moving;
                av=0;aa=0;
            }
        }
        if(state == RotatingLeft)
        {
            direction = -1;
            diff = angle;
            destinationAngle = 0;
            if(angle == 0) // we are there
            {
                timeoutStart = cover->frameTime();
                state = DoorOpening;
                av=0;aa=0;
            }
        }
        if(state ==RotatingLeft || state ==RotatingRight ) // not there yet
        {
            float v2 = av*av;
            if(diff > (v2/(2*aaMax))*1.5)
            { // beschleunigen
                aa+=0.5*dt;
                if(aa > aaMax)
                    aa=aaMax;
                av += aa*dt;
                if(av > avMax)
                    av=avMax;
                angle += direction*av*dt;
            }
            else
            { // verzögern
                if(diff > 0.0001)
                {
                    aa = v2/(2*diff);
                    av -= aa*dt;
                }
                else
                {
                    aa=0;av=0;
                }
                if(av <= 0)
                {
                    angle=destinationAngle;
                    v=0;
                }
                else
                {
                    angle += direction*av*dt;
                }
            }
            if(!(av>=0) || !(aa>=0) || !(av<10) || !(aa<4))
            {
                fprintf(stderr,"oops\n");
            }
            double timeStamp = System::the->time();
            d_ExchangerRotation.get()[3] = angle;
            eventOut(timeStamp, "ExchangerRotation", d_ExchangerRotation);
        }
    }*/

}
void VrmlNodeExchanger::setAngle(float a)
{
    angle = a;
    double timeStamp = System::the->time();
    d_Rotation.get()[3] = angle;
    eventOut(timeStamp, "Rotation", d_Rotation);
    d_Fraction.set(angle/3.1415);
    eventOut(timeStamp, "Fraction", d_Fraction);
}

int VrmlNodeExchanger::getCarNumber()
{
    if(currentCar!=NULL)
        return currentCar->getID();
    return -1;
}
void VrmlNodeExchanger::setCar(VrmlNodeCar *c)
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

void VrmlNodeExchanger::setElevator(VrmlNodeElevator *e)
{
    elevator = e;
    state = Idle;

}
