/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//

#include "Exchanger.h"
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


    t->addExposedField("ExchangerNumber", VrmlField::SFINT32);
    t->addExposedField("ExchangerPos", VrmlField::SFVEC3F);
    t->addEventOut("ExchangerDoorClose", VrmlField::SFTIME);
    t->addEventOut("ExchangerDoorOpen", VrmlField::SFTIME);
    t->addEventOut("ExchangerRotation", VrmlField::SFROTATION);

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
    aMax = 1;
    vMax = 5;
    aaMax = 0.3;
    avMax = 3;
    v=0;a=0;
    av=0;aa=0;
    angle=0;
    d_doorTimeout=1.0;
    d_ExchangerRotation.set(0,0,1,0);
}

VrmlNodeExchanger::VrmlNodeExchanger(const VrmlNodeExchanger &n)
    : VrmlNodeChild(n.d_scene)
{
    state=Uninitialized;
    aMax = 1;
    vMax = 5;
    v=0;a=0;
    aaMax = 0.3;
    avMax = 3;
    v=0;a=0;
    av=0;aa=0;
    angle=0;
    d_doorTimeout=1.0;
    d_ExchangerRotation.set(0,0,1,0);
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
        TRY_FIELD(ExchangerNumber, SFInt)
    else if
    TRY_FIELD(ExchangerPos, SFVec3f)
    else if
    TRY_FIELD(ExchangerDoorClose, SFTime)
    else if
    TRY_FIELD(ExchangerDoorOpen, SFTime)
    else if
    TRY_FIELD(ExchangerRotation, SFRotation)
    else
    VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeExchanger::getField(const char *fieldName)
{
    if (strcmp(fieldName, "ExchangerNumber") == 0)
        return &d_ExchangerNumber;
    else if (strcmp(fieldName, "ExchangerPos") == 0)
        return &d_ExchangerPos;
    else if (strcmp(fieldName, "ExchangerDoorClose") == 0)
        return &d_ExchangerDoorClose;
    else if (strcmp(fieldName, "ExchangerDoorOpen") == 0)
        return &d_ExchangerDoorOpen;
    else if (strcmp(fieldName, "ExchangerRotation") == 0)
        return &d_ExchangerRotation;
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
    if(state == Moving)
    {
        float dt = cover->frameDuration();
        if(dt > 1000) // first frameDuration is off because last FrameTime is 0
            dt=0.00001;
        if(d_ExchangerPos.x() != destinationX) //moving horizontally
        {
            float direction;
            float diff = fabs(destinationX - d_ExchangerPos.x());
            if(d_ExchangerPos.x() < destinationX)
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
                d_ExchangerPos.get()[0] += direction*v*dt;
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
                d_ExchangerPos.get()[0] += direction*v*dt;
                if(v <= 0)
                {
                    d_ExchangerPos.get()[0]=destinationX;
                    v=0;
                }
            }
            double timeStamp = System::the->time();
            eventOut(timeStamp, "ExchangerPos", d_ExchangerPos);
        }
        else if(d_ExchangerPos.y() != destinationY) // moving vertically
        {
            float direction;
            float diff = fabs(destinationY - d_ExchangerPos.y());
            if(d_ExchangerPos.y() < destinationY)
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
                d_ExchangerPos.get()[1] += direction*v*dt;
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
                    d_ExchangerPos.get()[1]=destinationY;
                    v=0;
                }
                else
                {
                    d_ExchangerPos.get()[1] += direction*v*dt;
                }
            }
            if(!(v>=0) || !(a>=0) || !(v<10) || !(a<4))
            {
                fprintf(stderr,"oops\n");
            }
            double timeStamp = System::the->time();
            eventOut(timeStamp, "ExchangerPos", d_ExchangerPos);
        }
        else // we are there
        {
            timeoutStart = cover->frameTime();
            state = DoorOpening;
            if(angle != 0)
            {
                state = RotatingLeft;
            }
            
            d_ExchangerDoorOpen = System::the->time();
            eventOut(d_ExchangerDoorOpen.get(), "ExchangerDoorOpen", d_ExchangerDoorOpen);
            v=0;a=0;
        }
    }
    else if(state == RotatingLeft || state == RotatingRight)
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
            d_ExchangerDoorClose = System::the->time();
            eventOut(d_ExchangerDoorOpen.get(), "ExchangerDoorClose", d_ExchangerDoorClose);
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

void VrmlNodeExchanger::setElevator(VrmlNodeElevator *e)
{
    elevator = e;
    for(int i=0;i<elevator->d_landingHeights.size();i++)
    {
        if(d_ExchangerPos.y()==elevator->d_landingHeights[i])
        {
            landingNumber = i;
        }
    }
    for(int i=0;i<elevator->d_shaftPositions.size();i++)
    {
        if(d_ExchangerPos.x()==elevator->d_shaftPositions[i])
        {
            shaftNumber = i;
        }
    }
    state = Idle;

}
void VrmlNodeExchanger::setDestination(int landing, int shaft)
{
    landingNumber = landing;
    if(shaftNumber != shaft)
    {
        // we have to travel horizontally
        state = RotatingRight;
    }
    else
    {
    state = Moving;
    }
    shaftNumber = shaft;
    
    destinationY = elevator->d_landingHeights[landing];
    destinationX = elevator->d_shaftPositions[shaft];
}
