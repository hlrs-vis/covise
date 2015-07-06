/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//

#include "Car.h"
#include "Elevator.h"
#include "Exchanger.h"
#include "Landing.h"

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
    t->addExposedField("stationList", VrmlField::MFINT32);
    t->addExposedField("currentStationIndex", VrmlField::SFINT32);
    t->addEventOut("carDoorClose", VrmlField::SFTIME);
    t->addEventOut("carDoorOpen", VrmlField::SFTIME);
    t->addEventOut("carRotation", VrmlField::SFROTATION);

    return t;
}

VrmlNodeType *VrmlNodeCar::nodeType() const
{
    return defineType(0);
}

int VrmlNodeCar::IDCounter=0;
VrmlNodeCar::VrmlNodeCar(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    state=Uninitialized;
    oldState=Uninitialized;
    chassisState=Idle;
    oldChassisState=Uninitialized;
    travelDirection=Uninitialized;
    oldTravelDirection=Uninitialized;
    aMax = 1;
    vMax = 5;
    aaMax = 0.3;
    avMax = 3;
    v=0;a=0;
    av=0;aa=0;
    angle=0;
    d_doorTimeout=1.0;
    d_carRotation.set(1,0,0,0);
    d_currentStationIndex=0;
    ID = IDCounter++;
}

VrmlNodeCar::VrmlNodeCar(const VrmlNodeCar &n)
    : VrmlNodeChild(n.d_scene)
{
    state=Uninitialized;
    oldState=Uninitialized;
    chassisState=Idle;
    oldChassisState=Uninitialized;
    travelDirection=Uninitialized;
    oldTravelDirection=Uninitialized;
    aMax = 1;
    vMax = 5;
    v=0;a=0;
    aaMax = 0.3;
    avMax = 3;
    v=0;a=0;
    av=0;aa=0;
    angle=0;
    d_doorTimeout=1.0;
    d_carRotation.set(1,0,0,0);
    d_currentStationIndex=0;
    ID = IDCounter++;
}

VrmlNodeCar::~VrmlNodeCar()
{
}

enum VrmlNodeCar::carState VrmlNodeCar::getState(){return state;}
void VrmlNodeCar::setState(enum carState s){oldState=state; state = s;}
enum VrmlNodeCar::carState VrmlNodeCar::getChassisState(){return chassisState;}
void VrmlNodeCar::setChassisState(enum carState s){oldChassisState=chassisState; chassisState = s;}
enum VrmlNodeCar::carState VrmlNodeCar::getTravelDirection(){return travelDirection;}
void VrmlNodeCar::setTravelDirection(enum carState t)
{
    oldTravelDirection=travelDirection;
    travelDirection = t;
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
    TRY_FIELD(carRotation, SFRotation)
    else if
    TRY_FIELD(stationList, MFInt)
    else if
    TRY_FIELD(currentStationIndex, SFInt)
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
    else if (strcmp(fieldName, "carRotation") == 0)
        return &d_carRotation;
    else if (strcmp(fieldName, "stationList") == 0)
        return &d_stationList;
    else if (strcmp(fieldName, "currentStationIndex") == 0)
        return &d_currentStationIndex;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeCar::eventIn(double timeStamp,
                          const char *eventName,
                          const VrmlField *fieldValue)
{
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
    startTurning(); // if necessarry
    if(state == Moving && chassisState == Idle)
    {
        
        float dt = cover->frameDuration();
        if(dt > 1000) // first frameDuration is off because last FrameTime is 0
            dt=0.00001;
        if(d_carPos.x() != destinationX) //moving horizontally
        {
            float direction;
            float diff = fabs(destinationX - d_carPos.x());
            float diffS = fabs(startingX - d_carPos.x());
            if(diffS > 2 && (oldLandingIndex >= 0))
            {
                elevator->release(oldLandingIndex);
                oldLandingIndex = -1;
            }
            if(diff < 2 && (destinationLandingIndex >= 0))
            {
                
                elevator->occupy(destinationLandingIndex,this);
                destinationLandingIndex = -1;
            }
            
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
            if(direction > 0 && d_carPos.get()[0]>destinationX)
            {
                d_carPos.get()[0]=destinationX;
                a=0;v=0;
            }
            if(direction < 0 && d_carPos.get()[0]<destinationX)
            {
                d_carPos.get()[0]=destinationX;
                a=0;v=0;
            }

            double timeStamp = System::the->time();
            eventOut(timeStamp, "carPos", d_carPos);
        }
        else if(d_carPos.y() != destinationY) // moving vertically
        {
            float direction;
            float diff = fabs(destinationY - d_carPos.y());

            float diffS = fabs(startingY - d_carPos.y());
            if(diffS > 2 && (oldLandingIndex >= 0))
            {
                elevator->release(oldLandingIndex);
                oldLandingIndex = -1;
            }
            if(diff < 2 && (destinationLandingIndex >= 0))
            {
                elevator->occupy(destinationLandingIndex,this);
                destinationLandingIndex = -1;
            }


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
            
            if(direction > 0 && d_carPos.get()[1]>destinationY)
            {
                d_carPos.get()[1]=destinationY;
                a=0;v=0;
            }
            if(direction < 0 && d_carPos.get()[1]<destinationY)
            {
                d_carPos.get()[1]=destinationY;
                a=0;v=0;
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
            v=0;a=0;
            arrivedAtDestination();
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
            d_carDoorClose = System::the->time();
            eventOut(d_carDoorOpen.get(), "carDoorClose", d_carDoorClose);
            if(elevator->landings[d_currentStationIndex.get()]!=NULL)
            {
                eventOut(d_carDoorOpen.get(), "doorClose",elevator->landings[ d_currentStationIndex.get()]->d_doorClose);
            }
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
    
    if(chassisState == RotatingLeft || chassisState == RotatingRight)
    {
        float dt = cover->frameDuration();
        if(dt > 1000) // first frameDuration is off because last FrameTime is 0
            dt=0.00001;
        float direction;
        float diff;
        double destinationAngle;
        if(chassisState == RotatingRight)
        {
            direction = 1;
            diff = M_PI_2 - angle;
            destinationAngle = M_PI_2;
            if(angle == M_PI_2) // we are there
            {
                chassisState = Idle;
                setTravelDirection(MoveUp);
                av=0;aa=0;
            }
        }
        if(chassisState == RotatingLeft)
        {
            direction = -1;
            diff = angle;
            destinationAngle = 0;
            if(angle == 0) // we are there
            {
                chassisState = Idle;
                setTravelDirection(MoveUp);
                av=0;aa=0;
            }
        }
        if(chassisState ==RotatingLeft || chassisState ==RotatingRight ) // not there yet
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
                    av=0;
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
            d_carRotation.get()[3] = angle;
            eventOut(timeStamp, "carRotation", d_carRotation);
            std::list<VrmlNodeExchanger *>::iterator it;
            for(it = currentExchangers.begin();it != currentExchangers.end();it++)
            {
                (*it)->setAngle(angle);
            }
        }
    }

}


void VrmlNodeCar::setElevator(VrmlNodeElevator *e)
{
    elevator = e;
 /*   for(int i=0;i<elevator->d_landingHeights.size();i++)
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
    }*/
    elevator->stations[d_stationList[d_currentStationIndex.get()]]=this;
    landingNumber = d_stationList[d_currentStationIndex.get()] % elevator->d_landingHeights.size();
    shaftNumber = d_stationList[d_currentStationIndex.get()] / elevator->d_landingHeights.size();
    d_carPos.set(elevator->d_shaftPositions[shaftNumber],elevator->d_landingHeights[landingNumber],0);
    double timeStamp = System::the->time();
    eventOut(timeStamp, "carPos", d_carPos);
    state = Idle;

}
void VrmlNodeCar::setDestination(int landing, int shaft)
{
    if(landing != landingNumber || shaft != shaftNumber)
    {
        oldLandingNumber=landingNumber;
        oldShaftNumber=shaftNumber;

        landingNumber = landing;
        state = Moving;
        shaftNumber = shaft;

        startingY = elevator->d_landingHeights[oldLandingNumber];
        startingX = elevator->d_shaftPositions[oldShaftNumber];

        destinationY = elevator->d_landingHeights[landing];
        destinationX = elevator->d_shaftPositions[shaft];
    }

}
void VrmlNodeCar::moveToNext()
{
    oldLandingIndex =  d_stationList[d_currentStationIndex.get()];
    d_currentStationIndex = d_currentStationIndex.get()+1;
    if(d_currentStationIndex.get()>=d_stationList.size())
        d_currentStationIndex=0;
    destinationLandingIndex =  d_stationList[d_currentStationIndex.get()];
    int landing = d_stationList[d_currentStationIndex.get()] % elevator->d_landingHeights.size();
    int shaft = d_stationList[d_currentStationIndex.get()] / elevator->d_landingHeights.size();
    setDestination(landing, shaft);
}

bool VrmlNodeCar::nextPositionIsEmpty() // return true if the destination landing is empty
{
    int nextIndex = d_currentStationIndex.get()+1;
    if(nextIndex>=d_stationList.size())
        nextIndex=0;
    int landing = d_stationList[nextIndex] % elevator->d_landingHeights.size();
    int shaft = d_stationList[nextIndex] / elevator->d_landingHeights.size();

    if(elevator->exchangers.size() > d_stationList[nextIndex] &&elevator->exchangers[d_stationList[nextIndex]] !=NULL)
    {
        if(elevator->exchangers[d_stationList[nextIndex]]->getCar()!=NULL)
            return false;
    }
    if(elevator->stations[d_stationList[nextIndex]]!=NULL)
        return false;
    
    if(elevator->landings.size() > d_stationList[nextIndex] &&elevator->landings[d_stationList[nextIndex]] !=NULL)
    {
        if(elevator->landings[d_stationList[nextIndex]]->getCar()!=NULL)
            return false;
    }
    if(elevator->stations[d_stationList[nextIndex]]!=NULL)
        return false;
    return true;
}


void VrmlNodeCar::startTurning() // turn if necessarry and possible
{
    if(chassisState == StartRotatingRight || chassisState == StartRotatingLeft)
    {
        timeoutStart = cover->frameTime();

        int nextIndex = d_currentStationIndex.get()+1;
        if(nextIndex>=d_stationList.size())
            nextIndex=0;
        

        if(travelDirection == MoveLeft || travelDirection == MoveRight)
        {
            if(elevator->stations[d_stationList[nextIndex]]==NULL)
            {
                // find all exchangers on the way to the next destination so that we can also turn them right and left
                int startShaft = d_stationList[d_currentStationIndex.get()] / elevator->d_landingHeights.size();
                int destinationShaft = d_stationList[nextIndex] / elevator->d_landingHeights.size();
                int startLanding = d_stationList[d_currentStationIndex.get()] % elevator->d_landingHeights.size();
                int destinationLanding = d_stationList[nextIndex] % elevator->d_landingHeights.size();
                currentExchangers.clear();
                if(startShaft < destinationShaft)
                {
                    for(int i=startShaft; i<=destinationShaft;i++)
                    {
                        currentExchangers.push_back(elevator->exchangers[i*elevator->d_landingHeights.size() + startLanding]);
                    }
                }
                else
                {
                    for(int i=destinationShaft; i<=startShaft;i++)
                    {
                        currentExchangers.push_back(elevator->exchangers[i*elevator->d_landingHeights.size() + startLanding]);
                    }
                }
                chassisState = RotatingRight;
            }
        }
        if(travelDirection == MoveUp || travelDirection == MoveDown)
        {
            chassisState = RotatingLeft;
        }
    }
}

void VrmlNodeCar::arrivedAtDestination() // the car arrived at its destination
{
    timeoutStart = cover->frameTime();
    state = DoorOpening;
    d_carDoorOpen = System::the->time();
    eventOut(d_carDoorOpen.get(), "carDoorOpen", d_carDoorOpen);
    
    if(elevator->landings[d_currentStationIndex.get()]!=NULL)
    {
        eventOut(d_carDoorOpen.get(), "doorOpen",elevator->landings[ d_currentStationIndex.get()]->d_doorOpen);
    }

    int nextIndex = d_currentStationIndex.get()+1;
    if(nextIndex>=d_stationList.size())
        nextIndex=0;
    int landing = d_stationList[nextIndex] % elevator->d_landingHeights.size();
    int shaft = d_stationList[nextIndex] / elevator->d_landingHeights.size();

    if(shaft!=shaftNumber) // next shaft != this shaft
    {
        if(shaft > shaftNumber)
            travelDirection = MoveRight;
        else
            travelDirection = MoveLeft;
    }
    else if(landing!=landingNumber) // next landing != this landing
    {
        if(landing > landingNumber)
            travelDirection = MoveUp;
        else
            travelDirection = MoveDown;
    }
    if(travelDirection!= oldTravelDirection)
    {

        if(travelDirection == MoveLeft || travelDirection == MoveRight)
        {
            chassisState = StartRotatingRight;
            oldTravelDirection = travelDirection;
        }
        if(travelDirection == MoveUp || travelDirection == MoveDown)
        {
            chassisState = StartRotatingLeft;
            oldTravelDirection = travelDirection;
        }
    }
    startTurning();
}