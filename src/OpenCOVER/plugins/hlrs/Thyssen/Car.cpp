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
    t->addExposedField("stationOpenTime", VrmlField::MFFLOAT);
    t->addExposedField("currentStationIndex", VrmlField::SFINT32);
    t->addEventOut("carDoorClose", VrmlField::SFTIME);
    t->addEventOut("carDoorOpen", VrmlField::SFTIME);
    t->addEventOut("carRotation", VrmlField::SFROTATION);
    t->addEventOut("carFraction", VrmlField::SFFLOAT);

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
    d_carFraction=0.0;
    d_currentStationIndex=0;
    ID = IDCounter++;
    currentPassingStation = passingStations.begin();
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
    d_carFraction=0.0;
    d_currentStationIndex=0;
    ID = IDCounter++;
    currentPassingStation = passingStations.begin();
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
    TRY_FIELD(carFraction, SFFloat)
    else if
    TRY_FIELD(stationList, MFInt)
    else if
    TRY_FIELD(stationOpenTime, MFFloat)
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
    else if (strcmp(fieldName, "carFraction") == 0)
        return &d_carFraction;
    else if (strcmp(fieldName, "stationList") == 0)
        return &d_stationList;
    else if (strcmp(fieldName, "stationOpenTime") == 0)
        return &d_stationOpenTime;
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
    //startTurning(); // if necessarry
    if(state == Moving && chassisState == Idle)
    {
        
        float dt = cover->frameDuration();
        if(dt > 1000) // first frameDuration is off because last FrameTime is 0
            dt=0.00001;
        VrmlNodeCar *nextCarOnRail;
        float distanceToNextCar = elevator->getNextCarOnRail(this,nextCarOnRail);
        if(d_carPos.x() != destinationX) //moving horizontally
        {
            float direction;
            float diff = fabs(destinationX - d_carPos.x());
            float diffS = fabs(startingX - d_carPos.x());
            float v2 = v*v;
            float bakeDistance = (v2/(2*aMax))*1.5; // distance the car travels until it stops at max decelleration
            
            if(d_carPos.x() < destinationX)
            {
                direction = 1;
                if(d_carPos.x() > elevator->stations[*currentPassingStation].x())
                {
                    if(currentPassingStation !=passingStations.end())
                        currentPassingStation++;
                }
            }
            else
            {
                direction = -1;
                if(d_carPos.x() < elevator->stations[*currentPassingStation].x())
                {
                    if(currentPassingStation !=passingStations.end())
                        currentPassingStation++;
                }
            }
            
            if(distanceToNextCar > 0)
            {
                float vd = v - nextCarOnRail->getV();
                if(vd > 0) // we only have to care if our car is faster than the next one, otherwise there is no chance to collide
                {
                    float vd2 = vd*vd;
                    float bakeDistance = (vd2/(2*aMax))*1.5; // distance the car travels until it reaches the velocity of the other car at max decelleration
                    if(diff < (distanceToNextCar - (CAR_WIDTH_2 + CAR_WIDTH_2 + SAFETY_DISTANCE + bakeDistance)))
                    {
                        diff = distanceToNextCar - (CAR_WIDTH_2 + CAR_WIDTH_2 + SAFETY_DISTANCE); // only travel to next car
                    }
                }
            }
            float passingDiff=0;
            passingDiff = (elevator->stations[*currentPassingStation].x() - d_carPos.x())*direction;
            for(std::list<int>::iterator it = occupiedStations.begin(); it != occupiedStations.end();)
            {
                float passingDiffS=0;
                passingDiffS = (d_carPos.x() - elevator->stations[*it].x())*direction;
                if(passingDiffS > (CAR_WIDTH_2 + LANDING_WIDTH_2 + SAFETY_DISTANCE))
                {
                    elevator->release(*it);
                    it = occupiedStations.erase(it);
                }
                else
                {
                    it++;
                }
            }
            if((passingDiff < (CAR_WIDTH_2 + LANDING_WIDTH_2 + SAFETY_DISTANCE + bakeDistance)) )
            {
                if(elevator->occupy(*currentPassingStation,this) == true)
                {
                    bool found=false;
                    for(std::list<int>::iterator it = occupiedStations.begin(); it != occupiedStations.end();it++)
                    {
                        if(*it == *currentPassingStation)
                        {
                            found = true;
                        }
                    }
                    if(!found)
                        occupiedStations.push_back(*currentPassingStation);
                }
                else
                {   // we can't occupy the landing yet, thus stop in front of the landing
                    diff = fabs(destinationX - d_carPos.x()) - (CAR_WIDTH_2 + LANDING_WIDTH_2 + SAFETY_DISTANCE);
                }
            }
            
            if(diff > bakeDistance)
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
                if(direction * (destinationX - d_carPos.get()[0]) < 0)
                {
                    d_carPos.get()[0]=destinationX;
                    v=0;
                }
                if(v < 0)
                {
                    v=0;
                }
                d_carPos.get()[0] += direction*v*dt;
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

            float v2 = v*v;
            float bakeDistance = (v2/(2*aMax))*1.5; // distance the car travels until it stops at max decelleration
            
            if(d_carPos.y() < destinationY)
            {
                if(d_carPos.y() > elevator->stations[*currentPassingStation].y())
                {
                    if(currentPassingStation !=passingStations.end())
                        currentPassingStation++;
                }
                direction = 1;
            }
            else
            {
                if(d_carPos.y() < elevator->stations[*currentPassingStation].y())
                {
                    if(currentPassingStation !=passingStations.end())
                        currentPassingStation++;
                }
                direction = -1;
            }

            if(distanceToNextCar > 0)
            {
                float vd = v - nextCarOnRail->getV();
                if(vd > 0) // we only have to care if our car is faster than the next one, otherwise there is no chance to collide
                {
                    float vd2 = vd*vd;
                    float bakeDistance = (vd2/(2*aMax))*1.5; // distance the car travels until it reaches the velocity of the other car at max decelleration
                    if(diff < (distanceToNextCar - (CAR_HEIGHT_2 + CAR_HEIGHT_2 + SAFETY_DISTANCE + bakeDistance)))
                    {
                        diff = distanceToNextCar - (CAR_HEIGHT_2 + CAR_HEIGHT_2 + SAFETY_DISTANCE); // only travel to next car
                    }
                }
            }
            float passingDiff=0;
            passingDiff = (elevator->stations[*currentPassingStation].y() - d_carPos.y())*direction;
            for(std::list<int>::iterator it = occupiedStations.begin(); it != occupiedStations.end();)
            {
                float passingDiffS=0;
                passingDiffS = (d_carPos.y() - elevator->stations[*it].y())*direction;
                if(passingDiffS > (CAR_HEIGHT_2 + LANDING_HEIGHT_2 + SAFETY_DISTANCE))
                {
                    elevator->release(*it);
                    it = occupiedStations.erase(it);
                }
                else
                {
                    it++;
                }
            }
            if((passingDiff < (CAR_HEIGHT_2 + LANDING_HEIGHT_2 + SAFETY_DISTANCE + bakeDistance)) )
            {
                if(elevator->occupy(*currentPassingStation,this) == true)
                {
                    bool found=false;
                    for(std::list<int>::iterator it = occupiedStations.begin(); it != occupiedStations.end();it++)
                    {
                        if(*it == *currentPassingStation)
                        {
                            found = true;
                        }
                    }
                    if(!found)
                        occupiedStations.push_back(*currentPassingStation);
                }
                else
                {   // we can't occupy the landing yet, thus stop in front of the landing
                    diff = fabs(destinationY - d_carPos.y()) - (CAR_WIDTH_2 + LANDING_WIDTH_2 + SAFETY_DISTANCE);
                }
            }


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
                if(direction * (destinationY - d_carPos.get()[1]) < 0)
                {
                    d_carPos.get()[1]=destinationY;
                    v=0;
                }
                if(v < 0)
                {
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
            currentPassingStation = passingStations.begin();
            v=0;a=0;
            for(std::list<int>::iterator it = occupiedStations.begin(); it != occupiedStations.end();)
            {
                if(*it != destinationLandingIndex) // release all but the station where we currently are
                {
                    elevator->release(*it);
                }
                it = occupiedStations.erase(it);
            }
            arrivedAtDestination();
        }
    }
    else if(state == DoorOpening)
    {
        
        nextPositionIsEmpty();
        if((cover->frameTime() - timeoutStart) > 1 )
        {
            timeoutStart = cover->frameTime();
            state = DoorOpen;
        }
    }
    else if(state == DoorOpen)
    {
        nextPositionIsEmpty();
        float doorOpenTime = d_doorTimeout.get();
        if(d_stationOpenTime.size() > d_currentStationIndex.get() && d_stationOpenTime[d_currentStationIndex.get()]>0)
        {
            doorOpenTime = d_stationOpenTime[d_currentStationIndex.get()];
        }
        if((cover->frameTime() - timeoutStart) > doorOpenTime )
        {
            timeoutStart = cover->frameTime();
            d_carDoorClose = System::the->time();
            eventOut(d_carDoorOpen.get(), "carDoorClose", d_carDoorClose);
            if(d_stationList[d_currentStationIndex.get()] < elevator->landings.size() && elevator->landings[d_stationList[d_currentStationIndex.get()]]!=NULL)
            {
                elevator->landings[ d_stationList[d_currentStationIndex.get()]]->closeDoor();
            }
            state = DoorClosing;
        }
    }
    else if(state == DoorClosing)
    {
        nextPositionIsEmpty();
        if((cover->frameTime() - timeoutStart) > 1 )
        {
            timeoutStart = cover->frameTime();
            state = Idle;
        }
    }
    

}


void VrmlNodeCar::setAngle(float a)
{
    double timeStamp = System::the->time();
    d_carRotation.get()[3] = a;
    eventOut(timeStamp, "carRotation", d_carRotation);
    d_carFraction.set(a/M_PI_2);
    eventOut(timeStamp, "carFraction", d_carFraction);
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
    elevator->stations[d_stationList[d_currentStationIndex.get()]].car=this;
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
    
    elevator->removeCarFromRail(this);
    oldLandingIndex =  d_stationList[d_currentStationIndex.get()];
    d_currentStationIndex = d_currentStationIndex.get()+1;
    if(d_currentStationIndex.get()>=d_stationList.size())
        d_currentStationIndex=0;


    destinationLandingIndex =  d_stationList[d_currentStationIndex.get()];
    int landing = d_stationList[d_currentStationIndex.get()] % elevator->d_landingHeights.size();
    passingStations.clear();
    if(landing > landingNumber)
    {
        setTravelDirection(VrmlNodeCar::MoveUp);
        for(int i=oldLandingIndex+1;i<=destinationLandingIndex;i++)
        {
            passingStations.push_back(i);
        }
    }
    if(landing < landingNumber)
    {
        setTravelDirection(VrmlNodeCar::MoveDown);
        for(int i=oldLandingIndex-1;i>=destinationLandingIndex;i--)
        {
            passingStations.push_back(i);
        }
    }
    int shaft = d_stationList[d_currentStationIndex.get()] / elevator->d_landingHeights.size();
    if(shaft > shaftNumber)
    {
        setTravelDirection(VrmlNodeCar::MoveRight);
        for(int i=oldLandingIndex+elevator->d_landingHeights.size();i<=destinationLandingIndex;i+=elevator->d_landingHeights.size())
        {
            passingStations.push_back(i);
        }
    }
    if(shaft < shaftNumber)
    {
        setTravelDirection(VrmlNodeCar::MoveLeft);
        for(int i=oldLandingIndex-elevator->d_landingHeights.size();i>=destinationLandingIndex;i-=elevator->d_landingHeights.size())
        {
            passingStations.push_back(i);
        }
    }
    setDestination(landing, shaft);
    
    currentPassingStation = passingStations.begin();
    occupiedStations.push_back(oldLandingIndex);
    elevator->putCarOnRail(this);
}

bool VrmlNodeCar::nextPositionIsEmpty() // return true if the destination landing is empty and all exchangers are in the right orientation (if not turn them)
{
    int nextIndex = d_currentStationIndex.get()+1;
    if(nextIndex>=d_stationList.size())
        nextIndex=0;
    
    int startShaft = d_stationList[d_currentStationIndex.get()] / elevator->d_landingHeights.size();
    int destinationShaft = d_stationList[nextIndex] / elevator->d_landingHeights.size();
    int startLanding = d_stationList[d_currentStationIndex.get()] % elevator->d_landingHeights.size();
    int destinationLanding = d_stationList[nextIndex] % elevator->d_landingHeights.size();
    currentExchangers.clear();
    if(startShaft < destinationShaft)
    {
        for(int i=startShaft; i<=destinationShaft;i++)
        {
            VrmlNodeExchanger *ex = elevator->exchangers[i*elevator->d_landingHeights.size() + startLanding];
            if(ex)
                currentExchangers.push_back(ex);
        }
    }
    else if(startShaft > destinationShaft)
    {
        for(int i=destinationShaft; i<=startShaft;i++)
        {
            VrmlNodeExchanger *ex = elevator->exchangers[i*elevator->d_landingHeights.size() + startLanding];
            if(ex)
                currentExchangers.push_back(ex);
        }
    }
    else if(startLanding < destinationLanding)
    {
        for(int i=startLanding; i<=destinationLanding;i++)
        {
            VrmlNodeExchanger *ex = elevator->exchangers[startShaft*elevator->d_landingHeights.size() + i];
            if(ex)
                currentExchangers.push_back(ex);
        }
    }
    else if(startLanding > destinationLanding)
    {
        for(int i=destinationLanding; i<=startLanding;i++)
        {
            VrmlNodeExchanger *ex = elevator->exchangers[startShaft*elevator->d_landingHeights.size() + i];
            if(ex)
                currentExchangers.push_back(ex);
        }
    }
    bool empty = true;
    if(startShaft != destinationShaft)
    {
        std::list<VrmlNodeExchanger *>::iterator it;
        for(it = currentExchangers.begin(); it != currentExchangers.end(); it++)
        {
            if((*it)->getAngle()!=(float)M_PI_2)
            {
                empty=false;
            }
            if((((*it)->getState()==VrmlNodeExchanger::Idle) && ((*it)->getCar()==NULL))||(*it)->getCar()==this)
                (*it)->rotateRight();
            else
            {
                if((*it)->getCar()!=this)
                    empty=false;
            }

        }
    }
    else
    {
        std::list<VrmlNodeExchanger *>::iterator it;
        for(it = currentExchangers.begin(); it != currentExchangers.end(); it++)
        {
            if((*it)->getAngle()!=0)
            {
                empty=false;
            }
            if((((*it)->getState()==VrmlNodeExchanger::Idle) && ((*it)->getCar()==NULL))||(*it)->getCar()==this)
                (*it)->rotateLeft();
            else
            {
                if((*it)->getCar()!=this)
                empty=false;
            }
        }
    }
    if(empty==false)
    {
        return false;
    }
    return true;
}


void VrmlNodeCar::startTurning() // turn if necessarry and possible
{
    
    int nextIndex = d_currentStationIndex.get()+1;
    if(nextIndex>=d_stationList.size())
        nextIndex=0;
    // find all exchangers on the way to the next destination so that we can also turn them right and left
    if(chassisState == StartRotatingRight || chassisState == StartRotatingLeft)
    {
        timeoutStart = cover->frameTime();

        

        if(travelDirection == MoveLeft || travelDirection == MoveRight)
        {
            int nextIndex = d_currentStationIndex.get()+1;
            if(nextIndex>=d_stationList.size())
                nextIndex=0;
            if(elevator->stations[d_stationList[nextIndex]].car==NULL)
            {
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
    
    if(d_stationList[d_currentStationIndex.get()] < elevator->landings.size() && elevator->landings[d_stationList[d_currentStationIndex.get()]]!=NULL)
    {
        elevator->landings[ d_stationList[d_currentStationIndex.get()]]->openDoor();
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
            //chassisState = StartRotatingRight;
            oldTravelDirection = travelDirection;
        }
        if(travelDirection == MoveUp || travelDirection == MoveDown)
        {
            //chassisState = StartRotatingLeft;
            oldTravelDirection = travelDirection;
        }
    }
    nextPositionIsEmpty();
    //startTurning();
}