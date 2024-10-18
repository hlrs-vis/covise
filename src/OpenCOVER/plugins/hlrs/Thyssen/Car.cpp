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

#include <boost/lexical_cast.hpp>

using namespace covise;


void VrmlNodeCar::initFields(VrmlNodeCar *node, vrml::VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("carNumber", node->d_carNumber),
        exposedField("carPos", node->d_carPos),
        exposedField("stationList", node->d_stationList),
        exposedField("stationOpenTime", node->d_stationOpenTime),
        exposedField("currentStationIndex", node->d_currentStationIndex));
    
    
    if (t)
    {

        t->addEventOut("carDoorClose", VrmlField::SFTIME);
        t->addEventOut("carDoorOpen", VrmlField::SFTIME);
        t->addEventOut("carRotation", VrmlField::SFROTATION);
        t->addEventOut("carFraction", VrmlField::SFFLOAT);
        t->addEventOut("Unlock", VrmlField::SFTIME);
        t->addEventOut("Lock", VrmlField::SFTIME);
    }
}

const char *VrmlNodeCar::name()
{
    return "Car";
}

int VrmlNodeCar::IDCounter=0;
VrmlNodeCar::VrmlNodeCar(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
{
    state=Uninitialized;
    oldState=Uninitialized;
    chassisState=Idle;
    oldChassisState=Uninitialized;
    travelDirection=Uninitialized;
    oldTravelDirection=Uninitialized;
    aMax = 1.2;
    vMax = 5;
	ahMax = 0.3;
	vhMax = 0.4;
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
    : VrmlNodeChild(n)
{
    state=Uninitialized;
    oldState=Uninitialized;
    chassisState=Idle;
    oldChassisState=Uninitialized;
    travelDirection=Uninitialized;
    oldTravelDirection=Uninitialized;
    aMax = 1.2;
    vMax = 5;
	ahMax = 0.3;
	vhMax = 0.4;
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

void VrmlNodeCar::lock()
{
	double timeStamp = System::the->time();
	d_lockTime.set(timeStamp);
	eventOut(timeStamp, "Lock", d_lockTime);
}
void VrmlNodeCar::unlock()
{
	double timeStamp = System::the->time();
	d_lockTime.set(timeStamp);
	eventOut(timeStamp, "Unlock", d_lockTime);
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

VrmlNodeCar *VrmlNodeCar::toCar() const
{
    return (VrmlNodeCar *)this;
}

const VrmlField *VrmlNodeCar::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "carDoorClose") == 0)
        return &d_carDoorClose;
    else if (strcmp(fieldName, "carDoorOpen") == 0)
        return &d_carDoorOpen;
    else if (strcmp(fieldName, "carRotation") == 0)
        return &d_carRotation;
    else if (strcmp(fieldName, "carFraction") == 0)
        return &d_carFraction;
    else
        return VrmlNodeChild::getField(fieldName);
}

void VrmlNodeCar::eventIn(double timeStamp,
                          const char *eventName,
                          const VrmlField *fieldValue)
{
    VrmlNode::eventIn(timeStamp, eventName, fieldValue);
}

void VrmlNodeCar::render(Viewer *)
{


}

void VrmlNodeCar::update()
{
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
            float bakeDistance = (v2/(2*ahMax))*1.5; // distance the car travels until it stops at max decelleration
            
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
                    float bakeDistance = (vd2/(2*ahMax))*1.5; // distance the car travels until it reaches the velocity of the other car at max decelleration
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
                    diff = passingDiff - (CAR_WIDTH_2 + LANDING_WIDTH_2 + SAFETY_DISTANCE);
                }
            }
            
            if(diff > bakeDistance)
            { // beschleunigen
                a+=0.5*dt;
                if(a > ahMax)
                    a=ahMax;
                v += a*dt;
                if(v > vhMax)
                    v=vhMax;
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
                    if(distanceToNextCar < (CAR_HEIGHT_2 + CAR_HEIGHT_2 + SAFETY_DISTANCE + bakeDistance))
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
                    diff = passingDiff - (CAR_HEIGHT_2 + LANDING_HEIGHT_2 + SAFETY_DISTANCE);
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
            /*if(!(v>=0) || !(a>=0) || !(v<10) || !(a<4))
            {
                fprintf(stderr,"oops\n");
            }*/
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
        if(abs(cover->frameTime() - timeoutStart) > 1 )
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
        if((abs(cover->frameTime() - timeoutStart) > doorOpenTime ) && openButton->getState()==false)
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
        if(abs(cover->frameTime() - timeoutStart) > 1 )
        {
            timeoutStart = cover->frameTime();
            state = Idle;
        }
    }
    

}


void VrmlNodeCar::tabletPressEvent(coTUIElement *tUIItem)
{
}

void VrmlNodeCar::tabletEvent(coTUIElement *tUIItem)
{
    if(tUIItem == stationListEdit)
    {
        temporaryStationList.clear();
        std::string listtext = stationListEdit->getText();
        istringstream iss(listtext);
        do
        {
            int st= -1;
            iss >> st;
            if(st >= 0 && st < elevator->stations.size())
            {
                temporaryStationList.push_back(st);
            }
        } while (iss);
        // check station list for consistency (can't travel diagonal)

        int numHeights = elevator->d_landingHeights.size();

        int oldLandingNumber=-1;
        int oldShaftNumber=-1;
        std::list<int>::iterator it;
        for(it=temporaryStationList.begin();it !=temporaryStationList.end(); it++)
        {
            int landingNumber = *it % numHeights;
            int shaftNumber = *it / numHeights;
            if(oldLandingNumber > 0)
            {
                if(oldLandingNumber != landingNumber && oldShaftNumber != shaftNumber) // this does not work.
                {
                    // add another step
                    temporaryStationList.insert(it,oldShaftNumber*numHeights+landingNumber);
                    oldLandingNumber = landingNumber;
                }
            }
            oldLandingNumber = landingNumber;
            oldShaftNumber = shaftNumber;
        }
        it=temporaryStationList.begin();
        int landingNumber = *it % numHeights;
        int shaftNumber = *it / numHeights;
        if(oldLandingNumber > 0)
        {
            if(oldLandingNumber != landingNumber && oldShaftNumber != shaftNumber) // this does not work.
            {
                // add another step
                temporaryStationList.insert(it,oldShaftNumber*numHeights+landingNumber);
                oldLandingNumber = landingNumber;
            }
        }
        std::string stationListString;
        for(std::list<int>::iterator it=temporaryStationList.begin();it !=temporaryStationList.end(); it++)
        {
            stationListString += boost::lexical_cast<std::string>(*it);
                stationListString += " ";
        }
        stationListEdit->setText(stationListString.c_str());

    }
}


bool VrmlNodeCar::stationListChanged()
{
    return temporaryStationList.size()>0;
}

void VrmlNodeCar::switchToNewStationList()// try to switch to new stationList
{
    int currentStation = d_stationList[d_currentStationIndex.get()];
    int index = 0;
    for(std::list<int>::iterator it=temporaryStationList.begin();it !=temporaryStationList.end(); it++)
    {
        if(*it == currentStation)
        {
            // the current station is also in the new station list --> we stay here.
            d_currentStationIndex.set(index);
            int *sl = new int[temporaryStationList.size()];
            int i=0;
            for(std::list<int>::iterator it=temporaryStationList.begin();it !=temporaryStationList.end(); it++)
            {
                sl[i] = *it;
                i++;
            }
            d_stationList.set(temporaryStationList.size(),sl);
            delete[] sl;
            temporaryStationList.clear();
            return;
        }
        index++;
    }
    index = 0;
    // the current station is not in the new station list, thus check if any station is empty and jump to it.
    for(std::list<int>::iterator it=temporaryStationList.begin();it !=temporaryStationList.end(); it++)
    {
        if(elevator->stations[*it].car == NULL) //this one is empty, jump there
        {
            elevator->release(currentStation); // we jump away, this release it.
            elevator->occupy(*it,this); // occupy the destination
            d_currentStationIndex.set(index);
            d_carPos.set(elevator->stations[*it].x(),elevator->stations[*it].y(),0);
            landingNumber = d_stationList[d_currentStationIndex.get()] % elevator->d_landingHeights.size();
            shaftNumber = d_stationList[d_currentStationIndex.get()] / elevator->d_landingHeights.size();
            oldLandingNumber = landingNumber;
            oldShaftNumber = shaftNumber;
            oldLandingIndex = *it; // is >=0 until we left the station
            destinationLandingIndex = *it; // is >=0 until we are close to the destination
            int *sl = new int[temporaryStationList.size()];
            int i=0;
            for(std::list<int>::iterator it=temporaryStationList.begin();it !=temporaryStationList.end(); it++)
            {
                sl[i] = *it;
                i++;
            }
            d_stationList.set(temporaryStationList.size(),sl);
            delete[] sl;
            temporaryStationList.clear();
            return;
        }
        index++;
    }
    // don't do anything and hope next time one of my stations is empty
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
    if(d_currentStationIndex.get() >= d_stationList.size())
    {
        fprintf(stderr,"currentStationIndex out of range\n");
		fprintf(stderr, "d_stationList.size %d\n", d_stationList.size());
		fprintf(stderr, "d_currentStationIndex.get %d\n", d_currentStationIndex.get());
        d_currentStationIndex.set(0);
    }
    elevator->stations[d_stationList[d_currentStationIndex.get()]].car=this;
    landingNumber = d_stationList[d_currentStationIndex.get()] % elevator->d_landingHeights.size();
    shaftNumber = d_stationList[d_currentStationIndex.get()] / elevator->d_landingHeights.size();
    d_carPos.set(elevator->d_shaftPositions[shaftNumber],elevator->d_landingHeights[landingNumber],0);
    double timeStamp = System::the->time();
    eventOut(timeStamp, "carPos", d_carPos);
    state = Idle;
    std::string name = "Car_";
    name += boost::lexical_cast<std::string>(ID);
    
    stationListEdit=new coTUIEditField("stationList",elevator->elevatorTab->getID());
    carLabel = new coTUILabel(name.c_str(),elevator->elevatorTab->getID());
    openButton = new coTUIToggleButton("Open",elevator->elevatorTab->getID());
    carLabel->setPos(ID,10);
    openButton->setPos(ID,11);
    stationListEdit->setPos(ID,12);
    stationListEdit->setEventListener(this);
    std::string stationListString;
    for(int i=0;i<d_stationList.size();i++)
    {
        stationListString += boost::lexical_cast<std::string>(d_stationList.get()[i]);
        if(i < d_stationList.size()-1)
            stationListString += " ";
    }
    stationListEdit->setText(stationListString.c_str());

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
		if (oldTravelDirection == Uninitialized && elevator->exchangers[oldLandingIndex] == NULL)
		{
			setAngle(M_PI_2); // if we start moving horizontally on a horizontal track we have to turn our chassis
		}
    }
    if(shaft < shaftNumber)
    {
        setTravelDirection(VrmlNodeCar::MoveLeft);
        for(int i=oldLandingIndex-elevator->d_landingHeights.size();i>=destinationLandingIndex;i-=elevator->d_landingHeights.size())
        {
            passingStations.push_back(i);
        }
		if (oldTravelDirection == Uninitialized && elevator->exchangers[oldLandingIndex] == NULL)
		{
			setAngle(M_PI_2); // if we start moving horizontally on a horizontal track we have to turn our chassis
		}
    }
    if(passingStations.size()==0)
    {
        // oops, invalid destination, can't reach it
        // work around: just place the destination in there

        passingStations.push_back(destinationLandingIndex);
    }
    setDestination(landing, shaft);
    
    currentPassingStation = passingStations.begin();
    occupiedStations.push_back(oldLandingIndex);
    elevator->putCarOnRail(this);
}

bool VrmlNodeCar::nextPositionIsEmpty() // return true if the destination landing is empty and all exchangers are in the right orientation (if not turn them)
    // if the current exchanger is not in the right orientation, false is returned
{
    int nextIndex = d_currentStationIndex.get()+1;
    if(nextIndex>=d_stationList.size())
        nextIndex=0;
    
    int currentStation = d_stationList[d_currentStationIndex.get()];
    int nextStation = d_stationList[nextIndex];
    
    int startShaft = currentStation / elevator->d_landingHeights.size();
    int destinationShaft = nextStation / elevator->d_landingHeights.size();
    int startLanding = currentStation % elevator->d_landingHeights.size();
    int destinationLanding = nextStation % elevator->d_landingHeights.size();
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
    int nextStationToCheck=0;
    if(startShaft != destinationShaft)
    {
        if(startShaft < destinationShaft)
        {
            nextStationToCheck = currentStation+(elevator->d_landingHeights.size());
        }
        else
        {
            nextStationToCheck = currentStation-(elevator->d_landingHeights.size());
        }
        std::list<VrmlNodeExchanger *>::iterator it;
        for(it = currentExchangers.begin(); it != currentExchangers.end(); it++)
        {
            if(((*it)->getStationNumber() == currentStation) && (*it)->getAngle()!=(float)M_PI_2)
            {
                empty=false;
            }
			if (((*it)->getRotatingState() == VrmlNodeExchanger::Idle) && ((((*it)->getState() == VrmlNodeExchanger::Idle) && ((*it)->getCar() == NULL)) || (*it)->getCar() == this))
				(*it)->rotateRight();
			// else
           // {
           //     if((*it)->getCar()!=this)
           //        empty=false;
           // }

        }
    }
    else
    {
        if(startLanding < destinationLanding)
        {
            nextStationToCheck = currentStation+1;
        }
        else
        {
            nextStationToCheck = currentStation-1;
        }
        std::list<VrmlNodeExchanger *>::iterator it;
        for(it = currentExchangers.begin(); it != currentExchangers.end(); it++)
        {
            if(((*it)->getStationNumber() == currentStation) && (*it)->getAngle()!=0)
            {
                empty=false;
            }

			if (((*it)->getRotatingState() == VrmlNodeExchanger::Idle) && ((((*it)->getState() == VrmlNodeExchanger::Idle) && ((*it)->getCar() == NULL)) || (*it)->getCar() == this))
			{
				(*it)->rotateLeft();
			}
            //else
            //{
            //    if((*it)->getCar()!=this)
            //    empty=false;
            //}
        }
    }
    if(elevator->stations[nextStationToCheck].car!=NULL)
        empty=false;
    if(empty==false)
    {
        return false;
    }
    return true;
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
            oldTravelDirection = travelDirection;
        }
        if(travelDirection == MoveUp || travelDirection == MoveDown)
        {
            oldTravelDirection = travelDirection;
        }
    }
    nextPositionIsEmpty(); // start rotating exchangers on the way to the next destination
}
