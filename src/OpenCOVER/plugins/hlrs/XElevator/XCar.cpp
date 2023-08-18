/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//

#include "XCar.h"
#include "XElevator.h"
#include "XLanding.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>

#include <boost/lexical_cast.hpp>

using namespace covise;


static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeXCar(scene);
}

// Define the built in VrmlNodeType:: "XCar" fields

VrmlNodeType *VrmlNodeXCar::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("XCar", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class


    t->addExposedField("carNumber", VrmlField::SFINT32);
    t->addExposedField("carPos", VrmlField::SFVEC3F);
    t->addExposedField("carOffset", VrmlField::SFVEC3F);
    t->addExposedField("stationList", VrmlField::MFINT32);
    t->addExposedField("stationOpenTime", VrmlField::MFFLOAT);
    t->addExposedField("currentLanding", VrmlField::SFINT32);
    t->addEventOut("carTransformPos", VrmlField::SFVEC3F);
    t->addEventOut("carDoorClose", VrmlField::SFTIME);
    t->addEventOut("carDoorOpen", VrmlField::SFTIME);
    t->addEventOut("carRotation", VrmlField::SFROTATION);
    t->addEventOut("carFraction", VrmlField::SFFLOAT);
	t->addEventOut("Unlock", VrmlField::SFTIME);
	t->addEventOut("Lock", VrmlField::SFTIME);

    return t;
}

VrmlNodeType *VrmlNodeXCar::nodeType() const
{
    return defineType(0);
}

int VrmlNodeXCar::IDCounter=0;
VrmlNodeXCar::VrmlNodeXCar(VrmlScene *scene)
    : VrmlNodeChild(scene)
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
    d_currentLanding=0;
    ID = IDCounter++;
    
}

VrmlNodeXCar::VrmlNodeXCar(const VrmlNodeXCar &n)
    : VrmlNodeChild(n.d_scene)
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
    d_doorTimeout=3.0;
    d_carRotation.set(1,0,0,0);
    d_carFraction=0.0;
    d_currentLanding=0;
    ID = IDCounter++;
}

VrmlNodeXCar::~VrmlNodeXCar()
{
}

void VrmlNodeXCar::lock()
{
	double timeStamp = System::the->time();
	d_lockTime.set(timeStamp);
	eventOut(timeStamp, "Lock", d_lockTime);
}
void VrmlNodeXCar::unlock()
{
	double timeStamp = System::the->time();
	d_lockTime.set(timeStamp);
	eventOut(timeStamp, "Unlock", d_lockTime);
}

enum VrmlNodeXCar::CarState VrmlNodeXCar::getState(){return state;}
void VrmlNodeXCar::setState(enum CarState s){oldState=state; state = s;}
enum VrmlNodeXCar::CarState VrmlNodeXCar::getChassisState(){return chassisState;}
void VrmlNodeXCar::setChassisState(enum CarState s){oldChassisState=chassisState; chassisState = s;}
enum VrmlNodeXCar::CarState VrmlNodeXCar::getTravelDirection(){return travelDirection;}
void VrmlNodeXCar::setTravelDirection(enum CarState t)
{
    oldTravelDirection=travelDirection;
    travelDirection = t;
}

VrmlNode *VrmlNodeXCar::cloneMe() const
{
    return new VrmlNodeXCar(*this);
}

VrmlNodeXCar *VrmlNodeXCar::toXCar() const
{
    return (VrmlNodeXCar *)this;
}

ostream &VrmlNodeXCar::printFields(ostream &os, int indent)
{

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeXCar::setField(const char *fieldName,
                           const VrmlField &fieldValue)
{

    if
        TRY_FIELD(carNumber, SFInt)
    else if
    TRY_FIELD(carPos, SFVec3f)
    else if
    TRY_FIELD(carOffset, SFVec3f)
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
    TRY_FIELD(currentLanding, SFInt)
    else
    VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeXCar::getField(const char *fieldName)
{
    if (strcmp(fieldName, "carNumber") == 0)
        return &d_carNumber;
    else if (strcmp(fieldName, "carPos") == 0)
        return &d_carPos;
    else if (strcmp(fieldName, "carOffset") == 0)
        return &d_carOffset;
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
    else if (strcmp(fieldName, "currentLanding") == 0)
        return &d_currentLanding;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeXCar::eventIn(double timeStamp,
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

void VrmlNodeXCar::goTo(int landing)
{
    stationList.insert(landing);
    if (d_currentLanding.get() == landing)
        arrivedAtDestination();
}

void VrmlNodeXCar::render(Viewer *)
{


}

void VrmlNodeXCar::update()
{
    if(state == Moving && chassisState == Idle)
    {
        
        float dt = cover->frameDuration();
        if(dt > 1000) // first frameDuration is off because last FrameTime is 0
            dt=0.00001;
        if(d_carPos.y() != destinationY) // moving vertically
        {
            float direction;
            float diff = fabs(destinationY - d_carPos.y());
            float diffS = fabs(startingY - d_carPos.y());

            float v2 = v*v;
            float bakeDistance = (v2/(2*aMax))*1.5; // distance the XCar travels until it stops at max decelleration
            
            if(d_carPos.y() < destinationY)
            {
                direction = 1;
            }
            else
            {
                direction = -1;
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
            d_carTransformPos.set(d_carPos.x() + d_carOffset.x(), d_carPos.y() + d_carOffset.y(), d_carPos.z() + d_carOffset.z());
            eventOut(timeStamp, "carTransformPos", d_carTransformPos);

        }
        else // we are there
        {
            v=0;a=0;
            d_currentLanding.set(*stationList.begin());
            arrivedAtDestination();
        }
    }
    else if(state == DoorOpening)
    {
        
        if(abs(cover->frameTime() - timeoutStart) > 3 )
        {
            timeoutStart = cover->frameTime();
            state = DoorOpen;
        }
    }
    else if(state == DoorOpen)
    {
        float doorOpenTime = d_doorTimeout.get();
        if(d_stationOpenTime.size() > d_currentLanding.get() && d_stationOpenTime[d_currentLanding.get()]>0)
        {
            doorOpenTime = d_stationOpenTime[d_currentLanding.get()];
        }
        if((abs(cover->frameTime() - timeoutStart) > doorOpenTime ) && openButton->getState()==false)
        {
            timeoutStart = cover->frameTime();
            d_carDoorClose = System::the->time();
            eventOut(d_carDoorOpen.get(), "carDoorClose", d_carDoorClose);
            if(d_currentLanding.get() < Elevator->Landings.size() && Elevator->Landings[d_currentLanding.get()]!=NULL)
            {
                Elevator->Landings[ d_currentLanding.get()]->closeDoor();
            }
            state = DoorClosing;
        }
    }
    else if(state == DoorClosing)
    {
        if(abs(cover->frameTime() - timeoutStart) > 3 )
        {
            timeoutStart = cover->frameTime();
            state = Idle;
        }
    }
    

}


void VrmlNodeXCar::tabletPressEvent(coTUIElement *tUIItem)
{
}

void VrmlNodeXCar::tabletEvent(coTUIElement *tUIItem)
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
            if(st >= 0 && st < Elevator->stations.size())
            {
                temporaryStationList.push_back(st);
            }
        } while (iss);
        // check station list for consistency (can't travel diagonal)

        int numHeights = Elevator->d_landingHeights.size();

        int oldLandingNumber=-1;
        int oldShaftNumber=-1;
        std::list<int>::iterator it;
        for(it=temporaryStationList.begin();it !=temporaryStationList.end(); it++)
        {
            int LandingNumber = *it % numHeights;
            int shaftNumber = *it / numHeights;
            if(oldLandingNumber > 0)
            {
                if(oldLandingNumber != LandingNumber && oldShaftNumber != shaftNumber) // this does not work.
                {
                    // add another step
                    temporaryStationList.insert(it,oldShaftNumber*numHeights+LandingNumber);
                    oldLandingNumber = LandingNumber;
                }
            }
            oldLandingNumber = LandingNumber;
            oldShaftNumber = shaftNumber;
        }
        it=temporaryStationList.begin();
        int LandingNumber = *it % numHeights;
        int shaftNumber = *it / numHeights;
        if(oldLandingNumber > 0)
        {
            if(oldLandingNumber != LandingNumber && oldShaftNumber != shaftNumber) // this does not work.
            {
                // add another step
                temporaryStationList.insert(it,oldShaftNumber*numHeights+LandingNumber);
                oldLandingNumber = LandingNumber;
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


void VrmlNodeXCar::setElevator(VrmlNodeXElevator *e)
{
    Elevator = e;
    if(d_currentLanding.get() >= Elevator->Landings.size())
    {
        fprintf(stderr,"currentLanding out of range\n");
		fprintf(stderr, "Landings.size %zd\n", Elevator->Landings.size());
		fprintf(stderr, "d_currentLanding.get %d\n", d_currentLanding.get());
        d_currentLanding.set(0);
    }
    Elevator->stations[d_currentLanding.get()].Car=this;
    LandingNumber = d_currentLanding.get();
    d_carPos.set(0,Elevator->d_landingHeights[LandingNumber],0);
    double timeStamp = System::the->time();
    eventOut(timeStamp, "carPos", d_carPos);
    d_carTransformPos.set(d_carPos.x() + d_carOffset.x(),d_carPos.y() + d_carOffset.y(),d_carPos.z() + d_carOffset.z());
    eventOut(timeStamp, "carTransformPos", d_carTransformPos);
    state = Idle;
    std::string name = "Car_";
    name += boost::lexical_cast<std::string>(ID);
    
    stationListEdit=new coTUIEditField("stationList",Elevator->XElevatorTab->getID());
    CarLabel = new coTUILabel(name.c_str(),Elevator->XElevatorTab->getID());
    openButton = new coTUIToggleButton("Open",Elevator->XElevatorTab->getID());
    CarLabel->setPos(ID,10);
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
void VrmlNodeXCar::setDestination(int landing)
{
    if(landing != LandingNumber )
    {   
        oldLandingNumber=LandingNumber;

        LandingNumber = landing;
        state = Moving;

        startingY = Elevator->d_landingHeights[oldLandingNumber];

        destinationY = Elevator->d_landingHeights[landing];
    }

}
void VrmlNodeXCar::moveToNext()
{
    if (stationList.size() == 0)
        return; // nowhere to go
    int landing = *stationList.begin();
    if(landing > LandingNumber)
    {
        setTravelDirection(VrmlNodeXCar::MoveUp);
    }
    if(landing < LandingNumber)
    {
        setTravelDirection(VrmlNodeXCar::MoveDown);
    }
    setDestination(landing);
    
}


void VrmlNodeXCar::arrivedAtDestination() // the XCar arrived at its destination
{
    timeoutStart = cover->frameTime();
    state = DoorOpening;
    d_carDoorOpen = System::the->time();
    eventOut(d_carDoorOpen.get(), "carDoorOpen", d_carDoorOpen);
    
    if(d_currentLanding.get() < Elevator->Landings.size() && Elevator->Landings[d_currentLanding.get()]!=NULL)
    {
        Elevator->Landings[ d_currentLanding.get()]->openDoor();
    }

    stationList.erase(d_currentLanding.get());
    if (stationList.size() > 0)
    {
        int Landing = *stationList.begin();

        if (Landing != d_currentLanding.get()) // next XLanding != this XLanding
        {
            if (Landing > d_currentLanding.get())
                travelDirection = MoveUp;
            else
                travelDirection = MoveDown;
        }
        if (travelDirection != oldTravelDirection)
        {
            oldTravelDirection = travelDirection;
        }
    }
}
