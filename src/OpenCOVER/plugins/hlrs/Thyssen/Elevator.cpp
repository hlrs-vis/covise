/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//

#include "Elevator.h"
#include "Car.h"
#include "Exchanger.h"
#include "Landing.h"

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
        
        stations.resize(d_landingHeights.size()*d_shaftPositions.size());
        for(int i=0;i<stations.size();i++)
        {
            stations[i]=NULL;
        }
        for(int i=0;i<d_landingHeights.size();i++)
        {
            hShafts.push_back(new Rail);
        }
        for(int i=0;i<d_shaftPositions.size();i++)
        {
            shafts.push_back(new Rail);
        }

        for(int i=0;i<d_children.size();i++)
        {
            VrmlNodeCar *car = dynamic_cast<VrmlNodeCar *>(d_children[i]);
            if(car)
            {
                if(car->d_carNumber.get() >= cars.size())
                {
                    cars.resize(car->d_carNumber.get()+1);
                }
                cars[car->d_carNumber.get()] = car;
                car->setElevator(this);
            }
            VrmlNodeExchanger *exchanger = dynamic_cast<VrmlNodeExchanger *>(d_children[i]);
            if(exchanger)
            {
                if(exchanger->d_LandingNumber.get() >= exchangers.size())
                {
                    exchangers.resize(exchanger->d_LandingNumber.get()+1);
                }
                exchangers[exchanger->d_LandingNumber.get()] = exchanger;
                exchanger->setElevator(this);
            }
            VrmlNodeLanding *landing = dynamic_cast<VrmlNodeLanding *>(d_children[i]);
            if(landing)
            {
                if(landing->d_LandingNumber.get() >= landings.size())
                {
                    int oldSize=landings.size();
                    int newSize=landing->d_LandingNumber.get()+1;
                    landings.resize(landing->d_LandingNumber.get()+1);
                    for(int i=oldSize;i<newSize;i++)
                    {
                        landings[i]=NULL;
                    }
                }
                landings[landing->d_LandingNumber.get()] = landing;
                landing->setElevator(this);
            }
        }
        //assign cars to exchangers
        for(int i=0;i<cars.size();i++)
        {
            if(cars[i])
            {
                int station = cars[i]->d_stationList[cars[i]->d_currentStationIndex.get()];
                if(exchangers.size() > station && exchangers[station] !=NULL)
                {
                    exchangers[station]->setCar(cars[i]);
                }
            }if(cars[i])
            {
                int station = cars[i]->d_stationList[cars[i]->d_currentStationIndex.get()];
                if(landings.size() > station && landings[station] !=NULL)
                {
                    landings[station]->setCar(cars[i]);
                }
            }
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
    
    for(int i=0;i<cars.size();i++)
    {
        VrmlNodeCar *car = cars[i];
        if(car!=NULL)
        {
            int lowerLanding=0;
            if(car->getID()%2)
                lowerLanding = 1;
            int upperLanding=d_landingHeights.size()-1;
            if(car->getID()%2)
                upperLanding--;
            
            if(car->getState()==VrmlNodeCar::Idle && car->nextPositionIsEmpty())
            {
                // tell it to move to next stop
                car->moveToNext();
            }
            car->update();
        }
    }
    for(int i=0;i<exchangers.size();i++)
    {
        VrmlNodeExchanger *exchanger = exchangers[i];
        if(exchanger!=NULL)
        {
            exchanger->update();
        }
    }

}


bool VrmlNodeElevator::occupy(int station,VrmlNodeCar *car)
{
    stations[station] = car;
    bool success=false;
    
    if(exchangers.size() > station && exchangers[station] !=NULL)
    {
        if(exchangers[station]->getCar() != NULL && exchangers[station]->getCar() != car)
            return false;
        exchangers[station]->setCar(car);
        success=true;
    }
    if(landings.size() > station && landings[station] !=NULL)
    {
        if(landings[station]->getCar() != NULL && landings[station]->getCar() != car)
            return false;
        landings[station]->setCar(car);
        success=true;
    }
    return success;
}
void VrmlNodeElevator::release(int station)
{
    stations[station] = NULL;
    if(exchangers.size() > station && exchangers[station] !=NULL)
    {
        exchangers[station]->setCar(NULL);
    }
    if(landings.size() > station && landings[station] !=NULL)
    {
        landings[station]->setCar(NULL);
    }
}


void VrmlNodeElevator::putCarOnRail(VrmlNodeCar *car)
{
    VrmlNodeCar::carState cs = car->getTravelDirection();
    if(cs==VrmlNodeCar::MoveDown || cs==VrmlNodeCar::MoveUp)
    {
        shafts[car->getShaftNumber()]->putCarOnRail(car);
    }
    else
    {
        hShafts[car->getLandingNumber()]->putCarOnRail(car);
    }
}
void VrmlNodeElevator::removeCarFromRail(VrmlNodeCar *car)
{
    VrmlNodeCar::carState cs = car->getTravelDirection();
    if(cs==VrmlNodeCar::MoveDown || cs==VrmlNodeCar::MoveUp)
    {
        shafts[car->getShaftNumber()]->removeCarFromRail(car);
    }
    else
    {
        hShafts[car->getLandingNumber()]->removeCarFromRail(car);
    }
}

float VrmlNodeElevator::getNextCarOnRail(VrmlNodeCar *car, VrmlNodeCar *&closestCar)
{
    VrmlNodeCar::carState cs = car->getTravelDirection();
    if(cs==VrmlNodeCar::MoveDown || cs==VrmlNodeCar::MoveUp)
    {
        return shafts[car->getShaftNumber()]->getNextCarOnRail(car,closestCar);
    }
    else
    {
        return hShafts[car->getLandingNumber()]->getNextCarOnRail(car,closestCar);
    }
}





void Rail::putCarOnRail(VrmlNodeCar *car)
{
    carsOnRail.push_back(car);
}
void Rail::removeCarFromRail(VrmlNodeCar *car)
{
    carsOnRail.remove(car);
}

float Rail::getNextCarOnRail(VrmlNodeCar *car, VrmlNodeCar *&closestCar)
{
     VrmlNodeCar::carState cs = car->getTravelDirection();
    closestCar = NULL;
    float myHeight = car->d_carPos.y();
    float myX = car->d_carPos.x();
#define INFINITE_HEIGHT 1000000.0
    float minDistance = INFINITE_HEIGHT;
    if(cs==VrmlNodeCar::MoveDown)
    {
        for(std::list<VrmlNodeCar *>::iterator it = carsOnRail.begin(); it != carsOnRail.end(); it++)
        {
            if(*it != car)
            {
                float height = (*it)->d_carPos.y();
                if(height < myHeight)
                {
                    if((myHeight - height) < minDistance)
                    {
                        minDistance = myHeight - height;
                        closestCar = (*it);
                    }
                }
            }
        }
            
    }
    else if(cs==VrmlNodeCar::MoveUp)
    {
        for(std::list<VrmlNodeCar *>::iterator it = carsOnRail.begin(); it != carsOnRail.end(); it++)
        {
            if(*it != car)
            {
                float height = (*it)->d_carPos.y();
                if(height > myHeight)
                {
                    if((height - myHeight) < minDistance)
                    {
                        minDistance = height - myHeight;
                        closestCar = (*it);
                    }
                }
            }
        }
    }
    
    if(cs==VrmlNodeCar::MoveLeft)
    {
        for(std::list<VrmlNodeCar *>::iterator it = carsOnRail.begin(); it != carsOnRail.end(); it++)
        {
            if(*it != car)
            {
                float currentX = (*it)->d_carPos.x();
                if(currentX < myX)
                {
                    if((myX - currentX) < minDistance)
                    {
                        minDistance = myX - currentX;
                        closestCar = (*it);
                    }
                }
            }
        }
            
    }
    else if(cs==VrmlNodeCar::MoveRight)
    {
        for(std::list<VrmlNodeCar *>::iterator it = carsOnRail.begin(); it != carsOnRail.end(); it++)
        {
            if(*it != car)
            {
                float currentX = (*it)->d_carPos.x();
                if(currentX > myX)
                {
                    if((currentX - myX) < minDistance)
                    {
                        minDistance = currentX - myX;
                        closestCar = (*it);
                    }
                }
            }
        }
    }
    if(minDistance == INFINITE_HEIGHT)
        return -1;
    return minDistance;
}