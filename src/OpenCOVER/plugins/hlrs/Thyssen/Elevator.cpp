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
#include <cover/coVRTui.h>

#include <net/covise_host.h>
#include <net/covise_socket.h>

using namespace covise;


void VrmlNodeElevator::initFields(VrmlNodeElevator *node, vrml::VrmlNodeType *t)
{
    VrmlNodeGroup::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("landingHeights", node->d_landingHeights),
        exposedField("shaftPositions", node->d_shaftPositions));
}

const char *VrmlNodeElevator::name()
{
    return "Elevator";
}

VrmlNodeElevator::VrmlNodeElevator(VrmlScene *scene)
    : VrmlNodeGroup(scene, name())
{
    setModified();
    elevatorTab = new coTUITab("Elevator", coVRTui::instance()->mainFolder->getID());
    elevatorTab->setPos(0, 0);
}

VrmlNodeElevator::VrmlNodeElevator(const VrmlNodeElevator &n)
    : VrmlNodeGroup(n)
{
    setModified();
    elevatorTab = n.elevatorTab;
}

VrmlNodeElevator *VrmlNodeElevator::toElevator() const
{
    return (VrmlNodeElevator *)this;
}

void VrmlNodeElevator::childrenChanged()
{
    
    stations.resize(d_landingHeights.size()*d_shaftPositions.size());
    exchangers.resize(d_landingHeights.size()*d_shaftPositions.size());
    landings.resize(d_landingHeights.size()*d_shaftPositions.size());
    for(int i=0;i<stations.size();i++)
    {
        stations[i].car=NULL;
        exchangers[i]=NULL;
        landings[i]=NULL;
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
                int oldSize=exchangers.size();
                int newSize=exchanger->d_LandingNumber.get()+1;
                exchangers.resize(newSize);
                for(int i=oldSize;i<newSize;i++)
                {
                    exchangers[i]=NULL;
                }
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
                landings.resize(newSize);
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
    
    for(int i=0;i<stations.size();i++)
    {
        
        int landing = i % d_landingHeights.size();
        int shaft = i / d_landingHeights.size();
        stations[i].setX(d_shaftPositions[shaft]);
        stations[i].setY(d_landingHeights[landing]);
    }
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
			VrmlNodeExchanger *ex = exchangers[car->d_stationList[car->d_currentStationIndex.get()]];
			
            if((car->getState()==VrmlNodeCar::Idle) && (ex == NULL || (ex->getRotatingState() == VrmlNodeExchanger::Idle)))
            {
                if(car->stationListChanged())
                {
                    // try to switch to new stationList
                    car->switchToNewStationList();
                }
                if(car->nextPositionIsEmpty())
                {
                    // tell it to move to next stop
                    car->moveToNext();
                }
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
    bool success=false;
    if(stations[station].car == NULL || stations[station].car == car)
    {
        success = true;
        stations[station].car = car;
    }
    else
    {
        return false;
    }
    
    if(exchangers.size() > station && exchangers[station] !=NULL)
    {
        VrmlNodeExchanger *ex = exchangers[station];
        if(ex->getCar() != NULL && ex->getCar() != car)
            return false;
        // find out if the exchanger is posittioned right, otherwise we can't occupy it.
        if(car->getTravelDirection() == VrmlNodeCar::MoveLeft || car->getTravelDirection() == VrmlNodeCar::MoveRight)
        {
            if(ex->getAngle()!=(float)M_PI_2)
            {
                // turn it right
				if((ex->getRotatingState() == VrmlNodeExchanger::Idle) && (((ex->getState() == VrmlNodeExchanger::Idle) && (ex->getCar() == NULL)) || ex->getCar() == car))
                {
                    exchangers[station]->setCar(car);
                    ex->rotateRight();
                }
                return false;
            }
        }
        else
        {
            if(ex->getAngle()!=0)
            {
                // turn it right
				if ((ex->getRotatingState() == VrmlNodeExchanger::Idle) && (((ex->getState() == VrmlNodeExchanger::Idle) && (ex->getCar() == NULL)) || ex->getCar() == car))
                {
                    exchangers[station]->setCar(car);
                    ex->rotateLeft();
                }
                return false;
            }
        }
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
    return success; // no landing and no exchanger
}
void VrmlNodeElevator::release(int station)
{
    stations[station].car = NULL;
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
    shafts[car->getShaftNumber()]->removeCarFromRail(car);
    hShafts[car->getLandingNumber()]->removeCarFromRail(car);
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