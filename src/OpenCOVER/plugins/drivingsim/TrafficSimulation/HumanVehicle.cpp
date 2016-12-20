/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "HumanVehicle.h"

#include <osg/Matrix>
#include <cover/coVRPluginSupport.h>
#include <set>
#include <limits>
#include "RoadSystem/RoadSystem.h"
#include "TrafficSimulation.h"

#ifdef __XENO__
#include "GasPedal.h"
#include <cover/coVRMSController.h>
#endif

osg::Vec2d HumanVehicle::human_pos(0, 0);
double HumanVehicle::human_v = 0.0;

HumanVehicle::HumanVehicle(std::string name)
    : Vehicle(name)
    , road(NULL)
    , u(-1.0)
    , v(0.0)
    , oldU(-1.0)
    , du(0.0)
    , currentLane(0)
    , currentSection(NULL)
    , hdg(0)
    , geometry(NULL)
//geometry(NULL),
//steeringWheelPlugin(NULL)
{
    geometry = new HumanVehicleGeometry(name);
}

Road *HumanVehicle::getRoad() const
{
    return road;
}

double HumanVehicle::getU() const
{
    return u;
}

double HumanVehicle::getDu() const
{
    return du;
}

double HumanVehicle::getV() const
{
    return v;
}

int HumanVehicle::getLane() const
{
    return currentLane;
}

bool HumanVehicle::isOnLane(int currentLane) const
{
    if (road && currentSection)
    {
        double boundRadius = (geometry == NULL) ? 0.0 : geometry->getBoundingCircleRadius();
        double extent = fabs(boundRadius * sin(hdg));
        int lowLane = currentSection->searchLane(u, v - extent);
        if (lowLane == Lane::NOLANE)
        {
            lowLane = currentSection->getTopRightLane();
        }
        int highLane = currentSection->searchLane(u, v + extent);
        if (highLane == Lane::NOLANE)
        {
            highLane = currentSection->getTopLeftLane();
        }
        //std::cout << "boundRadius: " << boundRadius << ", hdg: " << hdg << ", extent: " << extent << ", lowLane: " << lowLane << ", highLane: " << highLane << std::endl;

        if (currentLane >= lowLane && currentLane <= highLane)
        {
            return true;
        }
    }

    return false;
}

void HumanVehicle::move(double dt)
{
    Transform vehTrans = geometry->getVehicleTransformation();
    //std::cout << "Vehicle position: " << vehTrans.v();
    if (vehTrans.isNaT())
    {
        return;
    }
    human_pos = osg::Vec2d(vehTrans.v().x(), vehTrans.v().y());
    Road *newRoad = NULL;
    bool foundPos = false;

    Vector2D pos(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
    if (this->road)
    {
        pos = this->road->searchPosition(vehTrans.v(), u);
    }
    if (!pos.isNaV())
    {
        foundPos = true;
        newRoad = this->road;
    }
    else
    {
        std::set<Road *, bool (*)(Road *, Road *)> roadSet(Road::compare);
        for (int roadIt = 0; roadIt < RoadSystem::Instance()->getNumRoads(); ++roadIt)
        {
            Road *road = RoadSystem::Instance()->getRoad(roadIt);
            osg::BoundingBox roadBox = road->getRoadGeode()->getBoundingBox();
            roadBox.zMin() -= 1;
            roadBox.zMax() += 1;
            if (roadBox.contains(osg::Vec3(vehTrans.v().x(), vehTrans.v().y(), vehTrans.v().z())))
            {
                roadSet.insert(road);
            }
        }

        if (!roadSet.empty())
        {
            for (std::set<Road *, bool (*)(Road *, Road *)>::iterator roadSetIt = roadSet.begin(); roadSetIt != roadSet.end(); ++roadSetIt)
            {
                pos = (*roadSetIt)->searchPosition(vehTrans.v(), -1);

                if (!pos.isNaV())
                {
                    foundPos = true;
                    newRoad = (*roadSetIt);
                    break;
                }
            }
        }
    }

    if (!foundPos)
    {
        u = v = oldU = 0.0;
        currentLane = 0;
        currentSection = NULL;
        hdg = 0;
        if (newRoad != this->road)
        {
            TrafficSimulation::instance()->getVehicleManager()->changeRoad(this, road, newRoad, 1);
            this->road = newRoad;
        }
    }
    else
    {
        oldU = u;
        u = pos.u();
        v = pos.v();

        if (newRoad != this->road)
        {
            TrafficSimulation::instance()->getVehicleManager()->changeRoad(this, road, newRoad, 1);
            this->road = newRoad;
        }
        else if (road)
        {
            du = (u - oldU) / dt;
        }

        if (road)
        {
            currentLane = road->searchLane(u, v);
            currentSection = road->getLaneSection(u);
            Vector3D roadTangent = road->getTangentVector(u);
            Vector3D roadTangentRelVeh = (vehTrans.q().T() * roadTangent * vehTrans.q()).getVector();

            //Vector3D vehDir = (vehTrans.q()*Vector3D(1.0,0.0,0.0)*vehTrans.q().T()).getVector();
            //Vector3D vehDiry = (vehTrans.q()*Vector3D(0.0,1.0,0.0)*vehTrans.q().T()).getVector();
            //Vector3D vehDirz = (vehTrans.q()*Vector3D(0.0,0.0,1.0)*vehTrans.q().T()).getVector();
            //double hdgRoadRelVeh = atan2(roadTangentRelVeh.y(), roadTangentRelVeh.x());

            //hdg = acos((roadTangent.dot(vehDir))/(roadTangent.length()*vehDir.length()));
            //if(hdg!=hdg) {
            //   hdg = 0.0;
            //}
            /*std::cout << "Road tangent: " << roadTangent << "Road tangent relative to vehicle: " << roadTangentRelVeh << "Vehicle direction: " << vehDir;
         std::cout << "Vehicle direction y: " << vehDiry << "Vehicle direction z: " << vehDirz;
			std::cout << "RoadTangent dot VehicleDir: " << roadTangent.dot(vehDir) << std::endl;
			std::cout << "Road tangent length: " << roadTangent.length() << ", vehicle direction length: " << vehDir.length() << std::endl;
			std::cout << "Heading road relative to vehicle: " << hdgRoadRelVeh << std::endl;*/

            hdg = -atan2(roadTangentRelVeh.y(), roadTangentRelVeh.x());
            //std::cout << "Heading: " << hdg << std::endl;
        }
    }

/*std::cout << "road: " << road << ", id: " << (road ? road->getId() : "") << ", u: " << u << ", v: " << v << ", du: " << du;
	if(currentSection) {
		std::cout << ", numLanesLeft: " << currentSection->getNumLanesLeft() << ", numLanesRight: " << currentSection->getNumLanesRight();
	}
	std::cout << ", currentLane:";
	if(currentSection) {
			  for(int l = -currentSection->getNumLanesRight(); l<=currentSection->getNumLanesLeft(); ++l) {
						 if(this->isOnLane(l)) {
									std::cout << " " << l;
						 }
			  }
	}
	std::cout << std::endl;*/

/*if(!steeringWheelPlugin) {
      steeringWheelPlugin = cover->getPlugin("SteeringWheel");
   }
   if(steeringWheelPlugin && road) {
      int numextra = 10;
      int vehNumIt = 0;
      
      VehicleList vehList(TrafficSimulation::instance()->getVehicleManager()->getVehicleList(road));
      VehicleList::iterator thisVehIt = vehList.find(this);
      if(thisVehIt != vehList.end()) {
         VehicleList::iterator vehItRear = thisVehIt;
         VehicleList::iterator vehItFore = thisVehIt; ++vehItFore;
         while((vehItRear!=vehList.begin() && vehItFore!=vehList.end()) && vehNumIt<16) {
            osg::Matrix vehMat((*vehItRear)->getVehicleGeometry()->getVehicleTransformMatrix());
            steeringWheelPlugin->dataController->floatValuesOut[vehNumIt*3+numextra+0]=-vehMat.y();
            steeringWheelPlugin->dataController->floatValuesOut[vehNumIt*3+numextra+1]=vehMat.z();
            steeringWheelPlugin->dataController->floatValuesOut[vehNumIt*3+numextra+2]=-vehMat.x();
         }
      }
   }*/

/*std::map<double, Vehicle*> vehMap = TrafficSimulation::instance()->getVehicleManager()->getSurroundingVehicles(this);
   std::cout << "Surrounding Vehicles:" << std::endl;
   for(std::map<double, Vehicle*>::iterator mapIt = vehMap.begin(); mapIt != vehMap.end(); ++mapIt) {
      std::cout << "\tVehicle: " << mapIt->second->getName() << ", distance: " << mapIt->first << std::endl;
   }*/

//Gaspedal control
#ifdef __XENO__
#if true //uncommented
    if (coVRMSController::instance()->isMaster())
    {
        ObstacleRelation obstRel = locateNextVehicleOnLane();
        if (obstRel.vehicle)
        {
            double horScale = std::max(0.001, (10 + obstRel.diffDu) * 0.001);
            GasPedal::instance()->setTargetValueAngle(100.0 * tanh(std::max(0.0, obstRel.diffU - 10.0) * horScale));
            GasPedal::instance()->setTargetValueMaxTargetForce(100.0);
            GasPedal::instance()->setTargetValueMinTargetForce(50.0);
            GasPedal::instance()->setTargetValueStiffness(70.0);
            if (obstRel.diffDu < -5.0 && obstRel.diffU < 20.0)
            {
                GasPedal::instance()->setJitterAmplitude(6);
                GasPedal::instance()->setJitterFrequency(3);
            }
            else
            {
                GasPedal::instance()->setJitterAmplitude(0);
                GasPedal::instance()->setJitterFrequency(0);
            }
        }
        else
        {
            GasPedal::instance()->setTargetValueAngle(100.0);
            GasPedal::instance()->setTargetValueMaxTargetForce(0.0);
            GasPedal::instance()->setTargetValueMinTargetForce(0.0);

            GasPedal::instance()->setJitterAmplitude(0);
            GasPedal::instance()->setJitterFrequency(0);
        }
    }
#endif
#endif

    /*ObstacleRelation obstRel = locateNextVehicleOnLane();
   if(obstRel.vehicle) {
      std::cout << "Next vehicle: " << obstRel.vehicle->getName() << ", diffu: " << obstRel.diffU << ", diffDu: " << obstRel.diffDu << std::endl;
   }*/
    human_v = v;
}

VehicleGeometry *HumanVehicle::getVehicleGeometry()
{
    return geometry;
}

Transform HumanVehicle::getVehicleTransform()
{
    return geometry->getVehicleTransformation();
}

double HumanVehicle::getBoundingCircleRadius()
{
    return (geometry == NULL) ? 0.0 : geometry->getBoundingCircleRadius();
}

ObstacleRelation HumanVehicle::locateNextVehicleOnLane()
{
    double dis = 0;
    double dvel = 0;
    int dirTrans = (hdg < -M_PI * 0.5 || hdg > M_PI * 0.5) ? -1 : 1;
    int lane = currentLane;

    VehicleList vehList = VehicleManager::Instance()->getVehicleList(road);
    if (dirTrans < 0)
    {
        vehList.reverse();
    }
    VehicleList::iterator vehIt = std::find(vehList.begin(), vehList.end(), this);
    VehicleList::iterator nextVehIt = vehIt;
    double nextVehOldU = (*nextVehIt)->getU();
    ++nextVehIt;
    while (nextVehIt != vehList.end() && lane != Lane::NOLANE)
    {
        double nextVehNewU = (*nextVehIt)->getU();
        lane = road->traceLane(lane, nextVehOldU, nextVehNewU);
        if (lane == Lane::NOLANE)
        {
            return ObstacleRelation(NULL, 0, 0);
        }
        nextVehOldU = nextVehNewU;
        if ((*nextVehIt)->isOnLane(lane))
        {
            break;
        }
        ++nextVehIt;
    }

    if (nextVehIt != vehList.end())
    {
        double bodyExtent = getBoundingCircleRadius() + (*nextVehIt)->getBoundingCircleRadius();
        dis = dirTrans * ((*nextVehIt)->getU() - this->u) - bodyExtent;
        dvel = dirTrans * ((*nextVehIt)->getDu() - this->du);
        return ObstacleRelation((*nextVehIt), dis, dvel);
    }
    else
    {
        return ObstacleRelation(NULL, 0, 0);
    }
}
