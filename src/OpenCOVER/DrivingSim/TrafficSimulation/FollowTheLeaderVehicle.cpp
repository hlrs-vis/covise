/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FollowTheLeaderVehicle.h"

#include "Junction.h"
#include "TrafficSimulationPlugin.h"
#include <algorithm>
#include <set>

FollowTheLeaderVehicle::FollowTheLeaderVehicle(std::string n, CarGeometry *geo, Road *r, double startu, int startLane, double startVel, int startDir)
    : Vehicle(n)
    , currentLane(startLane)
    , u(startu)
    , geometry(NULL)
    , vel(startVel)
    , kv(5.0)
    , dampingV(5.0)
    , Tv(5)
    , dHdg(0.0)
    , Khdg(2.0)
    , Dhdg(20.0)
    , Kv(0.2)
    , Dv(2.0)
    , dist(2)
    ,
    //dist(10),
    velt(startVel)
    , velw(startVel)
    , aqmax(6.0)
    ,
    //T(0.0),
    T(1.4)
    ,
    //a(1.2),
    a(3.0)
    , b(1.5)
    , delta(4.0)
    , cDelta(0.8)
    , p(0.3)
    , brakeDec(0.0)
    , minTransitionListExtent(1000)
    , boundRadius(0.0)
    , junctionWaitTime(0.5)
    , junctionWaitTriggerDist(2)
    , sowedDetermineNextRoadVehicleAction(false)
    , timer(0.0)
{
    if (r)
    {
        s = 0;
        du = startVel;
        dv = 0;
        setRoadType(r->getRoadType(startu));

        roadTransitionList.push_back(RoadTransition(r, startDir));
        currentTransition = roadTransitionList.begin();

        currentSection = r->getLaneSection(startu);
        Vector2D laneCenter = currentSection->getLaneCenter(startLane, startu);
        v = laneCenter[0];
        //hdg =  M_PI*((currentTransition->direction - 1)/2) + laneCenter[1];
        hdg = M_PI * ((1 - currentTransition->direction) / 2);

        vehicleTransform = r->getRoadTransform(startu, laneCenter[0]);
    }

    if (geo)
    {
        geometry = geo;
        boundRadius = geo->getBoundingCircleRadius();
        geometry->setTransform(vehicleTransform, hdg);
        junctionWaitTriggerDist += boundRadius + dist;
    }
    //else {
    //   geometry = new CarGeometry(this->name);
    //}

    wMatMap.insert(std::pair<DrivingState, Matrix3D3D>(MOTORWAY, Matrix3D3D(this->p, 1.0, this->p,
                                                                            this->p, 1.0, 0.0,
                                                                            this->p, 1.0, this->p)));

    wMatMap.insert(std::pair<DrivingState, Matrix3D3D>(RURAL, Matrix3D3D(0.0, 1.0, this->p,
                                                                         0.0, 1.0, 0.0,
                                                                         0.0, 1.0, this->p)));

    wMatMap.insert(std::pair<DrivingState, Matrix3D3D>(OVERTAKE, Matrix3D3D(this->p, 1.0, this->p,
                                                                            this->p, 1.0, 0.0,
                                                                            this->p, 1.0, this->p)));

    wMatMap.insert(std::pair<DrivingState, Matrix3D3D>(EXIT_MOTORWAY, Matrix3D3D(this->p, 1.0, this->p,
                                                                                 this->p, 1.0, 0.0,
                                                                                 this->p, 1.0, this->p)));

    wMatMap.insert(std::pair<DrivingState, Matrix3D3D>(ENTER_MOTORWAY, Matrix3D3D(this->p, 1.0, this->p,
                                                                                  this->p, 1.0, 0.0,
                                                                                  this->p, 1.0, this->p)));
    //executeActionMap();

    /*std::cout << "Vehicle " << name << ": boundRadius: " << boundRadius << ", is on lane:";
   for(int laneIt=currentSection->getTopRightLane(); laneIt<=currentSection->getTopLeftLane(); ++laneIt) {
      if(isOnLane(laneIt)) std::cout << " " << laneIt;
   }
   std::cout << std::endl;*/
}

FollowTheLeaderVehicle::~FollowTheLeaderVehicle()
{
    delete geometry;
}

void FollowTheLeaderVehicle::move(double dt)
{
    if (!routeTransitionList.empty())
    {
        planRoute();
        std::cout << "Road transition list:";
        for (RoadTransitionList::iterator transIt = roadTransitionList.begin(); transIt != roadTransitionList.end(); ++transIt)
        {
            std::cout << " " << transIt->road->getId();
        }
        std::cout << std::endl;
    }
    else if (!sowedDetermineNextRoadVehicleAction && (--roadTransitionList.end() == currentTransition))
    {
        std::cout << "Inserting determine next road vehicle action at s: " << s << std::endl;
        vehiclePositionActionMap.insert(std::pair<double, VehicleAction *>(s, new DetermineNextRoadVehicleAction()));
        sowedDetermineNextRoadVehicleAction = true;
        executeActionMap();
        //std::cout << "Road transition list:";
        //for(RoadTransitionList::iterator transIt = roadTransitionList.begin(); transIt!=roadTransitionList.end(); ++transIt) {
        //   std::cout << " " << transIt->road->getId();
        //}
        //std::cout << std::endl;
    }

    double velc = sqrt(fabs(aqmax / currentTransition->road->getCurvature(u)));
    if (velc == velc)
    {
        velt = std::min(velc, velw);
    }
    else
    {
        velt = velw;
    }

    timer += dt;

    if (currentLane == Lane::NOLANE)
    {
        executeActionMap();
        return;
    }

    //double dis = 0;
    //double dvel = 0;
    double acc = 0;

    /*
   VehicleRelation nextVehRel = locateVehicle(currentLane, 1);
   Vehicle* nextVeh = nextVehRel.vehicle;
   dis = nextVehRel.diffU;
   dvel = nextVehRel.diffDu;
   if(nextVeh && nextVeh!=this) {
      acc = a*(1 - pow((vel/velt),delta) - pow(etd(vel,-dvel)/dis,2));
   }
   else {
      dis = 0;
      dvel = 0;
      acc = a*(1 - pow((vel/velt),delta));
   }
   if(acc!=acc) {
      acc = 0;
   }
   */

    setRoadType(currentTransition->road->getRoadType(this->u));
    if (currentLane * currentTransition->direction > 0 && currentDrivingState == RURAL)
    {
        setDrivingState(OVERTAKE);
    }
    else if (currentLane * currentTransition->direction < 0 && currentDrivingState == OVERTAKE)
    {
        setDrivingState(RURAL);
    }

    if (currentSection->getLane(currentLane)->getLaneType() == Lane::MWYENTRY)
    {
        setDrivingState(ENTER_MOTORWAY);
    }
    else if (currentSection->getLane(currentLane)->getLaneType() == Lane::MWYEXIT)
    {
        setDrivingState(EXIT_MOTORWAY);
    }
    else if (currentRoadType == Road::MOTORWAY && currentDrivingState != EXIT_MOTORWAY)
    {
        setDrivingState(MOTORWAY);
    }
    //if(name == "hotcar") {
    //if(name == "schmotzcar") {
    if (velt > 0)
    {
        //if(false) {
        std::vector<int> laneVector;
        std::vector<Matrix3D3D> accMatVector;
        std::vector<double> qualityVector;

        int rightLaneId = currentLane - currentTransition->direction;
        if (rightLaneId == 0)
        {
            rightLaneId -= currentTransition->direction;
        }
        Lane *rightLane = currentSection->getLane(rightLaneId);
        //if((rightLane) && (rightLane->getLaneType() == Lane::DRIVING || (rightLane->getLaneType() == Lane::MWYEXIT && currentDrivingState==EXIT_MOTORWAY))) {
        if ((rightLane) && (rightLane->getLaneType() == Lane::DRIVING || (rightLane->getLaneType() == Lane::MWYEXIT)))
        {
            laneVector.push_back(rightLaneId);
            Matrix3D3D Aright = buildAccelerationMatrix(-1);
            //std::cout << "Vehicle " << this->name << ", right lane, A:" << std::endl << Aright;
            accMatVector.push_back(Aright);
            DrivingState state = currentDrivingState;
            if (rightLaneId * currentTransition->direction > 0 && currentDrivingState != MOTORWAY)
            {
                state = OVERTAKE;
            }
            double bRight = abs(rightLaneId);
            //if(currentDrivingState==EXIT_MOTORWAY) {
            //   bRight += 2;
            //}
            qualityVector.push_back(getAccelerationQuality(Aright, bRight, state));
            //std::cout << "Vehicle " << name << " on lane " << currentLane << ": quality on lane " << laneVector.back() << ": " << qualityVector.back() << std::endl;
        }

        laneVector.push_back(currentLane);
        Matrix3D3D Acenter = buildAccelerationMatrix(0);
        //std::cout << "Vehicle " << this->name << ", center lane " << currentLane << ", A:" << std::endl << Acenter;
        accMatVector.push_back(Acenter);
        qualityVector.push_back(getAccelerationQuality(Acenter, abs(currentLane) + cDelta));
        //std::cout << "Vehicle " << name << " on lane " << currentLane << ": quality on lane " << laneVector.back() << ": " << qualityVector.back() << std::endl;
        acc = Acenter(1, 1);
        if (acc != acc)
        {
            acc = 0;
        }

        int leftLaneId = currentLane + currentTransition->direction;
        if (leftLaneId == 0)
        {
            leftLaneId += currentTransition->direction;
        }
        Lane *leftLane = currentSection->getLane(leftLaneId);
        if ((leftLane) && (leftLane->getLaneType() == Lane::DRIVING))
        {
            laneVector.push_back(leftLaneId);
            Matrix3D3D Aleft = buildAccelerationMatrix(1);
            //std::cout << "Vehicle " << this->name << ", left lane, A:" << std::endl << Aleft;
            accMatVector.push_back(Aleft);
            DrivingState state = currentDrivingState;
            if (leftLaneId * currentTransition->direction > 0 && currentDrivingState != MOTORWAY)
            {
                state = OVERTAKE;
            }
            double bLeft = abs(leftLaneId);
            //if(currentDrivingState==ENTER_MOTORWAY) {
            //   bLeft += 2;
            //}
            qualityVector.push_back(getAccelerationQuality(Aleft, bLeft, state));
            //std::cout << "Vehicle " << name << " on lane " << currentLane << ": quality on lane " << laneVector.back() << ": " << qualityVector.back() << std::endl;
        }

        //if(name=="hotcar") {
        /*if(true) {
         std::cout << "Vehicle " << this->name << ", current lane: " << currentLane << ", current state: " << currentDrivingState << std::endl;
         std::cout << "States: Rural: " << RURAL << ", overtake: " << OVERTAKE << ", motorway: " << MOTORWAY << std::endl;
         for(int it=0; it<qualityVector.size(); ++it) {
            std::cout << "\t Lane " << laneVector[it] << ", quality: " << qualityVector[it];
            std::cout << ", A:" << std::endl << accMatVector[it];
         }
         std::cout << std::endl;
      }*/

        int maxQualIt = 0;
        for (int it = 1; it < qualityVector.size(); ++it)
        {
            if (qualityVector[it] > qualityVector[maxQualIt])
            {
                maxQualIt = it;
            }
        }

        //if(currentLane!=laneVector[maxQualIt]) {
        //   TrafficSimulationPlugin::plugin->haltSimulation();
        //}

        currentLane = laneVector[maxQualIt];
    }

    /*
   if(name == "schmotzcar") {
      for(int i = 0; i > -3; --i) {
         VehicleRelation vehRel= locateVehicle(currentLane+2, i);
         if(vehRel.vehicle) {
            std::cout << "Next Vehicle " << i << ": " << vehRel.vehicle->getName() << ", dis: " << vehRel.diffU << ", dvel: " << vehRel.diffDu << std::endl;
         }
         else {
            std::cout << "No next vehicle " << i << "..." << std::endl;
         }
      }
   }

   if(name == "schmotzcar") {
      std::cout << "Vehicle " << name << ": " << std::endl;
      if(isOnLane(3)) {
         std::cout << "Is on lane 3!" << std::endl;
      }
      if(isOnLane(2)) {
         std::cout << "Is on lane 2!" << std::endl;
      }
   }
   */
    /*
   if(name == "schmotzcar") {
      std::cout << "Road " << currentTransition->road->getId() << ", s: " << s << ", lane: " << currentLane << std::endl;
      std::cout << "Vehicle " << name << ", u: " << u << ", s: " << s << ", lane: " << currentLane << ", br: " << boundRadius << ", dist: " << dist << std::endl;
      int i = 1;
      for(RoadTransitionList::iterator transIt = roadTransitionList.begin(); transIt != roadTransitionList.end(); ++transIt) {
         std::cout << "Transition " << i << ": " << std::endl;
         std::cout << "\t Road: " << transIt->road->getId() << ", Type: " << transIt->road->getRoadType((1-transIt->direction)/2*transIt->road->getLength()) << std::endl;
         std::cout << "\t Direction: " << transIt->direction << std::endl;
         ++i;
      }

      //std::cout << "Junctions for lookout: " << std::endl;
      //for(std::list<RoadTransitionList::iterator>::iterator lookIt = junctionPathLookoutList.begin(); lookIt != junctionPathLookoutList.end(); ++lookIt) {
      //   std::cout << "\t Junction " << (*lookIt)->road->getJunction()->getId() << std::endl; 
      //}
   }
   */
    /*
	if(dis < 0.0) {
		std::cout << "Something wrong: Vehicle " << name << ", u: " << u << "Following: " << nextVeh->getName() << ", dis = " << dis << " < 0.0" << std::endl;
      acc = 0;
      TrafficSimulationPlugin::plugin->haltSimulation();
      //TrafficSimulationPlugin::plugin->getVehicleManager()->sortVehicleList(currentTransition->road);
	}
   */
    { //Verskriklike metode om die bewegingsvergelyking op te los!!!
        acc += brakeDec;
        brakeDec = 0.0;

        vel += acc * dt;
        if (vel < 0 && acc < 0)
        { //Dis nie so goed nie om die !(vel<0) beperking te verdwing
            acc = 0;
            vel = 0;
        }

        Vector2D laneCenter = currentSection->getLaneCenter(currentLane, u);

        double vdiff = laneCenter[0] - v;

        //double ddHdg = currentTransition->direction*(vdiff*Kv*vel  - dHdg*Dhdg) + (M_PI*((1-currentTransition->direction)/2)-hdg)*Khdg - dv*Dv;
        //dHdg += ddHdg*dt;
        //hdg += dHdg*dt;

        //double dw = currentTransition->direction*vel*cos(hdg);
        //dv = currentTransition->direction*vel*sin(hdg);

        dv += (vdiff * kv - dampingV * dv) * dt;
        double dw = sqrt(vel * vel - dv * dv);
        if (dw != dw)
        {
            dw = 0;
        }
        else
        {
            hdg = M_PI * ((currentTransition->direction - 1) / 2) + currentTransition->direction * atan(dv / dw);
            if (hdg != hdg)
            {
                hdg = M_PI * ((currentTransition->direction - 1) / 2);
            }
        }
        du = 1 / (-currentTransition->road->getCurvature(u) * v * cos(currentTransition->road->getSuperelevation(u, v)) + 1) * dw;

        u += currentTransition->direction * du * dt;
        v += dv * dt;
        s += du * dt;

        if (s != s || v != v || hdg != hdg)
        {
            std::cout << "Vehicle " << name << ": s: " << s << ", dv: " << dv << ", dw: " << dw << ", du: " << du << ", hdg: " << hdg << ", u: " << u << ", v: " << v << std::endl;
            TrafficSimulationPlugin::plugin->haltSimulation();
        }
    }

    //Action Map
    executeActionMap();

    double roadLength = currentTransition->road->getLength();
    if ((u > roadLength) || (u < 0.0))
    {
        if (u > roadLength)
        {
            u -= roadLength;
            v -= currentSection->getDistanceToLane(roadLength, currentLane);
        }
        else if (u < 0.0)
        {
            u = -u;
            v -= currentSection->getDistanceToLane(0.0, currentLane);
        }

        std::list<RoadTransition>::iterator nextTransition = currentTransition;
        ++nextTransition;
        if (nextTransition != roadTransitionList.end())
        {
            u = (nextTransition->direction > 0) ? u : (nextTransition->road->getLength() - u);
            //Junction* junction = nextTransition->road->getJunction();
            Junction *junction = nextTransition->junction;
            if (junction)
            {
                int newLane = junction->getConnectingLane(currentTransition->road, nextTransition->road, currentLane);
                if (newLane != Lane::NOLANE)
                {
                    currentLane = newLane;
                }
            }
            else
            {
                currentLane = (currentTransition->direction > 0) ? currentSection->getLaneSuccessor(currentLane) : currentSection->getLanePredecessor(currentLane);
            }
            currentSection = nextTransition->road->getLaneSection(u);

            if (currentLane == Lane::NOLANE)
            {
                vel = 0;
                return;
            }

            v *= currentTransition->direction * nextTransition->direction;
            v += currentSection->getDistanceToLane(u, currentLane);
            hdg += M_PI * ((currentTransition->direction * nextTransition->direction - 1) / 2);

            TrafficSimulationPlugin::plugin->getVehicleManager()->changeRoad(this, currentTransition->road, nextTransition->road, nextTransition->direction);
            currentTransition = nextTransition;
        }
    }
    else
    {
        LaneSection *newSection = currentTransition->road->getLaneSection(u);
        if ((currentSection != newSection))
        {
            currentLane = (currentTransition->direction > 0) ? currentSection->getLaneSuccessor(currentLane) : currentSection->getLanePredecessor(currentLane);
            currentSection = newSection;
            if (currentLane == Lane::NOLANE)
            {
                vel = 0;
                return;
            }
        }
    }

    vehicleTransform = currentTransition->road->getRoadTransform(u, v);

    //if(name=="warthog_20_3") {
    //   std::cout << "Vehicle " << name << ": x: " << vehicleTransform.v().x() << ", y: " << vehicleTransform.v().y() << ", z: " << vehicleTransform.v().z() << std::endl;
    //}

    geometry->setTransform(vehicleTransform, hdg);

    //std::cout << "Current Road: " << currentTransition->road->getId() << ", s: " << s << ", pred: " << currentTransition->road->getPredecessorConnection() << ", succ: " << currentTransition->road->getSuccessorConnection() << ", dir: " << currentTransition->direction << ", lane: " << currentLane << std::endl;
    //std::cerr << "Vehicle: " << name << ", u: " << u << ", s: " << s << ", vel: " << vel << ", acc: " << acc << ", dt: " << dt << ", dis: " << dis << ", dvel: " << dvel << ", size rtl: " << roadTransitionList.size() << std::endl;
    // std::cout << "Vehicle " << name << ": Lane: " << currentLane << std::endl;
    //
    /*std::cout << "Vehicle " << name << ": boundRadius: " << boundRadius << ", is on lane:";
   for(int laneIt=currentSection->getTopRightLane(); laneIt<=currentSection->getTopLeftLane(); ++laneIt) {
      if(isOnLane(laneIt)) std::cout << " " << laneIt;
   }
   std::cout << std::endl;*/
}

Road *FollowTheLeaderVehicle::getRoad() const
{
    return currentTransition->road;
}

double FollowTheLeaderVehicle::getU() const
{
    return u;
}

double FollowTheLeaderVehicle::getDu() const
{
    return currentTransition->direction * du;
}

VehicleGeometry *FollowTheLeaderVehicle::getVehicleGeometry()
{
    return geometry;
}

CarGeometry *FollowTheLeaderVehicle::getCarGeometry()
{
    return geometry;
}

double FollowTheLeaderVehicle::getBoundingCircleRadius()
{
    return (geometry == NULL) ? 0.0 : geometry->getBoundingCircleRadius();
}

int FollowTheLeaderVehicle::getLane() const
{
    return currentLane;
}

/*std::set<int> FollowTheLeaderVehicle::getLaneSet() const
{
   double extent = fabs(boundRadius*sin(hdg));
   int lowLane = currentSection->searchLane(u, v - extent);
   if(lowLane==Lane::NOLANE) {
      lowLane=currentSection->getTopRightLane();
   }
   int highLane = currentSection->searchLane(u, v + extent);
   if(highLane==Lane::NOLANE) {
      highLane=currentSection->getTopLeftLane();
   }
   std::set<int> laneSet;
   for(int laneIt=lowLane; laneIt<=highLane; ++laneIt) {
      laneSet.insert(laneIt);
   }
   return laneSet;
}*/

bool FollowTheLeaderVehicle::isOnLane(int lane) const
{
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

    if (lane >= lowLane && lane <= highLane)
    {
        return true;
    }

    return false;
}

void FollowTheLeaderVehicle::brake()
{
    brakeDec = -20;
}

double FollowTheLeaderVehicle::etd(double vel, double dvel)
{
    //vel = fabs(vel);
    if (vel < 0.0)
    {
        return dist;
    }
    double etd = dist + T * vel + (vel * dvel) / (2 * sqrt(a * b));
    return (etd > dist) ? etd : dist;
}

bool FollowTheLeaderVehicle::executeActionMap()
{
    bool actionPerformed = false;

    VehicleActionMap::iterator actionEnd = vehiclePositionActionMap.upper_bound(s);
    for (VehicleActionMap::iterator actionIt = vehiclePositionActionMap.begin(); actionIt != actionEnd;)
    {
        (*(actionIt->second))(this);
        delete actionIt->second;
        vehiclePositionActionMap.erase(actionIt++);
        actionPerformed = true;
        actionEnd = vehiclePositionActionMap.upper_bound(s);
    }

    actionEnd = vehicleTimerActionMap.upper_bound(timer);
    for (VehicleActionMap::iterator actionIt = vehicleTimerActionMap.begin(); actionIt != actionEnd;)
    {
        (*(actionIt->second))(this);
        delete actionIt->second;
        vehicleTimerActionMap.erase(actionIt++);
        actionPerformed = true;
        actionEnd = vehicleTimerActionMap.upper_bound(timer);
    }

    return actionPerformed;
}

VehicleRelation FollowTheLeaderVehicle::locateVehicle(int lane, int vehOffset)
{
    if (vehOffset == 0)
    {
        if (lane == currentLane)
        {
            return VehicleRelation(this, 0.0, 0.0);
        }
        else
        {
            return VehicleRelation(NULL, 0.0, 0.0);
        }
    }

    double dis = 0;
    double dvel = 0;
    int dirSearch = (vehOffset < 0) ? -1 : 1;
    vehOffset = abs(vehOffset);
    int numVeh = 0;

    //RoadTransitionList::iterator transIt = roadTransitionList.begin();
    RoadTransitionList::iterator transIt = currentTransition;
    int dirTrans = transIt->direction * dirSearch;

    VehicleList vehList = VehicleManager::Instance()->getVehicleList(transIt->road);
    if (transIt->direction * dirSearch < 0)
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
        lane = transIt->road->traceLane(lane, nextVehOldU, nextVehNewU);
        if (lane == Lane::NOLANE)
        {
            return VehicleRelation(NULL, 0, 0);
        }
        nextVehOldU = nextVehNewU;
        //if( ((*nextVehIt)->getLane() == lane) && (++numVeh==vehOffset)) {
        if (((*nextVehIt)->isOnLane(lane)) && (++numVeh == vehOffset))
        {
            break;
        }
        ++nextVehIt;
    }

    if (nextVehIt == vehList.end())
    {
        dis += (1 - dirTrans) / 2 * transIt->road->getLength();

        while (nextVehIt == vehList.end())
        {
            dis += dirTrans * transIt->road->getLength();
            if (lane != Lane::NOLANE)
            {
                int newLane = transIt->road->traceLane(lane, nextVehOldU, (transIt->direction * dirSearch < 0) ? 0.0 : transIt->road->getLength());
                if (newLane != Lane::NOLANE)
                {
                    lane = newLane;
                }
                //if(lane==Lane::NOLANE) {
                //   return VehicleRelation(NULL,0,0);
                //}
            }
            RoadTransitionList::iterator lastTransIt = transIt;
            if ((dirSearch < 0) ? ((transIt--) != roadTransitionList.begin()) : ((++transIt) != roadTransitionList.end()))
            {
                //Junction* junction = transIt->road->getJunction();
                Junction *junction = transIt->junction;
                if (junction)
                {
                    lane = junction->getConnectingLane(lastTransIt->road, transIt->road, lane);
                    if (lane == Lane::NOLANE)
                    {
                        return VehicleRelation(NULL, 0, 0);
                    }
                }
                vehList = VehicleManager::Instance()->getVehicleList((transIt)->road);
                if (transIt->direction * dirSearch < 0)
                {
                    vehList.reverse();
                }
                nextVehOldU = 0.0;
                nextVehIt = vehList.begin();
                while (nextVehIt != vehList.end() && lane != Lane::NOLANE)
                {
                    double nextVehNewU = (*nextVehIt)->getU();
                    lane = transIt->road->traceLane(lane, nextVehOldU, nextVehNewU);
                    if (lane == Lane::NOLANE)
                    {
                        return VehicleRelation(NULL, 0, 0);
                    }
                    nextVehOldU = nextVehNewU;
                    //if( ((*nextVehIt)->getLane() == lane) && (++numVeh==vehOffset) ) {
                    if (((*nextVehIt)->isOnLane(lane)) && (++numVeh == vehOffset))
                    {
                        //std::cout << "Lane " << lane << ", dirSearch: " << dirSearch << ": found vehicle: " << (*nextVehIt)->getName() << std::endl;
                        break;
                    }
                    ++nextVehIt;
                }
            }
            else
            {
                //std::cout << "Lane " << lane << ", dirSearch: " << dirSearch << ": road transition list size: " << roadTransitionList.size() << std::endl;
                return VehicleRelation(NULL, 0, 0);
            }
        }
        dis += (1 - transIt->direction * dirSearch) / 2 * dirTrans * transIt->road->getLength();
        dis += dirTrans * transIt->direction * dirSearch * (*nextVehIt)->getU();
        double bodyExtent = boundRadius + (*nextVehIt)->getBoundingCircleRadius();
        dis = dirSearch * dirTrans * (dis - this->u) - dirSearch * bodyExtent;
        if (dirSearch * dis < 0)
        {
            dis = 0.0;
        }
        dvel = transIt->direction * (*nextVehIt)->getDu() - this->du;
    }
    else
    {
        double bodyExtent = boundRadius + (*nextVehIt)->getBoundingCircleRadius();
        dis = dirSearch * dirTrans * ((*nextVehIt)->getU() - this->u) - dirSearch * bodyExtent;
        if (dirSearch * dis < 0)
        {
            dis = 0.0;
        }
        dvel = dirSearch * dirTrans * (*nextVehIt)->getDu() - this->du;
    }

    return VehicleRelation((*nextVehIt), dis, dvel);
    //std::cout << "First vehicle of road " << currentTransition->road->getId() << ": " << vehList.front()->getName() << std::endl;
}

double FollowTheLeaderVehicle::locateLaneEnd(int lane)
{
    double pos = 0;
    bool entry = true;

    for (RoadTransitionList::iterator transIt = currentTransition; transIt != roadTransitionList.end(); ++transIt)
    {
        int newLane = lane;
        double startU = 0;
        if (entry)
        {
            startU = u;
            entry = false;
        }
        else
        {
            startU = (1 - transIt->direction) / 2 * transIt->road->getLength();
        }
        double roadPos = transIt->road->getLaneEnd(newLane, startU, transIt->direction);
        pos += (1 - transIt->direction) / 2 * transIt->road->getLength() + transIt->direction * roadPos;
        //std::cout << "\t Road: " << transIt->road->getId() << ", lane: " << lane << ", new lane: " << newLane << ", roadPos: " << roadPos << ", pos: " << pos << ", startU: " << startU << std::endl;

        RoadTransitionList::iterator nextTransIt = transIt;
        ++nextTransIt;
        //if((roadPos >= transIt->road->getLength() || roadPos<=0.0) && nextTransIt!=roadTransitionList.end() && nextTransIt->road->isJunctionPath() ) {
        //if(nextTransIt!=roadTransitionList.end() && nextTransIt->road->isJunctionPath() ) {
        if (nextTransIt != roadTransitionList.end() && nextTransIt->junction)
        {
            //Junction* junction = nextTransIt->road->getJunction();
            Junction *junction = nextTransIt->junction;
            lane = junction->getConnectingLane(transIt->road, nextTransIt->road, lane);
            if (lane == Lane::NOLANE)
            {
                lane = newLane;
            }
        }
        else
        {
            lane = newLane;
        }
        //std::cout << "\t Road: " << transIt->road->getId() << ", lane: " << lane << ", new lane: " << newLane << std::endl;

        if (lane == Lane::NOLANE)
        {
            pos -= (1 - currentTransition->direction) / 2 * currentTransition->road->getLength() + currentTransition->direction * u;
            return pos;
        }
    }

    return 1e10;
}

double FollowTheLeaderVehicle::locateNextJunction()
{
    double pos = 0;

    if (!junctionPathLookoutList.empty())
    {
        RoadTransitionList::iterator junctionPathIt = junctionPathLookoutList.front();

        for (RoadTransitionList::iterator transIt = currentTransition; transIt != junctionPathIt; ++transIt)
        {
            pos += transIt->road->getLength();
        }

        pos -= (1 - currentTransition->direction) / 2 * currentTransition->road->getLength() + currentTransition->direction * u;
    }
    else
    {
        pos = 1e10;
    }

    return pos;
}

Matrix3D3D FollowTheLeaderVehicle::buildAccelerationMatrix(int laneOffset)
{
    Matrix3D3D DV(0.0);
    Matrix3D3D DS(0.0);

    int rightLaneId = currentLane - currentTransition->direction;
    if (rightLaneId == 0)
    {
        rightLaneId -= currentTransition->direction;
    }
    //Lane* rightLane = currentSection->getLane(rightLaneId);

    int leftLaneId = currentLane + currentTransition->direction;
    if (leftLaneId == 0)
    {
        leftLaneId += currentTransition->direction;
    }

    std::vector<int> laneVector;
    laneVector.push_back(rightLaneId);
    laneVector.push_back(currentLane);
    laneVector.push_back(leftLaneId);

    int vehLane = (laneOffset == 0) ? currentLane : ((laneOffset < 0) ? rightLaneId : leftLaneId);

    for (int laneIt = 0; laneIt < 3; ++laneIt)
    {
        Lane *lane = currentSection->getLane(laneVector[laneIt]);
        if (lane)
        {
            VehicleRelation vehRel = locateVehicle(laneVector[laneIt], 1);
            double laneEndDis = locateLaneEnd(laneVector[laneIt]) - boundRadius;
            if (laneEndDis < 0)
            {
                laneEndDis = 0;
            }
            double junctionStopDis = locateNextJunction() - boundRadius;
            //std::cout << "Lane " << lane->getId() << ": vehRel.diffU: " << vehRel.diffU << ", laneEndDis: " << laneEndDis << ", junctionStopDis: " << junctionStopDis << std::endl;
            if (vehRel.vehicle && vehRel.diffU < laneEndDis && vehRel.diffU < junctionStopDis)
            {
                DS(laneIt, 0) = vehRel.diffU;
                DV(laneIt, 0) = vehRel.diffDu;
                /*if(name=="schmotzcar")
               std::cout << "DS(" << laneIt << ", 0)=" << DS(laneIt,0) << ", DV(" << laneIt << ", 0)=" << DV(laneIt,0) << ", du: " << this->du << ", dir: " << currentTransition->direction << std::endl;*/
            }
            else if (junctionStopDis < laneEndDis)
            {
                DS(laneIt, 0) = junctionStopDis;
                DV(laneIt, 0) = -this->du;
            }
            else
            {
                DS(laneIt, 0) = laneEndDis;
                DV(laneIt, 0) = -this->du;
            }

            vehRel = locateVehicle(laneVector[laneIt], -1);
            if (vehRel.vehicle)
            {
                DS(laneIt, 2) = vehRel.diffU;
                DV(laneIt, 2) = vehRel.diffDu;
            }
            else
            {
                DS(laneIt, 2) = -1e10;
                DV(laneIt, 2) = 0.0;
            }
        }
    }
    //std::cout << "DS:" << std::endl << DS;
    //std::cout << "DV:" << std::endl << DV;
    Matrix3D3D F(1, 1, 0,
                 0, 0, 0,
                 0, -1, -1);
    Matrix3D3D DSF = DS * F;
    Matrix3D3D DVF = DV * F;
    //std::cout << "Vehicle " << name << ": DSF:" << std::endl << DSF;
    //std::cout << "Vehicle " << name << ": DVF:" << std::endl << DVF;

    Matrix3D3D A(0.0);
    for (int laneIt = 0; laneIt < 3; ++laneIt)
    {
        if (currentSection->getLane(laneVector[laneIt]))
        {
            if (laneVector[laneIt] == vehLane)
            {
                double velFore = DV(laneIt, 0) + this->du;
                if (velFore < 0)
                {
                    double acc = -pow(etd(-velFore, -DVF(laneIt, 0)) / DSF(laneIt, 0), 2);
                    if (acc == acc)
                    {
                        A(laneIt, 0) = acc;
                    }
                    else
                    {
                        A(laneIt, 0) = -1e10;
                    }
                }

                if (velt != 0.0)
                {
                    double acc = a * (1 - pow((this->du / velt), delta) - pow(etd(this->du, -DVF(laneIt, 0)) / DSF(laneIt, 0), 2));
                    if (acc == acc)
                    {
                        A(laneIt, 1) = acc;
                    }
                    else
                    {
                        A(laneIt, 1) = -1e10;
                    }
                }
                else
                {
                    A(laneIt, 1) = 0.0;
                }

                double velBack = DV(laneIt, 2) + this->du;
                if (velBack > 0)
                {
                    double acc = -pow(etd(velBack, -DVF(laneIt, 2)) / DSF(laneIt, 2), 2);
                    if (acc == acc)
                    {
                        A(laneIt, 2) = acc;
                    }
                    else
                    {
                        A(laneIt, 2) = -1e10;
                    }
                }
            }
            else
            {
                double velFore = DV(laneIt, 0) + this->du;
                if (velFore < 0)
                {
                    double acc = -pow(etd(-velFore, -DVF(laneIt, 1)) / DSF(laneIt, 1), 2);
                    if (acc == acc)
                    {
                        A(laneIt, 0) = acc;
                    }
                    else
                    {
                        A(laneIt, 0) = -1e10;
                    }
                }

                double velBack = DV(laneIt, 2) + this->du;
                if (velBack > 0)
                {
                    double acc = -pow(etd(velBack, -DVF(laneIt, 1)) / DSF(laneIt, 1), 2);
                    if (acc == acc)
                    {
                        A(laneIt, 2) = acc;
                    }
                    else
                    {
                        A(laneIt, 2) = -1e10;
                    }
                }
            }
        }
        else
        {
            A(laneIt, 0) = 0;
            A(laneIt, 1) = 0;
            A(laneIt, 2) = 0;
        }
    }
    //std::cout << "Vehicle " << this->name << ", currentLane: " << currentLane << ", vehLane: " << vehLane << ", A:" << std::endl << A;

    return A;
}

void FollowTheLeaderVehicle::setDrivingState(DrivingState state)
{
    currentDrivingState = state;
}

void FollowTheLeaderVehicle::setRoadType(Road::RoadType type)
{
    currentRoadType = type;
    switch (type)
    {
    case Road::MOTORWAY:
        if (currentDrivingState != EXIT_MOTORWAY && currentDrivingState != ENTER_MOTORWAY)
        {
            setDrivingState(MOTORWAY);
        }
        break;
    case Road::RURAL:
        setDrivingState(RURAL);
        break;
    default:
        setDrivingState(RURAL);
        break;
    }
}

double FollowTheLeaderVehicle::getAccelerationQuality(const Matrix3D3D &A, const double &b) const
{
    //std::cout << "Vehicle " << this->name << ", A:" << std::endl << A;
    //std::cout << "wVec: " << std::endl << wVec;
    //std::cout << "pVec: " << std::endl << pVec;
    //std::cout << "b: " << b << std::endl;
    //return (A*wVecMap.find(currentDrivingState)->second).dot(pVecMap.find(currentDrivingState)->second) + b;
    return getAccelerationQuality(A, b, currentDrivingState);
}

double FollowTheLeaderVehicle::getAccelerationQuality(const Matrix3D3D &A, const double &b, DrivingState state) const
{
    return A.multiplyElementByElement(wMatMap.find(state)->second).getSumOverElements() + b;
    //return (A*wVecMap.find(state)->second).dot(pVecMap.find(state)->second) + b;
}

bool FollowTheLeaderVehicle::planRoute()
{
    //bool found = findRoute(RoadTransition(RoadSystem::Instance()->getRoad("5"), -1));
    bool oneFound = false;
    while (!routeTransitionList.empty())
    {
        bool found = findRoute(routeTransitionList.front());
        oneFound = oneFound || found;
        routeTransitionList.pop_front();

        /*
      std::cout << "Searching Transition, found: ";
      if(found) {
         std::cout << "true" << std::endl;
         std::cout << "Road transition list:";
         for(RoadTransitionList::iterator transIt = roadTransitionList.begin(); transIt!=roadTransitionList.end(); ++transIt) {
            std::cout << " " << transIt->road->getId();
         }
         std::cout << std::endl;
      }
      else {
         std::cout << "false" << std::endl;
      }
      */
    }

    return oneFound;
}

bool FollowTheLeaderVehicle::findRoute(RoadTransition goalTrans)
{
    double goalS = (goalTrans.direction < 0) ? goalTrans.road->getLength() : 0.0;
    Vector3D goalPos3D = goalTrans.road->getChordLinePlanViewPoint(goalS);
    Vector2D goalPos(goalPos3D.x(), goalPos3D.y());

    std::set<RouteFindingNode *, RouteFindingNodeCompare> fringeNodeSet;
    /*RouteFindingNode* baseNode = new RouteFindingNode((*currentTransition), goalTrans, goalPos);
   if(baseNode->foundGoal()) {
      RoadTransitionList::iterator eraseTrans = currentTransition;
      roadTransitionList.erase(++eraseTrans, roadTransitionList.end());
      roadTransitionList.splice(roadTransitionList.end(), baseNode->backchainTransitionList());
      delete baseNode;
      return true;
   }*/
    RouteFindingNode *baseNode = new RouteFindingNode(roadTransitionList.back(), goalTrans, goalPos);
    if (baseNode->foundGoal())
    {
        roadTransitionList.splice(roadTransitionList.end(), baseNode->backchainTransitionList());
        delete baseNode;
        return true;
    }
    else if (baseNode->isEndNode())
    {
        delete baseNode;
        return false;
    }
    fringeNodeSet.insert(baseNode);

    while (!fringeNodeSet.empty())
    {
        std::set<RouteFindingNode *, RouteFindingNodeCompare>::iterator currentNodeIt = fringeNodeSet.begin();
        RouteFindingNode *currentNode = (*currentNodeIt);
        Junction *currentJunction = currentNode->getEndJunction();
        PathConnectionSet connSet = currentJunction->getPathConnectionSet(currentNode->getLastTransition().road);
        for (PathConnectionSet::iterator connSetIt = connSet.begin(); connSetIt != connSet.end(); ++connSetIt)
        {

            RouteFindingNode *expandNode = new RouteFindingNode(RoadTransition((*connSetIt)->getConnectingPath(), (*connSetIt)->getConnectingPathDirection(), currentJunction), goalTrans, goalPos, currentNode);
            currentNode->addChild(expandNode);

            /*if(expandNode->foundGoal()) {
            RoadTransitionList::iterator eraseTrans = currentTransition;
            roadTransitionList.erase(++eraseTrans, roadTransitionList.end());
            roadTransitionList.splice(roadTransitionList.end(), expandNode->backchainTransitionList());
            delete baseNode;
            return true;
         }*/
            if (expandNode->foundGoal())
            {
                roadTransitionList.splice(roadTransitionList.end(), expandNode->backchainTransitionList());
                delete baseNode;
                return true;
            }

            else if (!(expandNode->isEndNode()))
            {
                fringeNodeSet.insert(expandNode);
            }
        }

        fringeNodeSet.erase(currentNodeIt);
    }

    std::cerr << "Vehicle " << name << ": findRoute(): Goal state not found using complete A*-search! So no route available to road " << goalTrans.road->getId() << " with direction " << goalTrans.direction << ", sorry..." << std::endl;
    return false;
}

void FollowTheLeaderVehicle::addRouteTransition(const RoadTransition &trans)
{
    routeTransitionList.push_back(trans);
}

void DetermineNextRoadVehicleAction::operator()(FollowTheLeaderVehicle *veh)
{
    RoadTransition trans = veh->roadTransitionList.back();

    TarmacConnection *conn = NULL;
    if (veh->roadTransitionList.back().direction > 0)
    {
        conn = veh->roadTransitionList.back().road->getSuccessorConnection();
    }
    else if (veh->roadTransitionList.back().direction < 0)
    {
        conn = veh->roadTransitionList.back().road->getPredecessorConnection();
    }
    if (conn)
    {
        Junction *junction;
        Road *road;
        double isJunctionPath = false;
        RoadTransition newTrans = trans;

        bool motorwayExit = false;
        if ((road = dynamic_cast<Road *>(conn->getConnectingTarmac())))
        {
            newTrans.road = road;
            newTrans.direction = conn->getConnectingTarmacDirection();
            newTrans.junction = NULL;
            /*if(newTrans->road->getId()=="1fr") {
            std::cout << "Vehicle " << name << ":" << std::endl;
            std::cout << "\t\tCurrent road is id " << roadTransitionList.back().road->getId() << ": current lane: " << roadTransitionList.back().lane << ", current roadTransitionList.back().direction: " << roadTransitionList.back().direction << std::endl;
            std::cout << "\t\tChoice road is id " << newTrans->road->getId() << ": lane of choice: " << newTrans->lane << ", roadTransitionList.back().direction of choice: " << newTrans->direction << std::endl;
         }*/
        }
        else if ((junction = dynamic_cast<Junction *>(conn->getConnectingTarmac())))
        {
            PathConnectionSet connSet = junction->getPathConnectionSet(veh->roadTransitionList.back().road);
            int path = rand() % connSet.size();
            PathConnectionSet::iterator connSetIt = connSet.begin();
            std::advance(connSetIt, path);
            PathConnection *conn = *connSetIt;
            newTrans.road = conn->getConnectingPath();
            newTrans.direction = conn->getConnectingPathDirection();
            newTrans.junction = junction;
            isJunctionPath = true;
            LaneSection *section = newTrans.road->getLaneSection((1 - newTrans.direction) / 2 * newTrans.road->getLength());
            for (int laneIt = section->getTopRightLane(); laneIt <= section->getTopLeftLane(); ++laneIt)
            {
                if (section->getLane(laneIt)->getLaneType() == Lane::MWYEXIT)
                {
                    motorwayExit = true;
                    break;
                }
            }
        }
        else
        {
            newTrans.direction = 0;
        }

        veh->roadTransitionList.push_back(newTrans);

        double transLength = veh->roadTransitionList.getLength(veh->currentTransition, veh->roadTransitionList.end())
                             + 0.5 * (veh->currentTransition->direction - 1) * veh->currentTransition->road->getLength()
                             - veh->currentTransition->direction * veh->u;
        veh->vehiclePositionActionMap.insert(std::pair<double, VehicleAction *>(veh->s + transLength - veh->minTransitionListExtent, new DetermineNextRoadVehicleAction()));

        if (isJunctionPath && newTrans.road->getRoadType((1 - newTrans.direction) / 2 * newTrans.road->getLength()) != Road::MOTORWAY)
        {
            veh->junctionPathLookoutList.push_back(--veh->roadTransitionList.end());
            veh->vehiclePositionActionMap.insert(std::pair<double, VehicleAction *>(veh->s + transLength - veh->roadTransitionList.back().road->getLength() - veh->junctionWaitTriggerDist, new WaitAtJunctionVehicleAction()));
            //std::cout << "Junction Trigger at s: " << veh->s + transLength - veh->roadTransitionList.back().road->getLength() - veh->junctionWaitTriggerDist << " with veh->s: " << veh->s << ", transLength: " << transLength << ", road length: " <<  veh->roadTransitionList.back().road->getLength() << ", trigger dist: " << veh->junctionWaitTriggerDist << std::endl;
        }

        if (motorwayExit)
        {
            veh->vehiclePositionActionMap.insert(std::pair<double, VehicleAction *>(veh->s + transLength - veh->roadTransitionList.back().road->getLength() - 500, new SetDrivingStateVehicleAction(FollowTheLeaderVehicle::EXIT_MOTORWAY)));
        }
    }
}

int DetermineNextRoadVehicleAction::removeAllActions(FollowTheLeaderVehicle *veh)
{
    int erased = 0;

    for (VehicleActionMap::iterator actionIt = veh->vehiclePositionActionMap.begin(); actionIt != veh->vehiclePositionActionMap.end();)
    {
        if (dynamic_cast<DetermineNextRoadVehicleAction *>(actionIt->second))
        {
            veh->vehiclePositionActionMap.erase(actionIt++);
            ++erased;
        }
        else
        {
            ++actionIt;
        }
    }

    for (VehicleActionMap::iterator actionIt = veh->vehicleTimerActionMap.begin(); actionIt != veh->vehicleTimerActionMap.end();)
    {
        if (dynamic_cast<DetermineNextRoadVehicleAction *>(actionIt->second))
        {
            veh->vehicleTimerActionMap.erase(actionIt++);
            ++erased;
        }
        else
        {
            ++actionIt;
        }
    }

    return erased;
}

void WaitAtJunctionVehicleAction::operator()(FollowTheLeaderVehicle *veh)
{
    Junction *junction = veh->junctionPathLookoutList.front()->road->getJunction();
    std::cout << "Junction " << junction->getId() << std::endl;
    if (VehicleManager::Instance()->isJunctionEmpty(junction))
    {
        veh->junctionPathLookoutList.pop_front();
    }
    else
    {
        veh->vehicleTimerActionMap.insert(std::pair<double, VehicleAction *>(veh->timer + veh->junctionWaitTime, new WaitAtJunctionVehicleAction()));
    }
}

void SetDrivingStateVehicleAction::operator()(FollowTheLeaderVehicle *veh)
{
    veh->currentDrivingState = state;
}
