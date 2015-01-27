/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Fiddleyard.h"

#include <sstream>

Carpool *Carpool::__instance = NULL;

FiddleRoad::FiddleRoad(Fiddleyard *fy, std::string id, std::string name, double l, Junction *junc, bool geo)
    : Road(id, name, l, junc)
    , fiddleyard(fy)
    , geometry(geo)
{
    xyMap[0.0] = (new PlaneStraightLine(0.0, l));
}

std::string FiddleRoad::getTypeSpecifier()
{
    return std::string("fiddleyard");
}

Fiddleyard *FiddleRoad::getFiddleyard()
{
    return fiddleyard;
}

osg::Geode *FiddleRoad::getRoadGeode()
{
    if (geometry)
        return Road::getRoadGeode();
    else
        return NULL;
}

VehicleSource::VehicleSource(std::string id, int l, double startt, double repeatt, double vel, double velDev)
    : Element(id)
    , startLane(l)
    , fiddleLane(l)
    , startTime(startt)
    , nextTime(startt)
    , repeatTime(repeatt)
    , startVel(vel)
    , startVelDev(velDev)
    , overallRatio(0.0)
{
}

bool VehicleSource::spawnNextVehicle(double time)
{
    if (time > nextTime)
    {
        nextTime += repeatTime;
        return true;
    }
    else
    {
        return false;
    }
}

void VehicleSource::setFiddleroad(FiddleRoad *r, TarmacConnection *conn)
{
    fiddleroad = r;
    int dir = 1;

    Road *road = dynamic_cast<Road *>(conn->getConnectingTarmac());
    if (road)
    {
        dir = conn->getConnectingTarmacDirection();
    }
    fiddleLane = -dir * startLane;

    LaneSection *section = fiddleroad->getLaneSection(0.0);

    /*
   int laneInc = (startLane<0) ? -1: 1;
   for(int laneIt = laneInc; laneIt!=startLane; laneIt+=laneInc) {
      Lane* lane = section->getLane(laneIt);
      if(!lane) {
         section->addLane(new Lane(laneIt, Lane::NONE));
      }
   }
   */

    Lane *lane = section->getLane(fiddleLane);
    /*
   if(!lane) {
      //fiddleroad->getLaneSection(0.0)->addLane(new Lane(startLane, Lane::DRIVING));
      lane = new Lane(startLane, Lane::NONE);
      section->addLane(lane);
   }
   */

    lane->setPredecessor(startLane);

    lane->setLaneType(Lane::DRIVING);

    /*for(int laneIt=section->getTopRightLane(); laneIt <= section->getTopLeftLane(); ++laneIt) {
      std::cout << "Lane " << laneIt << ": predecessor: " << section->getLane(laneIt)->getPredecessor() << std::endl;
   }*/
}

int VehicleSource::getLane()
{
    return startLane;
}

int VehicleSource::getFiddleLane() const
{
    return fiddleLane;
}

double VehicleSource::getStartTime()
{
    return startTime;
}

double VehicleSource::getRepeatTime()
{
    return repeatTime;
}

double VehicleSource::getStartVelocity()
{
    return startVel;
}

double VehicleSource::getStartVelocityDeviation()
{
    return startVelDev;
}

double VehicleSource::getOverallRatio()
{
    return overallRatio;
}

void VehicleSource::addVehicleRatio(std::string id, double ratio)
{
    overallRatio += fabs(ratio);
    ratioVehicleMap.insert(std::pair<double, std::string>(overallRatio, id));
}

bool VehicleSource::isRatioVehicleMapEmpty()
{
    return ratioVehicleMap.empty();
}

std::string VehicleSource::getVehicleByRatio(double targetRatio)
{
    return ratioVehicleMap.lower_bound(targetRatio)->second;
}

VehicleSink::VehicleSink(std::string id, int l)
    : Element(id)
    , endLane(l)
    , fiddleLane(l)
{
}

void VehicleSink::setFiddleroad(FiddleRoad *r, TarmacConnection *conn)
{
    fiddleroad = r;

    Road *road = dynamic_cast<Road *>(conn->getConnectingTarmac());
    if (road)
    {
        int dir = conn->getConnectingTarmacDirection();
        LaneSection *connSection = (dir < 0) ? road->getLaneSection(road->getLength()) : road->getLaneSection(0.0);

        fiddleLane = -dir * endLane;

        if (dir < 0)
        {
            connSection->getLane(endLane)->setSuccessor(fiddleLane);
        }
        else
        {
            connSection->getLane(endLane)->setPredecessor(fiddleLane);
        }

        fiddleroad->getLaneSection(0.0)->getLane(fiddleLane)->setLaneType(Lane::DRIVING);
    }
    /*
   LaneSection* section = fiddleroad->getLaneSection(0.0);

   int laneInc = (endLane<0) ? -1: 1;
   for(int laneIt = laneInc; laneIt!=endLane; laneIt+=laneInc) {
      Lane* lane = section->getLane(laneIt);
      if(!lane) {
         section->addLane(new Lane(laneIt, Lane::NONE));
      }
   }

   Lane* lane = section->getLane(endLane);
   if(!lane) {
      lane = new Lane(endLane, Lane::DRIVING);
      section->addLane(lane);
   }
   lane->setPredecessor(endLane);
   */
}

int VehicleSink::getLane()
{
    return endLane;
}

int VehicleSink::getFiddleLane() const
{
    return fiddleLane;
}

Fiddleyard::Fiddleyard(std::string id, std::string name, FiddleRoad *r)
    : Tarmac(id, name)
    , connection(NULL)
    , fiddleroad(r)
    , timer(0.0)
{
    if (!fiddleroad)
    {
        fiddleroad = new FiddleRoad(this, id.append("fr"), name.append("fr"), 30, NULL, false);
    }
}

void Fiddleyard::setTarmacConnection(TarmacConnection *conn)
{
    connection = conn;
    fiddleroad->setPredecessorConnection(conn);

    LaneSection *fiddleSection = fiddleroad->getLaneSection(0.0);
    Road *road = dynamic_cast<Road *>(conn->getConnectingTarmac());
    if (road)
    {
        int dir = conn->getConnectingTarmacDirection();
        LaneSection *connSection = road->getLaneSection((dir < 0) ? road->getLength() : 0.0);
        for (int laneIt = connSection->getTopRightLane(); laneIt <= connSection->getTopLeftLane(); ++laneIt)
        {
            //Lane* lane = new Lane(-dir*laneIt, connSection->getLane(laneIt)->getLaneType());
            Lane *lane = new Lane(-dir * laneIt, Lane::NONE);
            fiddleSection->addLane(lane);
            double width = connSection->getLaneWidth(((dir < 0) ? road->getLength() : 0.0), laneIt);
            //int laneDir = (laneIt<0) ? -1 : 1;
            //lane->addWidth(0.0, -dir*laneDir*width, 0.0, 0.0, 0.0);
            lane->addWidth(0.0, fabs(width), 0.0, 0.0, 0.0);
            //std::cout << "Fiddleyard: " << id << ": roadLane: " << laneIt << ", fiddleLane: " << lane->getId() << std::endl;
            //std::cout << "Fiddleyard " << this->id << ": added lane: " << lane->getId() << ", width: " << fiddleSection->getLaneWidth(0.0, lane->getId()) << ", should be: " << width << std::endl;
        }

        Vector3D planPoint = road->getChordLinePlanViewPoint((dir < 0) ? road->getLength() : 0.0);
        fiddleroad->addPlanViewGeometryLine(0.0, 0.0, planPoint.x(), planPoint.y(), (1 + dir) / 2 * M_PI + planPoint.z());
    }
}

TarmacConnection *Fiddleyard::getTarmacConnection()
{
    return connection;
}

FiddleRoad *Fiddleyard::getFiddleroad() const
{
    return fiddleroad;
}

double Fiddleyard::incrementTimer(double dt)
{
    return (timer += dt);
}

void Fiddleyard::addVehicleSource(VehicleSource *source)
{
    sourceMap[source->getLane()] = source;
    if (connection)
        source->setFiddleroad(fiddleroad, connection);
}

void Fiddleyard::addVehicleSink(VehicleSink *sink)
{
    sinkMap[sink->getLane()] = sink;
    if (connection)
        sink->setFiddleroad(fiddleroad, connection);
}

std::string Fiddleyard::getTypeSpecifier()
{
    return std::string("fiddleyard");
}

const std::map<int, VehicleSource *> &Fiddleyard::getVehicleSourceMap() const
{
    return sourceMap;
}

const std::map<int, VehicleSink *> &Fiddleyard::getVehicleSinkMap() const
{
    return sinkMap;
}

void Fiddleyard::accept(RoadSystemVisitor *visitor)
{
    visitor->visit(this);
}

/**
 *
 * Pedestrian extensions
 *
 */
// Pedestrian source
PedSource::PedSource(PedFiddleyard *fy, std::string id, double start, double repeat, double dev, int l, int d, double so, double vo, double v, double vd, double a, double ad)
    : VehicleSource(id, l, start, repeat, v, vd)
    , fiddleyard(fy)
    , startTime(start)
    , repeatTime(repeat)
    , nextTime(start)
    , timeDev(dev)
    , dir(d)
    , sOffset(so)
    , vOffset(vo)
    , acc(a)
    , accDev(ad)
{
}
bool PedSource::spawnNextPed(double time)
{
    return (time > nextTime);
}
void PedSource::updateTimer(double time, double rand)
{
    nextTime = time + repeatTime - timeDev + 2 * rand * timeDev;
    /*
   if (time > nextTime)
   {
      nextTime += repeatTime - timeDev + 2 * rand * timeDev;
      return true;
   }
   else
   {
      return false;
   }
*/
}
void PedSource::addPedRatio(std::string id, double ratio)
{
    VehicleSource::addVehicleRatio(id, ratio);
}
std::string PedSource::getPedByRatio(double targetRatio)
{
    if (isRatioMapEmpty())
        return "";
    else
        return VehicleSource::getVehicleByRatio(targetRatio);
}
bool PedSource::isRatioMapEmpty()
{
    return VehicleSource::isRatioVehicleMapEmpty();
}

// Pedestrian sink
PedSink::PedSink(PedFiddleyard *fy, std::string id, double p, int l, int d, double so, double vo)
    : VehicleSink(id, l)
    , fiddleyard(fy)
    , prob(p)
    , dir(d)
    , sOffset(so)
    , vOffset(vo)
{
}

// Pedestrian fiddleyard
PedFiddleyard::PedFiddleyard(std::string id, std::string name, std::string r)
    : Fiddleyard(id, name, NULL)
    , roadId(r)
{
}
void PedFiddleyard::addSource(PedSource *source)
{
    pedSourceMap[std::pair<int, double>(source->getLane(), source->getSOffset())] = source;
}
void PedFiddleyard::addSink(PedSink *sink)
{
    pedSinkMap[std::pair<int, double>(sink->getLane(), sink->getSOffset())] = sink;
}
const std::map<std::pair<int, double>, PedSource *> &PedFiddleyard::getSourceMap() const
{
    return pedSourceMap;
}
const std::map<std::pair<int, double>, PedSink *> &PedFiddleyard::getSinkMap() const
{
    return pedSinkMap;
}

// Erweiterung für die beweglichen Fiddleyards

Carpool::Carpool()
    : overallRatio(0.0)
{
}

//Carpool::Carpool(std::string name) : name(name){
//}

Carpool *Carpool::Instance()
{
    if (__instance == NULL)
    {
        __instance = new Carpool();
    }
    return __instance;
}

void Carpool::Destroy()
{
    delete __instance;
    __instance = NULL;
}

void Carpool::addPool(Pool *pool)
{
    //Pool * pool = new Pool(id, name, repeatTime, velocity,velocityDeviance);
    poolVector.push_back(pool);
}

std::vector<Pool *> Carpool::getPoolVector()
{
    return poolVector;
}

double Carpool::getOverallRatio()
{
    return overallRatio;
}

void Carpool::addPoolRatio(std::string id, double ratio)
{
    overallRatio += fabs(ratio);
    ratioPoolMap.insert(std::pair<double, std::string>(overallRatio, id));
}

bool Carpool::isRatioPoolMapEmpty()
{
    return ratioPoolMap.empty();
}

std::string Carpool::getPoolIdByRatio(double targetRatio)
{
    return ratioPoolMap.lower_bound(targetRatio)->second;
}

Pool *Carpool::getPoolById(std::string id)
{
    std::vector<Pool *> vector = Carpool::getPoolVector();
    std::vector<Pool *>::iterator it;
    for (it = vector.begin(); it != vector.end(); it++)
    {
        if (strcmp(((*it)->getId()).c_str(), id.c_str()) == 0)
        {
            return (*it);
        }
    }
    return NULL;
}

//double Pool::getRepeatTime(){
//	return repeatTime;
//}

double Pool::getStartVelocity()
{
    return startVel;
}

double Pool::getStartVelocityDeviation()
{
    return startVelDev;
}

double Pool::getOverallRatio()
{
    return overallRatio;
}

void Pool::addVehicle(std::string id, double ratio)
{
    overallRatio += fabs(ratio);
    ratioVehicleMap.insert(std::pair<double, std::string>(overallRatio, id));
}

bool Pool::isRatioVehicleMapEmpty()
{
    return ratioVehicleMap.empty();
}

std::string Pool::getVehicleByRatio(double targetRatio)
{
    //std::cout << "---> TEST VBR: " << ratioVehicleMap.lower_bound(targetRatio)->second<< std::endl;
    return ratioVehicleMap.lower_bound(targetRatio)->second;
}

Pool::Pool(std::string id, std::string name, double vel, double velDev, int direction, double poolNumerator)
    : id(id)
    , name(name)
    , startVel(vel)
    , startVelDev(velDev)
    , dir(direction)
    , numerator(poolNumerator)
    , overallRatio(0.0)
{
}

std::string Pool::getName()
{
    return name;
}

std::string Pool::getId()
{
    return id;
}

int Pool::getMapSize()
{
    return ratioVehicleMap.size();
}

int Pool::getDir()
{
    return dir;
}

void Pool::changeDir()
{
    this->dir *= -1;
}