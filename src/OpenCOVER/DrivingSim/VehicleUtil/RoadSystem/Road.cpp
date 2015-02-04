/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Road.h"
#include "Junction.h"
#include "Fiddleyard.h"
#include "OpenCRGSurface.h"

#include <limits>
#include <algorithm>
#include <map>
#include <deque>
//#include <iomanip>

#include <osg/Geometry>
#include <osg/PositionAttitudeTransform>
#include <osgViewer/Viewer>
#include <osg/StateAttribute>
#include <osg/StateSet>
#include <osg/PolygonMode>
#include <osgDB/ReadFile>
#include <osg/ShapeDrawable>
#include <osg/PolygonOffset>

#include <cover/coVRShader.h>
#include <cover/coVRFileManager.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

using namespace opencover;

osg::StateSet *Road::roadStateSet = NULL;
osg::StateSet *Road::batterStateSet = NULL;
osg::StateSet *Road::concreteBatterStateSet = NULL;
osg::Texture2D *Road::roadTex = NULL;
osg::Texture2D *Road::batterTex = NULL;
osg::Texture2D *Road::concreteTex = NULL;

float *Road::speedLimits = NULL;

Road::Road(std::string id, std::string name, double l, Junction *junc)
    : Tarmac(id, name)
{
    length = l;
    junction = junc;
    priority = 0;
    if (speedLimits == NULL)
    {
        speedLimits = new float[NUM_ROAD_TYPES];
        speedLimits[UNKNOWN] = 83.0;
        speedLimits[RURAL] = 27.77777;
        speedLimits[MOTORWAY] = 33.33333;
        speedLimits[TOWN] = 13.88888;
        speedLimits[LOWSPEED] = 8.3333333;
        speedLimits[PEDESTRIAN] = 1.388888;
        speedLimits[CUSTOM] = 22.222222;
    }

    xyMap[0.0] = (new PlaneStraightLine(0.0));
    zMap[0.0] = (new Polynom(0.0));
    laneSectionMap[0.0] = (new LaneSection(0.0));
    lateralProfileMap[0.0] = (new SuperelevationPolynom(0.0));
    roadTypeMap[0.0] = UNKNOWN;
    // Neu Andreas 27-11-2012
    speedLimitMap[0.0] = speedLimits[UNKNOWN];
    roadSurfaceMap[0.0] = (new RoadSurface());
    roadSensorMap[0.0] = (new RoadSensor("road_start_dummy", 0.0));

    predecessor = NULL;
    successor = NULL;
    leftNeighbour = NULL;
    leftNeighbourDirection = -1;
    rightNeighbour = NULL;
    rightNeighbourDirection = 1;

    roadGeode = NULL;
    batterGeode = NULL;
    guardRailGeode = NULL;
    roadGroup = NULL;
}

void Road::addRoadType(double s, RoadType rt)
{
    roadTypeMap[s] = rt;
}

void Road::addRoadType(double s, std::string rt)
{
    if (rt == "rural")
        roadTypeMap[s] = RURAL;
    else if (rt == "motorway")
        roadTypeMap[s] = MOTORWAY;
    else if (rt == "town")
        roadTypeMap[s] = TOWN;
    else if (rt == "lowspeed")
        roadTypeMap[s] = LOWSPEED;
    else if (rt == "pedestrian")
        roadTypeMap[s] = PEDESTRIAN;
    else if (rt == "bridge")
        roadTypeMap[s] = BRIDGE;
    else
        roadTypeMap[s] = UNKNOWN;
}
// Neu Andreas 27-11-2012
void Road::addSpeedLimit(double s, std::string rt)
{
    if (rt == "rural")
        speedLimitMap[s] = 27.77777; // 100 km/h
    else if (rt == "motorway")
        speedLimitMap[s] = 36.11111; // 130 km/h
    else if (rt == "town")
        speedLimitMap[s] = 13.88888; // 50 km/h
    else if (rt == "lowspeed")
        speedLimitMap[s] = 8.33333; // 30 km/h
    else if (rt == "pedestrian")
        speedLimitMap[s] = 1.38888; // 5  km/h
    else
        speedLimitMap[s] = 19.44444; // 70 km/h
}
void Road::addSpeedLimit(double s, double sl)
{
    speedLimitMap[s] = sl;
}
// Neu Andreas 27-11-2012
double Road::getSpeedLimit(double s, int lane)
{
    double speedLimit = Road::getLaneSection(s)->getLane(lane)->getSpeedLimit(s);
    if (speedLimit == 0)
    {
        speedLimit = (--speedLimitMap.upper_bound(s))->second;
    }
    return speedLimit;
}

Road::RoadType Road::getRoadType(double s)
{
    return (--roadTypeMap.upper_bound(s))->second;
}

void Road::setPredecessorConnection(TarmacConnection *tarmacCon)
{
    predecessor = tarmacCon;
}

void Road::setSuccessorConnection(TarmacConnection *tarmacCon)
{
    successor = tarmacCon;
}

void Road::setLeftNeighbour(Road *road, int dir)
{
    leftNeighbour = road;
    leftNeighbourDirection = dir;
    std::cerr << "Use of road neighbours records in OpenDRIVE is obsolete, please don't." << std::endl;
}

void Road::setRightNeighbour(Road *road, int dir)
{
    rightNeighbour = road;
    rightNeighbourDirection = dir;
    std::cerr << "Use of road neighbours records in OpenDRIVE is obsolete, please don't." << std::endl;
}

void Road::setJunction(Junction *junc)
{
    junction = junc;
}

void Road::addPlanViewGeometryLine(double s, double l, double x, double y, double hdg)
{
    xyMap[s] = (new PlaneStraightLine(s, l, x, y, hdg));
}

void Road::addPlanViewGeometrySpiral(double s, double l, double x, double y, double hdg, double curvStart, double curvEnd)
{
    xyMap[s] = (new PlaneClothoid(s, l, x, y, hdg, curvStart, curvEnd));
}

void Road::addPlanViewGeometryArc(double s, double l, double x, double y, double hdg, double curv)
{
    xyMap[s] = (new PlaneArc(s, l, x, y, hdg, 1 / curv));
}

void Road::addPlanViewGeometryPolynom(double s, double l, double x, double y, double hdg, double a, double b, double c, double d)
{
    xyMap[s] = (new PlanePolynom(s, l, x, y, hdg, a, b, c, d));
}

void Road::addElevationPolynom(double s, double a, double b, double c, double d)
{
    zMap[s] = (new Polynom(s, a, b, c, d));
}

void Road::addSuperelevationPolynom(double s, double a, double b, double c, double d)
{
    lateralProfileMap[s] = (new SuperelevationPolynom(s, a, b, c, d));
}

void Road::addCrossfallPolynom(double s, double a, double b, double c, double d, std::string side)
{
    if (side == "left")
    {
        lateralProfileMap[s] = (new CrossfallPolynom(s, a, b, c, d, -1.0, 0));
    }
    else if (side == "right")
    {
        lateralProfileMap[s] = (new CrossfallPolynom(s, a, b, c, d, 0, 1.0));
    }
    else
    {
        lateralProfileMap[s] = (new CrossfallPolynom(s, a, b, c, d));
    }
}

void Road::addLaneSection(LaneSection *section)
{
    laneSectionMap[section->getStart()] = section;
}

LaneSection *Road::getLaneSection(double s)
{
    /*std::cout << id << ": Lane sections: ";
	for(std::map<double, LaneSection*>::iterator lsIt = laneSectionMap.begin(); lsIt!=laneSectionMap.end(); ++lsIt) {
	std::cout << "\t " << lsIt->second << "(" << lsIt->second->getStart() << ")";
	}
	std::cout << std::endl;*/
    std::map<double, LaneSection *>::iterator lsIt = laneSectionMap.upper_bound(s);
    if (lsIt != laneSectionMap.begin())
        return (--lsIt)->second;
    else
        return lsIt->second; // toCheck or return NULL
}

int Road::traceLane(int lane, double from, double to)
{
    std::map<double, LaneSection *>::iterator sectionItFrom = --laneSectionMap.upper_bound(from);
    std::map<double, LaneSection *>::iterator sectionItTo = --laneSectionMap.upper_bound(to);
    if (to > from)
    {
        for (; sectionItFrom != sectionItTo; ++sectionItFrom)
        {
            lane = sectionItFrom->second->getLaneSuccessor(lane);
        }
        if (from < this->length && to >= this->length)
        {
            lane = sectionItTo->second->getLaneSuccessor(lane);
        }
    }
    else
    {
        for (; sectionItFrom != sectionItTo; --sectionItFrom)
        {
            lane = sectionItFrom->second->getLanePredecessor(lane);
        }
        if (from > 0.0 && to <= 0.0)
        {
            lane = sectionItTo->second->getLanePredecessor(lane);
        }
    }
    return lane;
}

double Road::getLaneEnd(int &lane, double from, int dir, bool &isSignal)
{
    std::map<double, LaneSection *>::iterator sectionIt = --laneSectionMap.upper_bound(from);
    if (!sectionIt->second->getLane(lane))
    {
        lane = Lane::NOLANE;
        return from;
    }

    //Road signal
    double signalLaneEnd = 1e10;
    for (unsigned int i = 0; i < getNumRoadSignals(); ++i)
    {
        RoadSignal *signal = getRoadSignal(i);

        //not traffic lights
        if (!(signal->getType() == 1000001))
            continue;

        //not valid in current direction
        if ((signal->getOrientation() + dir) == 0)
            continue;

        //not signaling stop
        if (signal->getValue() > 0.0)
            continue;

        if (dir >= 0)
        {
            if ((signal->getS() - from) >= 0.0)
            {
                signalLaneEnd = std::min(signalLaneEnd, signal->getS());
                isSignal = true;
            }
        }
        else
        {
            if ((signal->getS() - from) <= 0.0)
            {
                signalLaneEnd = std::min(signalLaneEnd, signal->getS());
                isSignal = true;
            }
        }
    }

    if (dir >= 0)
    {
        while (sectionIt != laneSectionMap.end())
        {
            lane = sectionIt->second->getLaneSuccessor(lane);
            if (lane == Lane::NOLANE || lane == 0)
            {
                ++sectionIt;
                return std::min(((sectionIt == laneSectionMap.end()) ? this->length : sectionIt->first), signalLaneEnd);
            }
            ++sectionIt;
        }
        return std::min(this->length, signalLaneEnd);
    }
    else
    {
        ++sectionIt;
        do
        {
            --sectionIt;
            lane = sectionIt->second->getLanePredecessor(lane);
            if (lane == Lane::NOLANE || lane == 0)
            {
                return std::min(sectionIt->first, signalLaneEnd);
            }
        } while (sectionIt != laneSectionMap.begin());

        return (signalLaneEnd > 0.0 && signalLaneEnd < 1e10) ? signalLaneEnd : 0.0;
    }
}

void Road::addRoadSurface(RoadSurface *surface, double sStart, double sEnd)
{
    std::map<double, RoadSurface *>::iterator firstCollide = roadSurfaceMap.lower_bound(sStart);
    if (firstCollide != roadSurfaceMap.end())
    {
        roadSurfaceMap.erase(firstCollide, roadSurfaceMap.lower_bound(sEnd));
    }

    roadSurfaceMap[sStart] = surface;
    roadSurfaceMap[sEnd] = new RoadSurface();
}

void Road::addRoadObject(RoadObject *obj)
{
    roadObjectVector.push_back(obj);
}

/**
 * Functions for handling crosswalks on a road
 */
void Road::addCrosswalk(Crosswalk *cw)
{
    crosswalkVector.push_back(cw);
}
Crosswalk *Road::getCrosswalk(double s, int lane)
{
    for (std::vector<Crosswalk *>::iterator cwIt = crosswalkVector.begin(); cwIt != crosswalkVector.end(); cwIt++)
    {
        Crosswalk *cw = (*cwIt);

        // Test whether position is in the crosswalk
        double start = cw->getS();
        double end = start + cw->getLength();
        if (s <= end && s >= start)
        {
            // Test whether current lane is included
            int from = cw->getValidity().first;
            int to = cw->getValidity().second;
            if (lane <= to && lane >= from)
                return cw;
        }
    }
    return NULL;
}
Crosswalk *Road::getNextCrosswalk(double s, int dir, int lane)
{
    // Return the crosswalk with the minimum distance from s in the given direction
    double minDist = 1.0e8;
    Crosswalk *closest = NULL;

    for (std::vector<Crosswalk *>::iterator cwIt = crosswalkVector.begin(); cwIt != crosswalkVector.end(); cwIt++)
    {
        Crosswalk *cw = (*cwIt);

        // Test whether position is before the crosswalk (in the given direction)
        double closestPoint = cw->getS() + (dir < 0 ? cw->getLength() : 0.0);
        if (dir > 0 ? s <= closestPoint : s >= closestPoint)
        {
            // Test whether current lane is included
            int from = cw->getValidity().first;
            int to = cw->getValidity().second;
            if (lane <= to && lane >= from)
            {
                double dist = (dir > 0 ? closestPoint - s : s - closestPoint);
                if (dist < minDist)
                {
                    minDist = dist;
                    closest = cw;
                }
            }
        }
    }
    return closest;
}
bool Road::isAnyCrosswalks(double firstPoint, double secondPoint, int lane)
{
    double searchStart = firstPoint <= secondPoint ? firstPoint : secondPoint;
    double searchEnd = firstPoint >= secondPoint ? firstPoint : secondPoint;

    for (std::vector<Crosswalk *>::iterator cwIt = crosswalkVector.begin(); cwIt != crosswalkVector.end(); cwIt++)
    {
        Crosswalk *cw = (*cwIt);

        // Test whether there is a crosswalk in this range
        double cwStart = cw->getS();
        double cwEnd = cwStart + cw->getLength();
        if (searchStart <= cwEnd && searchEnd >= cwStart)
        {
            // Test whether current lane is included
            int from = cw->getValidity().first;
            int to = cw->getValidity().second;
            if (lane <= to && lane >= from)
                return true;
        }
    }
    return false;
}

/**
 * Return the lane numbers of all sidewalks on this roadat the given s-position
 */
std::vector<int> Road::getSidewalks(double s)
{
    // Find sidewalks in this lane section
    std::vector<int> sidewalks;
    LaneSection *ls = getLaneSection(s);
    for (int i = 1; i <= ls->getNumLanesLeft(); i++)
    {
        // Is this a sidewalk?
        if (ls->isSidewalk(i))
        {
            sidewalks.push_back(i);
        }
    }
    for (int i = 1; i <= ls->getNumLanesRight(); i++)
    {
        // Is this a sidewalk?
        if (ls->isSidewalk(-1 * i))
        {
            sidewalks.push_back(-1 * i);
        }
    }
    return sidewalks;
}

void Road::addRoadSignal(RoadSignal *signal)
{
    Transform signalTransform = getRoadTransform(signal->getS(), signal->getT());
    signal->setTransform(signalTransform);
    roadSignalVector.push_back(signal);
}

RoadSignal *Road::getRoadSignal(const unsigned int &i)
{
    return roadSignalVector[i];
}

unsigned int Road::getNumRoadSignals()
{
    return roadSignalVector.size();
}

void Road::addRoadSensor(RoadSensor *sensor)
{
    roadSensorMap[sensor->getS()] = sensor;
}

std::map<double, RoadSensor *>::iterator Road::getRoadSensorMapEntry(const double &s)
{
    return (--roadSensorMap.upper_bound(s));
}

std::map<double, RoadSensor *>::iterator Road::getRoadSensorMapEnd()
{
    return roadSensorMap.end();
}

void Road::setLength(double l)
{
    length = l;
}

double Road::getLength()
{
    return length;
}

double Road::getHeading(double s)
{
    return ((--xyMap.upper_bound(s))->second)->getOrientation(s);
}

double Road::getCurvature(double s)
{
    return ((--xyMap.upper_bound(s))->second)->getCurvature(s);
}

double Road::getSuperelevation(double s, double t)
{
    return ((--lateralProfileMap.upper_bound(s))->second)->getAngle(s, t);
}

Vector3D Road::getCenterLinePoint(double s)
{
    Vector3D xyPoint = ((--xyMap.upper_bound(s))->second)->getPoint(s);
    Vector2D zPoint = ((--zMap.upper_bound(s))->second)->getPoint(s);

    return Vector3D(xyPoint.x(),
                    xyPoint.y(),
                    zPoint[0]);
}

Vector3D Road::getTangentVector(double s)
{
    return Vector3D(((--xyMap.upper_bound(s))->second)->getTangentVector(s),
                    ((--zMap.upper_bound(s))->second)->getSlope(s));
}

Vector3D Road::getNormalVector(double s)
{
    return Vector3D(((--xyMap.upper_bound(s))->second)->getNormalVector(s),
                    ((--zMap.upper_bound(s))->second)->getCurvature(s));
}

RoadPoint Road::getChordLinePoint(double s)
{
    Vector3D xyPoint(((--xyMap.upper_bound(s))->second)->getPoint(s));
    double zValue = (((--zMap.upper_bound(s))->second)->getValue(s));
    return RoadPoint(xyPoint.x(), xyPoint.y(), zValue);
}

Vector3D Road::getChordLinePlanViewPoint(double s)
{
    Vector3D xyPoint(((--xyMap.upper_bound(s))->second)->getPoint(s));
    //return Vector2D(xyPoint.x(), xyPoint.y());
    return xyPoint;
}

double Road::getChordLineElevation(double s)
{
    return (((--zMap.upper_bound(s))->second)->getValue(s));
}

double Road::getChordLineElevationSlope(double s)
{
    return (((--zMap.upper_bound(s))->second)->getSlope(s));
}

double Road::getChordLineElevationCurvature(double s)
{
    return (((--zMap.upper_bound(s))->second)->getCurvature(s));
}

RoadPoint Road::getRoadPoint(double s, double t)
{
    //RoadPoint center = getChordLinePoint(s);
    Vector3D xyPoint = ((--xyMap.upper_bound(s))->second)->getPoint(s);
    Vector2D zPoint = ((--zMap.upper_bound(s))->second)->getPoint(s);

    LaneSection *section = (--laneSectionMap.upper_bound(s))->second;
    double height = section->getHeight(s, t);

    RoadSurface *surface = (--roadSurfaceMap.upper_bound(s))->second;
    height += surface->height(s, t);
    //std::cout << "s: " << s << ", t: " << t << ", height: " << height << std::endl;
    //height = 0.0;

    //std::cerr << "Center: x: " << center.x() << ", y: " << center.y() << ", z: " << center.z() << std::endl;

    double alpha = ((--lateralProfileMap.upper_bound(s))->second)->getAngle(s, t);
    double beta = zPoint[1];
    double gamma = xyPoint[2];
    //std::cerr << "s: " << s << "Alpha: " << alpha << ", Beta: " << beta << ", Gamma: " << gamma << std::endl;
    double sinalpha = sin(alpha);
    double cosalpha = cos(alpha);
    double sinbeta = sin(beta);
    double cosbeta = cos(beta);
    double singamma = sin(gamma);
    double cosgamma = cos(gamma);

    return RoadPoint(xyPoint.x() + (sinalpha * sinbeta * cosgamma - cosalpha * singamma) * t,
                     xyPoint.y() + (sinalpha * sinbeta * singamma + cosalpha * cosgamma) * t,
                     zPoint[0] + height + (sinalpha * cosbeta) * t,
                     sinalpha * singamma + cosalpha * sinbeta * cosgamma,
                     cosalpha * sinbeta * singamma - sinalpha * cosgamma,
                     cosalpha * cosbeta);
}

bool Road::isOnRoad(Vector2D roadPos)
{
    double leftWidth, rightWidth;
    getLaneSection(roadPos.u())->getRoadWidth(roadPos.u(), leftWidth, rightWidth);
    if ((roadPos.v() < 0 && roadPos.v() < -rightWidth) || (roadPos.v() > 0 && roadPos.v() > leftWidth))
    {
        return false;
    }
    else
    {
        return true;
    }
}

void Road::getLaneWidthAndOffset(double s, int i, double &width, double &widthOffset)
{
    LaneSection *section = (--laneSectionMap.upper_bound(s))->second;
    widthOffset = section->getDistanceToLane(s, i);
    width = section->getLaneWidth(s, i);
}

void Road::getLaneRoadPoints(double s, int i, RoadPoint &pointIn, RoadPoint &pointOut, double &disIn, double &disOut)
{
    LaneSection *section = (--laneSectionMap.upper_bound(s))->second;
    disIn = section->getDistanceToLane(s, i);
    double laneWidth = section->getLaneWidth(s, i);
    disOut = disIn + laneWidth;
    //std::cerr << "s: " << s << ", i: " << i << ", distance: " << disIn << ", width: " << disOut-disIn << std::endl;
    Vector3D xyPoint = ((--xyMap.upper_bound(s))->second)->getPoint(s);
    Vector2D zPoint = ((--zMap.upper_bound(s))->second)->getPoint(s);

    double heightIn = 0.0;
    double heightOut = 0.0;
    section->getLaneBorderHeights(s, i, heightIn, heightOut);

    double alpha = ((--lateralProfileMap.upper_bound(s))->second)->getAngle(s, (disOut + disIn) * 0.5);
    double beta = zPoint[1];
    double gamma = xyPoint[2];
    double Tx = (sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma));
    double Ty = (sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma));
    double Tz = (sin(alpha) * cos(beta));
    //std::cout << "s: " << s << ", i: " << i << ", alpha: " << alpha << ", beta: " << beta << ", gamma: " << gamma << std::endl;

    double nx = sin(alpha) * sin(gamma) + cos(alpha) * sin(beta) * cos(gamma);
    double ny = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma);
    double nz = cos(alpha) * cos(beta);

    pointIn = RoadPoint(xyPoint.x() + Tx * disIn, xyPoint.y() + Ty * disIn, zPoint[0] + heightIn + Tz * disIn, nx, ny, nz);
    pointOut = RoadPoint(xyPoint.x() + Tx * disOut, xyPoint.y() + Ty * disOut, zPoint[0] + heightOut + Tz * disOut, nx, ny, nz);
    //std::cout << "s: " << s << ", i: " << i << ", inner: x: " << pointIn.x() << ", y: " << pointIn.y() << ", z: " << pointIn.z() << std::endl;
    //std::cout << "s: " << s << ", i: " << i << ", outer: x: " << pointOut.x() << ", y: " << pointOut.y() << ", z: " << pointOut.z() << std::endl << std::endl;
}

void Road::getBatterPoint(double s, int i, RoadPoint &pointOut, RoadPoint &pointCenter, RoadPoint &pointIn, double &width, int marker)
{

    LaneSection *section = (--laneSectionMap.upper_bound(s))->second;

    //RoadPoint batterpointIn = pointIn;
    //RoadPoint batterpointOut = pointOut;

    //pointIn = pointOut;
    //pointCenter = RoadPoint ( (pointOut.x() - batterpointIn.x())/(sqrt((pointOut.x()-batterpointIn.x())*(pointOut.x()-batterpointIn.x())+(pointOut.y()-batterpointIn.y())*(pointOut.y()-batterpointIn.y())+(pointOut.z()-batterpointIn.z())*(pointOut.z()-batterpointIn.z())))+pointOut.x() , (pointOut.y() - batterpointIn.y())/(sqrt((pointOut.x()-batterpointIn.x())*(pointOut.x()-batterpointIn.x())+(pointOut.y()-batterpointIn.y())*(pointOut.y()-batterpointIn.y())+(pointOut.z()-batterpointIn.z())*(pointOut.z()-batterpointIn.z())))+pointOut.y(), (pointOut.z() - batterpointIn.z())/(sqrt((pointOut.x()-batterpointIn.x())*(pointOut.x()-batterpointIn.x())+(pointOut.y()-batterpointIn.y())*(pointOut.y()-batterpointIn.y())+(pointOut.z()-batterpointIn.z())*(pointOut.z()-batterpointIn.z())))+pointOut.z(), pointOut.nx(), pointOut.ny(), pointOut.nz());
    //
    double disIn, disOut;
    getLaneRoadPoints(s, i, pointOut, pointIn, disIn, disOut);
    //Vector3D bitangent = pointIn.normal();
    Vector3D tangent = getTangentVector(s);
    Vector3D normal = pointIn.normal();
    Vector3D bitangent = tangent.cross(normal);
    bitangent.normalize();
    //std::cout << "tangent: " << tangent << ", normal: " << normal << ", bitangent: " << bitangent << std::endl;
    double bitangent_sign = (marker == 1) ? -1.0 : 1.0;
    pointCenter = RoadPoint(pointIn.x() + bitangent_sign * bitangent.x(), pointIn.y() + bitangent_sign * bitangent.y(), pointIn.z() + bitangent_sign * bitangent.z(), pointIn.nx(), pointIn.ny(), pointIn.nz());

    //int roadType = getRoadType(s);
    int roadType = 0;
    double batterWidthLeft;
    double batterFallLeft;
    double batterFallRight;
    double batterFall;
    double batterWidthRight;
    std::map<int, Batter *> batterMap = section->getBatterMap();

    if (batterMap.size() == 0)
    {

        switch (roadType)
        {
        case 0: //default
            batterWidthLeft = 6;
            batterFallLeft = 0.3f;
            batterWidthRight = 6;
            batterFallRight = 0.3f;
            width = (marker == 1) ? batterWidthLeft : batterWidthRight;
            batterFall = (marker == 1) ? batterFallLeft : batterFallRight;
            break;
        case 1: //rural
            batterWidthLeft = 5;
            batterFallLeft = 0.35f;
            batterWidthRight = 5;
            batterFallRight = 0.35f;
            width = (marker == 1) ? batterWidthLeft : batterWidthRight;
            batterFall = (marker == 1) ? batterFallLeft : batterFallRight;
            break;
        case 2: //motorway
            batterWidthLeft = 5;
            batterFallLeft = 0.35f;
            batterWidthRight = 5;
            batterFallRight = 0.35f;
            width = (marker == 1) ? batterWidthLeft : batterWidthRight;
            batterFall = (marker == 1) ? batterFallLeft : batterFallRight;
            break;
        case 3: //town
            batterWidthLeft = 3;
            batterFallLeft = 0.1f;
            batterWidthRight = 3;
            batterFallRight = 0.1f;
            width = (marker == 1) ? batterWidthLeft : batterWidthRight;
            batterFall = (marker == 1) ? batterFallLeft : batterFallRight;
            break;
        case 4: //lowspeed
            batterWidthLeft = 5;
            batterFallLeft = 0.5f;
            batterWidthRight = 5;
            batterFallRight = 0.5f;
            width = (marker == 1) ? batterWidthLeft : batterWidthRight;
            batterFall = (marker == 1) ? batterFallLeft : batterFallRight;
            break;
        case 5: //pedestrian
            batterWidthLeft = 10;
            batterFallLeft = 0.0f;
            batterWidthRight = 10;
            batterFallRight = 0.0f;
            width = (marker == 1) ? batterWidthLeft : batterWidthRight;
            batterFall = (marker == 1) ? batterFallLeft : batterFallRight;
            break;
        }
    }
    else
    {

        if (section->getBatterWidth(s, i) > 0)
        {
            width = section->getBatterWidth(s, i);
            batterFall = section->getBatterFall(s, i);
        }
        else
        {
            width = -section->getBatterWidth(s, i);
            batterFall = section->getBatterFall(s, i);
        }
    }

    //pointOut = RoadPoint ( width*(pointCenter.x() - batterpointOut.x())+pointCenter.x() , width*(pointCenter.y()-batterpointOut.y()) + pointCenter.y() ,(width*(pointCenter.z()-batterpointOut.z()) + pointCenter.z()) - (sin(batterFall)*width), batterpointOut.nx(), batterpointOut.ny(), batterpointOut.nz());
    //pointOut = RoadPoint ( width*(bitangent_sign*bitangent.x())+pointCenter.x() , width*(bitangent_sign*bitangent.y()) + pointCenter.y() ,(width*(bitangent_sign*bitangent.z()) + pointCenter.z()) - (sin(batterFall)*width), pointIn.nx(), pointIn.ny(), pointIn.nz());
    Quaternion q(batterFall * bitangent_sign, tangent);
    Vector3D pointOutPos = pointCenter.pos() + ((q * (bitangent * (width * bitangent_sign))) * q.T()).getVector();
    Vector3D pointOutNorm = ((q * pointIn.normal()) * q.T()).getVector();
    pointOut = RoadPoint(pointOutPos, pointOutNorm);
}

void Road::getGuardRailPoint(RoadPoint &pointDown, RoadPoint &pointUp, bool firstRound, bool lastRound, std::map<double, LaneSection *>::iterator lsIt)
{

    RoadPoint e;
    RoadPoint p;
    double width = 0;
    LaneSection *ls;

    if (firstRound == true)
    {
        if (lsIt == laneSectionMap.begin())
            width = 0.75;
        else
        {
            std::map<double, LaneSection *>::iterator previouslsIt = lsIt;
            --previouslsIt;
            ls = (*previouslsIt).second;
            if (getRoadType(ls->getStart()) != 2)
                width = 0.75;
        }
    }

    if (lastRound == true)
    {
        std::map<double, LaneSection *>::iterator nextlsIt = lsIt;
        ++nextlsIt;
        if (nextlsIt == laneSectionMap.end())
        {
            width = 0.75;
        }
        else
        {
            ls = (*nextlsIt).second;

            if (getRoadType(ls->getStart()) != 2)
                width = 0.75;
        }
    }

    e = RoadPoint((sqrt((pointDown.x() - pointUp.x()) * (pointDown.x() - pointUp.x()) + (pointDown.y() - pointUp.y()) * (pointDown.y() - pointUp.y()) + (pointDown.z() - pointUp.z()) * (pointDown.z() - pointUp.z()))), (sqrt((pointDown.x() - pointUp.x()) * (pointDown.x() - pointUp.x()) + (pointDown.y() - pointUp.y()) * (pointDown.y() - pointUp.y()) + (pointDown.z() - pointUp.z()) * (pointDown.z() - pointUp.z()))), (sqrt((pointDown.x() - pointUp.x()) * (pointDown.x() - pointUp.x()) + (pointDown.y() - pointUp.y()) * (pointDown.y() - pointUp.y()) + (pointDown.z() - pointUp.z()) * (pointDown.z() - pointUp.z()))));

    pointDown = RoadPoint(0.5 * ((pointDown.x() - pointUp.x()) / e.x()) + pointDown.x(), 0.5 * ((pointDown.y() - pointUp.y()) / e.y()) + pointDown.y(), 0.5 * ((pointDown.z() - pointUp.z()) / e.z()) + pointDown.z() + (0.44 - width), pointUp.x() - pointDown.x(), pointUp.y() - pointDown.y(), pointUp.z() - pointDown.z());

    pointUp = RoadPoint(pointDown.x(), pointDown.y(), pointDown.z() + 0.31, pointDown.nx(), pointDown.ny(), pointDown.nz());

    p = pointDown;
}

osg::Geometry *Road::getGuardRailPost(RoadPoint pointDown, RoadPoint pointUp, RoadPoint postPoint, RoadPoint nextrailPointDown)
{
    osg::Geometry *guardRailPostGeometry;
    guardRailPostGeometry = new osg::Geometry();

    osg::Vec3Array *guardRailPostVertices;
    guardRailPostVertices = new osg::Vec3Array;
    guardRailPostGeometry->setVertexArray(guardRailPostVertices);

    osg::Vec2Array *guardRailPostTexCoords;
    guardRailPostTexCoords = new osg::Vec2Array;
    guardRailPostGeometry->setTexCoordArray(3, guardRailPostTexCoords);

    osg::Vec3Array *guardRailPostNormals;
    guardRailPostNormals = new osg::Vec3Array;
    guardRailPostGeometry->setNormalArray(guardRailPostNormals);
    guardRailPostGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    RoadPoint e;
    RoadPoint p1Down;
    RoadPoint p2Down;
    RoadPoint p3Down;
    RoadPoint p4Down;
    RoadPoint p1Up;
    RoadPoint p2Up;
    RoadPoint p3Up;
    RoadPoint p4Up;

    e = RoadPoint((sqrt((pointDown.x() - pointUp.x()) * (pointDown.x() - pointUp.x()) + (pointDown.y() - pointUp.y()) * (pointDown.y() - pointUp.y()) + (pointDown.z() - pointUp.z()) * (pointDown.z() - pointUp.z()))), (sqrt((pointDown.x() - pointUp.x()) * (pointDown.x() - pointUp.x()) + (pointDown.y() - pointUp.y()) * (pointDown.y() - pointUp.y()) + (pointDown.z() - pointUp.z()) * (pointDown.z() - pointUp.z()))), (sqrt((pointDown.x() - pointUp.x()) * (pointDown.x() - pointUp.x()) + (pointDown.y() - pointUp.y()) * (pointDown.y() - pointUp.y()) + (pointDown.z() - pointUp.z()) * (pointDown.z() - pointUp.z()))));

    p1Down = RoadPoint(0.51 * ((pointDown.x() - pointUp.x()) / e.x()) + pointDown.x(), 0.51 * ((pointDown.y() - pointUp.y()) / e.y()) + pointDown.y(), 0.51 * ((pointDown.z() - pointUp.z()) / e.z()) + pointDown.z());

    p2Down = RoadPoint(0.7 * ((pointDown.x() - pointUp.x()) / e.x()) + pointDown.x(), 0.7 * ((pointDown.y() - pointUp.y()) / e.y()) + pointDown.y(), 0.7 * ((pointDown.z() - pointUp.z()) / e.z()) + pointDown.z(), pointUp.x() - pointDown.x(), pointUp.y() - pointDown.y(), pointUp.z() - pointDown.z());

    p3Down = p3Down = RoadPoint(0.04 * (postPoint.x() - nextrailPointDown.x()) + p2Down.x(), 0.04 * (postPoint.y() - nextrailPointDown.y()) + p2Down.y(), 0.04 * (postPoint.z() - nextrailPointDown.z()) + p2Down.z());

    p4Down = RoadPoint(0.04 * (postPoint.x() - nextrailPointDown.x()) + p1Down.x(), 0.04 * (postPoint.y() - nextrailPointDown.y()) + p1Down.y(), 0.04 * (postPoint.z() - nextrailPointDown.z()) + p1Down.z());

    p1Up = RoadPoint(p1Down.x(), p1Down.y(), p1Down.z() + 0.7);
    p2Up = RoadPoint(p2Down.x(), p2Down.y(), p2Down.z() + 0.7);
    p3Up = RoadPoint(p3Down.x(), p3Down.y(), p3Down.z() + 0.7);
    p4Up = RoadPoint(p4Down.x(), p4Down.y(), p4Down.z() + 0.7);

    guardRailPostVertices->push_back(osg::Vec3(p1Down.x(), p1Down.y(), p1Down.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(0.0, 0.0));
    guardRailPostNormals->push_back(osg::Vec3(p4Down.x(), p4Down.y(), p4Down.z()));
    guardRailPostVertices->push_back(osg::Vec3(p2Down.x(), p2Down.y(), p2Down.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(0.0, 1.0));
    guardRailPostNormals->push_back(osg::Vec3(p4Down.x(), p4Down.y(), p4Down.z()));
    guardRailPostVertices->push_back(osg::Vec3(p2Up.x(), p2Up.y(), p2Up.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(1.0, 1.0));
    guardRailPostNormals->push_back(osg::Vec3(p4Down.x(), p4Down.y(), p4Down.z()));
    guardRailPostVertices->push_back(osg::Vec3(p1Up.x(), p1Up.y(), p1Up.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(1.0, 0.0));
    guardRailPostNormals->push_back(osg::Vec3(p4Down.x(), p4Down.y(), p4Down.z()));
    osg::DrawArrays *guardRailPost1 = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4);
    guardRailPostGeometry->addPrimitiveSet(guardRailPost1);

    guardRailPostVertices->push_back(osg::Vec3(p2Down.x(), p2Down.y(), p2Down.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(0.0, 0.0));
    guardRailPostNormals->push_back(osg::Vec3(p1Down.x(), p1Down.y(), p1Down.z()));
    guardRailPostVertices->push_back(osg::Vec3(p3Down.x(), p3Down.y(), p3Down.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(1.0, 0.0));
    guardRailPostNormals->push_back(osg::Vec3(p1Down.x(), p1Down.y(), p1Down.z()));
    guardRailPostVertices->push_back(osg::Vec3(p3Up.x(), p3Up.y(), p3Up.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(1.0, 1.0));
    guardRailPostNormals->push_back(osg::Vec3(p1Down.x(), p1Down.y(), p1Down.z()));
    guardRailPostVertices->push_back(osg::Vec3(p2Up.x(), p2Up.y(), p2Up.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(0.0, 0.1));
    guardRailPostNormals->push_back(osg::Vec3(p1Down.x(), p1Down.y(), p1Down.z()));
    osg::DrawArrays *guardRailPost2 = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 4, 4);
    guardRailPostGeometry->addPrimitiveSet(guardRailPost2);

    guardRailPostVertices->push_back(osg::Vec3(p3Down.x(), p3Down.y(), p3Down.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(1.0, 1.0));
    guardRailPostNormals->push_back((-osg::Vec3(p4Down.x(), p4Down.y(), p4Down.z())));
    guardRailPostVertices->push_back(osg::Vec3(p4Down.x(), p4Down.y(), p4Down.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(0.0, 1.0));
    guardRailPostNormals->push_back(-(osg::Vec3(p4Down.x(), p4Down.y(), p4Down.z())));
    guardRailPostVertices->push_back(osg::Vec3(p4Up.x(), p4Up.y(), p4Up.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(1.0, 0.0));
    guardRailPostNormals->push_back(-(osg::Vec3(p4Down.x(), p4Down.y(), p4Down.z())));
    guardRailPostVertices->push_back(osg::Vec3(p3Up.x(), p3Up.y(), p3Up.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(0.0, 0.0));
    guardRailPostNormals->push_back(-(osg::Vec3(p4Down.x(), p4Down.y(), p4Down.z())));
    osg::DrawArrays *guardRailPost3 = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 8, 4);
    guardRailPostGeometry->addPrimitiveSet(guardRailPost3);

    guardRailPostVertices->push_back(osg::Vec3(p1Down.x(), p1Down.y(), p1Down.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(0.0, 0.0));
    guardRailPostNormals->push_back(-(osg::Vec3(p1Down.x(), p1Down.y(), p1Down.z())));
    guardRailPostVertices->push_back(osg::Vec3(p4Down.x(), p4Down.y(), p4Down.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(1.0, 0.0));
    guardRailPostNormals->push_back(-(osg::Vec3(p1Down.x(), p1Down.y(), p1Down.z())));
    guardRailPostVertices->push_back(osg::Vec3(p4Up.x(), p4Up.y(), p4Up.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(1.0, 1.0));
    guardRailPostNormals->push_back(-(osg::Vec3(p1Down.x(), p1Down.y(), p1Down.z())));
    guardRailPostVertices->push_back(osg::Vec3(p1Up.x(), p1Up.y(), p1Up.z()));
    guardRailPostTexCoords->push_back(osg::Vec2(0.0, 1.0));
    guardRailPostNormals->push_back(-(osg::Vec3(p1Down.x(), p1Down.y(), p1Down.z())));
    osg::DrawArrays *guardRailPost4 = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 12, 4);
    guardRailPostGeometry->addPrimitiveSet(guardRailPost4);

    return guardRailPostGeometry;
}

Transform Road::getRoadTransform(const double &s, const double &t)
{
    Vector3D xyPoint = ((--xyMap.upper_bound(s))->second)->getPoint(s);
    Vector2D zPoint = ((--zMap.upper_bound(s))->second)->getPoint(s);

    //std::cerr << "Center: x: " << center.x() << ", y: " << center.y() << ", z: " << center.z() << std::endl;

    double alpha = ((--lateralProfileMap.upper_bound(s))->second)->getAngle(s, t);
    //std::cerr << "s: " << s << "Alpha: " << alpha << ", Beta: " << beta << ", Gamma: " << gamma << std::endl;
    //double sinalpha = sin(alpha); double cosalpha=cos(alpha);
    //double sinbeta = sin(beta); double cosbeta=cos(beta);
    //double singamma = sin(gamma); double cosgamma=cos(gamma);

    Quaternion qz(xyPoint[2], Vector3D(0, 0, 1));
    Quaternion qya(zPoint[1], Vector3D(0, 1, 0));
    Quaternion qxaa(alpha, Vector3D(1, 0, 0));

    Quaternion q = qz * qya * qxaa;
    /*
	return RoadTransform(	xyPoint.x() + (sinalpha*sinbeta*cosgamma-cosalpha*singamma)*t,
							xyPoint.y() + (sinalpha*sinbeta*singamma+cosalpha*cosgamma)*t,
							zPoint.z() + (sinalpha*cosbeta)*t,
                     alpha, beta, gamma);
   */

    LaneSection *section = (--laneSectionMap.upper_bound(s))->second;
    return Transform(Vector3D(xyPoint.x(), xyPoint.y(), zPoint[0] + section->getHeight(s, t)) + (q * Vector3D(0, t, 0) * q.T()).getVector(), q);
}

int Road::getLaneNumber(double s, double t)
{
    LaneSection *section = getLaneSection(s);
    if (t > 0)
    {
        int lane = 1;
        int numLanesLeft = section->getNumLanesLeft();
        for (int i = 0; i < numLanesLeft; i++)
        {
            if (section->getDistanceToLane(s, lane) > t)
            {
                lane--;
                return lane;
            }
            lane++;
        }
        return lane;
    }
    else if (t < 0)
    {
        int lane = -1;
        int numLanesRight = section->getNumLanesRight();
        for (int i = 0; i < numLanesRight; i++)
        {
            if (section->getDistanceToLane(s, lane) < t)
            {
                lane++;

                return lane;
            }
            lane--;
        }

        return lane;
    }
    return 0;
}

double Road::getRoadLaneOuterPos(double s, int lane)
{
    LaneSection *section = (--laneSectionMap.upper_bound(s))->second;

    return section->getDistanceToLane(s, lane) + section->getLaneWidth(s, lane);
}

void Road::getRoadSideWidths(double s, double &r, double &l)
{
    LaneSection *section = (--laneSectionMap.upper_bound(s))->second;

    int numLanesRight = getLaneSection(s)->getNumLanesRight();
    r = section->getDistanceToLane(s, -numLanesRight) + section->getLaneWidth(s, -numLanesRight);

    int numLanesLeft = getLaneSection(s)->getNumLanesLeft();
    l = section->getDistanceToLane(s, numLanesLeft) + section->getLaneWidth(s, numLanesLeft);
}

Junction *Road::getJunction()
{
    return junction;
}
bool Road::isJunctionPath()
{
    return (junction != NULL);
}

void Road::addCrossingRoadPosition(Road *road, double s)
{
    crossingRoadPositionMap[road] = s;
}

const std::map<Road *, double> &Road::getCrossingRoadPositionMap()
{
    return crossingRoadPositionMap;
}

std::string Road::getTypeSpecifier()
{
    return std::string("road");
}

std::map<double, PlaneCurve *> Road::getPlaneCurveMap()
{
    return xyMap;
}

std::map<double, Polynom *> Road::getElevationMap()
{
    return zMap;
}

std::map<double, LateralProfile *> Road::getLateralMap()
{
    return lateralProfileMap;
}

std::map<double, LaneSection *> Road::getLaneSectionMap()
{
    return laneSectionMap;
}

TarmacConnection *Road::getPredecessorConnection()
{
    return predecessor;
}

TarmacConnection *Road::getSuccessorConnection()
{
    return successor;
}

TarmacConnection *Road::getConnection(int dir)
{
    if (dir == -1)
    {
        return getPredecessorConnection();
    }
    else if (dir == 1)
    {
        return getSuccessorConnection();
    }
    else
    {
        return NULL;
    }
}

Vector2D Road::searchPositionNoBorder(const Vector3D &pos, double sinit)
{
    double crit = 0.01;

    double sOld = 0;
    double s = 0;
    int numSecs = 5;

    if (sinit < 0)
    {
        //double ds = (pos - getCenterLinePoint(0)).length();
        //double de = (pos - getCenterLinePoint(this->length)).length();
        //s = (this->length)*ds/(ds+de);

        std::vector<double> supportPointVector;
        std::vector<Vector3D> supportVectorVector;
        for (int i = 0; i < numSecs + 1; ++i)
        {
            supportPointVector.push_back(i * 1.0 / numSecs * this->length);
            //supportVectorVector.push_back( getCenterLinePoint(supportPointVector[i]) - pos);
            supportVectorVector.push_back(getCenterLinePoint(supportPointVector[i]));
        }
        //double minArea = 1e10;
        //double minCircum = 1e10;
        double minDistance = 1e10;
        double minGamma = supportPointVector[0];
        int minPointer = 0;
        for (int i = 0; i < numSecs; ++i)
        {
            /*double area = ((supportVectorVector[i]).cross(supportVectorVector[i+1])).length();
         if(area < minArea) {
            minArea = area;
            minPointer = i;
         }*/
            /*double circum = supportVectorVector[i].length() + supportVectorVector[i+1].length() + (supportVectorVector[i] - supportVectorVector[i+1]).length();
         if(circum < minCircum) {
            minCircum = circum;
            minPointer = i;
         }*/
            Vector3D t = (supportVectorVector[i + 1] - supportVectorVector[i]);
            double gamma = t.dot(pos - supportVectorVector[i]) / t.dot(t);
            Vector3D n = pos - supportVectorVector[i] - t * gamma;
            double distance = n.dot(n);
            if (distance < minDistance)
            {
                minDistance = distance;
                minGamma = gamma;
                minPointer = i;
            }
        }

        //double dis1 = supportVectorVector[minPointer].length();
        //double dis2 = supportVectorVector[minPointer+1].length();
        //s = dis1/(dis1+dis2)*(supportPointVector[minPointer+1]-supportPointVector[minPointer]) + supportPointVector[minPointer];
        s = supportPointVector[minPointer] + minGamma * (supportPointVector[minPointer + 1] - supportPointVector[minPointer]);
        if (s < 0 || s > this->length)
        {
            return Vector2D(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
        }
        //std::cout << "Choosing " << minPointer+1 << ". road piece: initial s: " << s << ", ndist: " << minDistance << ", tgamma: " << minGamma << std::endl;
    }
    else
    {
        s = sinit;
    }

    Vector3D rc(0.0);
    Vector3D t(0.0);
    int it = 0;
    do
    {
        sOld = s;
        rc = pos - getCenterLinePoint(s);
        t = getTangentVector(s);
        Vector3D n = getNormalVector(s);
        //Flo old:
        //s = s - (rc.dot(t)/(rc.dot(n)-t.dot(t)));
        //    sApprox = sApprox - (QVector2D::dotProduct(rc, t) / (-fabs(QVector2D::dotProduct(rc, n)) - t.length())); // more iterations, wrong
        //Frank new:
        double kau = getCurvature(s);
        double ds = (rc.dot(t) / (-kau * rc.dot(n) - 1.0));
        s = s - ds;
        ++it;
        //std::cout << "Iteration " << it << ", s: " << s << ", old s: " << sOld << std::endl;
        //std::cout << ", rc: " << rc << ", t: " << t << ", n: " << n << std::endl;
        if (s < 0 || s > this->length || it >= 20)
        {
            //std::cout << "Lost position on road " << getId() << ", pos: " << pos << ", sinit: " << sinit << std::endl;
            return Vector2D(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
        }
    } while (fabs(sOld - s) > crit);
    if (s != s)
    {
        return Vector2D(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
    }

    //std::cout << "Iterations: " << it << ", ";
    double v = (t.cross(rc)).dot(Vector3D(0, 0, 1));
    v = v / fabs(v) * rc.length();
    if (v != v)
    {
        v = 0.0;
    }

    return Vector2D(s, v);
}

Vector2D Road::searchPosition(const Vector3D &pos, double sinit)
{
    Vector2D road_pos = searchPositionNoBorder(pos, sinit);
    const double &s = road_pos.u();
    const double &v = road_pos.v();

    double leftWidth, rightWidth;
    getLaneSection(s)->getRoadWidth(s, leftWidth, rightWidth);
    if ((v < 0 && v < -rightWidth) || (v > 0 && v > leftWidth))
    {
        return Vector2D(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
    }

    return road_pos;
}

int Road::searchLane(double s, double v)
{
    return (--laneSectionMap.upper_bound(s))->second->searchLane(s, v);
}

std::set<RoadTransition> Road::getConnectingRoadTransitionSet(const RoadTransition &trans)
{
    TarmacConnection *conn = getConnection(trans.direction);
    Tarmac *tarmac = conn->getConnectingTarmac();

    std::set<RoadTransition> transSet;
    if (Road *road = dynamic_cast<Road *>(tarmac))
    {
        transSet.insert(RoadTransition(road, conn->getConnectingTarmacDirection()));
    }
    else if (Junction *junction = dynamic_cast<Junction *>(tarmac))
    {
        PathConnectionSet connSet = junction->getPathConnectionSet(trans.road);
        for (PathConnectionSet::iterator connSetIt = connSet.begin(); connSetIt != connSet.end(); ++connSetIt)
        {
            transSet.insert(RoadTransition((*connSetIt)->getConnectingPath(), (*connSetIt)->getConnectingPathDirection()));
        }
    }
    else if (Fiddleyard *fiddleyard = dynamic_cast<Fiddleyard *>(tarmac))
    {
        transSet.insert(RoadTransition(fiddleyard->getFiddleroad(), 1));
    }

    return transSet;
}

osg::Geode *Road::createGuardRailGeode(std::map<double, LaneSection *>::iterator lsIt)
{

    guardRailGeode = new osg::Geode();
    guardRailGeode->setName("guardRailGeode");

    double h = 2.5;
    double texlength = 2.5;
    double texwidth = 0.33;
    LaneSection *ls;

    double lsStart;
    double lsEnd = length;
    std::map<double, LaneSection *>::iterator nextLsIt = lsIt;
    ++nextLsIt;

    ls = (*lsIt).second;
    lsStart = ls->getStart();

    if (nextLsIt == laneSectionMap.end())
    {
        lsEnd = length;
    }
    else
    {
        lsEnd = (*(nextLsIt)).second->getStart();
        if (lsEnd > length)
            lsEnd = length;
    }

    osg::Geometry *guardRailGeometry;
    guardRailGeometry = new osg::Geometry();
    guardRailGeode->addDrawable(guardRailGeometry);

    osg::Vec3Array *guardRailVertices;
    guardRailVertices = new osg::Vec3Array;
    guardRailGeometry->setVertexArray(guardRailVertices);

    osg::Vec3Array *guardRailNormals;
    guardRailNormals = new osg::Vec3Array;
    guardRailGeometry->setNormalArray(guardRailNormals);
    guardRailGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::Vec2Array *guardRailTexCoords;
    guardRailTexCoords = new osg::Vec2Array;
    guardRailGeometry->setTexCoordArray(3, guardRailTexCoords);

    int firstVertGuardRail = 0;
    int numVertsguardRail = 0;
    bool firstRound = true;
    bool lastRound = false;
    int i = 0;
    int k = 0;
    double disLeft, disRight;
    //double lastTess = lsStart;

    RoadPoint railPointUp;
    RoadPoint railPointDown;
    RoadPoint nextrailPointUp;
    RoadPoint nextrailPointDown;
    RoadPoint postPointDown;
    RoadPoint postPointUp;
    RoadPoint postPoint;

    osg::Vec3 v1;
    osg::Vec3 v2;
    osg::Vec3 n;
    osg::Vec3 lastN;
    osg::Vec3 averagedN;

    for (double s = lsStart; s < (lsEnd); s += h)
    {
        if (s >= (lsEnd - h))
        {
            s = (lsEnd - h);
            lastRound = true;
        }

        if (k == 0)
            i = -(ls->getNumLanesRight());
        else
            i = ls->getNumLanesLeft();

        if (firstRound)
        {
            getLaneRoadPoints(s, i, railPointUp, railPointDown, disLeft, disRight);
            getGuardRailPoint(railPointDown, railPointUp, firstRound, lastRound, lsIt);

            guardRailVertices->push_back(osg::Vec3(railPointDown.x(), railPointDown.y(), railPointDown.z()));
            guardRailTexCoords->push_back(osg::Vec2((s / texlength), 0));

            guardRailVertices->push_back(osg::Vec3(railPointUp.x(), railPointUp.y(), railPointUp.z()));
            guardRailTexCoords->push_back(osg::Vec2((s / texlength), 0.33 / texwidth));

            numVertsguardRail += 2;
        }
        firstRound = false;

        getLaneRoadPoints(s + h, i, nextrailPointUp, nextrailPointDown, disLeft, disRight);
        //	postPoint = nextrailPointDown;
        postPointDown = nextrailPointDown;
        postPointUp = nextrailPointUp;
        postPoint = railPointDown;
        getGuardRailPoint(nextrailPointDown, nextrailPointUp, firstRound, lastRound, lsIt);

        v1 = osg::Vec3(nextrailPointUp.x() - railPointUp.x(), nextrailPointUp.y() - railPointUp.y(), nextrailPointUp.z() - railPointUp.z());
        v2 = osg::Vec3(railPointDown.x() - railPointUp.x(), railPointDown.y() - railPointUp.y(), railPointDown.z() - railPointUp.z());

        if (k == 0)
            n = osg::Vec3(v1.y() * v2.z() - v1.z() * v2.y(), v1.z() * v2.x() - v1.x() * v2.z(), v1.x() * v2.y() - v1.y() * v2.x());
        else
            n = osg::Vec3(v1.y() * v2.z() - v1.z() * v2.y(), v1.z() * v2.x() - v1.x() * v2.z(), -(v1.x() * v2.y() - v1.y() * v2.x()));

        railPointUp = nextrailPointUp;
        railPointDown = nextrailPointDown;

        guardRailVertices->push_back(osg::Vec3(railPointDown.x(), railPointDown.y(), railPointDown.z()));
        guardRailTexCoords->push_back(osg::Vec2((s / texlength), 0));
        guardRailVertices->push_back(osg::Vec3(railPointUp.x(), railPointUp.y(), railPointUp.z()));
        guardRailTexCoords->push_back(osg::Vec2((s / texlength), 0.33 / texwidth));

        averagedN = n + lastN;

        if (s == lsStart)
        {
            guardRailNormals->push_back(osg::Vec3(n.x(), n.y(), n.z()));
            guardRailNormals->push_back(osg::Vec3(n.x(), n.y(), n.z()));
        }

        else
        {
            guardRailNormals->push_back(osg::Vec3(averagedN.x(), averagedN.y(), averagedN.z()));
            guardRailNormals->push_back(osg::Vec3(averagedN.x(), averagedN.y(), averagedN.z()));
        }

        numVertsguardRail += 2;

        if (lastRound == false)
        {
            /*
 	osg::Box* post = new osg::Box(osg::Vec3(1.22*(nextrailPointDown.x() - postPoint.x())+postPoint.x(),1.22*(nextrailPointDown.y() - postPoint.y())+postPoint.y(),1.22*((nextrailPointDown.z()-0.44) - postPoint.z())+postPoint.z()+0.25),0.2,0.1,0.7);
	osg::Quat quat;
	quat.makeRotate(osg::Vec3(1,0, 0),osg::Vec3(n.x(), n.y(), n.z()));
	
	post->setRotation(quat);	
	
	osg::ShapeDrawable* guardRailDrawable;
	guardRailDrawable = new osg::ShapeDrawable(post);
	guardRailGeode->addDrawable(guardRailDrawable);
*/
            guardRailGeode->addDrawable(getGuardRailPost(postPointDown, postPointUp, postPoint, nextrailPointDown));
        }

        lastN = n;

        if (lastRound)
        {
            if (k == 0)
            {

                osg::DrawArrays *guardRail = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, firstVertGuardRail, numVertsguardRail);
                guardRailGeometry->addPrimitiveSet(guardRail);

                firstVertGuardRail = numVertsguardRail;
                numVertsguardRail = 0;

                lastRound = false;
                firstRound = true;
                //lastTess = lsStart;
                s = lsStart - h;
                k = 1;
            }
            else
                s = lsEnd + h;
        }

        //          lastTess = s;
    }

    osg::DrawArrays *guardRail = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, firstVertGuardRail, numVertsguardRail);
    guardRailGeometry->addPrimitiveSet(guardRail);

    osg::StateSet *guardRailStateSet = guardRailGeode->getOrCreateStateSet();

    guardRailStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    guardRailStateSet->setMode(GL_LIGHT0, osg::StateAttribute::ON);
    //guardRailStateSet->setMode ( GL_LIGHT1, osg::StateAttribute::ON);

    const char *fileName = coVRFileManager::instance()->getName("share/covise/materials/guardRailTex.jpg");
    if (fileName)
    {
        osg::Image *guardRailTexImage = osgDB::readImageFile(fileName);
        osg::Texture2D *guardRailTex = new osg::Texture2D;
        guardRailTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
        guardRailTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
        if (guardRailTexImage)
            guardRailTex->setImage(guardRailTexImage);
        guardRailStateSet->setTextureAttributeAndModes(3, guardRailTex, osg::StateAttribute::ON);
    }
    else
    {
        std::cerr << "ERROR: no texture found named: share/covise/materials/guardRailTex.jpg";
    }

    return guardRailGeode;
}

osg::Geode *Road::getGuardRailGeode()
{
    return guardRailGeode;
}

osg::Geode *Road::getRoadGeode()
{
    if (roadGeode)
    {
        return roadGeode;
    }
    else
    {
        createRoadGroup(true, true);
        return roadGeode;
    }
}

osg::Group *Road::getRoadBatterGroup(bool tessellateBatters, bool tessellateObjects)
{
    //fprintf(stderr,"ytessellateBatters %d\n",tessellateBatters);
    return (roadGroup == NULL) ? createRoadGroup(tessellateBatters, tessellateObjects) : roadGroup;
}

osg::Group *Road::createRoadGroup(bool tessellateBatters, bool tessellateObjects)
{
    //tessellateBatters = tessellateBatters && (!isJunctionPath());

    if (isJunctionPath())
    {
        // calculate road priorities
        junction->setRoadPriorities();
    }

    roadGroup = new osg::Group();
    roadGroup->setName("roadGroup");

    roadGeode = new osg::Geode();
    roadGroup->addChild(roadGeode);
    //fprintf(stderr,"tessellateBatters %d\n",tessellateBatters);
    if (tessellateBatters)
    {
        //fprintf(stderr,"xtessellateBatters %d\n",tessellateBatters);
        std::cout << "Tessellating Batters!" << std::endl;
        batterGeode = new osg::Geode();
        batterGeode->setName("batterGeode");
        roadGroup->addChild(batterGeode);
    }

    //int n = 100;
    //double h = length/n;
    double h = 0.5;
    double texlength = 10.0;
    double texwidth = 10.0;
    double texLengthOffset = 0.0;
    double texWidthOffset = 0.0;
    double battertexlength = 10.0;
    double battertexwidth = 10.0;

    //crgSurface* crg_surface = dynamic_cast<crgSurface*>(roadSurfaceMap.begin()->second);
    //OpenCRGSurface* crg_surface = dynamic_cast<OpenCRGSurface*>(roadSurfaceMap.begin()->second);
    /*if(crg_surface) {
     texlength = crg_surface->getLength();
     texwidth = crg_surface->getWidth();
     texLengthOffset = crg_surface->getLongOffset();
     texWidthOffset = crg_surface->getLatOffset();
     }*/

    std::map<double, LaneSection *>::iterator lsIt;
    LaneSection *ls;
    std::list<RoadPoint> roadEdgeLeft;
    std::list<RoadPoint> roadEdgeRight;

    for (lsIt = laneSectionMap.begin(); lsIt != laneSectionMap.end(); ++lsIt)
    {
        double lsStart;
        double lsEnd = length;
        std::map<double, LaneSection *>::iterator nextLsIt = lsIt;
        ++nextLsIt;
        ls = (*lsIt).second;
        lsStart = ls->getStart();

        if (lsStart > length)
        {
            std::cerr << "Inconsistency: Lane Section defined with start position " << lsStart << " not in road boundaries 0 - " << lsEnd << "!" << std::endl;
            break;
        }
        else if (nextLsIt == laneSectionMap.end())
        {
            lsEnd = length;
        }
        else
        {
            lsEnd = (*(nextLsIt)).second->getStart();
            if (lsEnd > length)
                lsEnd = length;
        }

        osg::Geometry *roadGeometry;
        roadGeometry = new osg::Geometry();
        roadGeode->addDrawable(roadGeometry);

        osg::Vec3Array *roadVertices;
        roadVertices = new osg::Vec3Array;
        roadGeometry->setVertexArray(roadVertices);

        osg::Vec3Array *roadNormals;
        roadNormals = new osg::Vec3Array;
        roadGeometry->setNormalArray(roadNormals);
        roadGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

        //osg::Vec3Array* roadTangents = NULL;
        //osg::Vec3Array* roadBinormals = NULL;
        /*if(crg_surface) {
        roadTangents = new osg::Vec3Array;
        roadGeometry->setVertexAttribArray(6, roadTangents);
        roadGeometry->setVertexAttribBinding(6, osg::Geometry::BIND_PER_VERTEX );

        roadBinormals = new osg::Vec3Array;
        roadGeometry->setVertexAttribArray(7, roadBinormals);
        roadGeometry->setVertexAttribBinding(7, osg::Geometry::BIND_PER_VERTEX );
        }*/

        osg::Vec2Array *roadTexCoords;
        roadTexCoords = new osg::Vec2Array;
        roadGeometry->setTexCoordArray(0, roadTexCoords);
        /*if(crg_surface) {
        roadGeometry->setTexCoordArray(1, roadTexCoords);
        }*/

        osg::Vec4Array *roadMarks;
        roadMarks = new osg::Vec4Array;
        roadGeometry->setVertexAttribArray(5, roadMarks);
        roadGeometry->setVertexAttribBinding(5, osg::Geometry::BIND_PER_VERTEX);

        osg::DrawArrays *roadBase;

        int firstVertLane = 0;
        int numVertsLane = 0;

        for (int i = (ls->getNumLanesLeft()); i >= -(ls->getNumLanesRight()); --i)
        {
            if (i == 0)
                continue;
            bool firstRound = true;
            bool lastRound = false;
            //std::cout << "Lane: " << i << ", lsStart: " << lsStart << ", lsEnd: " << lsEnd << std::endl;

            double lastTess = lsStart;
            double lastWidthOffset = 0;
            double lastWidth = 0;
            for (double s = lsStart; s < (lsEnd + h); s += h)
            {
                if (s >= lsEnd)
                {
                    //std::cout << " Last round: s = " << s << ", lsEnd: " << lsEnd << std::endl;
                    s = lsEnd - 0.001;
                    if (s < 0)
                        s = 0;
                    lastRound = true;
                }
                double width;
                double widthOffset;
                getLaneWidthAndOffset(s, i, width, widthOffset);
                if ((0.01 * h / (s - lastTess) < fabs(getCurvature(s))) || (0.01 * h / (s - lastTess) < fabs(getChordLineElevationCurvature(s))) || lastWidth != width || lastWidthOffset != widthOffset || lastRound || firstRound)
                {
                    firstRound = false;
                    lastWidthOffset = widthOffset;
                    lastWidth = width;

                    RoadPoint pointLeft;
                    RoadPoint pointRight;
                    double disLeft, disRight;
                    RoadMark *markLeft;
                    RoadMark *markRight;

                    getLaneRoadPoints(s, i, pointLeft, pointRight, disLeft, disRight);
                    //std::cout << std::setprecision(15) << pointLeft.x() << " \t" << pointLeft.y() << std::endl;

                    //Vector3D tangent = getTangentVector(s).normalized();
                    //Vector3D binormalLeft = tangent.cross(pointLeft.normal());
                    //Vector3D binormalRight = tangent.cross(pointRight.normal());
                    if (i < 0)
                    {
                        markLeft = ls->getLaneRoadMark(s, i + 1);
                        markRight = ls->getLaneRoadMark(s, i);
                    }
                    else
                    {
                        getLaneRoadPoints(s, i, pointRight, pointLeft, disRight, disLeft);
                        markLeft = ls->getLaneRoadMark(s, i);
                        markRight = ls->getLaneRoadMark(s, i - 1);
                    }
                    RoadMark::RoadMarkType markTypeLeft = markLeft->getType();
                    RoadMark::RoadMarkType markTypeRight = markRight->getType();
                    /*if(this->isJunctionPath()) {
                  markTypeLeft = RoadMark::TYPE_NONE;
                  markTypeRight = RoadMark::TYPE_NONE;
               }*/

                    double width = fabs(disLeft - disRight);
                    //std::cerr << "s: " << s << ", i: " << "Outer mark width: " << markOut->getWidth() << ", inner mark width: " << markIn->getWidth() << std::endl;
                    //std::cerr << "s: " << s << ", i: " << i << ", width: " << width << std::endl;
                    //std::cout << "s: " << s << ", i: " << i << ", left Point: x: " << pointLeft.x() << ", y: " << pointLeft.y() << ", z: " << pointLeft.z() << std::endl;
                    //std::cerr << ", outer Point: x: " << pointOut.x() << ", y: " << pointOut.y() << ", z: " << pointOut.z() << std::endl;
                    //std::cout << "s: " << s << ", i: " << i << ", inner Point: x: " << pointIn.nx() << ", y: " << pointIn.ny() << ", z: " << pointIn.nz();
                    //std::cout << ", outer Point: x: " << pointOut.nx() << ", y: " << pointOut.ny() << ", z: " << pointOut.nz() << std::endl;

                    roadVertices->push_back(osg::Vec3(pointLeft.x(), pointLeft.y(), pointLeft.z()));
                    roadNormals->push_back(osg::Vec3(pointLeft.nx(), pointLeft.ny(), pointLeft.nz()));
                    /*if(crg_surface) {
                 roadTangents->push_back(osg::Vec3(tangent.x(), tangent.y(), tangent.z()));
                 roadBinormals->push_back(osg::Vec3(binormalLeft.x(), binormalLeft.y(), binormalLeft.z()));
                 }*/
                    roadTexCoords->push_back(osg::Vec2((s - texLengthOffset) / texlength, 0.5 + (disLeft - texWidthOffset) / texwidth));

                    roadMarks->push_back(osg::Vec4(-markLeft->getWidth() / width, -markRight->getWidth() / width, -markTypeLeft - (markLeft->getColor() * 1000.0), -markTypeRight - (getPriority() * 100.0) - (markRight->getColor() * 1000.0)));
                    roadVertices->push_back(osg::Vec3(pointRight.x(), pointRight.y(), pointRight.z()));
                    roadNormals->push_back(osg::Vec3(pointRight.nx(), pointRight.ny(), pointRight.nz()));
                    /*if(crg_surface) {
                 roadTangents->push_back(osg::Vec3(tangent.x(), tangent.y(), tangent.z()));
                 roadBinormals->push_back(osg::Vec3(binormalRight.x(), binormalRight.y(), binormalRight.z()));
                 }*/
                    roadTexCoords->push_back(osg::Vec2((s - texLengthOffset) / texlength, 0.5 + (disRight - texWidthOffset) / texwidth));
                    roadMarks->push_back(osg::Vec4(markLeft->getWidth() / width, markRight->getWidth() / width, markTypeLeft + (markLeft->getColor() * 1000.0), markTypeRight + (getPriority() * 100.0) + (markRight->getColor() * 1000.0)));

                    numVertsLane += 2;

                    if (lastRound)
                        s = lsEnd + h;

                    lastTess = s;
                }
            }

            roadBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, firstVertLane, numVertsLane);
            roadGeometry->addPrimitiveSet(roadBase);
            firstVertLane += numVertsLane;
            numVertsLane = 0;
            //reflectorPost

            if ((i == ls->getNumLanesLeft() || i == -(ls->getNumLanesRight())) && getRoadType(ls->getStart()) == 1 && tessellateObjects)
            {

                std::string id, name, type;
                double length, zOffset, validLength, width, radius, height, hdg = 0, pitch = 0, roll = 0;
                RoadObject::OrientationType orientation = RoadObject::BOTH_DIRECTIONS;
                double repeatLength = getLength();
                double repeatDistance = 50.0;
                double t;

                for (double sObj = 0; sObj < repeatLength; sObj += repeatDistance)
                {

                    LaneSection *section = (--laneSectionMap.upper_bound(sObj))->second;
                    double disIn = section->getDistanceToLane(sObj, i);
                    double laneWidth = section->getLaneWidth(sObj, i);
                    if (i < 0)
                        t = (disIn + laneWidth) - 0.5;
                    else
                        t = (disIn + laneWidth) + 0.5;

                    std::string file = "share/covise/materials/reflector_post.3ds";
                    const char *fn = coVRFileManager::instance()->getName(file.c_str());
                    std::string textureFile = "";
                    RoadObject *roadObject = new RoadObject(id, fn ? fn : file, textureFile, name, type, sObj, t, zOffset, validLength, orientation, length, width, radius, height, hdg, pitch, roll, this);
                    addRoadObject(roadObject);
                }
            }
        }

        //guardrail
        if (getRoadType(ls->getStart()) == 2 && tessellateObjects)
        {
            (createGuardRailGeode(lsIt));
        }
    }

    //batter
    if (tessellateBatters)
    {

        for (lsIt = laneSectionMap.begin(); lsIt != laneSectionMap.end(); ++lsIt)
        {
            if (ls->getNumLanesLeft() == 0 && ls->getNumLanesRight() == 0)
                continue;

            std::list<int> batterList;
            batterList.push_back(-ls->getNumLanesRight());
            batterList.push_back(ls->getNumLanesLeft());

            double lsStart;
            double lsEnd = length;
            std::map<double, LaneSection *>::iterator nextLsIt = lsIt;
            ++nextLsIt;
            ls = (*lsIt).second;
            lsStart = ls->getStart();

            if (lsStart > length)
            {
                std::cerr << "Inconsistency: Lane Section defined with start position " << lsStart << " not in road boundaries 0 - " << lsEnd << "!" << std::endl;
                break;
            }
            else if (nextLsIt == laneSectionMap.end())
            {
                lsEnd = length;
            }
            else
            {
                lsEnd = (*(nextLsIt)).second->getStart();
                if (lsEnd > length)
                    lsEnd = length;
            }

            osg::Geometry *firstbatterGeometry;
            firstbatterGeometry = new osg::Geometry();
            batterGeode->addDrawable(firstbatterGeometry);

            osg::Vec3Array *firstbatterVertices;
            firstbatterVertices = new osg::Vec3Array;
            firstbatterGeometry->setVertexArray(firstbatterVertices);

            osg::Vec3Array *firstbatterNormals;
            firstbatterNormals = new osg::Vec3Array;
            firstbatterGeometry->setNormalArray(firstbatterNormals);
            firstbatterGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

            osg::Vec2Array *firstbatterTexCoords;
            firstbatterTexCoords = new osg::Vec2Array;
            firstbatterGeometry->setTexCoordArray(2, firstbatterTexCoords);

            osg::Geometry *secondbatterGeometry;
            secondbatterGeometry = new osg::Geometry();
            batterGeode->addDrawable(secondbatterGeometry);

            osg::Vec3Array *secondbatterVertices;
            secondbatterVertices = new osg::Vec3Array;
            secondbatterGeometry->setVertexArray(secondbatterVertices);

            osg::Vec3Array *secondbatterNormals;
            secondbatterNormals = new osg::Vec3Array;
            secondbatterGeometry->setNormalArray(secondbatterNormals);
            secondbatterGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

            osg::Vec2Array *secondbatterTexCoords;
            secondbatterTexCoords = new osg::Vec2Array;
            secondbatterGeometry->setTexCoordArray(2, secondbatterTexCoords);

            int firstVertBatter = 0;
            int numVertsBatter = 0;

            for (std::list<int>::iterator batterIt = batterList.begin(); batterIt != batterList.end(); ++batterIt)
            {

                int i = (*batterIt);
                if (ls->getBatterShouldBeTessellated(i) == false)
                    continue;

                bool firstRound = true;
                bool lastRound = false;
                //std::cout << "Lane: " << i << ", lsStart: " << lsStart << ", lsEnd: " << lsEnd << std::endl;

                double lastTess = lsStart;
                for (double s = lsStart; s < (lsEnd + h); s += h)
                {

                    if (s >= lsEnd)
                    {
                        //std::cout << " Last round: s = " << s << ", lsEnd: " << lsEnd << std::endl;
                        s = lsEnd - 0.001;
                        lastRound = true;
                    }

                    if ((0.01 * h / (s - lastTess) < fabs(getCurvature(s))) || (0.01 * h / (s - lastTess) < fabs(getChordLineElevationCurvature(s))) || lastRound || firstRound)
                    {
                        firstRound = false;

                        //batter

                        RoadPoint pointCenter;
                        double width;

                        //double disRight, disLeft;
                        RoadPoint pointLeft;
                        RoadPoint pointLeftDummy;
                        RoadPoint pointRight;
                        RoadPoint pointRightDummy;

                        if (i > 0 || (i == 0 && ls->getNumLanesLeft() == 0))
                        {
                            getBatterPoint(s, i, pointLeft, pointCenter, pointRight, width, 1);
                            firstbatterTexCoords->push_back(osg::Vec2((s / battertexlength), (1 / battertexwidth)));
                            firstbatterTexCoords->push_back(osg::Vec2((s / battertexlength), 0));
                            secondbatterTexCoords->push_back(osg::Vec2((s / battertexlength), (width / battertexwidth)));
                            secondbatterTexCoords->push_back(osg::Vec2((s / battertexlength), 0));
                        }
                        else if (i < 0 || (i == 0 && ls->getNumLanesRight() == 0))
                        {
                            getBatterPoint(s, i, pointRight, pointCenter, pointLeft, width, 2);
                            firstbatterTexCoords->push_back(osg::Vec2((s / battertexlength), (width / battertexwidth)));
                            firstbatterTexCoords->push_back(osg::Vec2((s / battertexlength), 0));
                            secondbatterTexCoords->push_back(osg::Vec2((s / battertexlength), (1 / battertexwidth)));
                            secondbatterTexCoords->push_back(osg::Vec2((s / battertexlength), 0));
                        }

                        firstbatterVertices->push_back(osg::Vec3(pointCenter.x(), pointCenter.y(), pointCenter.z()));
                        firstbatterNormals->push_back(osg::Vec3(pointCenter.nx(), pointCenter.ny(), pointCenter.nz()));

                        firstbatterVertices->push_back(osg::Vec3(pointRight.x(), pointRight.y(), pointRight.z()));
                        firstbatterNormals->push_back(osg::Vec3(pointRight.nx(), pointRight.ny(), pointRight.nz()));

                        secondbatterVertices->push_back(osg::Vec3(pointLeft.x(), pointLeft.y(), pointLeft.z()));
                        secondbatterNormals->push_back(osg::Vec3(pointLeft.nx(), pointLeft.ny(), pointLeft.nz()));

                        secondbatterVertices->push_back(osg::Vec3(pointCenter.x(), pointCenter.y(), pointCenter.z()));
                        secondbatterNormals->push_back(osg::Vec3(pointCenter.nx(), pointCenter.ny(), pointCenter.nz()));

                        numVertsBatter += 2;

                        if (lastRound)
                            s = lsEnd + h;

                        lastTess = s;
                    }
                }

                if (numVertsBatter > 0)
                {
                    osg::DrawArrays *firstbatterBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, firstVertBatter, numVertsBatter);
                    firstbatterGeometry->addPrimitiveSet(firstbatterBase);
                    osg::DrawArrays *secondbatterBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, firstVertBatter, numVertsBatter);
                    secondbatterGeometry->addPrimitiveSet(secondbatterBase);
                    std::cout << "Batter: adding " << numVertsBatter << " vertices!" << std::endl;

                    firstVertBatter += numVertsBatter;
                }
                numVertsBatter = 0;
            }
        }
    }

    if (!roadStateSet)
    {
        roadStateSet = roadGeode->getOrCreateStateSet();
        //osg::StateSet* roadStateSet = roadGeode->getOrCreateStateSet();
        //osg::PolygonMode* polygonMode = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
        //roadStateSet->setAttributeAndModes(polygonMode);
        roadStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
        roadStateSet->setMode(GL_LIGHT0, osg::StateAttribute::ON);

        if (!roadTex)
        {
            const char *fileName = coVRFileManager::instance()->getName("share/covise/materials/roadTex.jpg");
            if (fileName)
            {
                osg::Image *roadTexImage = osgDB::readImageFile(fileName);
                roadTex = new osg::Texture2D;
                roadTex->ref();
                roadTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
                roadTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
                if (roadTexImage)
                    roadTex->setImage(roadTexImage);
                roadStateSet->setTextureAttributeAndModes(0, roadTex);
            }
            else
            {
                std::cerr << "ERROR: no texture found named: share/covise/materials/roadTex.jpg";
            }
        }
        else
        {
            roadStateSet->setTextureAttributeAndModes(0, roadTex);
        }

        coVRShader *shader;
        //if(!crg_surface)
        shader = coVRShaderList::instance()->get("roadMark");

        if (shader == NULL)
        {
            std::cerr << "ERROR: no shader found with name: roadMark/parallaxMapping" << std::endl;
        }
        if (shader)
        {
            shader->apply(roadStateSet);
        }
    }
    else
    {
        roadGeode->setStateSet(roadStateSet);
    }
    /*else {
     shader=coVRShaderList::instance()->get("parallaxMapping");

     osg::Texture::WrapMode wrapModeU;
     osg::Texture::WrapMode wrapModeV;
     switch(crg_surface->getSurfaceWrapModeU()) {
     case OpenCRGSurface::NONE:
     wrapModeU = osg::Texture::CLAMP;
     break;
     case OpenCRGSurface::EX_ZERO:
     wrapModeU = osg::Texture::CLAMP;
     break;
     case OpenCRGSurface::EX_KEEP:
     wrapModeU = osg::Texture::CLAMP;
     break;
     case OpenCRGSurface::REPEAT:
     wrapModeU = osg::Texture::REPEAT;
     break;
     case OpenCRGSurface::REFLECT:
     wrapModeU = osg::Texture::MIRROR;
     break;
     default:
     wrapModeU = osg::Texture::CLAMP;
     };
     switch(crg_surface->getSurfaceWrapModeV()) {
     case OpenCRGSurface::NONE:
     wrapModeV = osg::Texture::CLAMP;
     break;
     case OpenCRGSurface::EX_ZERO:
     wrapModeV = osg::Texture::CLAMP;
     break;
     case OpenCRGSurface::EX_KEEP:
     wrapModeV = osg::Texture::CLAMP;
     break;
     case OpenCRGSurface::REPEAT:
     wrapModeV = osg::Texture::REPEAT;
     break;
     case OpenCRGSurface::REFLECT:
     wrapModeV = osg::Texture::MIRROR;
     break;
     default:
     wrapModeV = osg::Texture::CLAMP;
     };

     osg::Texture2D* pavementTexture = new osg::Texture2D;
     pavementTexture->setWrap(osg::Texture2D::WRAP_S, wrapModeU);
     pavementTexture->setWrap(osg::Texture2D::WRAP_T, wrapModeV);
     pavementTexture->setImage(crg_surface->getPavementTextureImage());
     roadStateSet->setTextureAttributeAndModes(0, pavementTexture);

     osg::Texture2D* parallaxMapTexture = new osg::Texture2D;
     parallaxMapTexture->setWrap(osg::Texture2D::WRAP_S, wrapModeU);
     parallaxMapTexture->setWrap(osg::Texture2D::WRAP_T, wrapModeV);
     parallaxMapTexture->setImage(crg_surface->getParallaxMap());
     roadStateSet->setTextureAttributeAndModes(1, parallaxMapTexture);

     osg::Uniform* shaderUniformScale = new osg::Uniform("scale",0.0f);
     osg::Uniform* shaderUniformBias = new osg::Uniform("bias",0.0f);
     roadStateSet->addUniform(shaderUniformScale);
     roadStateSet->addUniform(shaderUniformBias);
     }*/
    if (tessellateBatters)
    {

        if (getRoadType(0.0) == BRIDGE)
        {
            if (!concreteBatterStateSet)
            {
                concreteBatterStateSet = batterGeode->getOrCreateStateSet();

                concreteBatterStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
                concreteBatterStateSet->setMode(GL_LIGHT0, osg::StateAttribute::ON);
                //concreteBatterStateSet->setMode ( GL_LIGHT1, osg::StateAttribute::ON);

                osg::PolygonOffset *polOffset = new osg::PolygonOffset(1.0, 1.0);
                concreteBatterStateSet->setAttributeAndModes(polOffset);

                if (!concreteTex)
                {
                    const char *fileName = coVRFileManager::instance()->getName("share/covise/materials/concrete_texture.jpg");
                    if (fileName)
                    {
                        osg::Image *concreteTexImage = osgDB::readImageFile(fileName);
                        concreteTex = new osg::Texture2D;
                        concreteTex->ref();
                        concreteTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
                        concreteTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
                        if (concreteTexImage)
                        {
                            concreteTex->setImage(concreteTexImage);
                            std::cerr << "Setting concrete batter StateSet texture image..." << std::endl;
                        }
                        concreteBatterStateSet->setTextureAttributeAndModes(2, concreteTex);
                    }
                    else
                    {
                        std::cerr << "ERROR: no texture found named: share/covise/materials/concrete_texture.jpg";
                    }
                }
            }
            else
            {
                batterGeode->setStateSet(concreteBatterStateSet);
            }
        }
        else
        {
            if (!batterStateSet)
            {
                batterStateSet = batterGeode->getOrCreateStateSet();

                batterStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
                batterStateSet->setMode(GL_LIGHT0, osg::StateAttribute::ON);
                //batterStateSet->setMode ( GL_LIGHT1, osg::StateAttribute::ON);

                osg::PolygonOffset *polOffset = new osg::PolygonOffset(2.0, 2.0);
                batterStateSet->setAttributeAndModes(polOffset);

                if (!batterTex)
                {
                    const char *fileName = coVRFileManager::instance()->getName("share/covise/materials/batter_texture.jpg");
                    if (fileName)
                    {
                        osg::Image *batterTexImage = osgDB::readImageFile(fileName);
                        batterTex = new osg::Texture2D;
                        batterTex->ref();
                        batterTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
                        batterTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
                        if (batterTexImage)
                        {
                            batterTex->setImage(batterTexImage);
                            std::cerr << "Setting batter StateSet texture image..." << std::endl;
                        }
                        batterStateSet->setTextureAttributeAndModes(2, batterTex);
                    }
                    else
                    {
                        std::cerr << "ERROR: no texture found named: share/covise/materials/grass_texture.jpg";
                    }
                }
            }
            else
            {
                batterGeode->setStateSet(batterStateSet);
            }
        }
    }

    roadGeode->setName(std::string("ROAD_") + this->id);

    /*bumpGeode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
     osg::StateSet* bumpGeodeState = bumpGeode->getOrCreateStateSet();
     if(texFile) {
     bumpGeodeState->setTextureAttributeAndModes(0, surfaceTex);
     }
     osg::DrawArrays* bumpBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, 0, 4);
     bumpGeometry->addPrimitiveSet(bumpBase);
     double bumpLength = surface->getNumberOfSurfaceDataLines() * reference_line_increment;
     double bumpWidth = surface->getNumberOfHeightDataElements() * long_section_v_increment;
     bumpVertices->push_back(osg::Vec3(0.0, 0.0, 0.0));
     bumpTexCoords->push_back(osg::Vec2(0.0, 0.0));
     bumpVertices->push_back(osg::Vec3(0.0, bumpWidth, 0.0));
     bumpTexCoords->push_back(osg::Vec2(0.0, 1.0));
     bumpVertices->push_back(osg::Vec3(bumpLength, 0.0, 0.0));
     bumpTexCoords->push_back(osg::Vec2(1.0, 0.0));
     bumpVertices->push_back(osg::Vec3(bumpLength, bumpWidth, 0.0));
     bumpTexCoords->push_back(osg::Vec2(1.0, 1.0));
     bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
     bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
     bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
     bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
     bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
     bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
     bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
     bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
     bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));
     bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));
     bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));
     bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));

     osg::Texture2D* parallaxMapTexture = new osg::Texture2D;
     parallaxMapTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
     parallaxMapTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
     parallaxMapTexture->setImage(surface->createParallaxMapTextureImage());
     bumpGeodeState->setTextureAttributeAndModes(1, parallaxMapTexture);

     coVRShader *parallaxShader=coVRShaderList::instance()->get("parallaxMapping");
     if(parallaxShader==NULL)
     {
     cerr << "ERROR: no shader found with name: parallaxMapping"<< endl;
     }
     else {
     parallaxShader->apply(bumpGeode);
     }*/

    return roadGroup;
}

osg::Group *Road::createObjectsGroup()
{
    osg::Group *objectsGroup = new osg::Group();
    objectsGroup->setName("objects_road_" + getId());
    for (std::vector<RoadObject *>::iterator objIt = roadObjectVector.begin(); objIt != roadObjectVector.end(); ++objIt)
    {
        RoadObject *obj = *objIt;
        osg::Node *objectNode = obj->getObjectNode();
        if (obj->isAbsolute())
        {
            objectsGroup->addChild(objectNode);
        }
        else
        {
            float ah = 0;
            if (obj->getT() < 0)
            {
                ah = M_PI;
            }
            osg::Quat objectAttitude(obj->getHdg() + ah, osg::Vec3(0.0, 0.0, 1.0),
                                     obj->getPitch(), osg::Vec3(0.0, 1.0, 0.0),
                                     obj->getRoll(), osg::Vec3(1.0, 0.0, 0.0));
            if (obj->isObjectRepeating())
            {
                osg::Group *objectRepeatGroup = new osg::Group();

                objectRepeatGroup->setName("object_repeat_" + obj->getId());
                objectsGroup->addChild(objectRepeatGroup);
                float dist = obj->getRepeatDistance();
                if (dist <= 0)
                    dist = 8;

                double wr, wl;
                double startWidth;
                getRoadSideWidths(obj->getS(), wr, wl);
                if (obj->getT() > 0)
                    startWidth = wl;
                else
                    startWidth = wr;
                for (double s = obj->getS(); s < (obj->getS() + obj->getRepeatLength()); s += dist)
                {

                    double currentT;
                    getRoadSideWidths(s, wr, wl);
                    if (obj->getT() > 0)
                        currentT = obj->getT() - startWidth + wl;
                    else
                        currentT = obj->getT() - startWidth + wr;
                    osg::PositionAttitudeTransform *objectTransform = new osg::PositionAttitudeTransform();
                    objectTransform->addChild(objectNode);
                    objectRepeatGroup->addChild(objectTransform);
                    Transform objectT = getRoadTransform(s, currentT);
                    objectTransform->setPosition(osg::Vec3(objectT.v().x(), objectT.v().y(), objectT.v().z() + obj->getZOffset()));
                    objectTransform->setAttitude(objectAttitude * osg::Quat(objectT.q().x(), objectT.q().y(), objectT.q().z(), objectT.q().w()) * obj->getOrientation());
                }
            }
            else
            {
                osg::PositionAttitudeTransform *objectTransform = new osg::PositionAttitudeTransform();
                objectTransform->addChild(objectNode);
                objectsGroup->addChild(objectTransform);

                Transform objectT = getRoadTransform(obj->getS(), obj->getT());
                objectTransform->setPosition(osg::Vec3(objectT.v().x(), objectT.v().y(), objectT.v().z() + obj->getZOffset()));
                objectTransform->setAttitude(objectAttitude * osg::Quat(objectT.q().x(), objectT.q().y(), objectT.q().z(), objectT.q().w()) * obj->getOrientation());
            }
        }
    }

    return objectsGroup;
}

void Road::accept(RoadSystemVisitor *visitor)
{
    visitor->visit(this);
}

bool Road::compare(Road *r1, Road *r2)
{
    return r1->getLength() < r2->getLength();
}
