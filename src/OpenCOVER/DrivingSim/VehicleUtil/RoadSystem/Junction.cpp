/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Junction.h"

#include <osgUtil/Optimizer>
#include <osgUtil/Tessellator>

bool PathConnectionCompare::operator()(const PathConnection *conna, const PathConnection *connb) const
{
    return conna->getAngleDifference() < connb->getAngleDifference();
}

PathConnectionSet::PathConnectionSet()
    : std::multiset<PathConnection *, PathConnectionCompare>()
    , connectionFrequencySum(0.0)
{
}

PathConnectionSet::iterator PathConnectionSet::insert(PathConnection *&conn)
{
    connectionFrequencySum += conn->getFrequency();
    frequencySumMap.insert(std::pair<double, PathConnection *>(connectionFrequencySum, conn));

    return std::multiset<PathConnection *, PathConnectionCompare>::insert(conn);
}

void PathConnectionSet::remove(PathConnectionSet::iterator it)
{
    PathConnection *p = *it;
    erase(it);
    connectionFrequencySum -= p->getFrequency();
    for(std::map<double, PathConnection *>::iterator fit = frequencySumMap.begin();fit != frequencySumMap.end();fit++)
    {
        if(fit->second == p)
        {
            frequencySumMap.erase(fit);
            break;
        }
    }
}
PathConnection *PathConnectionSet::choosePathConnection(double random)
{
    if(frequencySumMap.size()==0)
        return NULL;
    /*int path = rand() % this->size();
   PathConnectionSet::iterator connSetIt = this->begin();
   std::advance(connSetIt, path);
   return (*connSetIt);*/
    //double targetFreq = ((double)rand())/RAND_MAX * connectionFrequencySum;
    double targetFreq = random * connectionFrequencySum;

    std::map<double, PathConnection *>::iterator mapIt = frequencySumMap.lower_bound(targetFreq);
    return mapIt->second;
}

double PathConnectionSet::getConnectionFrequencySum()
{
    return connectionFrequencySum;
}

Junction::Junction(std::string id, std::string name)
    : Tarmac(id, name)
    , activeController(0)
    , toggleJunctionControllerTime(10.0)
    , timer(toggleJunctionControllerTime)
{
}

void Junction::addPathConnection(PathConnection *conn)
{
    (pathConnectionSetMap[conn->getIncomingRoad()]).insert(conn);
}

PathConnectionSet Junction::getPathConnectionSet(Road *incoming)
{
    return (*(pathConnectionSetMap.find(incoming))).second;
}

PathConnectionSet Junction::getPathConnectionSet(Road *incoming, int incomingLane) // only return connections to the current lane
{
    PathConnectionSet ps = (*(pathConnectionSetMap.find(incoming))).second;
    PathConnectionSet::iterator oldit;
    for(PathConnectionSet::iterator it=ps.begin();it != ps.end();)
    {
        PathConnection *p = *it;
        oldit = it++;
        if(p->getConnectingLane(incomingLane,false)==1000)
        {
            
            ps.remove(oldit);
        }
    }
    return ps;
}

Road *Junction::getIncomingRoad(int num)
{
    std::map<Road *, PathConnectionSet>::iterator it = pathConnectionSetMap.begin();
    for (int i = 0; i < num; ++i)
        ++it;
    return it->first;
}

double Junction::getNumIncomingRoads()
{
    return pathConnectionSetMap.size();
}

int Junction::getConnectingLane(Road *inRoad, Road *connPath, int lane)
{
    std::map<Road *, PathConnectionSet>::iterator mapIt = pathConnectionSetMap.find(inRoad);
    if (mapIt != pathConnectionSetMap.end())
    {
        for (PathConnectionSet::iterator setIt = mapIt->second.begin(); setIt != mapIt->second.end(); ++setIt)
        {
            if ((*setIt)->getConnectingPath() == connPath)
            {
                return (*setIt)->getConnectingLane(lane);
            }
        }
    }
    return Lane::NOLANE;
}

std::string Junction::getTypeSpecifier()
{
    return std::string("junction");
}

const std::map<Road *, PathConnectionSet> &Junction::getPathConnectionSetMap() const
{
    return pathConnectionSetMap;
}

void Junction::setRoadPriorities()
{
    for (std::map<Road *, PathConnectionSet>::iterator mapIt = pathConnectionSetMap.begin(); mapIt != pathConnectionSetMap.end(); ++mapIt)
    {
        PathConnectionSet &connSet = mapIt->second;
        for (PathConnectionSet::iterator setIt = connSet.begin(); setIt != connSet.end(); ++setIt)
        {
            (*setIt)->getConnectingPath()->setPriority(-1);
        }
    }
    int numPaths = 0;
    for (std::map<Road *, PathConnectionSet>::iterator mapIt = pathConnectionSetMap.begin(); mapIt != pathConnectionSetMap.end(); ++mapIt)
    {
        PathConnectionSet &connSet = mapIt->second;
        for (PathConnectionSet::iterator setIt = connSet.begin(); setIt != connSet.end(); ++setIt)
        {

            if ((*setIt)->getConnectingPath()->getPriority() == -1)
            {
                numPaths++;
                (*setIt)->getConnectingPath()->setPriority(numPaths);
            }
        }
    }
}

osg::Geode *Junction::getJunctionGeode()
{
    //std::cerr << "Junction tesselation not implemented yet..." << std::endl;
    //return NULL;

    osg::Group *junctionPaths = new osg::Group;

    for (std::map<Road *, PathConnectionSet>::iterator mapIt = pathConnectionSetMap.begin(); mapIt != pathConnectionSetMap.end(); ++mapIt)
    {
        PathConnectionSet &connSet = mapIt->second;
        for (PathConnectionSet::iterator setIt = connSet.begin(); setIt != connSet.end(); ++setIt)
        {
            junctionPaths->addChild((*setIt)->getConnectingPath()->getRoadGeode());
        }
    }

    osgUtil::Optimizer::MergeGeodesVisitor mergeGeodeVisit;
    mergeGeodeVisit.mergeGeodes(*junctionPaths);

    if (junctionPaths->getNumChildren() > 0)
    {
        osg::Geode *pathMergeGeode = junctionPaths->getChild(0)->asGeode();
        if (pathMergeGeode)
        {
            osgUtil::Optimizer::MergeGeometryVisitor mergeGeometryVisitor;
            mergeGeometryVisitor.mergeGeode(*pathMergeGeode);
            if (pathMergeGeode->getNumDrawables() > 0)
            {
                osg::Geometry *pathGeometry = pathMergeGeode->getDrawable(0)->asGeometry();
                if (pathGeometry)
                {
                    osgUtil::Tessellator pathTess;
                    pathTess.setTessellationType(osgUtil::Tessellator::TESS_TYPE_GEOMETRY);
                    //pathTess.setBoundaryOnly(true);
                    pathTess.setWindingType(osgUtil::Tessellator::TESS_WINDING_NEGATIVE);
                    pathTess.setTessellationNormal(osg::Vec3d(0.0, 0.0, 1.0));
                    std::cout << "Retessellating junction paths!" << std::endl;
                    pathTess.retessellatePolygons(*pathGeometry);
                    return pathMergeGeode;
                }
            }
        }
    }

    return NULL;
}

void Junction::accept(RoadSystemVisitor *visitor)
{
    visitor->visit(this);
}

void Junction::addJunctionController(Controller *controller, const std::string &controlType)
{
    junctionControllerVector.push_back(make_pair(controller, controlType));
}

void Junction::update(const double & /* dt*/)
{
    /*timer += dt; 
   if(junctionControllerVector.size()>0) {
      if(timer >= toggleJunctionControllerTime) {
         junctionControllerVector[activeController].first->allSignalsStop();
         timer = 0.0;
         ++activeController;
         if(activeController>=junctionControllerVector.size()) {
            activeController = 0;
         }
         junctionControllerVector[activeController].first->allSignalsGo();
      }
   }*/
}

PathConnection::PathConnection(std::string setId, Road *inRoad, Road *connPath, int dir, double f)
    //   :  laneConnectionMap(PathConnection::compare)
    : Element(setId)
{
    incomingRoad = inRoad;
    connectingPath = connPath;
    connectingPathDirection = dir;
    frequency = f;

    double inHead = connPath->getHeading(0.0);
    double outHead = connPath->getHeading(connPath->getLength());

    double cross = (cos(inHead) * sin(outHead) - cos(outHead) * sin(inHead));
    double scalar = (cos(inHead) * cos(outHead) + sin(inHead) * sin(outHead));

    angle = cross / fabs(cross) * acos(scalar);
    if (angle != angle)
        angle = 0;

    // indicators //
    //
    if (angle > 1.5708 / 3.0)
        connectingPathIndicator = 1;
    else if (angle < -1.5708 / 3.0)
        connectingPathIndicator = -1;
    else
        connectingPathIndicator = 0;

    //std::cout << "PathConnection: " << id << ", inHeading: " << inHead << ", outHeading: " << outHead << ", angle: " << angle << std::endl;
}

std::string PathConnection::getId()
{
    return id;
}

void PathConnection::addLaneLink(int from, int to)
{
    laneConnectionMap[from] = to;
}

Road *PathConnection::getIncomingRoad()
{
    return incomingRoad;
}

Road *PathConnection::getConnectingPath()
{
    return connectingPath;
}

int PathConnection::getConnectingPathDirection()
{
    return connectingPathDirection;
}

int PathConnection::getConnectingPathIndicator()
{
    return connectingPathIndicator;
}

double PathConnection::getAngleDifference() const
{
    return angle;
}

int PathConnection::getConnectingLane(int from, bool defaults)
{
    std::map<int, int>::iterator it = laneConnectionMap.find(from);
    if (it == laneConnectionMap.end())
    {
        if(defaults)
            return from;
        else
            return 1000;
    }
    else
    {
        return (*it).second;
    }
}

double PathConnection::getFrequency()
{
    return frequency;
}

bool PathConnection::operator<(const PathConnection *conn) const
{
    //std::cout << "Angle: " << angle << " < other angle: " << conn->getAngleDifference() << " = " << (angle < conn->getAngleDifference()) << std::endl;
    return angle < conn->getAngleDifference();
}

LaneConnectionMap PathConnection::getLaneConnectionMap()
{
    return laneConnectionMap;
}

void PathConnection::accept(RoadSystemVisitor *visitor)
{
    visitor->visit(this);
}
