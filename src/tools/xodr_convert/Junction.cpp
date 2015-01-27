/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Junction.h"

Junction::Junction(std::string id, std::string name)
    : Tarmac(id, name)
{
}

void Junction::addPathConnection(PathConnection *conn)
{
    pathConnectionIdMap[conn->getId()] = conn;

    (pathConnectionSetMap[conn->getIncomingRoad()]).insert(conn);
}

PathConnectionSet Junction::getPathConnectionSet(Road *incoming)
{
    return (*(pathConnectionSetMap.find(incoming))).second;
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

osg::Geode *Junction::getJunctionGeode()
{
    std::cerr << "Junction tesselation not implemented yet..." << std::endl;
    return NULL;
}

PathConnection::PathConnection(std::string setId, Road *inRoad, Road *connPath, int dir)
    //   :  laneConnectionMap(PathConnection::compare)
    : Element(setId)
{
    incomingRoad = inRoad;
    connectingPath = connPath;
    connectingPathDirection = dir;

    double inHead = connPath->getHeading(0.0);
    double outHead = connPath->getHeading(connPath->getLength());

    double cross = (cos(inHead) * sin(outHead) - cos(outHead) * sin(inHead));
    double scalar = (cos(inHead) * cos(outHead) + sin(inHead) * sin(outHead));

    angle = cross / fabs(cross) * acos(scalar);
    if (angle != angle)
        angle = 0;

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

double PathConnection::getAngleDifference() const
{
    return angle;
}

int PathConnection::getConnectingLane(int from)
{
    std::map<int, int>::iterator it = laneConnectionMap.find(from);
    if (it == laneConnectionMap.end())
    {
        return from;
    }
    else
    {
        return (*it).second;
    }
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
