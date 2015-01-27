/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Junction_h
#define Junction_h

#include <set>
#include <map>
#include <string>

#include "Tarmac.h"
#include "Road.h"

#include <osg/Geode>

class PathConnection;

struct PathConnectionCompare;

typedef std::set<PathConnection *, PathConnectionCompare> PathConnectionSet;

class Junction : public Tarmac
{
public:
    Junction(std::string, std::string = "no name");

    void addPathConnection(PathConnection *);
    PathConnectionSet getPathConnectionSet(Road *);

    Road *getIncomingRoad(int);
    double getNumIncomingRoads();

    osg::Geode *getJunctionGeode();

protected:
    std::map<Road *, PathConnectionSet> pathConnectionSetMap;
    std::map<std::string, PathConnection *> pathConnectionIdMap;
};

typedef std::map<int, int> LaneConnectionMap;

class PathConnection : public Element
{
public:
    PathConnection(std::string, Road *, Road *, int = 1);

    std::string getId();

    void addLaneLink(int, int);

    Road *getIncomingRoad();
    Road *getConnectingPath();
    int getConnectingPathDirection();
    int getConnectingLane(int);

    bool operator<(const PathConnection *) const;

    double getAngleDifference() const;

    LaneConnectionMap getLaneConnectionMap();

private:
    LaneConnectionMap laneConnectionMap;

    Road *incomingRoad;
    Road *connectingPath;
    int connectingPathDirection;

    double angle;
};

struct PathConnectionCompare
{
    bool operator()(const PathConnection *conna, const PathConnection *connb) const
    {
        return conna->getAngleDifference() < connb->getAngleDifference();
    }
};

#endif
