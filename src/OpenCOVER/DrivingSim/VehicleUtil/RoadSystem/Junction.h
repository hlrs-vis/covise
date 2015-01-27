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
#include "Controller.h"

#include <osg/Geode>
#include <util/coExport.h>

class PathConnection;

struct PathConnectionCompare
{
    bool operator()(const PathConnection *, const PathConnection *) const;
};

//typedef std::multiset<PathConnection*, PathConnectionCompare> PathConnectionSet;
class VEHICLEUTILEXPORT PathConnectionSet : public std::multiset<PathConnection *, PathConnectionCompare>
{
public:
    PathConnectionSet();

    PathConnectionSet::iterator insert(PathConnection *&);

    double getConnectionFrequencySum();

    PathConnection *choosePathConnection(double);

protected:
    std::map<double, PathConnection *> frequencySumMap;

    double connectionFrequencySum;
};

class VEHICLEUTILEXPORT Junction : public Tarmac
{
public:
    Junction(std::string, std::string = "no name");

    void addPathConnection(PathConnection *);
    PathConnectionSet getPathConnectionSet(Road *);

    Road *getIncomingRoad(int);
    double getNumIncomingRoads();

    int getConnectingLane(Road *, Road *, int);

    std::string getTypeSpecifier();

    const std::map<Road *, PathConnectionSet> &getPathConnectionSetMap() const;

    osg::Geode *getJunctionGeode();

    void accept(RoadSystemVisitor *);

    void addJunctionController(Controller *, const std::string &);

    void update(const double &);

    void setRoadPriorities();

protected:
    std::map<Road *, PathConnectionSet> pathConnectionSetMap;

    std::vector<std::pair<Controller *, std::string> > junctionControllerVector;
    unsigned int activeController;
    double toggleJunctionControllerTime;
    double timer;
};

typedef std::map<int, int> LaneConnectionMap;

class VEHICLEUTILEXPORT PathConnection : public Element
{
public:
    PathConnection(std::string, Road *, Road *, int = 1, double = 1);

    std::string getId();

    void addLaneLink(int, int);

    Road *getIncomingRoad();
    Road *getConnectingPath();
    int getConnectingPathDirection();
    int getConnectingPathIndicator();
    int getConnectingLane(int);
    double getFrequency();

    bool operator<(const PathConnection *) const;

    double getAngleDifference() const;

    LaneConnectionMap getLaneConnectionMap();

    void accept(RoadSystemVisitor *);

private:
    LaneConnectionMap laneConnectionMap;

    Road *incomingRoad;
    Road *connectingPath;
    int connectingPathDirection;
    int connectingPathIndicator;

    double angle;
    double frequency;
};

#endif
