/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PedestrianManager_h
#define PedestrianManager_h

#include "RoadSystem/Road.h"
#include "RoadSystem/RoadSystem.h"
#include <cover/coVRPluginSupport.h>

class Pedestrian;
class TRAFFICSIMULATIONEXPORT PedestrianManager
{
public:
    static PedestrianManager *Instance();
    static void Destroy();

    void generateAutoFiddleyards();
    void generateMovingFiddleyards();
    void enableMovingFiddleyards();

    void setSpawnRange(const double r)
    {
        spawnRange = r;
        spawnRangeSq = r * r;
    }

    void setReportInterval(const double i);
    void setAvoidCount(const int c);
    void setAvoidTime(const double l);

    void scheduleRemoval(Pedestrian *p);

    void addPedestrian(Pedestrian *p);
    void removePedestrian(Pedestrian *p, Road *r = NULL, const int l = 0);

    void addToRoad(Pedestrian *p, Road *r, const int l);
    void removeFromRoad(Pedestrian *p, Road *r, const int l);
    void changeRoad(Pedestrian *p, Road *fromR, const int fromL, Road *toR, const int toL);

    int numPeds() const;

    void moveAllPedestrians(const double dt);
    void updateFiddleyards(const double dt);

protected:
    PedestrianManager();
    static PedestrianManager *__instance;

private:
    RoadSystem *roadSystem;
    std::map<std::pair<Road *, int>, std::vector<Pedestrian *> > pedestrianListMap;
    std::vector<Pedestrian *> pedestrianOverallList;
    std::vector<Pedestrian *> pedestrianRemovalList;

    void performRemovals();

    std::vector<Pedestrian *> getClosestNeighbors(Pedestrian *p, int n);
    void performAvoidance(Pedestrian *p, const double time);
    double collisionIn(Pedestrian *a, Pedestrian *b) const;

    double squaredDistTo(double x, double y, double z);

    double spawnRange;
    double spawnRangeSq;

    double reportTimer;
    double repeatTime;
    double nextTime;
    int avoidCount;
    double avoidTime;

    bool autoFiddleyards;
    bool movingFiddleyards;
    osg::Vec3d viewPosLastUpdated;
};

#endif
