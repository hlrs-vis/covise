#ifndef REFERENCEPOSITION_H
#define REFERENCEPOSITION_H

#include<string>
#include<vector>
#include <osg/Vec3>
#include <math.h>
#include <iostream>
#include <Position.h>
#include "Entity.h"
#include <VehicleUtil/RoadSystem/RoadSystem.h>
#include <VehicleUtil/RoadSystem/Road.h>

class ReferencePosition
{

public:
    ReferencePosition();
    ReferencePosition(const ReferencePosition *oldRefPos); // copy constructor
    ~ReferencePosition();

    // Road
    std::string roadId;
    Road* road;
    double roadLength;
    RoadSystem* system;

    double s;
    double t;
    double hdg;

    // Lane
    int laneId;
    LaneSection *LS;

    // Absolute
    osg::Vec3 xyz;

    // intialize ReferencePosition:
    void init(std::string init_roadId, int init_laneId, double init_s, RoadSystem *system); // via Lane Coordinates
    void init(osg::Vec3 initPos, double init_hdg, RoadSystem* init_system); // via World Coordinates
    void init(std::string, double init_s, double init_t, RoadSystem* init_system); // via Road Coordinates

    void move(double ds, double dt, float step); // move Reference Position forward
    void move(osg::Vec3 dirVec,float step_distance); // move Reference Position forward (w/o Road)

    osg::Vec3 getPosition();
    void update(std::string init_roadId, double init_s, int init_laneId);
    void update(int init_dLane, double init_ds);
    void update(std::string init_roadId, double init_s, double init_t);
    void update(double init_ds, double init_dt);
    void update(double x, double y, double z);
    void update(double dx, double dy, double dz, bool dummy);
    void move(osg::Vec3 newPos);

    void getSuccessor();
    void getPredecessor();
};

#endif // SPLINE_H
