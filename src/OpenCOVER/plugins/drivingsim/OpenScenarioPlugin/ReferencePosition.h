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

    void initFromLane(std::string init_roadId, int init_laneId, double init_s, RoadSystem *system);
    void moveForward(float dt, float speed);
    void moveOnTrajectory(double ds, double dt, float step);

    osg::Vec3 getPosition();
    void update(std::string init_roadId, double init_s, int init_laneId);
    void update(int init_dLane, double init_ds);
    void update(std::string init_roadId, double init_s, double init_t);
    void update(double init_ds, double init_dt);
    void update(double x, double y, double z);
    void update(double dx, double dy, double dz, bool dummy);
    void move(osg::Vec3 newPos);
};

#endif // SPLINE_H
