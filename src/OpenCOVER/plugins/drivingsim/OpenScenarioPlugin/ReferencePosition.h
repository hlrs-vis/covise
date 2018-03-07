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
    ~ReferencePosition();

    // Road
    std::string roadId;
    Road* road;
    double roadLength;

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
    osg::Vec3 getPosition();
};

#endif // SPLINE_H
