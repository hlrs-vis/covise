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


    std::string roadId;
    Road* road;
    int laneId;
    double s;
    double t;

    osg::Vec3 xyz;

    void initFromLane(std::string init_roadId, int init_laneId, double init_s, RoadSystem *system);
};

#endif // SPLINE_H
