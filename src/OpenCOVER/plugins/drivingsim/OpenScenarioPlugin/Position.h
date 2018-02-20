#ifndef POSITION_H
#define POSITION_H

#include<string>
#include<vector>
#include <osg/Vec3>
#include <OpenScenario/schema/oscPosition.h>
#include <TrafficSimulation/AgentVehicle.h>
#include "../RoadTerrain/RoadTerrainPlugin.h"
#include <Entity.h>
#include <TrafficSimulation/CarGeometry.h>
#include <VehicleUtil/RoadSystem/RoadSystem.h>
class Road;


class Position : public OpenScenario::oscPosition
{

public:
    Position();
    ~Position();

    // Road Coordinates
    std::string roadId;
    int laneId;
    double offset;
    double s;

    // Relative Coordinates (Road/Object)
    double dx;
    double dy;
    double dz;
    osg::Vec3 relPosition;
    std::string relObject;

    //absolute Coordinates
    double x;
    double y;
    double z;
    osg::Vec3 absPosition;

    osg::Vec3 getAbsolutePosition(osg::Vec3 referencePosition, Entity* currentEntity, ::Road* road);
    double getDx();



};

#endif // POSITION_H
