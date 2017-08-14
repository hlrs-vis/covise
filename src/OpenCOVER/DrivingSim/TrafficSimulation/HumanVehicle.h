/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HumanVehicle_h
#define HumanVehicle_h

#include "Vehicle.h"
#include "HumanVehicleGeometry.h"
#include "VehicleUtils.h"

#include <string>
#include <list>

class Road;

class TRAFFICSIMULATIONEXPORT HumanVehicle : public Vehicle
{
public:
    HumanVehicle(std::string);
    ~HumanVehicle()
    {
    }

    Road *getRoad() const;

    double getU() const;
    double getDu() const;

    double getV() const;

    int getLane() const;
    bool isOnLane(int) const;

    void move(double);
	void setPosition(osg::Vec3 &pos, osg::Vec3 &direction);

    VehicleGeometry *getVehicleGeometry();
    Transform getVehicleTransform();
    double getBoundingCircleRadius();
    static osg::Vec2d human_pos;
    static double human_v;

protected:
    Road *road;
    double u, v;
    double oldU;
    double du;
    int currentLane;
    LaneSection *currentSection;
    double hdg;

    HumanVehicleGeometry *geometry;

    //coVRPlugin* steeringWheelPlugin;

    ObstacleRelation locateNextVehicleOnLane();
};

#endif
