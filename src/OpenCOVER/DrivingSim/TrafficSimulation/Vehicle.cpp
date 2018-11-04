/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Vehicle.h"

#include <VehicleUtil/RoadSystem/Road.h>
#include "VehicleUtils.h"

int Vehicle::vehicleIDs = 0;

Vehicle::Vehicle(std::string n)
    : name(n)
{
    vehicleID = vehicleIDs++; // give the vehicle an unique ID
}

void Vehicle::setName(std::string n)
{
    name = n;
}

std::string Vehicle::getName() const
{
    return name;
}

RoadTransition Vehicle::getRoadTransition() const
{
    return RoadTransition(getRoad(), getDu() < 0 ? -1 : 1);
}

double
Vehicle::getSquaredDistanceTo(Vector3D point)
{
    Vector3D v = point - getVehicleTransform().v();
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

//
