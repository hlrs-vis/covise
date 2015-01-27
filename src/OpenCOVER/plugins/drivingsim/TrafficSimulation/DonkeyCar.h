/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DonkeyCar_h
#define DonkeyCar_h

#include "Vehicle.h"
#include "CarGeometry.h"

#include <osg/MatrixTransform>

class DonkeyCar : public Vehicle
{
public:
    DonkeyCar(std::string, Road *, double, double, double, double);
    ~DonkeyCar();

    Road *getRoad() const;

    double getU() const;
    double getDu() const;

    void move(double);

    VehicleGeometry *getVehicleGeometry();
    double getBoundingCircleRadius();

protected:
    Road *currentRoad;
    double u, v, du, h;
    int currentLane;
    int direction;

    VehicleGeometry *geometry;

    Transform carTransform;
};

#endif
