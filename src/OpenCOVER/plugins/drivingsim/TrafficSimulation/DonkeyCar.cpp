/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DonkeyCar.h"

#include "RoadSystem/Road.h"

DonkeyCar::DonkeyCar(std::string n, Road *r, double startu, double startv, double starth, double startdu)
    : Vehicle(n)
    , currentRoad(r)
    , u(startu)
    , v(startv)
    , du(startdu)
    , h(starth)
    , carTransform(r->getRoadTransform(u, v))
{
    geometry = new CarGeometry();
}

DonkeyCar::~DonkeyCar()
{
    delete geometry;
}

void DonkeyCar::move(double dt)
{
    u += du * dt;
    double noise = v + 10 * sin(0.01 * u);
    //noise = t;

    if (u > currentRoad->getLength())
    {
        u -= currentRoad->getLength();
        currentRoad = dynamic_cast<Road *>(currentRoad->getSuccessorConnection()->getConnectingTarmac());
    }

    carTransform = currentRoad->getRoadTransform(u, noise);
    //carTransform.gamma() += dt;

    h = tan(0.1 * cos(0.01 * u));
    //heading = 0;

    geometry->setTransform(carTransform, h);
}

void DonkeyCar::setPosition(osg::Vec3&, osg::Vec3 &){
//geometry->setTransformByCoordinates(x, y, z);
}

Road *DonkeyCar::getRoad() const
{
    return currentRoad;
}

double DonkeyCar::getU() const
{
    return u;
}

double DonkeyCar::getDu() const
{
    return du;
}

VehicleGeometry *DonkeyCar::getVehicleGeometry()
{
    return geometry;
}

double DonkeyCar::getBoundingCircleRadius()
{
    return (geometry == NULL) ? 0.0 : geometry->getBoundingCircleRadius();
}
