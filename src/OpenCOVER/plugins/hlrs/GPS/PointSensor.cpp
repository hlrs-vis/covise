/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GPS.h"
#include "GPSPoint.h"


PointSensor::PointSensor(GPSPoint *p, osg::Node *n)
    : coPickSensor(n)
{
    myGPSPoint = p;
}

PointSensor::~PointSensor()
{
    if (active)
        disactivate();
}

void PointSensor::activate()
{
    myGPSPoint->activate();
}

void PointSensor::disactivate()
{
    myGPSPoint->disactivate();
}

