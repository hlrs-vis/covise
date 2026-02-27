/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GPS_PLUGIN_SENSOR_H
#define _GPS_PLUGIN_SENSOR_H

#include <PluginUtil/coSensor.h>

class PointSensor : public coPickSensor
{
private:
    GPSPoint *myGPSPoint;

public:
    PointSensor(GPSPoint *p, osg::Node *n);
    ~PointSensor();
    // this method is called if intersection just started
    // and should be overloaded
    void activate();

    // should be overloaded, is called if intersection finishes
    void disactivate();
};

#endif
