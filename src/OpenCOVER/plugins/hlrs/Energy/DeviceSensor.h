/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef _Energy_DeviceSensor_H
#define _Energy_DeviceSensor_H

#include "Device.h"
#include <PluginUtil/coSensor.h>
#include <cover/coVRPluginSupport.h>

using namespace covise;
using namespace opencover;

class DeviceSensor : public coPickSensor
{
private:
    Device *dev;

public:
    DeviceSensor(Device *h, osg::Node *n);
    ~DeviceSensor();

    void activate();
    void disactivate();
};

#endif
