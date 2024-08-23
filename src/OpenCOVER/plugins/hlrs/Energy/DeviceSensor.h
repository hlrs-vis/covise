/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef _Energy_DeviceSensor_H
#define _Energy_DeviceSensor_H

#include "Device.h"
#include <PluginUtil/coSensor.h>
#include <cover/coVRPluginSupport.h>
#include <memory>

using namespace opencover;

namespace energy {
class DeviceSensor: public coPickSensor {
private:
    energy::Device::ptr dev;

public:
    typedef std::shared_ptr<DeviceSensor> ptr;
    DeviceSensor(energy::Device::ptr d, osg::ref_ptr<osg::Node> n): coPickSensor(n), dev(d){};
    ~DeviceSensor()
    {
        if (active)
            disactivate();
    }
    DeviceSensor(const DeviceSensor &other) = delete;
    DeviceSensor &operator=(const DeviceSensor &) = delete;

    void activate() override { dev->activate(); }
    void disactivate() override { dev->disactivate(); }
    void update() override { 
        dev->update(); 
        coPickSensor::update();
    }
    energy::Device::ptr getDevice() const { return dev; }
};
}

#endif