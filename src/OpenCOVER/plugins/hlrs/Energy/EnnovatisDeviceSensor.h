#ifndef _ENNOVATISDEVICESENSOR_H
#define _ENNOVATISDEVICESENSOR_H

#include "EnnovatisDevice.h"
#include <PluginUtil/coSensor.h>

class EnnovatisDeviceSensor: public coPickSensor {
private:
    std::unique_ptr<EnnovatisDevice> m_dev;

public:
    EnnovatisDeviceSensor(std::unique_ptr<EnnovatisDevice> d, osg::Group *n): coPickSensor(n), m_dev(std::move(d)) {}

    ~EnnovatisDeviceSensor()
    {
        if (active)
            disactivate();
    }

    [[nodiscard]] auto getDevice() const { return m_dev.get(); }

    void activate() override
    {
        m_dev->activate();
        coPickSensor::activate();
    }
    void disactivate() override
    {
        m_dev->disactivate();
        coPickSensor::disactivate();
    }
    void update() override
    {
        m_dev->update();
        coPickSensor::update();
    }

    void setTimestep(int t) { m_dev->setTimestep(t); }
};

#endif
