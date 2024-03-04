#ifndef _ENNOVATISDEVICESENSOR_H
#define _ENNOVATISDEVICESENSOR_H

#include "EnnovatisDevice.h"
#include "ennovatis/rest.h"
#include <PluginUtil/coSensor.h>
#include <cover/coVRPluginSupport.h>
#include <memory>

class EnnovatisDeviceSensor: public coPickSensor {
private:
    std::unique_ptr<EnnovatisDevice> m_dev;

public:
    EnnovatisDeviceSensor(const EnnovatisDevice &d, osg::Group *n, std::shared_ptr<ennovatis::RESTRequest> req)
    : coPickSensor(n), m_dev(std::make_unique<EnnovatisDevice>(d))
    {
        n->addChild(m_dev->getDeviceGroup());
    }

    EnnovatisDeviceSensor(const EnnovatisDevice &d, osg::Group *n): EnnovatisDeviceSensor(d, n, nullptr) {}

    ~EnnovatisDeviceSensor()
    {
        if (active)
            disactivate();
    }

    [[nodiscard("Unused Getter")]] auto getDevice() const { return m_dev.get(); }

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
};

#endif
