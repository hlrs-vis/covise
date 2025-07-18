#pragma once
#include <PluginUtil/coSensor.h>
#include <cover/coVRPluginSupport.h>

#include "Device.h"

using namespace opencover;

namespace energy {
class DeviceSensor : public coPickSensor {
 public:
  // Make sure to follow Dependency Injection by using perfect forwarding
  template <
      typename... DeviceArgs,
      typename = std::enable_if_t<std::is_constructible_v<Device, DeviceArgs...>>>
  explicit DeviceSensor(osg::ref_ptr<osg::Group> parent, DeviceArgs &&...deviceArgs)
      : coPickSensor(parent), m_device(std::forward<DeviceArgs>(deviceArgs)...){};
  ~DeviceSensor() {
    if (active) disactivate();
  }
  DeviceSensor(DeviceSensor &&other)
      : coPickSensor(std::move(other)), m_device(std::move(other.m_device)) {}

  DeviceSensor &operator=(DeviceSensor &&other) {
    if (this != &other) {
      coPickSensor::operator=(std::move(other));
      m_device = std::move(other.m_device);
    }
    return *this;
  }

  DeviceSensor(const DeviceSensor &other)
      : coPickSensor(other), m_device(other.m_device) {
    // Both base class and member are explicitly initialized
  }

  DeviceSensor &operator=(const DeviceSensor &other) {
    if (this != &other) {
      m_device = other.m_device;  // Copy the device
      // Note: coPickSensor does not need to be copied, as it is already initialized
      // in the constructor and will be moved in the move constructor.
    }
    return *this;
  }

  void activate() override { m_device.activate(); }
  void disactivate() override { m_device.disactivate(); }
  void update() override {
    m_device.update();
    coPickSensor::update();
  }

  energy::Device &getDevice() { return m_device; }
  const energy::Device &getDevice() const { return m_device; }

 private:
  energy::Device m_device;
};

typedef std::vector<DeviceSensor> DeviceSensorList;
}  // namespace energy
