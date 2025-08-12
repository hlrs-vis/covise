#pragma once
#include <PluginUtil/coSensor.h>
#include <cover/ui/SelectionList.h>

#include <memory>

#include "EnnovatisDevice.h"

/**
 * @class EnnovatisDeviceSensor
 * @brief Sensor class for handling EnnovatisDevice objects within an OpenCOVER plugin.
 *
 * This class extends coPickSensor to provide interaction and update mechanisms
 * for EnnovatisDevice instances. It manages device activation, deactivation,
 * and updates, and integrates with a SelectionList UI element to track enabled devices.
 *
 * - Construction requires ownership of an EnnovatisDevice, a scene graph node, and a SelectionList pointer.
 * - Disallows copy construction and assignment.
 * - Provides device access, update, timestep setting, activation, and deactivation.
 *
 * @note The class assumes ownership of the EnnovatisDevice via std::unique_ptr.
 */
class EnnovatisDeviceSensor : public coPickSensor {
 public:
  EnnovatisDeviceSensor(std::unique_ptr<EnnovatisDevice> d, osg::Group *n,
                        opencover::ui::SelectionList *selList)
      : coPickSensor(n), m_dev(std::move(d)), m_enabledDevices(selList) {}

  ~EnnovatisDeviceSensor() {
    if (m_activated) disactivate();
  }

  EnnovatisDeviceSensor(const EnnovatisDeviceSensor &) = delete;
  EnnovatisDeviceSensor &operator=(const EnnovatisDeviceSensor &) = delete;

  [[nodiscard]] auto getDevice() const { return m_dev.get(); }

  void update() override {
    m_dev->update();
    coPickSensor::update();
  }

  void setTimestep(int t) { m_dev->setTimestep(t); }
  void activate() override;
  void disactivate() override;

 private:
  opencover::ui::SelectionList *m_enabledDevices;
  std::unique_ptr<EnnovatisDevice> m_dev;
  bool m_activated = false;
};
