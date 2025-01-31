#ifndef _ENNOVATISDEVICESENSOR_H
#define _ENNOVATISDEVICESENSOR_H

#include <PluginUtil/coSensor.h>

#include <memory>

#include "EnnovatisDevice.h"
#include "cover/ui/SelectionList.h"

class EnnovatisDeviceSensor : public coPickSensor {
 public:
  EnnovatisDeviceSensor(std::unique_ptr<EnnovatisDevice> d, osg::Group *n,
                        opencover::ui::SelectionList *selList)
      : coPickSensor(n), m_dev(std::move(d)), m_enabledDevices(selList) {}

  ~EnnovatisDeviceSensor() {
    if (active) disactivate();
  }

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

#endif
