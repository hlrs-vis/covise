#include "EnnovatisDeviceSensor.h"
#include <algorithm>

void EnnovatisDeviceSensor::activate() {
  if (!m_activated) {
    m_dev->activate();
    m_enabledDevices->append(m_dev->getBuildingInfo().building->getName());
    m_enabledDevices->select(m_enabledDevices->items().size() - 1);
  }
  m_activated = !m_activated;
}

void EnnovatisDeviceSensor::disactivate() {
  if (m_activated)
    return;

  m_dev->disactivate();
  auto selList = m_enabledDevices->items();
  auto name = m_dev->getBuildingInfo().building->getName();
  selList.erase(std::find(selList.begin(), selList.end(), name));
  m_enabledDevices->setList(selList);
}
