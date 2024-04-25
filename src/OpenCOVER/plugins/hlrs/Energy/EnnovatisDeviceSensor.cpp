#include "EnnovatisDeviceSensor.h"
#include <algorithm>

void EnnovatisDeviceSensor::activate()
{
    if (!m_activated) {
        m_dev->activate();
        auto selList_ptr = m_enabledDevices.lock();
        selList_ptr->append(m_dev->getBuildingInfo().building->getName());
        selList_ptr->select(selList_ptr->items().size() - 1);
    }
    m_activated = !m_activated;
}

void EnnovatisDeviceSensor::disactivate()
{
    if (m_activated)
        return;

    m_dev->disactivate();
    auto selList_ptr = m_enabledDevices.lock();
    auto selList = selList_ptr->items();
    auto name = m_dev->getBuildingInfo().building->getName();
    selList.erase(std::find(selList.begin(), selList.end(), name));
    selList_ptr->setList(selList);
}