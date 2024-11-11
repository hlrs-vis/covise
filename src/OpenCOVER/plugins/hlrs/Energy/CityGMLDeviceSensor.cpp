#include <CityGMLDeviceSensor.h>
#include <PluginUtil/coSensor.h>
#include <core/CityGMLBuilding.h>
#include <cstdint>

CityGMLDeviceSensor::CityGMLDeviceSensor(
    osg::ref_ptr<osg::Group> group,
    std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
    std::unique_ptr<core::interface::IBuilding> &&drawableBuilding)
    : coPickSensor(group), m_cityGMLBuilding(std::move(drawableBuilding)),
      m_infoBoard(std::move(infoBoard)) {

  m_cityGMLBuilding->initDrawable();

  // infoboard
  m_infoBoard->initInfoboard();
  m_infoBoard->initDrawable();
  group->addChild(m_infoBoard->getDrawable());
}

CityGMLDeviceSensor::~CityGMLDeviceSensor() {
  if (m_active)
    disactivate();
  auto parent = m_cityGMLBuilding->getDrawable()->getParent(0);
  parent->removeChild(m_infoBoard->getDrawable());
}

void CityGMLDeviceSensor::update() {
  m_cityGMLBuilding->updateDrawable();
  m_infoBoard->updateDrawable();
  coPickSensor::update();
}

void CityGMLDeviceSensor::activate() {
  if (!m_active) {
    m_infoBoard->updateInfo("DAS IST EIN TEST");
    m_infoBoard->showInfo();
  }
  m_active = !m_active;
}

void CityGMLDeviceSensor::disactivate() {
  if (m_active)
    return;
  m_infoBoard->hideInfo();
}

void CityGMLDeviceSensor::updateTime(int timestep) {
  static std::uint8_t r = 255;
  static std::uint8_t g = 0;
  static std::uint8_t b = 0;

  if (r == 255 && g < 255 && b == 0) {
    g++;
  } else if (r > 0 && g == 255 && b == 0) {
    r--;
  } else if (r == 0 && g == 255 && b < 255) {
    b++;
  } else if (r == 0 && g > 0 && b == 255) {
    g--;
  } else if (r < 255 && g == 0 && b == 255) {
    r++;
  } else if (r == 255 && g == 0 && b > 0) {
    b--;
  }

  m_cityGMLBuilding->updateColor(
      osg::Vec4(r / 255.0, g / 255.0, b / 255.0, 1.0));
  m_cityGMLBuilding->updateTime(timestep);
  m_infoBoard->updateTime(timestep);
}
