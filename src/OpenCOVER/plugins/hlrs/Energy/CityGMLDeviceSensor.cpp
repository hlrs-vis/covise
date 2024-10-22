#include <CityGMLDeviceSensor.h>
#include <PluginUtil/coSensor.h>
#include <core/CityGMLBuilding.h>

CityGMLDeviceSensor::CityGMLDeviceSensor(
    osg::ref_ptr<osg::Group> geo,
    std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
    std::unique_ptr<core::interface::IBuilding> &&drawableBuilding)
    : coPickSensor(geo), m_cityGMLBuilding(std::move(drawableBuilding)),
      m_infoBoard(std::move(infoBoard)) {
  m_cityGMLBuilding->initDrawable();
  m_infoBoard->initInfoboard();
}

CityGMLDeviceSensor::~CityGMLDeviceSensor() {
  if (active)
    disactivate();
}

void CityGMLDeviceSensor::update() {
  m_cityGMLBuilding->updateDrawable();
  m_infoBoard->updateDrawable();
  coPickSensor::update();
}

void CityGMLDeviceSensor::activate() {
  m_infoBoard->showInfo();
  m_cityGMLBuilding->initDrawable();
}

void CityGMLDeviceSensor::disactivate() { m_infoBoard->hideInfo(); }

void CityGMLDeviceSensor::updateTime(int timestep) {
  static uint r = 255;
  static uint g = 0;
  static uint b = 0;

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

  m_cityGMLBuilding->updateColor(osg::Vec4(r / 255.0, g / 255.0, b / 255.0, 1.0));
  m_cityGMLBuilding->updateTime(timestep);
  m_infoBoard->updateTime(timestep);
}
