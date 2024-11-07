#include <CityGMLDeviceSensor.h>
#include <PluginUtil/coSensor.h>
#include <core/CityGMLBuilding.h>

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
  if (active)
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
  m_infoBoard->updateInfo("const basic_string<char> &info");
  m_infoBoard->showInfo();
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

  m_cityGMLBuilding->updateColor(
      osg::Vec4(r / 255.0, g / 255.0, b / 255.0, 1.0));
  m_cityGMLBuilding->updateTime(timestep);
  m_infoBoard->updateTime(timestep);
}
