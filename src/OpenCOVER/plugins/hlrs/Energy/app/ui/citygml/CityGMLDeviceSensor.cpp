#include "CityGMLDeviceSensor.h"

#include <PluginUtil/coColorMap.h>
#include <PluginUtil/coSensor.h>
#include <PluginUtil/coShaderUtil.h>
#include <app/presentation/CityGMLBuilding.h>

#include <memory>
#include <osg/Geometry>

CityGMLDeviceSensor::CityGMLDeviceSensor(
    osg::ref_ptr<osg::Group> parent,
    std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
    std::unique_ptr<core::interface::IBuilding> &&drawableBuilding,
    std::shared_ptr<ColorMap> colorMap)
    : coPickSensor(parent),
      m_cityGMLBuilding(std::move(drawableBuilding)),
      m_infoBoard(std::move(infoBoard)),
      m_colorMapRef(colorMap) {
  m_cityGMLBuilding->initDrawables();

  // infoboard
  m_infoBoard->initInfoboard();
  m_infoBoard->initDrawable();
  parent->addChild(m_infoBoard->getDrawable());
}

CityGMLDeviceSensor::~CityGMLDeviceSensor() {
  if (m_active) disactivate();
  getParent()->removeChild(m_infoBoard->getDrawable());
}

void CityGMLDeviceSensor::update() {
  m_cityGMLBuilding->updateDrawables();
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
  if (m_active) return;
  m_infoBoard->hideInfo();
}

void CityGMLDeviceSensor::updateTimestepColors(const std::vector<float> &values) {
  auto color_map = m_colorMapRef.lock();
  m_colors.clear();
  m_colors.resize(values.size());
  const auto &max = m_colorMapRef;
  for (auto i = 0; i < m_colors.size(); ++i) {
    auto value = values[i];
    auto color =
        covise::getColor(value, *color_map, color_map->min, color_map->max);
    m_colors[i] = color;
  }
}

void CityGMLDeviceSensor::updateTime(int timestep) {
  if (timestep >= m_colors.size()) return;
  m_cityGMLBuilding->updateColor(m_colors[timestep]);
  m_cityGMLBuilding->updateTime(timestep);
  m_infoBoard->updateTime(timestep);
}
