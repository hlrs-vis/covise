#include "CityGMLDeviceSensor.h"

#include <PluginUtil/coSensor.h>
#include <PluginUtil/coShaderUtil.h>
#include <PluginUtil/colors/coColorMap.h>
#include <app/presentation/CityGMLBuilding.h>

#include <algorithm>
#include <memory>
#include <osg/Geometry>

CityGMLDeviceSensor::CityGMLDeviceSensor(
    osg::ref_ptr<osg::Group> parent,
    std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
    std::unique_ptr<core::interface::IBuilding> &&drawableBuilding,
    const std::vector<std::string> &textBoxTxt)
    : coPickSensor(parent),
      m_cityGMLBuilding(std::move(drawableBuilding)),
      m_infoBoard(std::move(infoBoard)),
      m_textBoxTxt(textBoxTxt),
      m_colors({}) {
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
    m_infoBoard->showInfo();
  }
  m_active = !m_active;
}

void CityGMLDeviceSensor::disactivate() {
  if (m_active) return;
  m_infoBoard->hideInfo();
}

void CityGMLDeviceSensor::updateTimestepColors(const std::vector<float> &values,
                                               const opencover::ColorMap &map) {
  m_colors.clear();
  m_colors.resize(values.size());
  std::transform(values.begin(), values.end(), m_colors.begin(),
                 [&map](const auto &v) {
                   return map.getColor(v);  // Initialize with the first value
                 });
}

void CityGMLDeviceSensor::updateTxtBoxTexts(const std::vector<std::string> &texts) {
  m_textBoxTxt = texts;
}

void CityGMLDeviceSensor::updateTitleOfInfoboard(const std::string &title) {
  auto txtInfoboard = dynamic_cast<TxtInfoboard *>(m_infoBoard.get());
  if (txtInfoboard) {
    txtInfoboard->setTitle(title);
  }
}

void CityGMLDeviceSensor::updateTime(int timestep) {
  auto gmlBuilding = dynamic_cast<CityGMLBuilding *>(m_cityGMLBuilding.get());
  if (gmlBuilding)
    if (gmlBuilding->hasShader()) gmlBuilding->updateTime(timestep);
}

void CityGMLDeviceSensor::setColorMapInShader(const opencover::ColorMap &colorMap) {
  auto gmlBuilding = dynamic_cast<CityGMLBuilding *>(m_cityGMLBuilding.get());
  if (gmlBuilding) {
    gmlBuilding->setColorMapInShader(colorMap);
    return;
  }
}
void CityGMLDeviceSensor::setDataInShader(const std::vector<double> &data, float min,
                                          float max) {
  auto gmlBuilding = dynamic_cast<CityGMLBuilding *>(m_cityGMLBuilding.get());
  if (gmlBuilding) {
    gmlBuilding->setDataInShader(data, min, max);
    return;
  }
}
