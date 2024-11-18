#include "CityGMLBuilding.h"

#include <osg/Geode>
#include <osg/MatrixTransform>

#include "utils/color.h"

namespace core {
using namespace utils;

CityGMLBuilding::CityGMLBuilding(const osgUtils::Geodes &geodes) {
  m_drawables.reserve(geodes.size());
  m_drawables.insert(m_drawables.begin(), geodes.begin(), geodes.end());
}

void CityGMLBuilding::initDrawables() {}

void CityGMLBuilding::updateColor(const osg::Vec4 &color) {
  for (auto drawable : m_drawables) {
    if (auto geo = drawable->asGeode()) color::overrideGeodeColor(geo, color);
  }
}

void CityGMLBuilding::updateTime(int timestep) { m_timestep = timestep; }

void CityGMLBuilding::updateDrawables() {}
std::unique_ptr<osg::Vec4> CityGMLBuilding::getColorInRange(float value,
                                                            float maxValue) {
  return nullptr;
}
}  // namespace core
