#include "CityGMLBuilding.h"
#include "utils/color.h"
#include <osg/Geode>
#include <osg/MatrixTransform>

namespace core {

CityGMLBuilding::CityGMLBuilding(osg::Geode *geode) { m_drawable = geode; }

void CityGMLBuilding::initDrawable() {}

void CityGMLBuilding::updateColor(const osg::Vec4 &color) {
  if (auto geode = dynamic_cast<osg::Geode *>(m_drawable.get()))
    utils::color::overrideGeodeColor(geode, color);
}

void CityGMLBuilding::updateTime(int timestep) { m_timestep = timestep; }

void CityGMLBuilding::updateDrawable() {}
std::unique_ptr<osg::Vec4> CityGMLBuilding::getColorInRange(float value,
                                                            float maxValue) {
  return nullptr;
}
} // namespace core
