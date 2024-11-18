#include <PrototypeBuilding.h>
#include <utils/color.h>
#include <utils/osgUtils.h>

#include <memory>

#include "osg/ref_ptr"

namespace core {

auto PrototypeBuilding::getColor(float val, float max) const {
  // RGB Colors 1,1,1 = white, 0,0,0 = black
  const auto &colMax = m_attributes.colorMap.max;
  const auto &colMin = m_attributes.colorMap.min;
  max = std::max(max, 1.f);
  float valN = val / max;

  auto col =
      std::make_unique<osg::Vec4>(colMax.r() * valN + colMin.r() * (1 - valN),
                                  colMax.g() * valN + colMin.g() * (1 - valN),
                                  colMax.b() * valN + colMin.b() * (1 - valN),
                                  colMax.a() * valN + colMin.a() * (1 - valN));
  return col;
}

void PrototypeBuilding::updateColor(const osg::Vec4 &color) {
  for (auto drawable : m_drawables)
    if (auto geode = dynamic_cast<osg::Geode *>(drawable.get()))
      utils::color::overrideGeodeColor(geode, color);
}

void PrototypeBuilding::initDrawables() {
  const osg::Vec3f bottom(m_attributes.position);
  osg::Vec3f top(bottom);
  top.z() += m_attributes.height;
  osg::ref_ptr<osg::Geode> drawable = utils::osgUtils::createCylinderBetweenPoints(
      bottom, top, m_attributes.radius, m_attributes.colorMap.defaultColor);
  m_drawables.push_back(drawable);
}

std::unique_ptr<osg::Vec4> PrototypeBuilding::getColorInRange(float value,
                                                              float maxValue) {
  return getColor(value, maxValue);
}

void PrototypeBuilding::updateDrawables() {}

void PrototypeBuilding::updateTime(int timestep) {
  // TODO: update for example the height of the cylinder with each
  // timestep
}

}  // namespace core
