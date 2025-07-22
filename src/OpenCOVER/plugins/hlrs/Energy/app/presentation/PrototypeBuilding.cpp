#include "PrototypeBuilding.h"

#include <lib/core/utils/color.h>
#include <lib/core/utils/osgUtils.h>

#include <memory>
#include <osg/ref_ptr>

using namespace core;

void PrototypeBuilding::updateColor(const osg::Vec4 &color) {
  for (auto drawable : m_drawables)
    if (auto geode = dynamic_cast<osg::Geode *>(drawable.get()))
      utils::color::overrideGeodeColor(geode, color);
}

void PrototypeBuilding::initDrawables() {
  const osg::Vec3f bottom(m_attributes.position);
  osg::Vec3f top(bottom);
  top.z() += m_attributes.height;
  osg::ref_ptr<osg::Geode> drawable = utils::osgUtils::createOsgCylinderBetweenPoints(
      bottom, top, m_attributes.radius, m_attributes.colorMap.defaultColor);
  m_drawables.push_back(drawable);
}

std::unique_ptr<osg::Vec4> PrototypeBuilding::getColorInRange(float value,
                                                              float maxValue) {
  return m_attributes.colorMap.getColor(value, maxValue);
}

void PrototypeBuilding::updateDrawables() {}

void PrototypeBuilding::updateTime(int timestep) {
  // TODO: update for example the height of the cylinder with each
  // timestep
}
