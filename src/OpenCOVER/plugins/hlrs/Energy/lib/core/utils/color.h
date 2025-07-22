#ifndef _CORE_UTILS_COLOR_H
#define _CORE_UTILS_COLOR_H

#include <memory>
#include <osg/Geode>
#include <osg/Material>

namespace core::utils::color {

struct ColorMap {
  ColorMap(const osg::Vec4 &max, const osg::Vec4 &min, const osg::Vec4 &def)
      : max(max), min(min), defaultColor(def) {}
  osg::Vec4 max;
  osg::Vec4 min;
  osg::Vec4 defaultColor;
  std::unique_ptr<osg::Vec4> getColor(float value, float maxValue) const;
};

osg::Vec4 getTrafficLightColor(float val, float max);
auto createMaterial(const osg::Vec4 &color, osg::Material::Face faceMask);
void overrideGeodeColor(osg::Geode *geode, const osg::Vec4 &color,
                        osg::Material::Face faceMask = osg::Material::FRONT);
void overrideGeodeMaterial(osg::Geode *geode, osg::Material *material);
}  // namespace core::utils::color
#endif
