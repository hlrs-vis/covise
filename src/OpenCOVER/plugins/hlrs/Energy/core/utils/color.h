#ifndef _CORE_UTILS_COLOR_H
#define _CORE_UTILS_COLOR_H

#include <osg/Geode>
#include <osg/Material>

namespace core::utils::color {
struct ColorMap {
  ColorMap(const osg::Vec4 &max, const osg::Vec4 &min, const osg::Vec4 &def)
      : max(max), min(min), defaultColor(def) {}
  osg::Vec4 max;
  osg::Vec4 min;
  osg::Vec4 defaultColor;
};

auto createMaterial(const osg::Vec4 &color, osg::Material::Face faceMask);
void overrideGeodeColor(osg::Geode *geode, const osg::Vec4 &color,
                        osg::Material::Face faceMask = osg::Material::FRONT);
} // namespace core::utils::color
#endif
