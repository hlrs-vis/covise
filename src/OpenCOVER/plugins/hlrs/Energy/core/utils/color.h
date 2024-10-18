#ifndef _CORE_UTILS_COLOR_H
#define _CORE_UTILS_COLOR_H
#endif

#include <osg/Geode>
#include <osg/Material>

namespace core::utils::color {
auto createMaterial(const osg::Vec4 &color, osg::Material::Face faceMask);
void overrideGeodeColor(osg::Geode *geode, const osg::Vec4 &color,
                        osg::Material::Face faceMask = osg::Material::FRONT);
} // namespace core::utils::color
