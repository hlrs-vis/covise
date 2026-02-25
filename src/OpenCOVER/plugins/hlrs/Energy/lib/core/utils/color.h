#pragma once
#include "../interfaces/IColorable.h"
#include <osg/Geode>
#include <osg/Material>

namespace core::utils::color
{

typedef core::interface::Color Color;
core::interface::Color getTrafficLightColor(float val, float max);
auto createMaterial(const osg::Vec4 &color, osg::Material::Face faceMask);
void overrideGeodeColor(osg::Geode *geode, const Color &color,
    osg::Material::Face faceMask = osg::Material::FRONT);
void overrideGeodeColor(osg::Geode *geode, const osg::Vec4 &color,
    osg::Material::Face faceMask = osg::Material::FRONT);
void overrideGeodeMaterial(osg::Geode *geode, osg::Material *material);
} // namespace core::utils::color
