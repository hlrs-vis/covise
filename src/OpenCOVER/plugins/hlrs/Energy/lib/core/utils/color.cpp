#include "color.h"
namespace core::utils::color {

std::unique_ptr<osg::Vec4> ColorMap::getColor(float value, float maxValue) const {
  // RGB Colors 1,1,1 = white, 0,0,0 = black
  maxValue = std::max(maxValue, 1.f);
  float valueNormiert = value / maxValue;

  auto col =
      std::make_unique<osg::Vec4>(max.r() * valueNormiert + min.r() * (1 - valueNormiert),
                                  max.g() * valueNormiert + min.g() * (1 - valueNormiert),
                                  max.b() * valueNormiert + min.b() * (1 - valueNormiert),
                                  max.a() * valueNormiert + min.a() * (1 - valueNormiert));
  return col;
}

auto createMaterial(const osg::Vec4 &color, osg::Material::Face faceMask) {
  osg::ref_ptr<osg::Material> mat = new osg::Material;
  mat->setDiffuse(faceMask, color);
  return mat;
}

void overrideGeodeColor(osg::Geode *geode, const osg::Vec4 &color,
                        osg::Material::Face faceMask) {
  auto mat = createMaterial(color, faceMask);
  overrideGeodeMaterial(geode, mat);
}

void overrideGeodeMaterial(osg::Geode *geode, osg::Material *material) {
  geode->getOrCreateStateSet()->setAttribute(material,
                                             osg::StateAttribute::OVERRIDE);
}
}  // namespace core::utils::color
