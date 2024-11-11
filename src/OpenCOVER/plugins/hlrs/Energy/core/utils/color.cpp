#include "color.h"

namespace core::utils::color {

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
} // namespace core::utils::color
