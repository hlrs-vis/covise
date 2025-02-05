#include "grid.h"

#include <utils/osgUtils.h>

namespace core::simulation::grid {
Point::Point(const std::string &name, const float &x, const float &y, const float &z,
             const float &radius, const Data &additionalData)
    : osg::Group(),
      m_point(new osg::Sphere(osg::Vec3(x, y, z), radius)),
      m_additionalData(additionalData) {
  osg::ref_ptr<osg::TessellationHints> hints = new osg::TessellationHints;
  hints->setDetailRatio(1.5f);
  m_shape = new osg::ShapeDrawable(m_point, hints);
  osg::ref_ptr<osg::Geode> geode = new osg::Geode;
  geode->addChild(m_shape);
  addChild(geode);
  setName(name);
}

DirectedConnection::DirectedConnection(const std::string &name,
                                       const osg::Vec3 &start, const osg::Vec3 &end,
                                       const float &radius,
                                       osg::ref_ptr<osg::TessellationHints> hints,
                                       const Data &additionalData)
    : osg::Group(),
      m_start(new osg::Vec3(start)),
      m_end(new osg::Vec3(end)),
      m_additionalData(additionalData) {
  m_geode = utils::osgUtils::createCylinderBetweenPoints(
      start, end, radius, osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f), hints);
  addChild(m_geode);
  setName(name);
}
}  // namespace core::simulation::grid
