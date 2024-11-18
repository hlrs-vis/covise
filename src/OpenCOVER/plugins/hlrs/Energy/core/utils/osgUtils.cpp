#include "osgUtils.h"

#include <utils/color.h>

#include <memory>
#include <osg/BoundingBox>
#include <osg/Geode>
#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osg/Vec4>
#include <osg/ref_ptr>

namespace core::utils::osgUtils {
std::unique_ptr<Geodes> getGeodes(osg::Group *grp) {
  Geodes geodes{};
  for (auto i = 0; i < grp->getNumChildren(); ++i) {
    auto child = grp->getChild(i);
    if (osg::ref_ptr<osg::Geode> child_geode = dynamic_cast<osg::Geode *>(child)) {
      geodes.push_back(child_geode);
      continue;
    }
    if (osg::ref_ptr<osg::Group> child_group = dynamic_cast<osg::Group *>(child)) {
      auto child_geodes = getGeodes(child_group);
      std::move(child_geodes->begin(), child_geodes->end(),
                std::back_inserter(geodes));
    }
  }
  return std::make_unique<Geodes>(geodes);
}

osg::BoundingBox getBoundingBox(
    const std::vector<osg::ref_ptr<osg::Geode>> &geodes) {
  osg::BoundingBox bb;
  for (auto geode : geodes) {
    bb.expandBy(geode->getBoundingBox());
  }
  return bb;
}

void deleteChildrenFromOtherGroup(osg::Group *grp, osg::Group *other) {
  if (!grp || !other) return;

  for (int i = 0; i < other->getNumChildren(); ++i) {
    auto child = other->getChild(i);
    if (grp->containsNode(child)) grp->removeChild(child);
  }
}

void deleteChildrenRecursive(osg::Group *grp) {
  if (!grp) return;

  for (int i = 0; i < grp->getNumChildren(); ++i) {
    auto child = grp->getChild(i);
    if (auto child_group = dynamic_cast<osg::Group *>(child))
      deleteChildrenRecursive(child_group);
    grp->removeChild(child);
  }
}

/**
 * @brief Adds a cylinder between two points.
 * Source: http://www.thjsmith.com/40/cylinder-between-two-points-opengl-c
 *
 * @param start The starting point of the cylinder.
 * @param end The ending point of the cylinder.
 * @param radius The radius of the cylinder.
 * @param cylinderColor The color of the cylinder.
 * @param group The group to which the cylinder will be added.
 */
osg::ref_ptr<osg::Geode> createCylinderBetweenPoints(osg::Vec3 start, osg::Vec3 end,
                                                     float radius,
                                                     osg::Vec4 cylinderColor) {
  osg::ref_ptr geode = new osg::Geode;
  osg::Vec3 center;
  float height;
  osg::ref_ptr<osg::Cylinder> cylinder;
  osg::ref_ptr<osg::ShapeDrawable> cylinderDrawable;
  osg::ref_ptr<osg::Material> pMaterial;

  height = (start - end).length();
  center = osg::Vec3((start.x() + end.x()) / 2, (start.y() + end.y()) / 2,
                     (start.z() + end.z()) / 2);

  // This is the default direction for the cylinders to face in OpenGL
  osg::Vec3 z = osg::Vec3(0, 0, 1);

  // Get diff between two points you want cylinder along
  osg::Vec3 p = start - end;

  // Get CROSS product (the axis of rotation)
  osg::Vec3 t = z ^ p;

  // Get angle. length is magnitude of the vector
  double angle = acos((z * p) / p.length());

  // Create a cylinder between the two points with the given radius
  cylinder = new osg::Cylinder(center, radius, height);
  cylinder->setRotation(osg::Quat(angle, osg::Vec3(t.x(), t.y(), t.z())));

  cylinderDrawable = new osg::ShapeDrawable(cylinder);
  geode->addDrawable(cylinderDrawable);

  // Set the color of the cylinder that extends between the two points.
  color::overrideGeodeColor(geode, cylinderColor);

  return geode;
}
}  // namespace core::utils::osgUtils
