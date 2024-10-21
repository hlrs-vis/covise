#ifndef _CORE_UTILS_OSGUTILS_H
#define _CORE_UTILS_OSGUTILS_H

#include <osg/Geode>

namespace core::utils::osgUtils {
void deleteChildrenRecursive(osg::Group *grp);
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
osg::ref_ptr<osg::Geode> createCylinderBetweenPoints(osg::Vec3 start, osg::Vec3 end, float radius, osg::Vec4 cylinderColor);
}
#endif
