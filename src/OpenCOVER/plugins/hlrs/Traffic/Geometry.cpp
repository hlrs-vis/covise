#include "Geometry.h"

void Geometry::setTransform(osg::Vec3 position, double heading, double pitch)
{
    setTransform(osg::Matrix::rotate(pitch, osg::Vec3d(0, 1, 0)) * osg::Matrix::rotate(heading, osg::Vec3d(0, 0, 1)) * osg::Matrix::translate(position));
}
