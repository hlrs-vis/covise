#include "Geometry.h"

void Geometry::setTransform(osg::Vec3 position, double heading)
{
    osg::Matrix translation;
    translation.makeTranslate(position);

    osg::Matrix rotation;
    rotation.makeRotate(heading, osg::Vec3d(0, 0, 1));

    setTransform(rotation * translation);
}
