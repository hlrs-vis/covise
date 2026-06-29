#include <osgAnimation/RigGeometry>

#include "RigGeometryBoundsFixer.h"

void RigGeometryBoundsFixer::apply(osg::Geode &geode)
{
    for (unsigned int i = 0; i < geode.getNumDrawables(); ++i)
    {
        if (auto rig = dynamic_cast<osgAnimation::RigGeometry *>(geode.getDrawable(i)))
        {
            rig->setComputeBoundingBoxCallback(
                new osgAnimation::RigComputeBoundingBoxCallback(1.0));
            static_cast<osgAnimation::RigComputeBoundingBoxCallback *>(
                rig->getComputeBoundingBoxCallback())
                ->reset();
            rig->dirtyBound();
        }
    }
    traverse(geode);
}