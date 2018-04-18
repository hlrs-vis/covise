/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_VR_3D_ROT_INTERACTOR_H
#define _CO_VR_3D_ROT_INTERACTOR_H

#include <cover/coVRIntersectionInteractor.h>

namespace opencover
{
// selectable cone which can be positioned in 3D
// the cone is modeled in the origin (along z axis)
// the interactor rotates around the rotation point
//
//                               |---|
//            laser sword        | x | interactor center
//    hand ----------------------o___|-------->
//                               hitPoint
//
//    d: distance hand-hitPoint
//    diff: Vector hitPoint_0-interactorcenter
//    laserSword Direction.xformVec(yaxis, handMat)
//    laserSwordDirection_o.xformVec(laserSwordDirection, cover->getInvBase)
//    new interactor center = handPos +laserSwordDirection_o * d + diff

class PLUGIN_UTILEXPORT coVR3DRotInteractor : public coVRIntersectionInteractor
{
private:
    osg::Vec3 _interPos; // position in object coordinates
    osg::Vec3 _rotationPoint; // position in object coordinates
    float _d;
    osg::Vec3 _diff;
    osg::Matrix _oldHandMat;
    void createGeometry();

public:
    // pos: position in object coordinates
    // (it is intended for positioning COVISE module parameters like the
    // startpoints for tracers and these parameters are always in object
    // coordinates
    coVR3DRotInteractor(osg::Vec3 rotationPoint, osg::Vec3 position, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium);

    virtual ~coVR3DRotInteractor();

    void startInteraction();

    void doInteraction();

    // stop the interaction
    // implemented in base class void stopInteraction();

    // set a new position
    void updateTransform(osg::Vec3 pos, osg::Vec3 rotationPoint);

    // set the rotation point
    void updateRotationPoint(osg::Vec3 rotPoint);

    // return the current position
    osg::Vec3 getPos()
    {
        return _interPos;
    }

    // return the current rotation point
    osg::Vec3 getRotationPoint()
    {
        return _rotationPoint;
    }
};
}
#endif
