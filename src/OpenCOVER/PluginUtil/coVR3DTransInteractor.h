/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_VR_3D_TRANS_INTERACTOR_H
#define _CO_VR_3D_TRANS_INTERACTOR_H

#include <cover/coVRIntersectionInteractor.h>

namespace opencover
{
// selectable sphere which can be positioned in 3D
// the sphere is modeled in the origin
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

class PLUGIN_UTILEXPORT coVR3DTransInteractor : public coVRIntersectionInteractor
{
private:
    osg::Vec3 _interPos; // position in object coordinates
    float _d;
    osg::Vec3 _diff;
    osg::Matrix _oldHandMat;

protected:
    virtual void createGeometry();

public:
    // pos: position in object coordinates
    // (it is intended for positioning COVISE module parameters like the
    // startpoints for tracers and these parameters are always in object
    // coordinates
    coVR3DTransInteractor(osg::Vec3 pos, float size, vrui::coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority);

    virtual ~coVR3DTransInteractor();

    void startInteraction();

    void doInteraction();

    // stop the interaction
    // implemented in base class void stopInteraction();

    // set a new position
    void updateTransform(osg::Vec3 pos);

    // set a new position
    void setPos();

    // return the current position
    osg::Vec3 getPos()
    {
        return _interPos;
    };
};
}
#endif
