/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_VR_3D_ROT_CENTER_INTERACTOR_H
#define _CO_VR_3D_ROT_CENTER_INTERACTOR_H

#include <cover/coVRIntersectionInteractor.h>
#include <osg/Matrix>
#include <osg/Vec3>
// selectable cone which rotates according to hand around its origin
// it can be places (not interactive) with pos
// the cone is modeled in the origin (along z axis)

namespace opencover
{

class PLUGIN_UTILEXPORT coVR3DRotCenterInteractor : public opencover::coVRIntersectionInteractor
{
private:
    osg::Vec3 p_;
    osg::Matrix m_, invOldHandMat_, startMat_, frameDiffMat_;
    void createGeometry();

public:
    // mat: 4x4 matrix, but it's position is ignored and replaced by pos
    // pos: position in object coordinates
    coVR3DRotCenterInteractor(osg::Matrix mat, osg::Vec3 pos, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium);

    virtual ~coVR3DRotCenterInteractor();

    virtual void doInteraction();
    virtual void startInteraction();
    virtual void stopInteraction();

    // set a new position
    void updateTransform(osg::Matrix mat, osg::Vec3 pos);
    void updatePosition(osg::Vec3 pos);

    // return the current position
    osg::Matrix getMatrix()
    {
        return m_;
    }
    osg::Vec3 getPosition()
    {
        return p_;
    }
    osg::Matrix getFrameDiffMatrix()
    {
        return frameDiffMat_;
    }
};

}
#endif
