/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_VR_1D_TRANS_INTERACTOR_H
#define _CO_VR_1D_TRANS_INTERACTOR_H

#include <cover/coVRIntersectionInteractor.h>
#include "coPlane.h"
#include <osg/Vec3>
#include <osg/MatrixTransform>

namespace opencover
{
// selectable cylinder which can be positioned on a plane
// cylinder and plane are positioned in object coordinates

class PLUGIN_UTILEXPORT coVR1DTransInteractor : public opencover::coVRIntersectionInteractor
{

private:
    osg::Vec3 _diff;
    osg::Vec3 _normal;
    osg::Matrix _oldHandMat;
    void createGeometry();

public:
    // position and normal on object coordinates
    // size in world coordinates (mm)
    coVR1DTransInteractor(osg::Vec3 pos, osg::Vec3 normal, float size, coInteraction::InteractionType type, const char *iconName, const char *interactorName, enum coInteraction::InteractionPriority priority);

    virtual ~coVR1DTransInteractor();

    virtual void startInteraction();

    virtual void doInteraction();

    // stop the interaction
    // implemented in base class void stopInteraction();

    // set a new position,normal
    virtual void updateTransform(osg::Vec3 pos);

    virtual osg::Vec3 getPosition()
    {
        return moveTransform->getMatrix().getTrans();
    };
    virtual osg::MatrixTransform *getNode()
    {
        return moveTransform;
    };
};
}

#endif
