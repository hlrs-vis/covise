/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_VR_2D_TRANS_INTERACTOR_H
#define _CO_VR_2D_TRANS_INTERACTOR_H

#include <osg/Vec3>
#include <osg/MatrixTransform>
#include <cover/coVRIntersectionInteractor.h>
#include "coPlane.h"

namespace opencover
{

// selectable cylinder which can be positioned on a plane
// cylinder and plane are positioned in object coordinates

class PLUGIN_UTILEXPORT coVR2DTransInteractor : public coVRIntersectionInteractor, public coPlane
{

private:
    osg::Vec3 _diff;

protected:
    virtual void createGeometry();

public:
    // position and normal on object coordinates
    // size in world coordinates (mm)
    coVR2DTransInteractor(osg::Vec3 pos, osg::Vec3 normal, float size, coInteraction::InteractionType type, const char *iconName, const char *interactorName, enum coInteraction::InteractionPriority priority);

    virtual ~coVR2DTransInteractor();

    void startInteraction();

    void doInteraction();

    // stop the interaction
    // implemented in base class void stopInteraction();

    // set a new position,normal
    void updateTransform(osg::Vec3 pos, osg::Vec3 normal);

    osg::Vec3 getPosition()
    {
        return moveTransform->getMatrix().getTrans();
    };
};
}
#endif
