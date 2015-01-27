/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef POINT_H_
#define POINT_H_

#include <osg/Vec3>
#include <osg/Matrix>
#include <PluginUtil/coVR3DTransInteractor.h>
#include <cover/coInteractor.h>

using namespace vrui;
using namespace opencover;

class InteractionPoint
{
private:
    osg::Vec3 position;
    coVR3DTransInteractor *interactor;
    float interactorSize;

public:
    bool interact;
    osg::Vec3 interpos;
    InteractionPoint();
    InteractionPoint(osg::Vec3 initialPosition);
    ~InteractionPoint();
    osg::Vec3 getPosition()
    {
        return position;
    }
    coVR3DTransInteractor *getInteractor()
    {
        return interactor;
    }
    void updatePosition(osg::Vec3 newPosition);
    void preFrame();
    void deleteInteractor();
    void showInteractor(bool state);
};

#endif /* POINT_H_ */
