/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "InteractionPoint.h"
#include <cover/coVRPluginSupport.h>

InteractionPoint::InteractionPoint()
{
    InteractionPoint(osg::Vec3(0, 0, 0));
}

InteractionPoint::InteractionPoint(osg::Vec3 initialPosition)
{
    position = initialPosition;
    interactorSize = cover->getSceneSize() / 30;
    interactor = new coVR3DTransInteractor(position, interactorSize, coInteraction::ButtonA, "hand", "vpTanOutInteractor", coInteraction::Medium);
    interactor->show();
    interactor->enableIntersection();
}

InteractionPoint::~InteractionPoint()
{
    delete interactor;
}

void InteractionPoint::updatePosition(osg::Vec3 newPosition)
{
    position = newPosition;
}
void InteractionPoint::preFrame()
{
    interactor->preFrame();

    if (interactor->isRunning())
    {
        interact = true;
        osg::Matrix m = interactor->getMatrix();
        interpos = m.getTrans();
        updatePosition(interpos);
    }
    else
    {
        interact = false;
    }
}
void InteractionPoint::showInteractor(bool state)
{
    if (state == true)
    {
        interactor->show();
        interactor->enableIntersection();
    }
    else
    {
        interactor->hide();
        interactor->disableIntersection();
    }
}
