/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Point.cpp
 *
 *  Created on: Dec 7, 2010
 *      Author: tm_te
 */

#include "Point.h"
#include <cover/coVRPluginSupport.h>

#include <osg/Matrix>

Point::Point()
{
    Point(osg::Vec3(0, 0, 0));
}

Point::Point(osg::Vec3 initialPosition)
{
    position = initialPosition;
    interactorSize = cover->getSceneSize() / 15;
    interactor = new coVR3DTransInteractor(position, interactorSize, coInteraction::ButtonA, "hand", "vpTanOutInteractor", coInteraction::Medium);
    interactor->show();
    interactor->enableIntersection();
}

Point::~Point()
{
    delete interactor;
}

void Point::updatePosition(osg::Vec3 newPosition)
{
    position = newPosition;
}

void Point::preFrame()
{
    interactor->preFrame();

    if (interactor->isRunning())
    {
        osg::Matrix m = interactor->getMatrix();
        osg::Vec3 interpos = m.getTrans();

        updatePosition(interpos);
    }
}

void Point::showInteractor(bool state)
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
