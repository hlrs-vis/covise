/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CHARGED_POINT_H
#define _CHARGED_POINT_H

#include "ChargedObject.h"

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>

const float MINIMUM_DISTANCE = 0.05;

class ChargedPoint : public ChargedObject
{
public:
    ChargedPoint(std::string name, float initialCharge);
    virtual ~ChargedPoint();

    virtual osg::Vec3 getFieldAt(osg::Vec3 point);
    virtual float getPotentialAt(osg::Vec3 point);
    virtual osg::Vec4 getFieldAndPotentialAt(osg::Vec3 point);

protected:
    virtual void createGeometry(); // interactor

private:
    void activeStateChanged();
};

#endif
