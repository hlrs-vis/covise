/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SET_TRANSFORM_AXIS_EVENT_H
#define SET_TRANSFORM_AXIS_EVENT_H

#include "Event.h"

#include <osg/Vec3>

class SetTransformAxisEvent : public Event
{
public:
    SetTransformAxisEvent();
    virtual ~SetTransformAxisEvent();

    void setTranslateAxis(osg::Vec3 axis);
    osg::Vec3 getTranslateAxis();
    bool hasTranslateAxis();
    void resetTranslate();
    bool hasResetTranslate();

    void setRotateAxis(osg::Vec3 axis);
    osg::Vec3 getRotateAxis();
    bool hasRotateAxis();
    void resetRotate();
    bool hasResetRotate();

private:
    osg::Vec3 _translateAxis;
    osg::Vec3 _rotateAxis;
    bool _hasTranslateAxis;
    bool _hasRotateAxis;
    bool _hasResetTranslate;
    bool _hasResetRotate;
};

#endif
