/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SetTransformAxisEvent.h"

SetTransformAxisEvent::SetTransformAxisEvent()
{
    _type = EventTypes::SET_TRANSFORM_AXIS_EVENT;
    _hasTranslateAxis = false;
    _hasRotateAxis = false;
    _hasResetTranslate = false;
    _hasResetRotate = false;
}

SetTransformAxisEvent::~SetTransformAxisEvent()
{
}

void SetTransformAxisEvent::setTranslateAxis(osg::Vec3 axis)
{
    _translateAxis = axis;
    _hasTranslateAxis = true;
}

osg::Vec3 SetTransformAxisEvent::getTranslateAxis()
{
    return _translateAxis;
}

bool SetTransformAxisEvent::hasTranslateAxis()
{
    return _hasTranslateAxis;
}

void SetTransformAxisEvent::resetTranslate()
{
    _hasResetTranslate = true;
}

bool SetTransformAxisEvent::hasResetTranslate()
{
    return _hasResetTranslate;
}

void SetTransformAxisEvent::setRotateAxis(osg::Vec3 axis)
{
    _rotateAxis = axis;
    _hasRotateAxis = true;
}

osg::Vec3 SetTransformAxisEvent::getRotateAxis()
{
    return _rotateAxis;
}

bool SetTransformAxisEvent::hasRotateAxis()
{
    return _hasRotateAxis;
}

void SetTransformAxisEvent::resetRotate()
{
    _hasResetRotate = true;
}

bool SetTransformAxisEvent::hasResetRotate()
{
    return _hasResetRotate;
}
