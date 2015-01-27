/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_ARROW_H_
#define _CUI_ARROW_H_

// OSG:
#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osg/BoundingBox>

// CUI:
#include "Widget.h"
#include "Events.h"

namespace cui
{
class Interaction;
class Measure;

/** 
    This is an implementation of a cone-shaped arrow head. It can be used
    at the end of a line.
*/
class CUIEXPORT Arrow : public Widget, public Events
{
public:
    Arrow(Interaction *interaction, Measure *);
    ~Arrow();
    void setColor(osg::Vec4);
    osg::Vec3 getTipPos();

    virtual void cursorEnter(InputDevice *);
    virtual void cursorUpdate(InputDevice *);
    virtual void cursorLeave(InputDevice *);
    virtual void buttonEvent(InputDevice *, int);
    virtual void joystickEvent(InputDevice *);
    virtual void wheelEvent(InputDevice *, int);

protected:
    Interaction *_interaction;
    osg::Matrix _initWand2w;
    osg::ShapeDrawable *_drawable;
    osg::Geode *_geode;
    osg::Vec3 _tip;
    osg::Vec4 _color;
    osg::Vec4 _highlightCol;
    Measure *_measure;

    bool isTipInside(osg::Matrix);
};
}

#endif
