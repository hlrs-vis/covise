/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_BAR_H_
#define _CUI_BAR_H_

// OSG:
#include "osg/ShapeDrawable"

// CUI:
#include "Events.h"
#include "Widget.h"

namespace cui
{
class Interaction;
class Measure;
class InputDevice;

/** 
      This class is the implementation of a cylinder of variable length.
      It can be grabbed and moved by the user.
  */
class CUIEXPORT Bar : public Widget, public cui::Events
{
public:
    Bar(cui::Interaction *, cui::Measure *);
    ~Bar();
    void setColor(osg::Vec4);
    void setVertices(osg::Vec3, osg::Vec3);
    virtual void cursorEnter(InputDevice *);
    virtual void cursorUpdate(InputDevice *);
    virtual void cursorLeave(InputDevice *);
    virtual void buttonEvent(InputDevice *, int);
    virtual void joystickEvent(InputDevice *);
    virtual void wheelEvent(InputDevice *, int);

    osg::Vec3 getRight()
    {
        return (*_vertices)[0];
    }
    osg::Vec3 getLeft()
    {
        return (*_vertices)[1];
    }
    osg::Vec3 getVector()
    {
        return _vector;
    }

protected:
    cui::Interaction *_interaction;
    cui::Measure *_measure;

    osg::Geometry *_geom;
    osg::Matrix _initWand2w;
    osg::Vec3Array *_vertices;
    osg::Vec4Array *_colors;

    osg::Vec3 _vector;
};
}
#endif
