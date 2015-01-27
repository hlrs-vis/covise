/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_TFColorWidget_H_
#define _CUI_TFColorWidget_H_

// Inspace:
#include <osgDrawObj.H>

// OSG:
#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osgText/Text>
#include <osg/LineSegment>

// CUI:
#include "CheckBox.H"
#include "Interaction.H"
#include "Dial.H"
#include "Events.H"
#include "Card.H"
#include "TextureWidget.H"
#include "Panel.H"
#include "Widget.H"

// Virvo:
#include <vvtransfunc.h>
#include <vvtfwidget.h>

class osgDrawObj;

namespace cui
{
class Interaction;
class Measure;
class InputDevice;

class TFColorWidget : public Widget, public cui::Events
{
public:
    TFColorWidget(cui::Interaction *, cui::Measure *, int);
    ~TFColorWidget();
    void setColor(osg::Vec4);
    void setVertices(osg::Vec3, osg::Vec3);
    virtual void cursorEnter(InputDevice *);
    virtual void cursorUpdate(InputDevice *);
    virtual void cursorLeave(InputDevice *);
    virtual void buttonEvent(InputDevice *, int);
    virtual void joystickEvent(InputDevice *);
    virtual void wheelEvent(InputDevice *, int);
    virtual void setSize(int);

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
