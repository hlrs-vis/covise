/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_RECTANGLE_H
#define _CUI_RECTANGLE_H

#include <iostream>
#include <fstream>

using namespace std;

// OSG
#include <osg/Geometry>
#include <osg/Vec4>
#include <osg/Vec3>

// CUI:
#include "Widget.h"
#include "Events.h"
#include "Interaction.h"

namespace cui
{
class RectangleListener;

class Rectangle : public Widget, public Events
{
public:
    Rectangle(Interaction *);
    ~Rectangle();

    void addListener(RectangleListener *);

    virtual void cursorEnter(InputDevice *);
    virtual void cursorUpdate(InputDevice *);
    virtual void cursorLeave(InputDevice *);
    virtual void buttonEvent(InputDevice *, int);
    virtual void joystickEvent(InputDevice *);
    virtual void wheelEvent(InputDevice *, int);

    void getCorners(float[4][3]);

private:
    osg::Vec3Array *_vertices;
    osg::Vec3Array *_corners;
    osg::Vec3Array *_edges;
    osg::Vec4Array *_rectColor;
    osg::Matrix _initI2W;
    osg::Geode *_rectGeode;
    osg::Geometry *_rectGeom;
    osg::Geode *_sphere1Geode;
    osg::Geode *_sphere2Geode;
    osg::ShapeDrawable *_sphere1Drawable;
    osg::ShapeDrawable *_sphere2Drawable;
    osg::Sphere *_sphere1;
    osg::Sphere *_sphere2;
    osg::Vec4 _sphere1Color;
    osg::Vec4 _sphere2Color;

    list<RectangleListener *> _listeners;
    std::list<RectangleListener *>::iterator _iter;

    float plane[4];

    Interaction *_interaction;

    void updateRectangle();
    void scaleRectangle(osg::Vec3, osg::Geode *);
};

class RectangleListener
{
public:
    virtual ~RectangleListener()
    {
    }
    virtual void rectangleUpdate() = 0;
};
}

#endif
