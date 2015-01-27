/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_Paintbrush_H_
#define _CUI_Paintbrush_H_

// OSG:
#include <osg/ShapeDrawable>

// CUI:
#include "Widget.h"
#include "Events.h"
#include "InputDevice.h"

namespace cui
{
class Interaction;
class Measure;
class PaintbrushListener;

/** This is an implementation of a cone-shaped Paintbrush. It provides a pop-up
    menu to change its color, size, and to remove it.
    @author Devon Penney
    @author Jurgen Schulze
  */
class CUIEXPORT Paintbrush : public Widget, public Events
{
public:
    enum GeomType
    {
        CONE,
        SPHERE,
        BOX
    };
    Paintbrush(GeomType, Interaction * = NULL);
    Paintbrush(GeomType, Interaction *, float, osg::Vec4 &);
    virtual ~Paintbrush();

    // Inherited:
    virtual void cursorEnter(InputDevice *);
    virtual void cursorUpdate(InputDevice *);
    virtual void cursorLeave(InputDevice *);
    virtual void buttonEvent(InputDevice *, int);
    virtual void joystickEvent(InputDevice *);
    virtual void wheelEvent(InputDevice *, int);

    // Local:
    virtual void init(GeomType, Interaction *, float, osg::Vec4 &);
    virtual void setSize(float);
    virtual float getSize();
    virtual void setColor(osg::Vec4);
    virtual void setHue(float);
    virtual float getHue();
    virtual void setOpacity(float);
    virtual float getOpacity();
    virtual osg::Vec4 getColor();
    virtual void setPosition(osg::Vec3);
    virtual osg::Vec3 getPosition();
    virtual osg::Vec3 getDirection();
    virtual void addPaintbrushListener(PaintbrushListener *);
    virtual void invertColor();

protected:
    static const float RATIO_LENGTH2RADIUS;
    osg::Matrix _lastWand2w;
    osg::ShapeDrawable *_shapeDrawable;
    osg::Cone *_coneShape;
    osg::Sphere *_sphereShape;
    osg::Box *_boxShape;
    Interaction *_interaction;
    std::list<PaintbrushListener *> _listeners;
    osg::ref_ptr<osg::MatrixTransform> _PaintbrushXF;
    cui::InputDevice::CursorType _prevType; ///< pointer style when pointer entered box
    GeomType _gt;
    float _size;
    osg::Vec4 _color;

    osg::Geode *createCone(osg::Vec4 &);
    osg::Geode *createSphere(osg::Vec4 &);
    osg::Geode *createBox(osg::Vec4 &);
};

class CUIEXPORT PaintbrushListener
{
public:
    virtual ~PaintbrushListener()
    {
    }
    virtual void PaintbrushEvent(Paintbrush *, int, int) = 0;
};
}

#endif
