/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_INPUT_DEVICE_H_
#define _CUI_INPUT_DEVICE_H_

// C++:
#include <list>

// OSG:
#include <osg/Matrix>
#include <osg/ShapeDrawable>

// CUI:
#include "Widget.h"
#include "WidgetInfo.h"

namespace cui
{
class Events;
class Interaction;
class Marker;
class Paintbrush;

/** Intersection information.
   */
class CUIEXPORT IsectInfo
{
public:
    bool found; ///< false: no intersection found, rest of attributes are undefined
    WidgetInfo widget; ///< intersected widget
    osg::Vec3 point; ///< intersection point
    osg::Vec3 normal; ///< intersection normal

    IsectInfo &operator=(const IsectInfo &right)
    {
        if (&right == this)
            return *this;
        found = right.found;
        widget._events = right.widget._events;
        widget._widget = right.widget._widget;
        widget._geodeList = right.widget._geodeList;
        widget._box = right.widget._box;
        point = right.point;
        normal = right.normal;
        return *this;
    }
};

/** This class provides information for the state
    of input devices like wand, head tracker, etc.
    The device can have up to 6 degrees of freedom (DOF),
    a 2D joystick, an arbitrary number of buttons with an arbitrary
    number of states, and a mouse wheel.
    Multiple types of cursors are supported. The default cursor
    is a laser pointer.
  */
class CUIEXPORT InputDevice
{
public:
    static const float DEFAULT_LASER_LENGTH;
    static const float DEFAULT_LASER_THICKNESS;
    enum CursorType
    {
        NONE, ///< no visible cursor
        LASER, ///< line long enough to reach any point in the scene
        POINTER, ///< line with ball at tip
        CONE_MARKER, ///< line with cone-shaped marker at tip
        SPHERE_MARKER, ///< line with sphere-shaped marker at tip
        BOX_MARKER, ///< line with box-shaped marker at tip
        INVISIBLE, ///< invisible, infinitely long laser
        GAZE, ///< normally invisible, drawing crosshair cursor if on UI widget
        CONE_BRUSH, ///< line with cone-shaped brush at tip
        SPHERE_BRUSH, ///< line with sphere-shaped brush at tip
        BOX_BRUSH ///< line with box-shaped brush at tip
    };
    enum DeviceType
    {
        HEAD = 0, ///< numbers are important for log file!
        WAND_R,
        WAND_L,
        MOUSE
    };
    cui::Interaction *_interaction;

    InputDevice(cui::Interaction *, osg::Group *, int);
    virtual ~InputDevice();
    virtual bool anyButtonPressed();
    virtual int getButtonState(int);
    virtual int getLastButtonState(int);
    virtual bool buttonJustPressed(int);
    virtual void setButtonState(int, int);
    virtual float getJoystick(int);
    virtual void setJoystick(int, float);
    virtual bool isJoystickCentered();
    virtual void setI2W(const osg::Matrix &);
    virtual osg::Matrix getI2W();
    virtual osg::Matrix getLastI2W();
    virtual osg::Vec3 getCursorPos();
    virtual osg::Vec3 getCursorDir();
    virtual osg::Matrix getPressedI2W(int);
    virtual void buttonStateChanged(int, int);
    virtual void joystickValueChanged(float, float);
    virtual void wheelTurned(int);
    virtual bool action();
    virtual void createCursorGeometry();
    virtual void setCursorType(CursorType);
    virtual CursorType getCursorType();
    virtual CursorType getLastCursorType();
    virtual void updateCursorGeometry();
    virtual void setLaserLength(float);
    virtual void setLaserThickness(float);
    virtual void setPointerLength(float);
    virtual float getPointerLength();
    virtual void setPointerThicknessFactor(float);
    virtual void setBallRadiusFactor(float);
    virtual void setGazeGeometry();
    virtual osg::Vec3 getIsectPoint();
    virtual cui::Marker *getConeMarker();
    virtual cui::Marker *getSphereMarker();
    virtual cui::Marker *getBoxMarker();
    virtual cui::Paintbrush *getConeBrush();
    virtual cui::Paintbrush *getSphereBrush();
    virtual cui::Paintbrush *getBoxBrush();
    virtual void setConeMarkerPosition(float, float, float);
    virtual void setSphereMarkerPosition(float, float, float);
    virtual void setBoxMarkerPosition(float, float, float);
    virtual void setConeBrushPosition(float, float, float);
    virtual void setSphereBrushPosition(float, float, float);
    virtual void setBoxBrushPosition(float, float, float);
    virtual void widgetDeleted(osg::Node *);
    virtual void widgetDeleted();
    virtual osg::Geode *getIsectGeode();
    virtual Widget *getIsectWidget();

protected:
    int _numButtons; ///< number of buttons on device
    osg::Matrix _lastI2W; ///< device matrix in previous frame
    int *_buttonState; ///< current button state: 0=released, 1=pressed, 2=popped through (only for pop-through buttons)
    int *_lastButtonState; ///< button state at last call to 'action': 0=released, 1=pressed, 2=popped through (only for pop-through buttons)
    float _joystick[2]; ///< x/y joystick/trackball position
    osg::Matrix *_pressedI2W; ///< device matrix at moment of button press, one for each button
    WidgetInfo _currentWidget; ///< widget currently selected by cursor, or last selected when button was pressed
    osg::Group *_worldRoot;
    IsectInfo _isect, _lastIsect;
    osg::ref_ptr<osg::MatrixTransform> _cursorNode;
    osg::Box *_laserBox; ///< laser geometry
    osg::Box *_pointerBox; ///< pointer geometry
    osg::Sphere *_pointerBall; ///< ball at end of pointer
    osg::ShapeDrawable *_sphereDrawable; ///< drawable of ball at end of pointer
    osg::ShapeDrawable *_pointerBoxDrawable;
    osg::ShapeDrawable *_laserBoxDrawable;
    osg::ref_ptr<osg::MatrixTransform> _coneMarkerXF;
    osg::ref_ptr<osg::MatrixTransform> _sphereMarkerXF;
    osg::ref_ptr<osg::MatrixTransform> _boxMarkerXF;
    osg::ref_ptr<osg::MatrixTransform> _coneBrushXF;
    osg::ref_ptr<osg::MatrixTransform> _sphereBrushXF;
    osg::ref_ptr<osg::MatrixTransform> _boxBrushXF;
    osg::ref_ptr<osg::MatrixTransform> _gazeCursorXF;
    osg::ref_ptr<osg::Switch> _cursorSwitch;
    CursorType _lastCursorType; ///< last selected cursor type
    CursorType _cursorType; ///< currently selected cursor type
    cui::Marker *_coneMarker;
    cui::Marker *_sphereMarker;
    cui::Marker *_boxMarker;
    cui::Paintbrush *_coneBrush;
    cui::Paintbrush *_boxBrush;
    cui::Paintbrush *_sphereBrush;
    float _laserLength;
    float _laserThickness;
    float _pointerLength;
    float _pointerThickFactor;
    float _ballRadiusFactor;
    Events *_lastWheelEvent;

    osg::Vec3 amplifyHeadDirection(osg::Vec3 &, bool, bool);
    osg::Geometry *createTriangle(osg::Vec4 &, osg::Vec3 &, osg::Vec3 &, osg::Vec3 &);
    osg::Geode *createSphere(float);
};
}
#endif
