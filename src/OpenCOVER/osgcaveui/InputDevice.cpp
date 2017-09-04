/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <fstream>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <algorithm>
using namespace std;

// C++:
#include <assert.h>

// OSG:
#include <osg/MatrixTransform>
#include <osg/Switch>
#include <osg/Geometry>

// Local:
#include "InputDevice.h"
#include "Interaction.h"
#include "Events.h"
#include "Widget.h"
#include "Marker.h"
#include "Paintbrush.h"
#include "CUI.h"

using namespace osg;
using namespace cui;

// default length of laser pointer. should appear as infinitely long
const float InputDevice::DEFAULT_LASER_LENGTH = 3000.0f;
// default thickness of laser pointer
const float InputDevice::DEFAULT_LASER_THICKNESS = 10.0f;

InputDevice::InputDevice(Interaction *interaction, Group *worldRoot, int numButtons)
{
    int i;

    _interaction = interaction;
    _worldRoot = worldRoot;
    _numButtons = numButtons;
    _buttonState = new int[_numButtons];
    _lastButtonState = new int[_numButtons];
    _pressedI2W = new Matrix[_numButtons];
    _coneMarkerXF = new MatrixTransform();
    _sphereMarkerXF = new MatrixTransform();
    _boxMarkerXF = new MatrixTransform();
    _coneBrushXF = new MatrixTransform();
    _sphereBrushXF = new MatrixTransform();
    _boxBrushXF = new MatrixTransform();
    _gazeCursorXF = new MatrixTransform();
    _lastWheelEvent = NULL;

    for (i = 0; i < _numButtons; ++i)
    {
        _buttonState[i] = 0;
        _lastButtonState[i] = 0;
    }

    for (i = 0; i < 2; ++i)
    {
        _joystick[i] = 0.0f;
    }

    // Init cursor parameters:
    _laserLength = DEFAULT_LASER_LENGTH;
    _laserThickness = DEFAULT_LASER_THICKNESS;
    _pointerLength = 6.0f;
    _pointerThickFactor = 0.1f;
    _ballRadiusFactor = 0.001f;

    // Create cursor geometry:
    createCursorGeometry();
    _cursorNode = new MatrixTransform();
    //_cursorNode->setNodeMask(~1);                   // ignore in intersection test
    _cursorNode->setNodeMask(_cursorNode->getNodeMask() & (~2));
    _cursorNode->addChild(_cursorSwitch.get());
    _worldRoot->addChild(_cursorNode.get());
    setCursorType(NONE);
}

InputDevice::~InputDevice()
{
    delete[] _buttonState;
    delete[] _lastButtonState;
    delete[] _pressedI2W;
}

/** @return true if any button is pressed (i.e., buttonState!=0)
 */
bool InputDevice::anyButtonPressed()
{
    int i;
    for (i = 0; i < _numButtons; ++i)
    {
        if (_buttonState[i] != 0)
            return true;
    }
    return false;
}

/** @param coordinate 0 for x, 1 for y
 */
float InputDevice::getJoystick(int coordinate)
{
    assert(coordinate == 0 || coordinate == 1);
    return _joystick[coordinate];
}

/** @param coordinate 0 for x, 1 for y
  @ param value new joystick value
*/
void InputDevice::setJoystick(int coordinate, float value)
{
    assert(coordinate == 0 || coordinate == 1);
    // Constrain values to -1..1:
    if (value < -1.0f)
        value = -1.0f;
    else if (value > 1.0f)
        value = 1.0f;
    _joystick[coordinate] = value;
}

/** Returns true if joystick/trackball is in neutral position.
 */
bool InputDevice::isJoystickCentered()
{
    if (_joystick[0] == 0.0f && _joystick[1] == 0.0f)
        return true;
    else
        return false;
}

/** @return button state for a particular button
  @param button button ID [0.._numButtons-1]
*/
int InputDevice::getButtonState(int button)
{
    if (button >= 0 && button < _numButtons)
    {
        return _buttonState[button];
    }
    else
        return -1;
}

/** @return previous button state for a particular button
  @param button button ID [0.._numButtons-1]
*/
int InputDevice::getLastButtonState(int button)
{
    if (button >= 0 && button < _numButtons)
    {
        return _lastButtonState[button];
    }
    else
        return -1;
}

/** @return true if a button has just been pressed since the
  last call to 'action'
  @param button button ID [0.._numButtons-1]
*/
bool InputDevice::buttonJustPressed(int button)
{
    if (button >= 0 && button < _numButtons)
    {
        return ((_lastButtonState[button] == 0) && (_buttonState[button] > 0));
    }
    else
        return false;
}

void InputDevice::setButtonState(int button, int newState)
{
    _buttonState[button] = newState;
    if (newState > 0)
    {
        _pressedI2W[button] = _cursorNode->getMatrix();
    }
}

void InputDevice::setI2W(const osg::Matrix &i2w)
{
    _lastI2W = _cursorNode->getMatrix();
    _cursorNode->setMatrix(i2w);
}

/** @return input device to world matrix
 */
Matrix InputDevice::getI2W()
{
    return _cursorNode->getMatrix();
}

/** @return input device to world matrix from previous frame
 */
Matrix InputDevice::getLastI2W()
{
    return _lastI2W;
}

/** @return wand position in world space
*/
Vec3 InputDevice::getCursorPos()
{
    return _cursorNode->getMatrix().getTrans();
}

/** @return normalized wand direction in world space
 */
Vec3 InputDevice::getCursorDir()
{
    const double *dWandXF = _cursorNode->getMatrix().ptr();
    Vec3 wDir(dWandXF[8], dWandXF[9], dWandXF[10]); // 3rd row equals z direction
    wDir.normalize();
    return wDir;
}

Matrix InputDevice::getPressedI2W(int button)
{
    return _pressedI2W[button];
}

void InputDevice::setCursorType(CursorType type)
{
    _lastCursorType = getCursorType();
    _cursorType = type;
    _cursorSwitch->setAllChildrenOff();
    switch (_cursorType)
    {
    case NONE:
        break;
    case LASER:
        _cursorSwitch->setValue(0, true);
        break;
    case POINTER:
        _cursorSwitch->setValue(1, true);
        _cursorSwitch->setValue(2, true);
        break;
    case CONE_MARKER:
        _cursorSwitch->setValue(1, true);
        _cursorSwitch->setValue(3, true);
        break;
    case SPHERE_MARKER:
        _cursorSwitch->setValue(1, true);
        _cursorSwitch->setValue(4, true);
        break;
    case INVISIBLE:
        break;
    case GAZE:
        _cursorSwitch->setValue(5, true);
        break;
    case BOX_MARKER:
        _cursorSwitch->setValue(1, true);
        _cursorSwitch->setValue(6, true);
        break;
    case CONE_BRUSH:
        _cursorSwitch->setValue(1, true);
        _cursorSwitch->setValue(7, true);
        break;
    case SPHERE_BRUSH:
        _cursorSwitch->setValue(1, true);
        _cursorSwitch->setValue(8, true);
        break;
    case BOX_BRUSH:
        _cursorSwitch->setValue(1, true);
        _cursorSwitch->setValue(9, true);
        break;
    default:
        assert(0);
        break;
    }
}

InputDevice::CursorType InputDevice::getCursorType()
{
    return _cursorType;
}

InputDevice::CursorType InputDevice::getLastCursorType()
{
    return _lastCursorType;
}

/** Called when a button state changes.
@param button physical button pressed by the user
@param newState new state of physical button
*/
void InputDevice::buttonStateChanged(int button, int newState)
{
    DeviceType dt = MOUSE;

    // Log the button press:
    if (this == _interaction->_head)
        dt = HEAD;
    else if (this == _interaction->_wandR)
        dt = WAND_R;
    else if (this == _interaction->_wandL)
        dt = WAND_L;
    else if (this == _interaction->_mouse)
        dt = MOUSE;
    if (_interaction->getLogFile())
    {
        _interaction->getLogFile()->addButtonStateLog(int(dt), button, newState);
    }

    // Process the button press:
    if (button >= 0 && button < _numButtons)
    {
        if (getButtonState(button) != newState)
        {
            setButtonState(button, newState);

            std::list<WidgetInfo *>::iterator iter;
            for (iter = _interaction->_anyButtonListeners.begin(); iter != _interaction->_anyButtonListeners.end(); iter++)
                (*iter)->_events->buttonEvent(this, button);

            if (_currentWidget._events)
            {
                _currentWidget._events->buttonEvent(this, button);
            }
        }
    }
}

/** Called when the 2D joystick/trackball on the input device changed
  its value.
*/
void InputDevice::joystickValueChanged(float x, float y)
{
    setJoystick(0, x);
    setJoystick(1, y);
    if (_currentWidget._events)
        _currentWidget._events->joystickEvent(this);
}

void InputDevice::wheelTurned(int direction)
{
    cerr << "wheelTurned" << endl;
    if (_currentWidget._events && !_currentWidget._box)
    {
        _currentWidget._events->wheelEvent(this, direction);
        _lastWheelEvent = _currentWidget._events;
    }
    else if (_lastWheelEvent) // snarf last used widget
    {
        _lastWheelEvent->wheelEvent(this, direction);
    }
}

/** A good value for the cave is 100.0.
  The laser pointer should appear as if it's infinitely long.
  @param length [feet]
*/
void InputDevice::setLaserLength(float length)
{
    _laserLength = length;
    updateCursorGeometry();
}

/** A good value for the cave is 3.0
  @param length [feet]
*/
void InputDevice::setPointerLength(float length)
{
    _pointerLength = length;
    updateCursorGeometry();
}

float InputDevice::getPointerLength()
{
    return _pointerLength;
}

Marker *InputDevice::getConeMarker()
{
    return _coneMarker;
}

Marker *InputDevice::getSphereMarker()
{
    return _sphereMarker;
}

Marker *InputDevice::getBoxMarker()
{
    return _boxMarker;
}

Paintbrush *InputDevice::getConeBrush()
{
    return _coneBrush;
}

Paintbrush *InputDevice::getSphereBrush()
{
    return _sphereBrush;
}

Paintbrush *InputDevice::getBoxBrush()
{
    return _boxBrush;
}

/// A good value for the cave is 0.005
void InputDevice::setPointerThicknessFactor(float factor)
{
    _pointerThickFactor = factor;
    updateCursorGeometry();
}

/// A good value for the cave is 0.01
void InputDevice::setLaserThickness(float thick)
{
    _laserThickness = thick;
    updateCursorGeometry();
}

/// A good value for the cave is 0.02
void InputDevice::setBallRadiusFactor(float factor)
{
    _ballRadiusFactor = factor;
    updateCursorGeometry();
}

void InputDevice::setGazeGeometry()
{
    // Update gaze cursor
    if (!_interaction->getGazeInteraction())
    {
        _gazeCursorXF->setNodeMask(0); // hide
    }
}

/** Compute intersection of cursor with intersectable world.
  @return true if intersection was found
*/
bool InputDevice::action()
{
    Vec3 wStart, wEnd;

    // Compute start and end of laser:
    Matrix i2w = getI2W();
    Vec3 wPos = i2w.getTrans();
    double *dInputXF = i2w.ptr();
    //  Vec3 wDir(dInputXF[8], dInputXF[9], dInputXF[10]); // 3rd row equals z direction
    // 2nd row equals y direction
    Vec3 wDir(dInputXF[4], dInputXF[5], dInputXF[6]);

    if (this == _interaction->_head) // amplify head orientation
    {
        amplifyHeadDirection(wDir, true, true);
    }

    wDir.normalize();
    switch (_cursorType)
    {
    case LASER:
        wStart = wPos;
        wEnd = wPos + wDir * _laserLength;
        break;
    case POINTER:
        //wStart = wPos;
        //wEnd   = wPos + wDir * _pointerLength;
        wStart = wPos + wDir * (_pointerLength - _pointerBall->getRadius());
        wEnd = wPos + wDir * (_pointerLength + _pointerBall->getRadius());
        break;
    case CONE_MARKER:
        wStart = wPos + wDir * (_pointerLength - _coneMarker->getSize());
        wEnd = wPos + wDir * (_pointerLength + _coneMarker->getSize());
        break;
    case SPHERE_MARKER:
        wStart = wPos + wDir * (_pointerLength - _sphereMarker->getSize());
        wEnd = wPos + wDir * (_pointerLength + _sphereMarker->getSize());
        break;
    case INVISIBLE:
        wStart = wPos;
        wEnd = wPos + wDir * _laserLength;
        break;
    case GAZE:
        wStart = wPos;
        wEnd = wPos + wDir * _laserLength;
        break;
    case BOX_MARKER:
        wStart = wPos + wDir * (_pointerLength - _boxMarker->getSize());
        wEnd = wPos + wDir * (_pointerLength + _boxMarker->getSize());
        break;
    case CONE_BRUSH:
        wStart = wPos + wDir * (_pointerLength - _coneBrush->getSize());
        wEnd = wPos + wDir * (_pointerLength + _coneBrush->getSize());
    case SPHERE_BRUSH:
        wStart = wPos + wDir * (_pointerLength - _sphereBrush->getSize());
        wEnd = wPos + wDir * (_pointerLength + _sphereBrush->getSize());
    case BOX_BRUSH:
        wStart = wPos + wDir * (_pointerLength - _boxBrush->getSize());
        wEnd = wPos + wDir * (_pointerLength + _boxBrush->getSize());
    default:
        wStart = wPos;
        wEnd = wPos + wDir * 0.001f;
        break;
    }

    //_interaction->getFirstIntersection(wStart, wEnd, _isect);
    _interaction->getFirstIntersection(wStart, wEnd, _isect);

    // Process pickbox events only with main wand:
    if (this != _interaction->_wandR && !_currentWidget._box)
    {
        //    return _isect.found;
    }

    // Process user input:
    if (anyButtonPressed()) // button pressed: update currentWidget, not the intersected widget
    {
        if (_currentWidget._events)
        {
            _currentWidget._events->cursorUpdate(this);
        }
    }
    else if (_isect.found) // no button pressed, and intersecting: update _currentWidget no matter what
    {
        if (_isect.widget._events == _currentWidget._events && _isect.widget._widget == _currentWidget._widget && _isect.widget._box == _currentWidget._box) // same object intersected as before?
        {
            if (_currentWidget._events)
                _currentWidget._events->cursorUpdate(this);
        }
        else
        {
            if (_currentWidget._events)
            {
                _currentWidget._events->cursorLeave(this);
                //std::cerr << "cursor leave event called - Pointer: " << _currentWidget._widget << " " << _currentWidget._box << endl;
            }
            if (_isect.widget._events)
            {
                _isect.widget._events->cursorEnter(this);
                //std::cerr << "cursor enter event called - Pointer: " << _isect.widget._widget << " " << _isect.widget._box << endl;
            }
        }
        _currentWidget._events = _isect.widget._events;
        _currentWidget._widget = _isect.widget._widget;
        _currentWidget._geodeList = _isect.widget._geodeList;
        _currentWidget._box = _isect.widget._box;
    }
    else // nothing intersected, no button pressed: reset currentWidget
    {
        if (_currentWidget._events || _currentWidget._box)
        {
            if (_currentWidget._events)
            {
                _currentWidget._events->cursorLeave(this);
                //std::cerr << "no intersection - cursor leave event called - Pointer: " << _currentWidget._widget << " " << _currentWidget._box << endl;
            }
            _currentWidget.reset();
        }
    }

    // Update gaze cursor:
    if (_cursorType == GAZE)
    {
        if (_isect.found && !_isect.widget._box) // is gaze intersecting a UI widget
        {
            Matrix w2i = Matrix::inverse(getI2W());
            Vec3 iIsectPoint = _isect.point * w2i;

            Matrix mat;
            mat.setTrans(iIsectPoint);
            _gazeCursorXF->setMatrix(mat);

            _gazeCursorXF->setNodeMask(~1); // show, but ignore in intersection test
        }
        else
        {
            _gazeCursorXF->setNodeMask(0); // hide
        }
    }
    _lastIsect = _isect;

    int button;
    for (button = 0; button < _numButtons; ++button)
    {
        _lastButtonState[button] = _buttonState[button];
    }

    return _isect.found;
}

void InputDevice::createCursorGeometry()
{
    // Create laser pointer:
    _laserBox = new Box();
    _laserBoxDrawable = new ShapeDrawable(_laserBox);
    _laserBoxDrawable->setColor(Widget::COL_WHITE);
    _laserBoxDrawable->setUseDisplayList(false); // turn off display list so that we can change the pointer length
    Geode *laserGeode = new Geode();
    laserGeode->addDrawable(_laserBoxDrawable);

    // Create pointer for use with ball or marker:
    _pointerBox = new Box();
    _pointerBoxDrawable = new ShapeDrawable(_pointerBox);
    _pointerBoxDrawable->setColor(Widget::COL_WHITE);
    _pointerBoxDrawable->setUseDisplayList(false); // turn off display list so that we can change the pointer length
    Geode *pointerGeode = new Geode();
    pointerGeode->addDrawable(_pointerBoxDrawable);

    // Create ball:
    _pointerBall = new Sphere();
    _sphereDrawable = new ShapeDrawable(_pointerBall);
    _sphereDrawable->setColor(Widget::COL_RED);
    _sphereDrawable->setUseDisplayList(false); // turn off display list so that we can change the pointer length
    Geode *sphereGeode = new Geode();
    sphereGeode->addDrawable(_sphereDrawable);

    // Create cone-shaped marker:
    _coneMarker = new Marker(Marker::CONE, _interaction);
    _coneMarkerXF->addChild(_coneMarker->getNode());

    // Create sphere-shaped marker:
    _sphereMarker = new Marker(Marker::SPHERE, _interaction);
    _sphereMarkerXF->addChild(_sphereMarker->getNode());

    // Create box-shaped marker:
    _boxMarker = new Marker(Marker::BOX, _interaction);
    _boxMarkerXF->addChild(_boxMarker->getNode());

    // Create cone-shaped brush:
    _coneBrush = new Paintbrush(Paintbrush::CONE, _interaction);
    _coneBrushXF->addChild(_coneBrush->getNode());

    // Create sphere-shaped brush:
    _sphereBrush = new Paintbrush(Paintbrush::SPHERE, _interaction);
    _sphereBrushXF->addChild(_sphereBrush->getNode());

    // Create box-shaped brush:
    _boxBrush = new Paintbrush(Paintbrush::BOX, _interaction);
    _boxBrushXF->addChild(_boxBrush->getNode());

    // Create crosshairs:
    Vec3 p1, p2, p3;
    Geode *crosshairsGeode = new Geode();
    Vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
    p1.set(0.1f, 0.0f, 0.0f);
    p2.set(0.3f, 0.2f, 0.0f);
    p3.set(0.3f, -0.2f, 0.0f);
    crosshairsGeode->addDrawable(createTriangle(color, p1, p2, p3));
    p1.set(-0.1f, 0.0f, 0.0f);
    p2.set(-0.3f, -0.2f, 0.0f);
    p3.set(-0.3f, 0.2f, 0.0f);
    crosshairsGeode->addDrawable(createTriangle(color, p1, p2, p3));
    p1.set(0.0f, -0.1f, 0.0f);
    p2.set(0.2f, -0.3f, 0.0f);
    p3.set(-0.2f, -0.3f, 0.0f);
    crosshairsGeode->addDrawable(createTriangle(color, p1, p2, p3));
    p1.set(0.0f, 0.1f, 0.0f);
    p2.set(-0.2f, 0.3f, 0.0f);
    p3.set(0.2f, 0.3f, 0.0f);
    crosshairsGeode->addDrawable(createTriangle(color, p1, p2, p3));
    crosshairsGeode->getOrCreateStateSet()->setMode(GL_LIGHTING, StateAttribute::OFF);
    //  _gazeCursorXF->addChild(crosshairsGeode);
    _gazeCursorXF->setNodeMask(0); // hide by default

    // Create sphere:
    Geode *gazeSphereGeode = createSphere(0.05f);
    _gazeCursorXF->addChild(gazeSphereGeode);

    // Apply current size values:
    updateCursorGeometry();

    // Add to switch:
    _cursorSwitch = new Switch();
    _cursorSwitch->addChild(laserGeode);
    _cursorSwitch->addChild(pointerGeode);
    _cursorSwitch->addChild(sphereGeode);
    _cursorSwitch->addChild(_coneMarkerXF.get());
    _cursorSwitch->addChild(_sphereMarkerXF.get());
    _cursorSwitch->addChild(_gazeCursorXF.get());
    _cursorSwitch->addChild(_boxMarkerXF.get());
    _cursorSwitch->addChild(_coneBrushXF.get());
    _cursorSwitch->addChild(_sphereBrushXF.get());
    _cursorSwitch->addChild(_boxBrushXF.get());

    // Initialize switch:
    _cursorSwitch->setSingleChildOn(0);
}

Geode *InputDevice::createSphere(float radius)
{
    Vec3 sphereCenter(0.0f, 0.0f, 0.0f);
    Sphere *sphere = new Sphere(sphereCenter, radius);
    ShapeDrawable *drawable = new ShapeDrawable(sphere);
    drawable->setColor(Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    Geode *geode = new Geode();
    geode->addDrawable(drawable);
    return geode;
}

/// Create a triangle
Geometry *InputDevice::createTriangle(Vec4 &col, Vec3 &p1, Vec3 &p2, Vec3 &p3)
{
    Geometry *geom = new Geometry();

    // Create vertices:
    Vec3Array *vertices = new Vec3Array(3);
    (*vertices)[0] = p1;
    (*vertices)[1] = p2;
    (*vertices)[2] = p3;
    geom->setVertexArray(vertices);

    // Create normals:
    Vec3Array *normals = new Vec3Array(1);
    (*normals)[0].set(0.0f, 0.0f, -1.0f);
    geom->setNormalArray(normals);
    geom->setNormalBinding(Geometry::BIND_OVERALL);

    // Create colors:
    Vec4Array *colors = new Vec4Array(1);
    (*colors)[0].set(col[0], col[1], col[2], col[3]);
    geom->setColorArray(colors);
    geom->setColorBinding(Geometry::BIND_OVERALL);

    // Create triangle:
    geom->addPrimitiveSet(new DrawArrays(GL_TRIANGLES, 0, 3));

    return geom;
}

void InputDevice::updateCursorGeometry()
{
    // Update laser length:
    _laserBox->set(Vec3(0.0f, _laserLength / 2.0f, 0.0f),
                   Vec3(_laserThickness / 2.0f, // set command expects half lengths
                        _laserLength / 2.0f,
                        _laserThickness / 2.0f));
    _laserBoxDrawable->dirtyBound();

    // Update pointer length:
    float thickness = _pointerThickFactor * _pointerLength;
    thickness = std::min(thickness, _laserThickness);
    //  _pointerBox->set(Vec3(0.0f, 0.0f, _pointerLength / 2.0f),
    //    Vec3(thickness / 2.0f, thickness / 2.0f, _pointerLength / 2.0f));
    _pointerBox->set(Vec3(0.0f, _pointerLength / 2.0f, 0.0f),
                     Vec3(thickness / 2.0f, _pointerLength / 2.0f, thickness / 2.0f));
    _pointerBoxDrawable->dirtyBound();

    // Update ball:
    // _pointerBall->setCenter(Vec3(0.0f, 0.0f, _pointerLength));
    _pointerBall->setCenter(Vec3(0.0f, _pointerLength, 0.0f));
    _pointerBall->setRadius(_ballRadiusFactor * _pointerLength);
    _sphereDrawable->dirtyBound(); // update bounding box
    Matrix rot, trans;
    rot.makeRotate(M_PI / 2.0f, 1.0f, 0.0f, 0.0f);
    trans.makeTranslate(0.0f, 0.0f, _pointerLength);

    // Update marker:
    setConeMarkerPosition(0, 0, _pointerLength);
    setSphereMarkerPosition(0, 0, _pointerLength);
    setBoxMarkerPosition(0, 0, _pointerLength);

    setConeBrushPosition(0, 0, _pointerLength);
    setSphereBrushPosition(0, 0, _pointerLength);
    setBoxBrushPosition(0, 0, _pointerLength);
}

void InputDevice::setConeMarkerPosition(float x, float y, float z)
{
    Matrix rot;
    if (CUI::_display == CUI::FISHTANK)
    {
        rot.makeRotate(M_PI / 2.0f, 0.0f, 1.0f, 0.0f);
    }
    else
    {
        rot.makeRotate(M_PI / 2.0f, 1.0f, 0.0f, 0.0f);
    }

    Matrix trans;
    trans.makeTranslate(x, y, z);

    _coneMarkerXF->setMatrix(rot * trans);
}

void InputDevice::setConeBrushPosition(float x, float y, float z)
{
    Matrix rot;
    if (CUI::_display == CUI::FISHTANK)
    {
        rot.makeRotate(M_PI / 2.0f, 0.0f, 1.0f, 0.0f);
    }
    else
    {
        rot.makeRotate(M_PI / 2.0f, 1.0f, 0.0f, 0.0f);
    }

    Matrix trans;
    trans.makeTranslate(x, y, z);

    _coneBrushXF->setMatrix(rot * trans);
}

void InputDevice::setBoxMarkerPosition(float x, float y, float z)
{
    Matrix trans;
    trans.makeTranslate(x, y, z);
    _boxMarkerXF->setMatrix(trans);
}

void InputDevice::setBoxBrushPosition(float x, float y, float z)
{
    Matrix trans;
    trans.makeTranslate(x, y, z);
    _boxBrushXF->setMatrix(trans);
}

void InputDevice::setSphereMarkerPosition(float x, float y, float z)
{
    Matrix trans;
    trans.makeTranslate(x, y, z);
    _sphereMarkerXF->setMatrix(trans);
}

void InputDevice::setSphereBrushPosition(float x, float y, float z)
{
    Matrix trans;
    trans.makeTranslate(x, y, z);
    _sphereBrushXF->setMatrix(trans);
}

Vec3 InputDevice::getIsectPoint()
{
    return _isect.point;
}

/** Amplifies the vertical component of the viewing direction so that
  less strain is put on the neck in gaze directed techniques.
*/
Vec3 InputDevice::amplifyHeadDirection(Vec3 &viewDir, bool, bool)
{
    const float amplification = 0.2; // in hundreds percent
    Vec3 dirNew, horiz;

    horiz = viewDir;
    horiz[1] = 0.0; // create vector in x-z plane
    horiz.normalize();
    float angle = acosf(viewDir * horiz); // dot product
    Vec3 rotAxis = horiz ^ viewDir; // cross product
    if (angle > M_PI / 12.0)
        angle *= 1.0 + amplification;
    else
        angle *= 1.0 + amplification * (fabs(angle) / (M_PI / 12.0));
    Matrix rotMatrix;
    rotMatrix.makeRotate(angle, rotAxis);
    dirNew = rotMatrix * horiz;
    dirNew.normalize();
    return dirNew;
}

void InputDevice::widgetDeleted(Node *node)
{
    list<Geode *>::const_iterator iterGeode;
    for (iterGeode = _currentWidget._geodeList.begin(); iterGeode != _currentWidget._geodeList.end(); iterGeode++)
    {
        if ((*iterGeode) == node)
        {
            cerr << "deleted the widget" << endl;
            _currentWidget.reset();
            return;
        }
    }
}

// reset the current widget (used when deleteing a widget)
void InputDevice::widgetDeleted()
{
    _currentWidget.reset();
}

Geode *InputDevice::getIsectGeode()
{
    return _currentWidget._isectGeode;
}

Widget *InputDevice::getIsectWidget()
{
    return _currentWidget._widget;
}
