/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// C++:
#include <iostream>
#include <math.h>

// Virvo:
#include <virvo/vvtoolshed.h>

// Local:
#include "Marker.h"
#include "Measure.h"
#include "Interaction.h"

// Cover:
#include <OpenVRUI/coTrackerButtonInteraction.h>

// OSG:
#include <osg/BlendFunc>

// CUI:
#include "CUI.h"

using namespace cui;
using namespace osg;
using namespace std;

const float Marker::RATIO_LENGTH2RADIUS = 5.0f;

/** Constructor. 
  @param interaction interaction class; NULL for no interaction
*/
Marker::Marker(GeometryType gt, Interaction *interaction)
    : Widget()
    , Events()
{
    Vec4 col = Widget::COL_YELLOW;
    init(gt, interaction, 0.5f, col);
}

/** Constructor. 
  @param interaction interaction class; NULL for no interaction
*/
Marker::Marker(GeometryType gt, Interaction *interaction, float size, Vec4 &color)
    : Widget()
    , Events()
{
    init(gt, interaction, size, color);
}

Marker::~Marker()
{
    _listeners.clear();
    _interaction->removeListener(this);
    if (_interactionA)
    {
        if (_interactionA->isRegistered())
            vrui::coInteractionManager::the()->unregisterInteraction(_interactionA);
        delete _interactionA;
        _interactionA = NULL;
    }

    if (_interactionB)
    {
        if (_interactionB->isRegistered())
            vrui::coInteractionManager::the()->unregisterInteraction(_interactionB);
        delete _interactionB;
        _interactionB = NULL;
    }
}

void Marker::init(GeometryType gt, Interaction *interaction, float size, Vec4 &color)
{
    Geode *geode;

    _gt = gt;
    if (_gt == CONE)
        geode = createCone(color);
    else if (_gt == BOX)
        geode = createBox(color);
    else
        geode = createSphere(color);

    setSize(size);
    setColor(color);

    // Make sure lighting is correct when marker is scaled:
    StateSet *stateSet = geode->getOrCreateStateSet();
    stateSet->setMode(GL_RESCALE_NORMAL, StateAttribute::ON);
    //stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF); // Philip added this else the color is not visible in opencover

    _interaction = interaction;
    _node->addChild(geode);

    if (_interaction)
        _interaction->addListener(this, this);
    _interactionA = new vrui::coTrackerButtonInteraction(vrui::coInteraction::ButtonA, "MarkerMove", vrui::coInteraction::Medium);
    _interactionB = NULL;
    assert(_interactionA);
}

Geode *Marker::createCone(Vec4 &color)
{
    Geode *geode = new Geode();

    // setDetailRatio is a factor to multiply the default values for
    // numSegments (40) and numRows (10).
    // They won't go below the minimum values of MIN_NUM_SEGMENTS = 5, MIN_NUM_ROWS = 3
    TessellationHints *hints = new TessellationHints();
    hints->setDetailRatio(0.3f);

    // Create cone geometry:
    _coneShape = new Cone(Vec3(0, 0, 0), 0.1f, 0.5f); // center, radius, height
    _shapeDrawable = new ShapeDrawable(_coneShape);
    _shapeDrawable->setTessellationHints(hints);
    _shapeDrawable->setColor(color);
    _shapeDrawable->setUseDisplayList(false); // allow changes to color and shape

    geode->addDrawable(_shapeDrawable);
    return geode;
}

Geode *Marker::createSphere(Vec4 &color)
{
    Geode *geode = new Geode();

    // setDetailRatio is a factor to multiply the default values for
    // numSegments (40) and numRows (10).
    // They won't go below the minimum values of MIN_NUM_SEGMENTS = 5, MIN_NUM_ROWS = 3
    TessellationHints *hints = new TessellationHints();
    hints->setDetailRatio(0.3f);

    Vec3 sphereCenter(0.0f, 0.0f, 0.0f);
    _sphereShape = new Sphere(sphereCenter, 1.0f);
    _shapeDrawable = new ShapeDrawable(_sphereShape);
    _shapeDrawable->setTessellationHints(hints);
    _shapeDrawable->setColor(color);
    geode->addDrawable(_shapeDrawable);
    _shapeDrawable->setUseDisplayList(false); // allow changes to color and shape
    return geode;
}

Geode *Marker::createBox(Vec4 &color)
{
    Geode *geode = new Geode();

    TessellationHints *hints = new TessellationHints();
    hints->setDetailRatio(0.3f);

    Vec3 boxCenter(0.0f, 0.0f, 0.0f);
    _boxShape = new Box(boxCenter, 1.0f);
    _shapeDrawable = new ShapeDrawable(_boxShape);
    _shapeDrawable->setTessellationHints(hints);
    _shapeDrawable->setColor(color);
    geode->addDrawable(_shapeDrawable);
    _shapeDrawable->setUseDisplayList(false); // allow changes to color and shape
    return geode;
}

void Marker::setSize(float size)
{
    _size = size;
    if (_gt == CONE)
    {
        _coneShape->setHeight(size);
        Vec3 center(0.0f, 0.0f, -0.75f * size); // shift along z to make tip to be at the origin
        _coneShape->setCenter(center);
        _coneShape->setRadius(size / RATIO_LENGTH2RADIUS);
    }
    else if (_gt == SPHERE)
    {
        _sphereShape->setRadius(size / 1.8f);
    }
    else if (_gt == BOX)
    {
        _boxShape->setHalfLengths(Vec3(size / 7.0f, size / 7.0f, size / 7.0f));
        //TODO Fix size changing of box marker
    }
    _shapeDrawable->dirtyBound();
}

void Marker::setColor(Vec4 color)
{
    _color = color;
    _shapeDrawable->setColor(_color);
}

void Marker::setHue(float hue)
{
    float r, g, b;
    Vec4 color;

    vvToolshed::HSBtoRGB(hue, 1, 1, &r, &g, &b);
    color[0] = r;
    color[1] = g;
    color[2] = b;
    color[3] = 1.0;

    setColor(color);
}

void Marker::setOpacity(float opacity)
{
    Vec4 color = _shapeDrawable->getColor();
    color.set(color[0], color[1], color[2], opacity);
    setColor(color);
}

void Marker::cursorEnter(InputDevice *dev)
{
    _prevType = dev->getCursorType();
    //dev->setCursorType(dev->POINTER);

    invertColor();
    _lastWand2w = dev->getI2W();
}

void Marker::cursorUpdate(InputDevice *input)
{
    _lastWand2w = input->getI2W();
}

void Marker::cursorLeave(InputDevice *input)
{
    invertColor();
    _lastWand2w = input->getI2W();
}

void Marker::invertColor()
{
    int i;
    Vec4 color = _shapeDrawable->getColor();
    for (i = 0; i < 3; ++i)
        color[i] = 1 - color[i];
    setColor(color);
}

void Marker::buttonEvent(InputDevice *evt, int button)
{
    std::list<MarkerListener *>::iterator iter;
    for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
    {
        (*iter)->markerEvent(this, button, evt->getButtonState(button));
    }
}

void Marker::joystickEvent(InputDevice *)
{
}

void Marker::wheelEvent(InputDevice *, int)
{
}

void Marker::setPosition(osg::Vec3 point)
{
    Matrix mat = _node->getMatrix();
    mat.setTrans(point);
    _node->setMatrix(mat);
}

osg::Vec3 Marker::getPosition()
{
    Matrix mat = _node->getMatrix();
    return mat.getTrans();
}

osg::Vec3 Marker::getDirection()
{
    Vec3 zDir(0, 0, 1);
    Matrix mat = _node->getMatrix();
    return mat * zDir;
}

void Marker::addMarkerListener(MarkerListener *listener)
{
    _listeners.push_back(listener);
}

float Marker::getSize()
{
    return _size;
}

Vec4 Marker::getColor()
{
    return _color;
}

float Marker::getHue()
{
    float h, s, v;

    Vec4 color = _shapeDrawable->getColor();
    vvToolshed::RGBtoHSB(color[0], color[1], color[2], &h, &s, &v);
    return h;
}

float Marker::getOpacity()
{
    Vec4 color = _shapeDrawable->getColor();
    return color[3];
}
