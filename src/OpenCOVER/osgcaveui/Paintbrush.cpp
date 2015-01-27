/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// C++:
#include <math.h>

// Virvo:
#include <virvo/vvtoolshed.h>

// Local:
#include "Paintbrush.h"
#include "Measure.h"
#include "Interaction.h"

// OSG:
#include <osg/BlendFunc>

// CUI:
#include "CUI.h"

using namespace cui;
using namespace osg;

const float Paintbrush::RATIO_LENGTH2RADIUS = 5.0f;

/** Constructor. 
  @param interaction interaction class; NULL for no interaction
*/
Paintbrush::Paintbrush(GeomType gt, Interaction *interaction)
    : Widget()
    , Events()
{
    Vec4 col = Widget::COL_YELLOW;
    init(gt, interaction, 0.5f, col);
}

/** Constructor. 
  @param interaction interaction class; NULL for no interaction
*/
Paintbrush::Paintbrush(GeomType gt, Interaction *interaction, float size, Vec4 &color)
    : Widget()
    , Events()
{
    init(gt, interaction, size, color);
}

Paintbrush::~Paintbrush()
{
}

void Paintbrush::init(GeomType gt, Interaction *interaction, float size, Vec4 &color)
{
    Geode *geode;

    _gt = gt;
    if (_gt == CONE)
        geode = createCone(color);
    else if (_gt == BOX)
        geode = createBox(color);
    else
        geode = createSphere(color);

    // Make sure lighting is correct when Paintbrush is scaled:
    StateSet *stateSet = geode->getOrCreateStateSet();
    stateSet->setMode(GL_RESCALE_NORMAL, StateAttribute::ON);

    _interaction = interaction;
    _node->addChild(geode);
    if (_interaction)
        _interaction->addListener(this, this);

    setSize(size);
    setColor(color);
}

Geode *Paintbrush::createCone(Vec4 &color)
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

    // Turn transparency on:
    /*	
  StateSet* stateSet = geode->getOrCreateStateSet();
  BlendFunc *bf = new BlendFunc(GL_SRC_ALPHA, GL_ONE);
  stateSet->setMode(GL_BLEND, StateAttribute::ON);
  stateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
  stateSet->setAttribute(bf);
  geode->setStateSet(stateSet); 	// comment this out to use opaque Paintbrushs
*/

    geode->addDrawable(_shapeDrawable);
    return geode;
}

Geode *Paintbrush::createSphere(Vec4 &color)
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

Geode *Paintbrush::createBox(Vec4 &color)
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

void Paintbrush::setSize(float size)
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
        _sphereShape->setRadius(size / 2.5f);
    }
    else if (_gt == BOX)
    {
        _boxShape->setHalfLengths(Vec3(size / 7.0f, size / 7.0f, size / 7.0f));
        //TODO Fix size changing of box Paintbrush
    }
    _shapeDrawable->dirtyBound();
}

void Paintbrush::setColor(Vec4 color)
{
    _color = color;
    _shapeDrawable->setColor(_color);
}

void Paintbrush::setHue(float hue)
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

void Paintbrush::setOpacity(float opacity)
{
    Vec4 color = _shapeDrawable->getColor();
    color.set(color[0], color[1], color[2], opacity);
    setColor(color);
}

void Paintbrush::cursorEnter(InputDevice *dev)
{
    if (dev == _interaction->_wandR)
    {
        _prevType = dev->getCursorType();
        dev->setCursorType(dev->POINTER);

        invertColor();
        _lastWand2w = dev->getI2W();
    }
}

void Paintbrush::cursorUpdate(InputDevice *input)
{
    if (input->getButtonState(0) != 0)
    {
        Matrix local2parent = _node->getMatrix();

        Matrix local2w = CUI::computeLocal2Root(_node.get());

        Matrix w2local = Matrix::inverse(local2w);

        Matrix w2parent = w2local * local2parent;

        Matrix initInverse = Matrix::inverse(_lastWand2w);
        Matrix wDiff = initInverse * input->getI2W();

        local2w = local2w * wDiff; // move Paintbrush with pointer
        local2parent = local2w * w2parent;

        _node->setMatrix(local2parent);
    }
    _lastWand2w = input->getI2W();
}

void Paintbrush::cursorLeave(InputDevice *dev)
{
    if (dev == _interaction->_wandR)
    {
        dev->setCursorType(_prevType);
        invertColor();
    }
}

void Paintbrush::invertColor()
{
    int i;
    Vec4 color = _shapeDrawable->getColor();
    for (i = 0; i < 3; ++i)
        color[i] = 1 - color[i];
    setColor(color);
}

void Paintbrush::buttonEvent(InputDevice *evt, int button)
{
    std::list<PaintbrushListener *>::iterator iter;
    for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
    {
        (*iter)->PaintbrushEvent(this, button, evt->getButtonState(button));
    }
}

void Paintbrush::joystickEvent(InputDevice *)
{
}

void Paintbrush::wheelEvent(InputDevice *, int)
{
}

void Paintbrush::setPosition(osg::Vec3 point)
{
    Matrix mat = _node->getMatrix();
    mat.setTrans(point);
    _node->setMatrix(mat);
}

osg::Vec3 Paintbrush::getPosition()
{
    Matrix mat = _node->getMatrix();
    return mat.getTrans();
}

osg::Vec3 Paintbrush::getDirection()
{
    Vec3 zDir(0, 0, 1);
    Matrix mat = _node->getMatrix();
    return mat * zDir;
}

void Paintbrush::addPaintbrushListener(PaintbrushListener *listener)
{
    _listeners.push_back(listener);
}

float Paintbrush::getSize()
{
    return _size;
}

Vec4 Paintbrush::getColor()
{
    return _color;
}

float Paintbrush::getHue()
{
    float h, s, v;

    Vec4 color = _shapeDrawable->getColor();
    vvToolshed::RGBtoHSB(color[0], color[1], color[2], &h, &s, &v);
    return h;
}

float Paintbrush::getOpacity()
{
    Vec4 color = _shapeDrawable->getColor();
    return color[3];
}
