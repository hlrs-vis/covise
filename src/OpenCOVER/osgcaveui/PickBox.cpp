/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;

// OSG:
#include <osg/Geode>
#include <osg/Switch>
#include <osg/TexEnv>
#include <osg/Depth>
#include <osg/LineWidth>
#include <osgText/Text>
#include <osgDB/ReadFile>

// Cover:
#include <OpenVRUI/coTrackerButtonInteraction.h>

// Local:
#include "PickBox.h"
#include "Interaction.h"
#include "CUI.h"

using namespace cui;
using namespace osg;

const int PickBox::NUM_BOX_COLORS = 3;

/** Constructor.
  @param interaction interaction instance that deals with this widget
  @param c1 non-picked boundary color
  @param c2 picked boundary color
*/
PickBox::PickBox(Interaction *interaction, const Vec3 &min, const Vec3 &max,
                 const Vec4 &c1, const Vec4 &c2, const Vec4 &c3)
    : Widget()
    , Events()
{
    _switch = new Switch();
    _scale = new MatrixTransform();
    _node->addChild(_scale.get());
    _geom[0] = _geom[1] = NULL;
    _isMovable = false;
    _isIntersected = false;
    _showWireframe = false;
    _isSelected = false;
    //_scale->setNodeMask(_scale->getNodeMask() & (~2));
    _scale->setNodeMask(~2);

    // Set bounding box:
    _bbox.set(min, max);
    if (!_bbox.valid())
    {
        cerr << "PickBox::createWireframe: invalid bounding box size, using default" << endl;
        _bbox.set(Vec3(0, 0, 0), Vec3(1, 1, 1));
    }

    createGeometry(c1, c2, c3);

    _interaction = interaction;
    _interaction->addListener(this);

    _interactionA = new vrui::coTrackerButtonInteraction(vrui::coInteraction::ButtonA, "MoveMode", vrui::coInteraction::Medium);
    _interactionB = new vrui::coTrackerButtonInteraction(vrui::coInteraction::ButtonB, "MoveMode", vrui::coInteraction::Medium);
    _interactionC = new vrui::coTrackerButtonInteraction(vrui::coInteraction::ButtonC, "MoveMode", vrui::coInteraction::Medium);

    assert(_interactionA);
    assert(_interactionB);
    assert(_interactionC);
};

PickBox::~PickBox()
{
    //cerr << "PickBox destructor called" << endl;
    _listeners.clear();
    _interaction->removeListener(this);

    //need to remove widget from the list

    if (_node->getNumParents())
    {
        _node->getParent(0)->removeChild(_node.get());
    }
    else
    {
        _node->removeChild(_scale.get());
    }

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

    if (_interactionC)
    {
        if (_interactionC->isRegistered())
            vrui::coInteractionManager::the()->unregisterInteraction(_interactionC);
        delete _interactionC;
        _interactionC = NULL;
    }
}

void PickBox::setScale(float scale)
{
    Matrix scaleMat;
    scaleMat.makeScale(scale, scale, scale);
    _scale->setMatrix(scaleMat);
}

float PickBox::getScale()
{
    Matrix mat = _scale->getMatrix();
    double *matPtr = mat.ptr();
    return matPtr[0];
}

void PickBox::setPosition(Vec3 &pos)
{
    setPosition(pos[0], pos[1], pos[2]);
}

void PickBox::setPosition(float x, float y, float z)
{
    Matrix mat = _node->getMatrix();
    mat.setTrans(x, y, z);
    _node->setMatrix(mat);
}

/** @return PickBox to World matrix.
*/
Matrix PickBox::getB2W()
{
    /*
  Matrix xf = _node->getMatrix();
  Matrix scale = _scale->getMatrix();
  return scale * xf;
*/
    Matrix b2w = CUI::computeLocal2Root(_scale.get());

    return b2w;
}

/** Create PickBox geometry.
*/
void PickBox::createGeometry(const Vec4 &c1, const Vec4 &c2, const Vec4 &c3)
{
    Geode *frame[NUM_BOX_COLORS];
    int i;

    // Create geometries:
    _geom[0] = createWireframe(c1);
    _geom[1] = createWireframe(c2);
    _geom[2] = createWireframe(c3);

    // Create geodes:
    for (i = 0; i < NUM_BOX_COLORS; ++i)
    {
        frame[i] = new Geode();
        frame[i]->addDrawable(_geom[i]);
        frame[i]->setNodeMask(~0);
        _switch->addChild(frame[i]);
    }
    updateWireframe(); // initialize wireframe color

    // Thick lines and lighting off:
    LineWidth *lineWidth = new LineWidth();
    lineWidth->setWidth(4.0f);
    StateSet *stateset = _switch->getOrCreateStateSet();
    stateset->setAttribute(lineWidth);
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);

    _scale->addChild(_switch.get());
}

void PickBox::cursorEnter(InputDevice *dev)
{
    _isIntersected = true;
    updateWireframe();
    Matrix i2w = dev->getI2W();
    _lastWand2w = i2w;
}

void PickBox::cursorLeave(InputDevice *dev)
{
    _isIntersected = false;
    updateWireframe();
    Matrix i2w = dev->getI2W();
    _lastWand2w = i2w;
}

void PickBox::cursorUpdate(InputDevice *dev)
{
    _lastWand2w = dev->getI2W();
}

void PickBox::updateWireframe()
{
    if (_showWireframe)
    {
        if (!_isSelected && !_isIntersected)
        {
            _switch->setSingleChildOn(0);
        }
        else if (_isIntersected)
        {
            _switch->setSingleChildOn(1);
        }
        else
        {
            _switch->setSingleChildOn(2);
        }
    }
    else
    {
        _switch->setAllChildrenOff();
    }
}

void PickBox::move(Matrix &lastWand2w, Matrix &wand2w)
{
    // Compute difference matrix between last and current wand:
    Matrix invLastWand2w = Matrix::inverse(lastWand2w);
    Matrix wDiff = invLastWand2w * wand2w;

    // Volume follows wand movement:
    Matrix box2w = getB2W();

    Matrix w2o = _interaction->getW2O();

    // need to make adjust due to _scale node used in calculation
    Matrix invScale = Matrix::inverse(_scale->getMatrix());
    _node->setMatrix(invScale * box2w * wDiff * w2o);
}

void PickBox::addListener(PickBoxListener *pb)
{
    _listeners.push_back(pb);
}

void PickBox::buttonEvent(InputDevice *dev, int button)
{
    // Notify event listeners:
    std::list<PickBoxListener *>::iterator iter;
    for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
    {
        (*iter)->pickBoxButtonEvent(this, dev, button);
    }
}

void PickBox::joystickEvent(InputDevice *)
{
}

void PickBox::wheelEvent(InputDevice *, int)
{
}

void PickBox::setBoxSize(const Vec3 &size)
{
    int i;

    Vec3 min, max;
    min = -size / 2.0f;
    max = size / 2.0f;
    _bbox.set(min, max);
    for (i = 0; i < NUM_BOX_COLORS; ++i)
    {
        updateVertices(_geom[i]);
    }
}

void PickBox::setBoxSize(Vec3 &center, Vec3 &size)
{

    int i;

    Vec3 min, max;
    min = center - (size / 2.0f);
    max = center + (size / 2.0f);
    _bbox.set(min, max);
    for (i = 0; i < NUM_BOX_COLORS; ++i)
    {
        updateVertices(_geom[i]);
    }
}

Vec3 PickBox::getBoxSize()
{
    Vec3 size = _bbox._max - _bbox._min;
    return size;
}

void PickBox::updateVertices(Geometry *geom)
{
    Vec3 min, max;

    // Create lines vertices:
    Vec3Array *vertices = (Vec3Array *)geom->getVertexArray();
    if (!vertices)
        vertices = new Vec3Array(24);

    min = _bbox._min;
    max = _bbox._max;

    (*vertices)[0].set(min[0], max[1], max[2]); // 0
    (*vertices)[1].set(min[0], min[1], max[2]); // 1
    (*vertices)[2].set(min[0], max[1], max[2]); // 0
    (*vertices)[3].set(max[0], max[1], max[2]); // 3
    (*vertices)[4].set(min[0], max[1], max[2]); // 0
    (*vertices)[5].set(min[0], max[1], min[2]); // 4
    (*vertices)[6].set(min[0], min[1], max[2]); // 1
    (*vertices)[7].set(max[0], min[1], max[2]); // 2
    (*vertices)[8].set(min[0], min[1], max[2]); // 1
    (*vertices)[9].set(min[0], min[1], min[2]); // 5
    (*vertices)[10].set(max[0], min[1], max[2]); // 2
    (*vertices)[11].set(max[0], max[1], max[2]); // 3
    (*vertices)[12].set(max[0], min[1], max[2]); // 2
    (*vertices)[13].set(max[0], min[1], min[2]); // 6
    (*vertices)[14].set(max[0], min[1], min[2]); // 6
    (*vertices)[15].set(max[0], max[1], min[2]); // 7
    (*vertices)[16].set(max[0], min[1], min[2]); // 6
    (*vertices)[17].set(min[0], min[1], min[2]); // 5
    (*vertices)[18].set(max[0], max[1], max[2]); // 3
    (*vertices)[19].set(max[0], max[1], min[2]); // 7
    (*vertices)[20].set(min[0], max[1], min[2]); // 4
    (*vertices)[21].set(max[0], max[1], min[2]); // 7
    (*vertices)[22].set(min[0], max[1], min[2]); // 4
    (*vertices)[23].set(min[0], min[1], min[2]); // 5

    // Pass the created vertex array to the points geometry object:
    geom->setVertexArray(vertices);
}

/** This creates the wireframe PickBox around the widget.
  Volume vertex names:
  <PRE>
      4____ 7        y
     /___ /|         |
   0|   3| |         |___x
    | 5  | /6       /
    |/___|/        z
    1    2
  </PRE>
*/
Geometry *PickBox::createWireframe(const Vec4 &color)
{
    Geometry *geom = new Geometry();

    updateVertices(geom);

    // Set colors:
    Vec4Array *colors = new Vec4Array();
    colors->push_back(color);
    geom->setColorArray(colors);
    geom->setColorBinding(Geometry::BIND_OVERALL);

    // Set normals:
    Vec3Array *normals = new Vec3Array();
    normals->push_back(Vec3(0.0f, 0.0f, 1.0f));
    geom->setNormalArray(normals);
    geom->setNormalBinding(Geometry::BIND_OVERALL);

    // This time we simply use primitive, and hardwire the number of coords
    // to use since we know up front:
    geom->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 24));

    geom->setUseDisplayList(false); // allow dynamic changes

    return geom;
}

void PickBox::setMovable(bool movable)
{
    _isMovable = movable;
}

bool PickBox::getMovable()
{
    return _isMovable;
}

void PickBox::setShowWireframe(bool show)
{
    _showWireframe = show;
    updateWireframe();
}

void PickBox::setSelected(bool selected)
{
    _isSelected = selected;
    updateWireframe();
}

bool PickBox::getSelected()
{
    return _isSelected;
}

bool PickBox::getIntersected()
{
    return _isIntersected;
}
