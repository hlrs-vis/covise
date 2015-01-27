/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// CUI:
#include "CUI.h"
#include "Rectangle.h"
#include "InputDevice.h"

// OSG:
#include <osg/Geode>

using namespace cui;
using namespace osg;

Rectangle::Rectangle(Interaction *interaction)
    : Widget()
    , Events()
{
    StateSet *state;

    _interaction = interaction;

    // create rectangle geometry
    _rectGeode = new Geode();
    _node->addChild(_rectGeode);

    _rectGeom = new Geometry();
    _rectGeode->addDrawable(_rectGeom);

    _vertices = new Vec3Array(4);
    (*_vertices)[0].set(-0.5, -0.5, 0.0);
    (*_vertices)[1].set(0.5, -0.5, 0.0);
    (*_vertices)[2].set(0.5, 0.5, 0.0);
    (*_vertices)[3].set(-0.5, 0.5, 0.0);
    _rectGeom->setVertexArray(_vertices);

    _rectColor = new Vec4Array(1);
    (*_rectColor)[0].set(1.0, 1.0, 1.0, 1.0);
    _rectGeom->setColorArray(_rectColor);
    _rectGeom->setColorBinding(Geometry::BIND_OVERALL);

    _rectGeom->setUseDisplayList(false);
    _rectGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4));

    state = _rectGeode->getOrCreateStateSet();
    state->setMode(GL_LIGHTING, StateAttribute::OFF);

    // create spheres
    _sphere1Color.set(0.0, 0.0, 1.0, 1.0);
    _sphere2Color.set(0.0, 0.0, 1.0, 1.0);

    _sphere1Geode = new Geode();
    _node->addChild(_sphere1Geode);
    _sphere2Geode = new Geode();
    _node->addChild(_sphere2Geode);

    _sphere1 = new Sphere((*_vertices)[0], 0.05);
    _sphere1Drawable = new ShapeDrawable(_sphere1);
    _sphere1Drawable->setColor(_sphere1Color);
    _sphere1Drawable->setUseDisplayList(false);

    _sphere2 = new Sphere((*_vertices)[2], 0.05);
    _sphere2Drawable = new ShapeDrawable(_sphere2);
    _sphere2Drawable->setColor(_sphere2Color);
    _sphere2Drawable->setUseDisplayList(false);

    _sphere1Geode->addDrawable(_sphere1Drawable);
    _sphere2Geode->addDrawable(_sphere2Drawable);

    state = _sphere1Geode->getOrCreateStateSet();
    state->setMode(GL_LIGHTING, StateAttribute::OFF);

    state = _sphere2Geode->getOrCreateStateSet();
    state->setMode(GL_LIGHTING, StateAttribute::OFF);

    _corners = new Vec3Array(4);
    _edges = new Vec3Array(4);
    updateRectangle();

    _interaction->addListener(this, this);
}

Rectangle::~Rectangle()
{
    _corners->unref();
    _edges->unref();
}

void Rectangle::addListener(RectangleListener *listener)
{
    _listeners.push_back(listener);
}

void Rectangle::cursorEnter(InputDevice *input)
{
    if (input->getIsectGeode() == _sphere1Geode)
    {
        _sphere1Color.set(1.0, 1.0, 0.0, 1.0);
        _sphere1Drawable->setColor(_sphere1Color);
    }
    else if (input->getIsectGeode() == _sphere2Geode)
    {
        _sphere2Color.set(1.0, 1.0, 0.0, 1.0);
        _sphere2Drawable->setColor(_sphere2Color);
    }
    else if (input->getIsectGeode() == _rectGeode)
    {
        (*_rectColor)[0].set(1.0, 1.0, 0.0, 1.0);
        _rectGeom->setColorArray(_rectColor);
    }

    _initI2W = input->getI2W();
}

void Rectangle::cursorUpdate(InputDevice *input)
{
    // R = Rectangle, W = World, I = Input
    Matrix R2W, R2I, tmp, rot, diff, trans, transToO, transFromO, local;
    Vec3 tmpVec1, tmpVec2, transVec;

    if (input->getButtonState(0))
    {
        R2W = CUI::computeLocal2Root(getNode());
        R2I = Matrix::inverse(_initI2W * Matrix::inverse(R2W));
        diff = R2I * input->getI2W() * Matrix::inverse(R2W);

        if (input->getIsectGeode() == _rectGeode)
        {
            // translate
            local = _node->getMatrix();
            tmp.setTrans(local.getTrans());
            trans = tmp * diff;
            local.setTrans(trans.getTrans());
            _node->setMatrix(local);
        }
        else if (input->getIsectGeode() == _sphere1Geode)
        {
            // scale
            scaleRectangle(diff.getTrans(), _sphere1Geode);
        }
        else if (input->getIsectGeode() == _sphere2Geode)
        {
            // scale
            scaleRectangle(diff.getTrans(), _sphere2Geode);
        }

        updateRectangle();
    }

    if (input->getButtonState(1) && (input->getIsectGeode() == _rectGeode))
    {
        // rotate
        R2W = CUI::computeLocal2Root(getNode());
        R2I = Matrix::inverse(_initI2W * Matrix::inverse(R2W));
        diff = R2I * input->getI2W() * Matrix::inverse(R2W);

        local = _node->getMatrix();
        transVec = local.getTrans();
        transFromO.makeTranslate(transVec);
        transVec *= -1;
        transToO.makeTranslate(transVec);

        diff.setTrans(Vec3(0, 0, 0));

        rot.makeRotate(tmpVec1, tmpVec2);

        _node->setMatrix(transToO * diff * transFromO * _node->getMatrix());

        updateRectangle();
    }

    _initI2W = input->getI2W();
}

void Rectangle::cursorLeave(InputDevice *input)
{
    if (input->getIsectGeode() == _sphere1Geode)
    {
        _sphere1Color.set(0.0, 0.0, 1.0, 1.0);
        _sphere1Drawable->setColor(_sphere1Color);
    }
    else if (input->getIsectGeode() == _sphere2Geode)
    {
        _sphere2Color.set(0.0, 0.0, 1.0, 1.0);
        _sphere2Drawable->setColor(_sphere2Color);
    }
    else if (input->getIsectGeode() == _rectGeode)
    {
        (*_rectColor)[0].set(1.0, 1.0, 1.0, 1.0);
        _rectGeom->setColorArray(_rectColor);
    }
}

void Rectangle::buttonEvent(InputDevice *, int)
{
}

void Rectangle::joystickEvent(InputDevice *)
{
}

void Rectangle::wheelEvent(InputDevice *, int)
{
}

void Rectangle::getCorners(float points[4][3])
{
    int i;

    for (i = 0; i < 4; i++)
    {
        points[i][0] = (*_corners)[i][0];
        points[i][1] = (*_corners)[i][1];
        points[i][2] = (*_corners)[i][2];
    }
}

void Rectangle::updateRectangle()
{
    Vec3 tmp;

    (*_corners)[0] = (*_vertices)[0] * _node->getMatrix();
    (*_corners)[1] = (*_vertices)[1] * _node->getMatrix();
    (*_corners)[2] = (*_vertices)[2] * _node->getMatrix();
    (*_corners)[3] = (*_vertices)[3] * _node->getMatrix();

    (*_edges)[0] = (*_corners)[1] - (*_corners)[0];
    (*_edges)[1] = (*_corners)[2] - (*_corners)[1];
    (*_edges)[2] = (*_corners)[3] - (*_corners)[2];
    (*_edges)[3] = (*_corners)[0] - (*_corners)[3];

    (*_edges)[0].normalize();
    (*_edges)[1].normalize();
    (*_edges)[2].normalize();
    (*_edges)[3].normalize();

    tmp = (*_edges)[0].operator^((*_edges)[3]);

    plane[0] = tmp[0];
    plane[1] = tmp[1];
    plane[2] = tmp[2];
    plane[3] = -(plane[0] * (*_corners)[0][0] + plane[1] * (*_corners)[0][1] + plane[2] * (*_corners)[0][2]);

    for (_iter = _listeners.begin(); _iter != _listeners.end(); _iter++)
    {
        (*_iter)->rectangleUpdate();
    }
}

void Rectangle::scaleRectangle(Vec3 trans, Geode *corner)
{
    float lengthA, lengthB, lengthC;
    Matrix id;
    Vec3 transA, transC;

    lengthB = trans.length();
    trans.normalize();

    lengthA = lengthB * (*_edges)[0].operator*(trans);
    lengthC = lengthB * (*_edges)[1].operator*(trans);

    transA = (*_edges)[0] * lengthA;
    transC = (*_edges)[1] * lengthC;

    if (corner == _sphere1Geode)
    {
        (*_vertices)[0] = (*_vertices)[0] * _node->getMatrix();
        (*_vertices)[0] += transA;
        (*_vertices)[0] += transC;
        (*_vertices)[1] = (*_vertices)[1] * _node->getMatrix();
        (*_vertices)[1] += transC;
        (*_vertices)[3] = (*_vertices)[3] * _node->getMatrix();
        (*_vertices)[3] += transA;
        (*_vertices)[2] = (*_vertices)[2] * _node->getMatrix();
        _sphere1->setCenter((*_vertices)[0]);
        _sphere2->setCenter((*_vertices)[2]);
    }
    else if (corner == _sphere2Geode)
    {
        (*_vertices)[2] = (*_vertices)[2] * _node->getMatrix();
        (*_vertices)[2] += transA;
        (*_vertices)[2] += transC;
        (*_vertices)[1] = (*_vertices)[1] * _node->getMatrix();
        (*_vertices)[1] += transA;
        (*_vertices)[3] = (*_vertices)[3] * _node->getMatrix();
        (*_vertices)[3] += transC;
        (*_vertices)[0] = (*_vertices)[0] * _node->getMatrix();
        _sphere2->setCenter((*_vertices)[2]);
        _sphere1->setCenter((*_vertices)[0]);
    }

    id.makeIdentity();
    _node->setMatrix(id);

    _rectGeom->dirtyBound();
    _sphere1Drawable->dirtyBound();
    _sphere2Drawable->dirtyBound();
}
