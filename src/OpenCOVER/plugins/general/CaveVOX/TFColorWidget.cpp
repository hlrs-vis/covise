/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// OSG:
#include <osg/Geometry>
#include <osg/LineWidth>

// Local:
#include "TFColorWidget.H"
#include "Interaction.H"
#include "Measure.H"

using namespace cui;
using namespace osg;

TFColorWidget::TFColorWidget(Interaction *interaction, Measure *b, int size)
    : Widget()
    , Events()
{
    _interaction = interaction;

    Geode *geode = new Geode();

    _geom = new Geometry();
    _geom->setUseDisplayList(false);

    _vertices = new Vec3Array();
    _vertices->push_back(Vec3(0, 0, 0));
    _vertices->push_back(Vec3(0, 0, .2));
    _vector = (*_vertices)[1] - (*_vertices)[0];
    _vector.normalize();
    _geom->setVertexArray(_vertices);

    DrawElementsUInt *line = new DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    line->push_back(0);
    line->push_back(1);
    _geom->addPrimitiveSet(line);

    _colors = new Vec4Array(1);
    (*_colors)[0].set(0.0, 0.0, 0.0, 1.0);
    _geom->setColorArray(_colors);
    _geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    geode->addDrawable(_geom);

    _node->addChild(geode);

    LineWidth *width = new LineWidth();
    width->setWidth(size);
    StateSet *state = _geom->getOrCreateStateSet();
    state->setAttribute(width, osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, StateAttribute::OFF);

    _interaction->addListener(this, this);
}

TFColorWidget::~TFColorWidget()
{
}

void TFColorWidget::setSize(int size)
{
    LineWidth *width = new LineWidth();
    width->setWidth(size);
    StateSet *state = _geom->getOrCreateStateSet();
    state->setAttribute(width, osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, StateAttribute::OFF);
}

void TFColorWidget::setVertices(Vec3 v1, Vec3 v2)
{
    (*_vertices)[0] = v1;
    (*_vertices)[1] = v2;

    _vector = (*_vertices)[1] - (*_vertices)[0];
    _vector.normalize();

    _geom->dirtyBound();
}

void TFColorWidget::setColor(Vec4 color)
{
    (*_colors)[0] = color;
}

void TFColorWidget::cursorEnter(InputDevice *input)
{
}

void TFColorWidget::cursorUpdate(InputDevice *input)
{
}

void TFColorWidget::cursorLeave(InputDevice *)
{
}

void TFColorWidget::buttonEvent(InputDevice *, int)
{
}

void TFColorWidget::joystickEvent(InputDevice *)
{
}

void TFColorWidget::wheelEvent(InputDevice *, int)
{
}
