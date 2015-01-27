/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Local:
#include "CUI.h"
#include "Arrow.h"
#include "Measure.h"
#include "Interaction.h"

using namespace cui;
using namespace osg;

Arrow::Arrow(Interaction *interaction, Measure *measure)
    : Widget()
    , Events()
{
    _interaction = interaction;
    _measure = measure;

    osg::Cone *cone = new osg::Cone(Vec3(0, 0, 0), 0.08, 0.08);
    _tip = Vec3(0, 0, 0.08);

    _drawable = new osg::ShapeDrawable(cone);
    _color.set(0.0, 0.0, 1.0, 1.0);
    _highlightCol.set(1.0, 1.0, 0.0, 1.0);
    _drawable->setColor(_color);
    _drawable->setUseDisplayList(false);

    _geode = new osg::Geode();
    _geode->addDrawable(_drawable);

    StateSet *stateSet = _geode->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);

    _node->addChild(_geode);
    _interaction->addListener(this, this);
}

Arrow::~Arrow()
{
}

void Arrow::setColor(Vec4 color)
{
    _color = color;
    _drawable->setColor(_color);
}

Vec3 Arrow::getTipPos()
{
    Vec3 ret;
    Matrix local2Root = CUI::computeLocal2Root(getNode());
    ret = _tip * _node->getMatrix();
    return ret;
}

void Arrow::cursorEnter(InputDevice *input)
{
    _drawable->setColor(_highlightCol);
    _initWand2w = input->getI2W();
}

void Arrow::cursorUpdate(InputDevice *input)
{
    if (input->getButtonState(0))
    {
        Matrix m, initInverse, diff, trans, tmp;

        m = _node->getMatrix();
        initInverse = Matrix::inverse(_initWand2w * Matrix::inverse(_measure->getPickBox()->getB2W()));
        diff = initInverse * input->getI2W() * Matrix::inverse(_measure->getPickBox()->getB2W());
        tmp.setTrans(m.getTrans());
        trans = tmp * diff;

        m.setTrans(trans.getTrans());
        if (!isTipInside(m))
            return;
        _node->setMatrix(m);

        _measure->setRotate();
    }

    _initWand2w = input->getI2W();
}

void Arrow::cursorLeave(InputDevice *)
{
    _drawable->setColor(_color);
}

void Arrow::buttonEvent(InputDevice *, int)
{
}

void Arrow::joystickEvent(InputDevice *)
{
}

void Arrow::wheelEvent(InputDevice *, int)
{
}

bool Arrow::isTipInside(Matrix m)
{
    Vec3 tip;
    float scale = 1.2;

    tip = _tip * m;

    if (((tip[0] < (scale * _measure->getBBox()._min[0]))
         || (tip[1] < (scale * _measure->getBBox()._min[1]))
         || (tip[2] < (scale * _measure->getBBox()._min[2])))
        || ((tip[0] > (scale * _measure->getBBox()._max[0]))
            || (tip[1] > (scale * _measure->getBBox()._max[1]))
            || (tip[2] > (scale * _measure->getBBox()._max[2]))))
        return false;
    else
    {
        return true;
    }
}
