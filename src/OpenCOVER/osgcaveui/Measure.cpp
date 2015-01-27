/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Local:
#include "Measure.h"
#include "Arrow.h"
#include "Bar.h"
#include "Interaction.h"

using namespace cui;
using namespace osg;

Measure::Measure(Interaction *interaction, PickBox *box)
    : Widget()
{
    _bar = new Bar(interaction, this);
    _rightEnd = new Arrow(interaction, this);
    _rightEnd->setColor(Vec4(1.0, 0.0, 0.0, 1.0));
    _leftEnd = new Arrow(interaction, this);
    _leftEnd->setColor(Vec4(0.0, 0.0, 1.0, 1.0));

    _pickBox = box;
    setBBox(_pickBox->_bbox);

    _node->addChild(_bar->getNode());
    _node->addChild(_rightEnd->getNode());
    _node->addChild(_leftEnd->getNode());

    Vec3 posMin, posMax;
    Matrix transLeft, transRight;

    posMin = Vec3(0.0, 0.0, -0.1);
    posMax = Vec3(0.0, 0.0, 0.1);

    transRight.makeTranslate(posMax);
    _rightEnd->getNode()->setMatrix(transRight);

    transLeft.makeTranslate(posMin);
    _leftEnd->getNode()->setMatrix(transLeft);

    setRotate();
}

Measure::~Measure()
{
}

void Measure::setRotate()
{
    Vec3 right, left, rotAxis;
    Matrix rotMat, transMat;

    // get measure coordinates of right and left arrow
    right = _rightEnd->getNode()->getMatrix().getTrans();
    left = _leftEnd->getNode()->getMatrix().getTrans();
    right = right * _node->getMatrix();
    left = left * _node->getMatrix();

    _bar->setVertices(left, right);

    rotAxis = _bar->getVector();

    rotMat.makeRotate(Vec3(0, 0, 1), rotAxis);
    transMat.makeTranslate(right);
    _rightEnd->getNode()->setMatrix(rotMat * transMat);

    rotMat.makeRotate(Vec3(0, 0, -1), rotAxis);
    transMat.makeTranslate(left);
    _leftEnd->getNode()->setMatrix(rotMat * transMat);

    for (_iter = _listeners.begin(); _iter != _listeners.end(); ++_iter)
    {
        (*_iter)->measureUpdate();
    }
}

void Measure::addMeasureListener(MeasureListener *m)
{
    _listeners.push_back(m);
}

void Measure::setBBox(BoundingBox bbox)
{
    _bbox = bbox;

    Vec3 posMin, posMax;
    Matrix transLeft, transRight;

    posMin = _bbox._min;
    posMin[1] *= -1;
    posMin[2] *= -1;
    posMax = _bbox._max;
    posMax[1] *= -1;
    posMax[2] *= -1;

    transRight.makeTranslate(posMax);
    _rightEnd->getNode()->setMatrix(transRight);

    transLeft.makeTranslate(posMin);
    _leftEnd->getNode()->setMatrix(transLeft);

    setRotate();
}

BoundingBox Measure::getBBox()
{
    return _bbox;
}

PickBox *Measure::getPickBox()
{
    return _pickBox;
}

Vec3 Measure::getRightEnd()
{
    return _rightEnd->getTipPos();
}

Vec3 Measure::getLeftEnd()
{
    return _leftEnd->getTipPos();
}
