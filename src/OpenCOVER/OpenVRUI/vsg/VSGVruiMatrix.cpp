/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiMatrix.h>

#include "mathUtils.h"

using namespace vsg;

namespace vrui
{

VSGVruiMatrix::VSGVruiMatrix()
{
}

VSGVruiMatrix::~VSGVruiMatrix()
{
}

vruiMatrix *VSGVruiMatrix::makeIdentity()
{
    this->matrix = vsg::dmat4();
    return this;
}

vruiMatrix *VSGVruiMatrix::preTranslated(double x, double y, double z, vruiMatrix *matrix)
{
    vsg::dmat4 translate= vsg::translate(x, y, z);
    VSGVruiMatrix *VSGVruiMatrix = dynamic_cast<vrui::VSGVruiMatrix *>(matrix);
    this->matrix = VSGVruiMatrix->matrix * translate;
    return this;
}

vruiMatrix *VSGVruiMatrix::makeTranslate(double x, double y, double z)
{
    this->matrix = translate(x, y, z);
    return this;
}

vruiMatrix *VSGVruiMatrix::makeRotate(double degrees, double xAxis, double yAxis, double zAxis)
{
    this->matrix = rotate(vsg::radians(degrees), xAxis, yAxis, zAxis);
    return this;
}

vruiMatrix *VSGVruiMatrix::makeScale(double x, double y, double z)
{
    this->matrix = scale(x, y, z);
    return this;
}

vruiMatrix *VSGVruiMatrix::makeEuler(double h, double p, double r)
{
    dmat4 hm, pm, rm;
    pm = rotate(vsg::radians(p), 1.0, 0.0, 0.0);
    rm = rotate(vsg::radians(r), 0.0, 1.0, 0.0);
    hm = rotate(vsg::radians(h), 0.0, 0.0, 1.0);
    this->matrix = hm * pm * rm;
    return this;
}

vruiMatrix *VSGVruiMatrix::mult(const vruiMatrix *mat)
{
    const VSGVruiMatrix *VSGVruiMatrix = dynamic_cast<const vrui::VSGVruiMatrix *>(mat);
    matrix = const_cast<vrui::VSGVruiMatrix *>(VSGVruiMatrix)->matrix * matrix;
    return this;
}

vruiMatrix *VSGVruiMatrix::makeInverse(const vruiMatrix *source)
{
    const VSGVruiMatrix *sourceMat = dynamic_cast<const VSGVruiMatrix *>(source);
    this->matrix = inverse(sourceMat->matrix);
    return this;
}

double &VSGVruiMatrix::operator()(int row, int column)
{
    return this->matrix(row, column);
}

double VSGVruiMatrix::operator()(int row, int column) const
{
    return this->matrix(row, column);
}

coVector VSGVruiMatrix::getFullXformPt(const coVector &point) const
{
    dvec3 pnt(point[0], point[1], point[2]);
    dvec3 transformedPnt = this->matrix * pnt;
    //transformedPnt /= transformedPnt[3];

    return coVector(transformedPnt[0], transformedPnt[1], transformedPnt[2]);
}

coVector VSGVruiMatrix::getHPR() const
{
    coCoord coord(this->matrix);
    vsg::dvec3 &v = coord.hpr;
    return coVector(v[0], v[1], v[2]);
}

void VSGVruiMatrix::setMatrix(const dmat4 &matrix)
{
    this->matrix = matrix;
}

const dmat4 &VSGVruiMatrix::getMatrix() const
{
    return this->matrix;
}

dmat4 VSGVruiMatrix::getMatrix()
{
    return this->matrix;
}
}
