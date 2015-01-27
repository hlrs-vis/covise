/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <osg/Vec4>
#include "mathUtils.h"

using namespace osg;

namespace vrui
{

OSGVruiMatrix::OSGVruiMatrix()
{
}

OSGVruiMatrix::~OSGVruiMatrix()
{
}

vruiMatrix *OSGVruiMatrix::makeIdentity()
{
    this->matrix.makeIdentity();
    return this;
}

vruiMatrix *OSGVruiMatrix::preTranslated(double x, double y, double z, vruiMatrix *matrix)
{
    Matrix translate;
    translate.makeTranslate(x, y, z);
    OSGVruiMatrix *osgVruiMatrix = dynamic_cast<OSGVruiMatrix *>(matrix);
    this->matrix = translate * osgVruiMatrix->matrix;
    return this;
}

vruiMatrix *OSGVruiMatrix::makeTranslate(double x, double y, double z)
{
    this->matrix.makeTranslate(x, y, z);
    return this;
}

vruiMatrix *OSGVruiMatrix::makeRotate(double degrees, double xAxis, double yAxis, double zAxis)
{
    this->matrix.makeRotate(osg::inDegrees(degrees), xAxis, yAxis, zAxis);
    return this;
}

vruiMatrix *OSGVruiMatrix::makeScale(double x, double y, double z)
{
    this->matrix.makeScale(x, y, z);
    return this;
}

vruiMatrix *OSGVruiMatrix::makeEuler(double h, double p, double r)
{
    Matrix hm, pm, rm;
    pm.makeRotate(osg::inDegrees(p), 1.0, 0.0, 0.0);
    rm.makeRotate(osg::inDegrees(r), 0.0, 1.0, 0.0);
    hm.makeRotate(osg::inDegrees(h), 0.0, 0.0, 1.0);
    this->matrix = rm * pm * hm;
    return this;
}

vruiMatrix *OSGVruiMatrix::mult(const vruiMatrix *matrix)
{
    const OSGVruiMatrix *osgVruiMatrix = dynamic_cast<const OSGVruiMatrix *>(matrix);
    this->matrix.postMult(const_cast<OSGVruiMatrix *>(osgVruiMatrix)->matrix);
    return this;
}

vruiMatrix *OSGVruiMatrix::makeInverse(const vruiMatrix *source)
{
    const OSGVruiMatrix *sourceMat = dynamic_cast<const OSGVruiMatrix *>(source);
    this->matrix.invert(sourceMat->matrix);
    return this;
}

double &OSGVruiMatrix::operator()(int row, int column)
{
    return this->matrix(row, column);
}

double OSGVruiMatrix::operator()(int row, int column) const
{
    return this->matrix(row, column);
}

coVector OSGVruiMatrix::getFullXformPt(const coVector &point) const
{
    Vec3 pnt(point[0], point[1], point[2]);
    Vec3 transformedPnt = pnt * this->matrix;
    //transformedPnt /= transformedPnt[3];

    return coVector(transformedPnt[0], transformedPnt[1], transformedPnt[2]);
}

coVector OSGVruiMatrix::getHPR() const
{
    coCoord coord(this->matrix);
    osg::Vec3 &v = coord.hpr;
    return coVector(v[0], v[1], v[2]);
}

void OSGVruiMatrix::setMatrix(const Matrix &matrix)
{
    this->matrix = matrix;
}

const Matrix &OSGVruiMatrix::getMatrix() const
{
    return this->matrix;
}

Matrix OSGVruiMatrix::getMatrix()
{
    return this->matrix;
}
}
