/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <OpenVRUI/sginterface/vruiMatrix.h>

#include <vsg/maths/vec3.h>
#include <vsg/maths/mat4.h>
#include <vsg/maths/transform.h>

namespace vrui
{

class VSGVRUIEXPORT VSGVruiMatrix : public virtual vruiMatrix
{

public:
    VSGVruiMatrix();
    virtual ~VSGVruiMatrix();

    virtual vruiMatrix *makeIdentity();
    virtual vruiMatrix *preTranslated(double x, double y, double z, vruiMatrix *matrix);
    virtual vruiMatrix *makeTranslate(double x, double y, double z);
    virtual vruiMatrix *makeRotate(double degrees, double xAxis, double yAxis, double zAxis);
    virtual vruiMatrix *makeScale(double x, double y, double z);
    virtual vruiMatrix *makeEuler(double h, double p, double r);
    virtual vruiMatrix *makeInverse(const vruiMatrix *source);
    virtual vruiMatrix *mult(const vruiMatrix *matrix);

    virtual double &operator()(int row, int column);
    virtual double operator()(int row, int column) const;

    virtual coVector getFullXformPt(const coVector &point) const;
    virtual coVector getHPR() const;

    vsg::dmat4 getMatrix();
    const vsg::dmat4 &getMatrix() const;
    void setMatrix(const vsg::dmat4 &matrix);

private:
    vsg::dmat4 matrix;
};
}
