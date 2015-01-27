/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_MATRIX_H
#define OSG_VRUI_MATRIX_H

#include <OpenVRUI/sginterface/vruiMatrix.h>

#include <osg/Matrix>

namespace vrui
{

class OSGVRUIEXPORT OSGVruiMatrix : public virtual vruiMatrix
{

public:
    OSGVruiMatrix();
    virtual ~OSGVruiMatrix();

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

    osg::Matrix getMatrix();
    const osg::Matrix &getMatrix() const;
    void setMatrix(const osg::Matrix &matrix);

private:
    osg::Matrix matrix;
};
}
#endif
