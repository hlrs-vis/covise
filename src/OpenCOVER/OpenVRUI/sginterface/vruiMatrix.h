/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_MATRIX_H
#define VRUI_MATRIX_H

#include <util/coTypes.h>
#include <util/coVector.h>

namespace vrui
{

using covise::coVector;

class OPENVRUIEXPORT vruiMatrix
{

public:
    vruiMatrix()
    {
    }
    virtual ~vruiMatrix();

    virtual vruiMatrix *makeIdentity() = 0;
    virtual vruiMatrix *makeEuler(double h, double p, double r) = 0;
    virtual vruiMatrix *makeRotate(double degrees, double xAxis, double yAxis, double zAxis) = 0;
    virtual vruiMatrix *makeScale(double x, double y, double z) = 0;
    virtual vruiMatrix *makeTranslate(double x, double y, double z) = 0;
    virtual vruiMatrix *makeInverse(const vruiMatrix *source) = 0;

    virtual vruiMatrix *setTranslation(double x, double y, double z)
    {
        (*this)(3, 0) = x;
        (*this)(3, 1) = y;
        (*this)(3, 2) = z;
        return this;
    }

    virtual vruiMatrix *preTranslated(double x, double y, double z, vruiMatrix *matrix) = 0;
    virtual vruiMatrix *mult(const vruiMatrix *matrix) = 0;

    virtual double &operator()(int row, int column) = 0;
    virtual double operator()(int row, int column) const = 0;

    virtual coVector getFullXformPt(const coVector &point) const = 0;

    virtual coVector getTranslate() const
    {
        coVector rv;
        rv[0] = (*this)(3, 0);
        rv[1] = (*this)(3, 1);
        rv[2] = (*this)(3, 2);
        return rv;
    }

    virtual coVector getHPR() const = 0;

    bool isIdentity() const;
};

class OPENVRUIEXPORT vruiCoord
{

public:
    vruiCoord()
    {
    }
    vruiCoord(const vruiMatrix *right)
    {
        hpr = right->getHPR();
        xyz = right->getTranslate();
    }

    inline vruiCoord &operator=(const vruiMatrix *right)
    {
        hpr = right->getHPR();
        xyz = right->getTranslate();
        return *this;
    }

    inline void makeCoordMatrix(vruiMatrix *right)
    {
        right->makeEuler(hpr[0], hpr[1], hpr[2]);
        right->setTranslation(xyz[0], xyz[1], xyz[2]);
    }

    coVector hpr;
    coVector xyz;
};
}
#endif
