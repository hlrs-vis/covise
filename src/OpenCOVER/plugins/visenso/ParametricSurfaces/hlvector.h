/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PV_H_
#define _PV_H_

#include <math.h>

#include <iostream>
using namespace std;

#define HL_PI 3.14159265358979323846
#define WINKELEPSILON 0.00001
#define EPSILON 0.00001
#define TOGRAD(x) ((x) / HL_PI * 180.0)
#define TORAD(x) ((x)*HL_PI / 180.0)

class HlVector;
class HlPoint;

HlVector operator/(const HlVector &v, double a);

class HlVector
{

public:
    double mX, mY, mZ;

public:
    HlVector()
        : mX(0)
        , mY(0)
        , mZ(0)
    {
    }
    HlVector(double x, double y, double z = 0)
        : mX(x)
        , mY(y)
        , mZ(z)
    {
    }
    HlVector(const HlVector &v)
    {
        *this = v;
    }

    void setXYZ(double x, double y, double z)
    {
        mX = x;
        mY = y;
        mZ = z;
    }

    const HlVector &operator=(const HlVector &v)
    {
        if (this == &v)
            return (*this);
        mX = v.mX;
        mY = v.mY;
        mZ = v.mZ;
        return (*this);
    }

    bool equalQ(const HlVector &v) const
    {
        if (this == &v)
            return true;
        return (mX == v.mX) && (mY == v.mY) && (mZ == v.mZ);
    }

    bool nullQ() const
    {
        return (mX == 0) && (mY == 0) && (mZ == 0);
    }

    void out(ostream &strm) const
    {
        strm << "(" << mX << "," << mY << "," << mZ << ")";
    }

    HlVector plus(const HlVector &v) const
    {
        return HlVector(mX + v.mX, mY + v.mY, mZ + v.mZ);
    }

    HlVector minus(const HlVector &v) const
    {
        return HlVector(mX - v.mX, mY - v.mY, mZ - v.mZ);
    }

    HlVector scalarmult(double a) const
    {
        return HlVector(a * mX, a * mY, a * mZ);
    }

    double skalarprodukt(const HlVector &v) const
    {
        return (mX * v.mX + mY * v.mY + mZ * v.mZ);
    }

    HlVector kreuzprodukt(const HlVector &v) const
    {
        return HlVector(mY * v.mZ - mZ * v.mY, mZ * v.mX - mX * v.mZ, mX * v.mY - mY * v.mX);
    }

    HlVector operator-() const
    {
        return HlVector(-mX, -mY, -mZ);
    }

    const HlVector &operator+=(const HlVector &v)
    {
        mX += v.mX;
        mY += v.mY;
        mZ += v.mZ;
        return *this;
    }

    const HlVector &operator-=(const HlVector &v)
    {
        mX -= v.mX;
        mY -= v.mY;
        mZ -= v.mZ;
        return *this;
    }

    const HlVector &operator*=(const double a)
    {
        mX *= a;
        mY *= a;
        mZ *= a;
        return *this;
    }

    double betrag() const
    {
        return sqrt(quadrat());
    }

    double quadrat() const
    {
        return skalarprodukt(*this);
    }

    int normierbarQ() const
    {
        return (quadrat() != 0);
    }

    HlVector normiert() const
    {
        if (normierbarQ())
            return *this / betrag();
        else
            return HlVector(0, 0, 0);
    }
};

inline ostream &operator<<(ostream &strm, const HlVector &v)
{
    v.out(strm);
    return strm;
}

inline bool operator==(const HlVector &vl, const HlVector &vr)
{
    return vl.equalQ(vr);
}

inline bool operator!=(const HlVector &vl, const HlVector &vr)
{
    return !vl.equalQ(vr);
}

inline HlVector operator+(const HlVector &vl, const HlVector &vr)
{
    return vl.plus(vr);
}

inline HlVector operator-(const HlVector &vl, const HlVector &vr)
{
    return vl.minus(vr);
}

inline HlVector operator*(double a, const HlVector &vr)
{
    return vr.scalarmult(a);
}

inline HlVector operator*(const HlVector &vl, double a)
{
    return vl.scalarmult(a);
}

inline double operator*(const HlVector &vl, const HlVector &vr)
{
    return vl.skalarprodukt(vr);
}

inline HlVector operator%(const HlVector &vl, const HlVector &vr)
{
    return vl.kreuzprodukt(vr);
}

inline HlVector operator/(const HlVector &v, double a)
{
    return v.scalarmult(1.0 / a);
}

inline double winkel(const HlVector &v1, const HlVector &v2)
{
    HlVector v1n = HlVector(v1).normiert();
    HlVector v2n = HlVector(v2).normiert();
    return atan2((v1n % v2n).betrag(), v1n * v2n);
}

inline double spat(const HlVector &a, const HlVector &b, const HlVector &c)
{
    return (a % b) * c;
}

inline double HlAngle2(const HlVector &a, const HlVector &b)
{
    double angle = atan2(b.mY - a.mY, b.mX - a.mX);
    if (angle < 0.)
        angle += 2 * HL_PI;
    return angle;
}

inline double HlAngle2(const HlVector &a, const HlVector &b, const HlVector &c, const HlVector &d)
{
    double ang12 = HlAngle2(a, b);
    double ang13 = HlAngle2(c, d);

    return (ang12 > ang13) ? (ang13 - ang12 + 2 * HL_PI) : (ang13 - ang12);
}

inline double HlAngle2(const HlVector &a, const HlVector &b, const HlVector &c)
{
    return HlAngle2(a, b, a, c);
}

#endif
