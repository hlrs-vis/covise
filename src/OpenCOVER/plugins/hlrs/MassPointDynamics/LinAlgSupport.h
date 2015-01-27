/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef LinAlgSupport_h
#define LinAlgSupport_h

#include <cmath>

struct Vec3d
{
    Vec3d()
        : x(0.0)
        , y(0.0)
        , z(0.0)
    {
    }

    Vec3d(double xset, double yset, double zset)
        : x(xset)
        , y(yset)
        , z(zset)
    {
    }

    Vec3d operator+(const Vec3d &addVec) const
    {
        return Vec3d(x + addVec.x, y + addVec.y, z + addVec.z);
    }

    Vec3d operator-(const Vec3d &addVec) const
    {
        return Vec3d(x - addVec.x, y - addVec.y, z - addVec.z);
    }

    Vec3d &operator+=(const Vec3d &addVec)
    {
        x += addVec.x;
        y += addVec.y;
        z += addVec.z;
        return *this;
    }

    Vec3d operator*(const double &multScal) const
    {
        return Vec3d(x * multScal, y * multScal, z * multScal);
    }

    double operator*(const Vec3d &multVec) const
    {
        return x * multVec.x + y * multVec.y + z * multVec.z;
    }

    Vec3d cross(const Vec3d &other) const
    {
        return Vec3d(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    double norm() const
    {
        return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
    }

    double squarenorm() const
    {
        return (pow(x, 2) + pow(y, 2) + pow(z, 2));
    }

    double x, y, z;
};

#endif
