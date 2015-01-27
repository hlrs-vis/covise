/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description: Interface classes for application modules to the COVISE   **
 **              software environment                                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C)1997 RUS                                **
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 ** Author: D. Rantzau                                                     **
 ** Date:  15.08.97  V1.0                                                  **
\**************************************************************************/

#include "coMatrix.h"

using namespace covise;

// This function build the invers matrix

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::invers() const
{
    coMatrix im;
    double a0 = val[0][0];
    double a1 = val[0][1];
    double a2 = val[0][2];
    double a3 = val[0][3];
    double a4 = val[1][0];
    double a5 = val[1][1];
    double a6 = val[1][2];
    double a7 = val[1][3];
    double a8 = val[2][0];
    double a9 = val[2][1];
    double a10 = val[2][2];
    double a11 = val[2][3];
    double a12 = val[3][0];
    double a13 = val[3][1];
    double a14 = val[3][2];
    double a15 = val[3][3];
    double a1a11a14a4 = a1 * a11 * a14 * a4;
    double a1a10a15a4 = a1 * a10 * a15 * a4;
    double a11a13a2a4 = a11 * a13 * a2 * a4;
    double a10a13a3a4 = a10 * a13 * a3 * a4;
    double a0a11a14a5 = a0 * a11 * a14 * a5;
    double a0a10a15a5 = a0 * a10 * a15 * a5;
    double a11a12a2a5 = a11 * a12 * a2 * a5;
    double a10a12a3a5 = a10 * a12 * a3 * a5;
    double a1a11a12a6 = a1 * a11 * a12 * a6;
    double a0a11a13a6 = a0 * a11 * a13 * a6;
    double a1a10a12a7 = a1 * a10 * a12 * a7;
    double a0a10a13a7 = a0 * a10 * a13 * a7;
    double a15a2a5a8 = a15 * a2 * a5 * a8;
    double a14a3a5a8 = a14 * a3 * a5 * a8;
    double a1a15a6a8 = a1 * a15 * a6 * a8;
    double a13a3a6a8 = a13 * a3 * a6 * a8;
    double a1a14a7a8 = a1 * a14 * a7 * a8;
    double a13a2a7a8 = a13 * a2 * a7 * a8;
    double a15a2a4a9 = a15 * a2 * a4 * a9;
    double a14a3a4a9 = a14 * a3 * a4 * a9;
    double a0a15a6a9 = a0 * a15 * a6 * a9;
    double a12a3a6a9 = a12 * a3 * a6 * a9;
    double a0a14a7a9 = a0 * a14 * a7 * a9;
    double a12a2a7a9 = a12 * a2 * a7 * a9;
    double a11a14a5 = a11 * a14 * a5;
    double a10a15a5 = a10 * a15 * a5;
    double a11a13a6 = a11 * a13 * a6;
    double a10a13a7 = a10 * a13 * a7;
    double a15a6a9 = a15 * a6 * a9;
    double a14a7a9 = a14 * a7 * a9;
    double a1a11a14 = a1 * a11 * a14;
    double a1a10a15 = a1 * a10 * a15;
    double a11a13a2 = a11 * a13 * a2;
    double a10a13a3 = a10 * a13 * a3;
    double a15a2a9 = a15 * a2 * a9;
    double a14a3a9 = a14 * a3 * a9;
    double a15a2a5 = a15 * a2 * a5;
    double a14a3a5 = a14 * a3 * a5;
    double a1a15a6 = a1 * a15 * a6;
    double a13a3a6 = a13 * a3 * a6;
    double a1a14a7 = a1 * a14 * a7;
    double a13a2a7 = a13 * a2 * a7;
    double a11a2a5 = a11 * a2 * a5;
    double a10a3a5 = a10 * a3 * a5;
    double a1a11a6 = a1 * a11 * a6;
    double a1a10a7 = a1 * a10 * a7;
    double a3a6a9 = a3 * a6 * a9;
    double a2a7a9 = a2 * a7 * a9;
    double a11a14a4 = a11 * a14 * a4;
    double a10a15a4 = a10 * a15 * a4;
    double a11a12a6 = a11 * a12 * a6;
    double a10a12a7 = a10 * a12 * a7;
    double a15a6a8 = a15 * a6 * a8;
    double a14a7a8 = a14 * a7 * a8;
    double a0a11a14 = a0 * a11 * a14;
    double a0a10a15 = a0 * a10 * a15;
    double a11a12a2 = a11 * a12 * a2;
    double a10a12a3 = a10 * a12 * a3;
    double a15a2a8 = a15 * a2 * a8;
    double a14a3a8 = a14 * a3 * a8;
    double a15a2a4 = a15 * a2 * a4;
    double a14a3a4 = a14 * a3 * a4;
    double a0a15a6 = a0 * a15 * a6;
    double a12a3a6 = a12 * a3 * a6;
    double a0a14a7 = a0 * a14 * a7;
    double a12a2a7 = a12 * a2 * a7;
    double a11a2a4 = a11 * a2 * a4;
    double a10a3a4 = a10 * a3 * a4;
    double a0a11a6 = a0 * a11 * a6;
    double a0a10a7 = a0 * a10 * a7;
    double a3a6a8 = a3 * a6 * a8;
    double a2a7a8 = a2 * a7 * a8;
    double a11a13a4 = a11 * a13 * a4;
    double a11a12a5 = a11 * a12 * a5;
    double a15a5a8 = a15 * a5 * a8;
    double a13a7a8 = a13 * a7 * a8;
    double a15a4a9 = a15 * a4 * a9;
    double a12a7a9 = a12 * a7 * a9;
    double a1a11a12 = a1 * a11 * a12;
    double a0a11a13 = a0 * a11 * a13;
    double a1a15a8 = a1 * a15 * a8;
    double a13a3a8 = a13 * a3 * a8;
    double a0a15a9 = a0 * a15 * a9;
    double a12a3a9 = a12 * a3 * a9;
    double a1a15a4 = a1 * a15 * a4;
    double a13a3a4 = a13 * a3 * a4;
    double a0a15a5 = a0 * a15 * a5;
    double a12a3a5 = a12 * a3 * a5;
    double a1a12a7 = a1 * a12 * a7;
    double a0a13a7 = a0 * a13 * a7;
    double a1a11a4 = a1 * a11 * a4;
    double a0a11a5 = a0 * a11 * a5;
    double a3a5a8 = a3 * a5 * a8;
    double a1a7a8 = a1 * a7 * a8;
    double a3a4a9 = a3 * a4 * a9;
    double a0a7a9 = a0 * a7 * a9;
    double a10a13a4 = a10 * a13 * a4;
    double a10a12a5 = a10 * a12 * a5;
    double a14a5a8 = a14 * a5 * a8;
    double a13a6a8 = a13 * a6 * a8;
    double a14a4a9 = a14 * a4 * a9;
    double a12a6a9 = a12 * a6 * a9;
    double a1a10a12 = a1 * a10 * a12;
    double a0a10a13 = a0 * a10 * a13;
    double a1a14a8 = a1 * a14 * a8;
    double a13a2a8 = a13 * a2 * a8;
    double a0a14a9 = a0 * a14 * a9;
    double a12a2a9 = a12 * a2 * a9;
    double a1a14a4 = a1 * a14 * a4;
    double a13a2a4 = a13 * a2 * a4;
    double a0a14a5 = a0 * a14 * a5;
    double a12a2a5 = a12 * a2 * a5;
    double a1a12a6 = a1 * a12 * a6;
    double a0a13a6 = a0 * a13 * a6;
    double a1a10a4 = a1 * a10 * a4;
    double a0a10a5 = a0 * a10 * a5;
    double a2a5a8 = a2 * a5 * a8;
    double a1a6a8 = a1 * a6 * a8;
    double a2a4a9 = a2 * a4 * a9;
    double a0a6a9 = a0 * a6 * a9;

    double det = a1a11a14a4 - a1a10a15a4 - a11a13a2a4 + a10a13a3a4 - a0a11a14a5 + a0a10a15a5 + a11a12a2a5 - a10a12a3a5 - a1a11a12a6 + a0a11a13a6 + a1a10a12a7 - a0a10a13a7 - a15a2a5a8 + a14a3a5a8 + a1a15a6a8 - a13a3a6a8 - a1a14a7a8 + a13a2a7a8 + a15a2a4a9 - a14a3a4a9 - a0a15a6a9 + a12a3a6a9 + a0a14a7a9 - a12a2a7a9;

    im.val[0][0] = (-a11a14a5 + a10a15a5 + a11a13a6
                    - a10a13a7 - a15a6a9 + a14a7a9) / det;
    im.val[0][1] = (a1a11a14 - a1a10a15 - a11a13a2
                    + a10a13a3 + a15a2a9 - a14a3a9) / det;
    im.val[0][2] = (-a15a2a5 + a14a3a5 + a1a15a6
                    - a13a3a6 - a1a14a7 + a13a2a7) / det;
    im.val[0][3] = (a11a2a5 - a10a3a5 - a1a11a6
                    + a1a10a7 + a3a6a9 - a2a7a9) / det;
    im.val[1][0] = (a11a14a4 - a10a15a4 - a11a12a6
                    + a10a12a7 + a15a6a8 - a14a7a8) / det;
    im.val[1][1] = (-a0a11a14 + a0a10a15 + a11a12a2
                    - a10a12a3 - a15a2a8 + a14a3a8) / det;
    im.val[1][2] = (a15a2a4 - a14a3a4 - a0a15a6
                    + a12a3a6 + a0a14a7 - a12a2a7) / det;
    im.val[1][3] = (-a11a2a4 + a10a3a4 + a0a11a6
                    - a0a10a7 - a3a6a8 + a2a7a8) / det;
    im.val[2][0] = (-a11a13a4 + a11a12a5 - a15a5a8 + a13a7a8 + a15a4a9 - a12a7a9) / det;
    im.val[2][1] = (-a1a11a12 + a0a11a13 + a1a15a8
                    - a13a3a8 - a0a15a9 + a12a3a9) / det;
    im.val[2][2] = (-a1a15a4 + a13a3a4 + a0a15a5
                    - a12a3a5 + a1a12a7 - a0a13a7) / det;
    im.val[2][3] = (a1a11a4 - a0a11a5 + a3a5a8
                    - a1a7a8 - a3a4a9 + a0a7a9) / det;
    im.val[3][0] = (a10a13a4 - a10a12a5 + a14a5a8
                    - a13a6a8 - a14a4a9 + a12a6a9) / det;
    im.val[3][1] = (a1a10a12 - a0a10a13 - a1a14a8
                    + a13a2a8 + a0a14a9 - a12a2a9) / det;
    im.val[3][2] = (a1a14a4 - a13a2a4 - a0a14a5
                    + a12a2a5 - a1a12a6 + a0a13a6) / det;
    im.val[3][3] = (-a1a10a4 + a0a10a5 - a2a5a8
                    + a1a6a8 + a2a4a9 - a0a6a9) / det;

    im.changed = 1;
    return im;
}

//==========================================================================
//
//==========================================================================
// build the transposed matrix of the current matrix
coMatrix coMatrix::transpose() const
{
    coMatrix tm;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            tm.val[i][j] = val[j][i];
    tm.changed = 1;
    return tm;
}

//==========================================================================
//
//==========================================================================

//generates a rotation matrix from a quaternion. See documentation in function.
void coMatrix::fromQuat(const float quat0, const float quat1, const float quat2, const float quat3)
{
    float _v[4];
    _v[0] = quat0;
    _v[1] = quat1;
    _v[2] = quat2;
    _v[3] = quat3;
    float length2 = sqrt(_v[0] * _v[0] + _v[1] * _v[1] + _v[2] * _v[2] + _v[3] * _v[3]);
    if (length2 != 1.0 && length2 != 0)
    {
        // normalize quat if required.
        _v[0] /= sqrt(length2);
        _v[1] /= sqrt(length2);
        _v[2] /= sqrt(length2);
        _v[3] /= sqrt(length2);
    }

    // Source: Gamasutra, Rotating Objects Using Quaternions
    //
    //http://www.gamasutra.com/features/programming/19980703/quaternions_01.htm

    float wx, wy, wz, xx, yy, yz, xy, xz, zz, x2, y2, z2;

    // calculate coefficients
    x2 = _v[0] + _v[0];
    y2 = _v[1] + _v[1];
    z2 = _v[2] + _v[2];

    xx = _v[0] * x2;
    xy = _v[0] * y2;
    xz = _v[0] * z2;

    yy = _v[1] * y2;
    yz = _v[1] * z2;
    zz = _v[2] * z2;

    wx = _v[3] * x2;
    wy = _v[3] * y2;
    wz = _v[3] * z2;

    // Note.  Gamasutra gets the matrix assignments inverted, resulting
    // in left-handed rotations, which is contrary to OpenGL and OSG's
    // methodology.  The matrix assignment has been altered in the next
    // few lines of code to do the right thing.
    // Don Burns - Oct 13, 2001

    val[0][0] = 1.0f - (yy + zz);
    val[1][0] = xy - wz;
    val[2][0] = xz + wy;
    val[3][0] = 0.0f;

    val[0][1] = xy + wz;
    val[1][1] = 1.0f - (xx + zz);
    val[2][1] = yz - wx;
    val[3][1] = 0.0f;

    val[0][2] = xz - wy;
    val[1][2] = yz + wx;
    val[2][2] = 1.0f - (xx + yy);
    val[3][2] = 0.0f;

    val[0][3] = 0.0f;
    val[1][3] = 0.0f;
    val[2][3] = 0.0f;
    val[3][3] = 1.0f;
}

//==========================================================================
//
//==========================================================================
int coMatrix::operator==(const coMatrix &m) const
{
    double r = 0.0;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            r += (val[i][j] - m.val[i][j]) * (val[i][j] - m.val[i][j]);

    return (r == 0);
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::operator+(const coMatrix &b) const
{
    coMatrix m;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            m.val[i][j] = val[i][j] + b.val[i][j];
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::operator-(const coMatrix &b) const
{
    coMatrix m;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            m.val[i][j] = val[i][j] - b.val[i][j];
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::operator*(const coMatrix &b) const
{
    coMatrix m;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            m.val[i][j] = 0.0;
            for (int k = 0; k < 4; k++)
                m.val[i][j] += val[i][k] * b.val[k][j];
        }
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coVector coMatrix::operator*(const coVector &v) const
{
    coVector r;
    r.vec[0] = v.vec[0] * val[0][0] + v.vec[1] * val[0][1] + v.vec[2] * val[0][2] + val[0][3];
    r.vec[1] = v.vec[0] * val[1][0] + v.vec[1] * val[1][1] + v.vec[2] * val[1][2] + val[1][3];
    r.vec[2] = v.vec[0] * val[2][0] + v.vec[1] * val[2][1] + v.vec[2] * val[2][2] + val[2][3];

    double w = v.vec[0] * val[3][0]
               + v.vec[1] * val[3][1]
               + v.vec[2] * val[3][2]
               + val[3][3];

    r.vec[0] /= w;
    r.vec[1] /= w;
    r.vec[2] /= w;
    r.vec[3] = 1.;

    return r;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::translation(const coVector &v) const
{
    coMatrix tmp, m;
    tmp.val[0][3] = v.vec[0];
    tmp.val[1][3] = v.vec[1];
    tmp.val[2][3] = v.vec[2];
    m = *this * tmp;
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::invTranslation(const coVector &v) const
{
    coMatrix tmp, m;
    tmp.val[0][3] = -v.vec[0];
    tmp.val[1][3] = -v.vec[1];
    tmp.val[2][3] = -v.vec[2];
    m = tmp * (*this);
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::rotationX(const double angle) const
{
    coMatrix tmp, m;
    double cos_val = cos(angle);
    double sin_val = sin(angle);
    tmp.val[1][1] = cos_val;
    tmp.val[2][2] = cos_val;
    tmp.val[1][2] = sin_val;
    tmp.val[2][1] = -sin_val;

    m = *this * tmp;
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::invRotationX(const double angle) const
{
    double xangle = angle * (-1);
    coMatrix tmp, m;
    double cos_val = cos(xangle);
    double sin_val = sin(xangle);
    tmp.val[1][1] = cos_val;
    tmp.val[2][2] = cos_val;
    tmp.val[1][2] = sin_val;
    tmp.val[2][1] = -sin_val;

    m = tmp * *this;
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::rotationY(const double angle) const
{
    coMatrix tmp, m;
    double cos_val = cos(angle);
    double sin_val = sin(angle);
    tmp.val[0][0] = cos_val;
    tmp.val[2][2] = cos_val;
    tmp.val[0][2] = -sin_val;
    tmp.val[2][0] = sin_val;

    m = *this * tmp;
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::invRotationY(const double angle) const
{
    double xangle = angle * (-1);
    coMatrix tmp, m;
    double cos_val = cos(xangle);
    double sin_val = sin(xangle);
    tmp.val[0][0] = cos_val;
    tmp.val[2][2] = cos_val;
    tmp.val[0][2] = -sin_val;
    tmp.val[2][0] = sin_val;

    m = tmp * *this;
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::rotationZ(const double angle) const
{
    coMatrix tmp, m;
    double cos_val = cos(angle);
    double sin_val = sin(angle);
    tmp.val[0][0] = cos_val;
    tmp.val[1][1] = cos_val;
    tmp.val[0][1] = sin_val;
    tmp.val[1][0] = -sin_val;

    m = *this * tmp;
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::invRotationZ(const double angle) const
{
    double xangle = angle * (-1);
    coMatrix tmp, m;
    double cos_val = cos(xangle);
    double sin_val = sin(xangle);
    tmp.val[0][0] = cos_val;
    tmp.val[1][1] = cos_val;
    tmp.val[0][1] = sin_val;
    tmp.val[1][0] = -sin_val;

    m = tmp * *this;
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix coMatrix::operator*(const double &d) const
{
    coMatrix m;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            m.val[i][j] = val[i][j] * d;
    m.changed = 1;
    return m;
}

//==========================================================================
//
//==========================================================================
coMatrix &coMatrix::setRotation(const coVector &axis, double angle)
{

    double ca = cos(angle);
    double sa = sin(angle);
    double ta = 1 - ca;

    val[0][0] = ta * axis[0] * axis[0] + ca;
    val[0][1] = ta * axis[0] * axis[1] - axis[2] * sa;
    val[0][2] = ta * axis[0] * axis[2] + axis[1] * sa;
    val[1][0] = ta * axis[0] * axis[1] + axis[2] * sa;
    val[1][1] = ta * axis[1] * axis[1] + ca;
    val[1][2] = ta * axis[1] * axis[2] - axis[0] * sa;
    val[2][0] = ta * axis[0] * axis[2] - axis[1] * sa;
    val[2][1] = ta * axis[1] * axis[2] + axis[0] * sa;
    val[2][2] = ta * axis[2] * axis[2] + ca;

    return *this;
}
