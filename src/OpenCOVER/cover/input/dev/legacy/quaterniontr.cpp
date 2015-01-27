/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// class stripped down for tracking by Marcel

// **********************************************************************
// * \author Sven Havemann, Institute for Computer Graphics,            *
// *         TU Braunschweig, Muehlenpfordtstrasse 23,                  *
// *         38106 Braunschweig, Germany                                *
// *         http://www.cg.cs.tu-bs.de      mailto:s.havemann@tu-bs.de  *
// * \date   01.10.2000                                                 *
// **********************************************************************

#include <stdio.h>
#include <string.h>
/*#ifdef USE_OPENGLEAN
#include <GL/openglean.h>
#else
#include <GL/glut.h>
#endif*/

#include "quaterniontr.hpp"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// **********************************************************************
// *   Quaternion class
// *

Quaternion::Quaternion()
    : v_s(1.0)
{
}

Quaternion::Quaternion(float s, const CGVVec3 &v)
    : v_s(s)
    , v_v(v)
{
}

void Quaternion::assign(float s, const CGVVec3 &v)
{
    v_s = s;
    v_v = v;
}

void Quaternion::assign(float s, float x, float y, float z)
{
    v_s = s;
    v_v.assign(x, y, z);
}
/*
float Quaternion::dot (const Quaternion& q2) const
{ 
   return ( v_s*q2.v_s + v_v.x*q2.v_v.x + v_v.y*q2.v_v.y + v_v.z*q2.v_v.z); 
}
*/

std::ostream &operator<<(std::ostream &os, const Quaternion &q)
{
    return (os << '(' << q.v_s << ',' << q.v_v << ')');
}

std::istream &operator>>(std::istream &is, Quaternion &q)
{
    char c;
    float s;
    CGVVec3 v;
    is >> c;
    if (c == '(')
    {
        is >> s >> c;
        if (c == ',')
        {
            is >> v >> c;
            if (c != ')')
            {
                is.clear(std::ios_base::badbit);
            }
        }
        else
        {
            is.clear(std::ios_base::badbit);
        }
    }
    else
    {
        is.clear(std::ios_base::badbit);
    }
    if (is)
    {
        q = Quaternion(s, v);
    }
    return is;
}

QuaternionRot::QuaternionRot()
    : Quaternion(1.0f, CGVVec3(0, 0, 0))
{
}

QuaternionRot::QuaternionRot(float s, const CGVVec3 &v)
    :

    Quaternion((float)cos(s / 2.0f), CGVVec3((float)sin(s / 2.0f) * v.x, (float)sin(s / 2.0f) * v.y, (float)sin(s / 2.0f) * v.z))
{
}

void QuaternionRot::assignRot(float s, const CGVVec3 &v)
{
    v_s = (float)cos(s / 2.0f);
    v_v.x = (float)sin(s / 2.0f) * v.x;
    v_v.y = (float)sin(s / 2.0f) * v.y;
    v_v.z = (float)sin(s / 2.0f) * v.z;
}

CGVVec3 QuaternionRot::rotate(const CGVVec3 &v) const
{
    float a = v_s * v_s - (v_v.x * v_v.x + v_v.y * v_v.y + v_v.z * v_v.z); //  4m 3a
    float b = 2.0f * (v.x * v_v.x + v.y * v_v.y + v.z * v_v.z); //  4m 2a
    float c = 2.0f * v_s; //  1m 0a
    return CGVVec3(
        a * v.x + b * v_v.x + c * (v_v.y * v.z - v.y * v_v.z), // 5m 3a
        a * v.y + b * v_v.y + c * (v_v.z * v.x - v.z * v_v.x), // 5m 3a
        a * v.z + b * v_v.z + c * (v_v.x * v.y - v.x * v_v.y) // 5m 3a
        );
}

void QuaternionRot::toRotMatrix(float *matrix) const // 4x4 matrix: first row, then second, ... last row
{
    float qx = v_v.x;
    float qy = v_v.y;
    float qz = v_v.z;
    float qw = v_s;

    float s = qx * qx + qy * qy + qz * qz + qw * qw; // 4m 3a

    if (s < 0.0000001)
    {
        matrix[0] = 1;
        matrix[1] = 0;
        matrix[2] = 0;
        matrix[3] = 0;
        matrix[4] = 0;
        matrix[5] = 1;
        matrix[6] = 0;
        matrix[7] = 0;
        matrix[8] = 0;
        matrix[9] = 0;
        matrix[10] = 1;
        matrix[11] = 0;
        matrix[12] = 0;
        matrix[13] = 0;
        matrix[14] = 0;
        matrix[15] = 1;
    }
    s = 2.0f / s; // 1m 0a

    float xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;
    xs = qx * s;
    ys = qy * s;
    zs = qz * s; // 3m 0a
    wx = qw * xs;
    wy = qw * ys;
    wz = qw * zs; // 3m 0a
    xx = qx * xs;
    xy = qx * ys;
    xz = qx * zs; // 3m 0a
    yy = qy * ys;
    yz = qy * zs;
    zz = qz * zs; // 3m 0a

    // 0m 12a
    matrix[0] = 1.0f - yy - zz;
    matrix[1] = xy + wz;
    matrix[2] = xz - wy;
    matrix[3] = 0;
    matrix[4] = xy - wz;
    matrix[5] = 1.0f - xx - zz;
    matrix[6] = yz + wx;
    matrix[7] = 0;
    matrix[8] = xz + wy;
    matrix[9] = yz - wx;
    matrix[10] = 1.0f - xx - yy;
    matrix[11] = 0;
    matrix[12] = 0;
    matrix[13] = 0;
    matrix[14] = 0;
    matrix[15] = 1;
}

Quaternion &QuaternionRot::fromRotMatrix(const float *rotmat) // 4x4 matrix: first row, then second, ... last row
{
    float s;
    float trace = rotmat[0] + rotmat[5] + rotmat[10] + 1.0f; // 0m 3a

    if (trace > 0.1)
    { // this is an epsilon but it doesn't hurt if it's big like 0.1
        s = (float)sqrt(trace); // 1m 1a   ...sqrt as mult?
        v_s = s * 0.5f; // 1m 0a
        s = 0.5f / s; // 1m 0a
        v_v.assign(s * (rotmat[6] - rotmat[9]), // 1m 1a
                   s * (rotmat[8] - rotmat[2]), // 1m 1a
                   s * (rotmat[1] - rotmat[4])); // 1m 1a
    }
    else
    {
        if ((rotmat[0] > rotmat[5]) && (rotmat[0] > rotmat[10]))
        { // Column 0:
            s = (float)sqrt(1.0 + rotmat[0] - rotmat[5] - rotmat[10]); // 0m 3a 1s
            if (s < 0.000001)
            {
                return *this;
            }
            v_v.x = 0.5f * s; // 1m 0a
            s = 0.5f / s; // 1m 0a
            v_v.y = s * (rotmat[1] + rotmat[4]); // 1m 1a
            v_v.z = s * (rotmat[8] + rotmat[2]); // 1m 1a
            v_s = s * (rotmat[6] - rotmat[9]); // 1m 1a
        }
        else if (rotmat[5] > rotmat[10])
        { // Column 1:
            s = (float)sqrt(1.0 + rotmat[5] - rotmat[0] - rotmat[10]); // 0m 3a 1s
            if (s < 0.000001)
            {
                return *this;
            }
            v_v.y = 0.5f * s; // 1m 0a
            s = 0.5f / s; // 1m 0a
            v_v.x = s * (rotmat[1] + rotmat[4]); // 1m 1a
            v_v.z = s * (rotmat[6] + rotmat[9]); // 1m 1a
            v_s = s * (rotmat[8] - rotmat[2]); // 1m 1a
        }
        else
        { // Column 2:
            s = (float)sqrt(1.0 + rotmat[10] - rotmat[0] - rotmat[5]); // 0m 3a 1s
            if (s < 0.000001)
            {
                return *this;
            }
            v_v.z = 0.5f * s; // 1m 0a
            s = 0.5f / s; // 1m 0a
            v_v.x = s * (rotmat[8] + rotmat[2]); // 1m 1a
            v_v.y = s * (rotmat[6] + rotmat[9]); // 1m 1a
            v_s = s * (rotmat[1] - rotmat[4]); // 1m 1a
        }
    }
    return *this;
}
/*
void QuaternionRot::applyToCurrentGLMatrix() {
    float angle = acos(v_s)*2 * 180/M_PI;
	// (axis not normalized)
	glRotatef(angle, v_v.x, v_v.y, v_v.z);
}
*/
