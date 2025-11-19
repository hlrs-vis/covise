/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <util/coExport.h>
#include <vsg/maths/vec3.h>
#include <vsg/maths/mat4.h>
#include <vsg/maths/transform.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.141592653
#endif
template<typename T>
constexpr vsg::t_vec3<T> getTrans(const vsg::t_mat4<T>& mat)
{
    return vsg::t_vec3<T>(mat[3][0], mat[3][1], mat[3][2]);
}
template<typename T>
constexpr void setTrans(vsg::t_mat4<T>& mat, vsg::t_vec3<T>vec)
{
    mat[3][0] = vec[0];
    mat[3][1] = vec[1];
    mat[3][2] = vec[2];
}
template<typename T>
constexpr vsg::t_mat4<T> makeEulerMat(T h, T p, T r)
{
    vsg::t_mat4<T> m;
    double sr = sin(r / 180.0 * M_PI);
    double sp = sin(p / 180.0 * M_PI);
    double sh = sin(h / 180.0 * M_PI);
    double cr = cos(r / 180.0 * M_PI);
    double cp = cos(p / 180.0 * M_PI);
    double ch = cos(h / 180.0 * M_PI);
    m[0][0] = ch * cr - sh * sr * sp;
    m[1][0] = cr * sh + ch * sr * sp;
    m[2][0] = -sr * cp;
    m[3][0] = 0;
    m[0][1] = -sh * cp;
    m[1][1] = ch * cp;
    m[2][1] = sp;
    m[3][1] = 0;
    m[0][2] = sp * cr * sh + sr * ch;
    m[1][2] = sr * sh - sp * cr * ch;
    m[2][2] = cp * cr;
    m[3][2] = 0;
    m[0][3] = 0;
    m[1][3] = 0;
    m[2][3] = 0;
    m[3][3] = 1;
    return m;
}
template<typename T>
constexpr vsg::t_mat4<T> makeEulerMat(vsg::t_vec3<T> hpr)
{
    return(makeEulerMat(hpr[0], hpr[1], hpr[2]));
}


class VSGVRUIEXPORT coCoord
{
public:
    coCoord(){};
    coCoord(const vsg::dmat4 &right);
    ~coCoord();
    coCoord(const coCoord &c);
    vsg::dvec3 xyz;
    vsg::dvec3 hpr;
    coCoord &operator=(const vsg::dmat4 &right);
    void makeMat(vsg::dmat4 &right) const;

private:
    void initFromMatrix(const vsg::dmat4 &right);
};

// snap matrix 45 degrees in orientation
void VSGVRUIEXPORT snapTo45Degrees(vsg::dmat4 *mat);
// snap matrix 5 degrees in orientation
void VSGVRUIEXPORT snapToDegrees(float degree, vsg::dmat4 *mat);
// modulo for doubles
double VSGVRUIEXPORT mod(double a, double b);
