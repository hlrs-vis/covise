/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_MATH_UTILS_H_
#define CO_MATH_UTILS_H_
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <util/coExport.h>
#include <osg/Vec3>
#include <osg/Matrix>
#include <math.h>

#ifndef M_PI
#define M_PI 3.141592653
#endif

//#define MAKE_EULER_MAT(m,h,p,r)	  m.makeRotate(h,osg::Y_AXIS, p,osg::X_AXIS, r,osg::Z_AXIS)

#define MAKE_EULER_MAT(m, h, p, r)        \
    {                                     \
        double sr, sp, sh, cr, cp, ch;    \
        sr = sin(r / 180.0 * M_PI);       \
        sp = sin(p / 180.0 * M_PI);       \
        sh = sin(h / 180.0 * M_PI);       \
        cr = cos(r / 180.0 * M_PI);       \
        cp = cos(p / 180.0 * M_PI);       \
        ch = cos(h / 180.0 * M_PI);       \
        m(0, 0) = ch * cr - sh * sr * sp; \
        m(0, 1) = cr * sh + ch * sr * sp; \
        m(0, 2) = -sr * cp;               \
        m(0, 3) = 0;                      \
        m(1, 0) = -sh * cp;               \
        m(1, 1) = ch * cp;                \
        m(1, 2) = sp;                     \
        m(1, 3) = 0;                      \
        m(2, 0) = sp * cr * sh + sr * ch; \
        m(2, 1) = sr * sh - sp * cr * ch; \
        m(2, 2) = cp * cr;                \
        m(2, 3) = 0;                      \
        m(3, 0) = 0;                      \
        m(3, 1) = 0;                      \
        m(3, 2) = 0;                      \
        m(3, 3) = 1;                      \
    }
#define MAKE_EULER_MAT_VEC(m, hpr) MAKE_EULER_MAT(m, hpr[0], hpr[1], hpr[2])

#if 0
//#define GET_HPR(m,h,p,r)          { float cp; p= asin(m(1,2)); cp = cos(p); r = acos(m(2,2)/cp); h = -asin(m(1,0)/cp);  }
#define GET_HPR(m, h, p, r)                                                                       \
    {                                                                                             \
        float cp;                                                                                 \
        osg::Vec3 v1(m(0, 0), m(0, 1), m(0, 2));                                                  \
        osg::Vec3 v2(m(1, 0), m(1, 1), m(1, 2));                                                  \
        osg::Vec3 v3(m(2, 0), m(2, 1), m(2, 2));                                                  \
        v1.normalize();                                                                           \
        v2.normalize();                                                                           \
        v3.normalize();                                                                           \
        m(0, 0) = v1[0];                                                                          \
        m(0, 1) = v1[1];                                                                          \
        m(0, 2) = v1[2];                                                                          \
        m(1, 0) = v2[0];                                                                          \
        m(1, 1) = v2[1];                                                                          \
        m(1, 2) = v2[2];                                                                          \
        m(2, 0) = v3[0];                                                                          \
        m(2, 1) = v3[1];                                                                          \
        m(2, 2) = v3[2];                                                                          \
        p = asin(m(1, 2));                                                                        \
        cp = cos(p);                                                                              \
        float d = m(1, 0) / cp;                                                                   \
        if (d > 1.0)                                                                              \
        {                                                                                         \
            h = -M_PI_2;                                                                          \
        }                                                                                         \
        else if (d < -1.0)                                                                        \
        {                                                                                         \
            h = M_PI_2;                                                                           \
        }                                                                                         \
        else                                                                                      \
            h = -asin(d);                                                                         \
        float diff = cos(h) * cp - m(1, 1);                                                       \
        if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other Heading angle */         \
        {                                                                                         \
            h = M_PI_2 - h;                                                                       \
            diff = cos(h) * cp - m(1, 1);                                                         \
            if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other pitch angle */       \
            {                                                                                     \
                p = M_PI - p;                                                                     \
                cp = cos(p);                                                                      \
                d = m(1, 0) / cp;                                                                 \
                if (d > 1.0)                                                                      \
                {                                                                                 \
                    h = -M_PI_2;                                                                  \
                }                                                                                 \
                else if (d < -1.0)                                                                \
                {                                                                                 \
                    h = M_PI_2;                                                                   \
                }                                                                                 \
                else                                                                              \
                    h = -asin(d);                                                                 \
                diff = cos(h) * cp - m(1, 1);                                                     \
                if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other Heading angle */ \
                {                                                                                 \
                    h = M_PI - h;                                                                 \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        d = m(2, 2) / cp;                                                                         \
        if (d > 1.0)                                                                              \
            r = 0;                                                                                \
        else if (d > 1.0)                                                                         \
            r = M_PI;                                                                             \
        else                                                                                      \
            r = acos(d);                                                                          \
        diff = -sin(r) * cp - m(0, 2);                                                            \
        if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other roll angle */            \
            r = -r;                                                                               \
        h = h / M_PI * 180.0;                                                                     \
        p = p / M_PI * 180.0;                                                                     \
        r = r / M_PI * 180.0;                                                                     \
    }

#define GET_HPR_VEC(mx, hpr)                                                                      \
    {                                                                                             \
        float cp;                                                                                 \
        osg::Matrix m;                                                                            \
        osg::Vec3 v1(mx(0, 0), mx(0, 1), mx(0, 2));                                               \
        osg::Vec3 v2(mx(1, 0), mx(1, 1), mx(1, 2));                                               \
        osg::Vec3 v3(mx(2, 0), mx(2, 1), mx(2, 2));                                               \
        v1.normalize();                                                                           \
        v2.normalize();                                                                           \
        v3.normalize();                                                                           \
        m(0, 0) = v1[0];                                                                          \
        m(0, 1) = v1[1];                                                                          \
        m(0, 2) = v1[2];                                                                          \
        m(1, 0) = v2[0];                                                                          \
        m(1, 1) = v2[1];                                                                          \
        m(1, 2) = v2[2];                                                                          \
        m(2, 0) = v3[0];                                                                          \
        m(2, 1) = v3[1];                                                                          \
        m(2, 2) = v3[2];                                                                          \
        hpr[1] = asin(m(1, 2));                                                                   \
        cp = cos(hpr[1]);                                                                         \
        hpr[0] = -asin(m(1, 0) / cp);                                                             \
        float diff = cos(hpr[0]) * cp - m(1, 1);                                                  \
        if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other Heading angle */         \
        {                                                                                         \
            hpr[0] = M_PI - hpr[0];                                                               \
            diff = cos(hpr[0]) * cp - m(1, 1);                                                    \
            if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other pitch angle */       \
            {                                                                                     \
                hpr[1] = M_PI - hpr[1];                                                           \
                cp = cos(hpr[1]);                                                                 \
                hpr[0] = -asin(m(1, 0) / cp);                                                     \
                diff = cos(hpr[0]) * cp - m(1, 1);                                                \
                if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other Heading angle */ \
                {                                                                                 \
                    hpr[0] = M_PI - hpr[0];                                                       \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        hpr[2] = acos(m(2, 2) / cp);                                                              \
        diff = -sin(hpr[2]) * cp - m(0, 2);                                                       \
        if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other roll angle */            \
            hpr[2] = -hpr[2];                                                                     \
        hpr[0] = hpr[0] / M_PI * 180.0;                                                           \
        hpr[1] = hpr[1] / M_PI * 180.0;                                                           \
        hpr[2] = hpr[2] / M_PI * 180.0;                                                           \
    }
#endif

class OSGVRUIEXPORT coCoord
{
public:
    coCoord(){};
    coCoord(const osg::Matrix &right);
    ~coCoord();
    coCoord(const coCoord &c);
    osg::Vec3 xyz;
    osg::Vec3 hpr;
    coCoord &operator=(const osg::Matrix &right);
    void makeMat(osg::Matrix &right);

private:
    void initFromMatrix(const osg::Matrix &right);
};

// snap matrix 45 degrees in orientation
void OSGVRUIEXPORT snapTo45Degrees(osg::Matrix *mat);
// snap matrix 5 degrees in orientation
void OSGVRUIEXPORT snapToDegrees(float degree, osg::Matrix *mat);
// modulo for doubles
double OSGVRUIEXPORT mod(double a, double b);

#endif
