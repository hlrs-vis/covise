/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "mathUtils.h"

// snap matrix 45 degrees in orientation
void snapTo45Degrees(osg::Matrix *mat)
{
    coCoord coord = *mat;
    //fprintf(stderr,"H: %f P: %f R: %f\n",coord.hpr[0],coord.hpr[1],coord.hpr[2]);
    if (coord.hpr[0] > 0.0)
        coord.hpr[0] = 45.0 * ((int)(coord.hpr[0] + 22.5) / 45);
    else
        coord.hpr[0] = 45.0 * ((int)(coord.hpr[0] - 22.5) / 45);
    if (coord.hpr[1] > 0.0)
        coord.hpr[1] = 45.0 * ((int)(coord.hpr[1] + 22.5) / 45);
    else
        coord.hpr[1] = 45.0 * ((int)(coord.hpr[1] - 22.5) / 45);
    if (coord.hpr[2] > 0.0)
        coord.hpr[2] = 45.0 * ((int)(coord.hpr[2] + 22.5) / 45);
    else
        coord.hpr[2] = 45.0 * ((int)(coord.hpr[2] - 22.5) / 45);
    //fprintf(stderr,"H2: %f P2: %f R2: %f\n",coord.hpr[0],coord.hpr[1],coord.hpr[2]);

    coord.makeMat(*mat);
}

double mod(double a, double b)
{
    int c = int(a / b);
    return a - (double)(c * b);
}

// snap matrix 15 degrees in orientation
void snapToDegrees(float degree, osg::Matrix *mat)
{
    coCoord coord = *mat;
    //fprintf(stderr,"H: %f P: %f R: %f\n",coord.hpr[0],coord.hpr[1],coord.hpr[2]);
    float mod5 = mod(coord.hpr[0], degree);
    if (mod5 < (degree / 2))
        coord.hpr[0] = coord.hpr[0] - mod5;
    else
        coord.hpr[0] = coord.hpr[0] + (degree - mod5);
    mod5 = mod(coord.hpr[1], degree);
    if (mod5 < (degree / 2))
        coord.hpr[1] = coord.hpr[1] - mod5;
    else
        coord.hpr[1] = coord.hpr[1] + (degree - mod5);
    mod5 = mod(coord.hpr[2], degree);
    if (mod5 < (degree / 2))
        coord.hpr[2] = coord.hpr[2] - mod5;
    else
        coord.hpr[2] = coord.hpr[2] + (degree - mod5);
    //fprintf(stderr,"H2: %f P2: %f R2: %f\n",coord.hpr[0],coord.hpr[1],coord.hpr[2]);

    coord.makeMat(*mat);
}

coCoord::~coCoord() {}

coCoord::coCoord(const coCoord &c)
{
    xyz = c.xyz;
    hpr = c.hpr;
}

coCoord &coCoord::operator=(const osg::Matrix &right)
{
    initFromMatrix(right);
    return *this;
}

void coCoord::makeMat(osg::Matrix &right)
{
    MAKE_EULER_MAT_VEC(right, hpr);
    right.setTrans(xyz);
}

coCoord::coCoord(const osg::Matrix &right)
{
    initFromMatrix(right);
}

void
coCoord::initFromMatrix(const osg::Matrix &right)
{
    //GET_HPR_VEC(right,hpr);
    float cp;
    osg::Matrix m;
    osg::Vec3 v1(right(0, 0), right(0, 1), right(0, 2));
    osg::Vec3 v2(right(1, 0), right(1, 1), right(1, 2));
    osg::Vec3 v3(right(2, 0), right(2, 1), right(2, 2));
    v1.normalize();
    v2.normalize();
    v3.normalize();
    m(0, 0) = v1[0];
    m(0, 1) = v1[1];
    m(0, 2) = v1[2];
    m(1, 0) = v2[0];
    m(1, 1) = v2[1];
    m(1, 2) = v2[2];
    m(2, 0) = v3[0];
    m(2, 1) = v3[1];
    m(2, 2) = v3[2];
    hpr[1] = asin(m(1, 2));
    cp = cos(hpr[1]);
    float d;
    if (cp > -0.0000001 && cp < 0.0000001)
    {
        if (m(0, 0) > -0.00001 && m(0, 0) < 0.00001 && m(0, 2) > -0.00001 && m(0, 2) < 0.00001)
        {
            hpr[0] = asin(m(0, 1));
        }
        else
        {
            hpr[0] = atan2(m(0, 2), m(0, 0));
        }
        hpr[2] = 0;
    }
    else
    {
        d = m(1, 0) / cp;
        if (d > 1.0)
        {
            hpr[0] = (osg::Vec3f::value_type) - M_PI_2;
        }
        else if (d < -1.0)
        {
            hpr[0] = (osg::Vec3f::value_type)M_PI_2;
        }
        else
            hpr[0] = -asin(d);
        float diff = cos(hpr[0]) * cp - m(1, 1);
        if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other Heading angle */
        {
            hpr[0] = M_PI - hpr[0];
            diff = cos(hpr[0]) * cp - m(1, 1);
            if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other pitch angle */
            {
                hpr[1] = M_PI - hpr[1];
                cp = cos(hpr[1]);
                d = m(1, 0) / cp;
                if (d > 1.0)
                {
                    hpr[0] = (osg::Vec3f::value_type) - M_PI_2;
                }
                else if (d < -1.0)
                {
                    hpr[0] = (osg::Vec3f::value_type)M_PI_2;
                }
                else
                    hpr[0] = -asin(d);
                diff = cos(hpr[0]) * cp - m(1, 1);
                if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other Heading angle */
                {
                    hpr[0] = M_PI - hpr[0];
                }
            }
        }
        if (cp > -0.0000001 && cp < 0.0000001)
        {
            cp = 0.0;
            d = 1.0;
        }
        else
        {
            d = m(2, 2) / cp;
        }
        if (d > 1.0)
            hpr[2] = 0;
        else if (d < -1.0)
            hpr[2] = (osg::Vec3f::value_type)M_PI;
        else
            hpr[2] = acos(d);
        diff = -sin(hpr[2]) * cp - m(0, 2);
        if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other roll angle */
            hpr[2] = -hpr[2];
    }
    hpr[0] = hpr[0] / M_PI * 180.0;
    hpr[1] = hpr[1] / M_PI * 180.0;
    hpr[2] = hpr[2] / M_PI * 180.0;

    xyz = right.getTrans();
}
