/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coCoverConfig.h"

#include <config/CoviseConfig.h>

#include <string.h>
#include <stdio.h>
#include <osg/Matrix>
#include <osg/Vec3>

using namespace covise;
using namespace opencover;
coCoverConfig::~coCoverConfig()
{
}

bool coCoverConfig::getScreenConfigEntry(int pos, std::string &name, float *hsize, float *vsize, float *x, float *y, float *z, float *h, float *p, float *r)
{
    char str[200];
    snprintf(str, sizeof(str), "COVER.ScreenConfig.Screen:%d", pos);
    name = coCoviseConfig::getEntry("comment", str, "NoNameWindow");
    std::string frustum = coCoviseConfig::getEntry("frustum", str, "");
    if (frustum.length() == 0)
    {
        *hsize = coCoviseConfig::getFloat("width", str, 2000.f);
        *vsize = coCoviseConfig::getFloat("height", str, 1500.f);
        *x = coCoviseConfig::getFloat("originX", str, 0.f);
        *y = coCoviseConfig::getFloat("originY", str, 0.f);
        *z = coCoviseConfig::getFloat("originZ", str, 0.f);
        *h = coCoviseConfig::getFloat("h", str, 0.f);
        *p = coCoviseConfig::getFloat("p", str, 0.f);
        *r = coCoviseConfig::getFloat("r", str, 0.f);
    }
    else
    {
        float radius = coCoviseConfig::getFloat("radius", str, 2800.f);
        float cx, cy, cz, left, right, bottom, top, hdeg, pdeg, rdeg;
        sscanf(frustum.c_str(), "%f;%f;%f;%f;%f;%f;%f;%f;%f;%f", &cx, &cy, &cz, &hdeg, &pdeg, &rdeg, &left, &right, &bottom, &top);
        *h = hdeg;
        *p = pdeg;
        *r = rdeg;
        osg::Matrix m;
        // MAKE_EULER_MAT(m,hdeg/180.0*M_PI,pdeg/180.0*M_PI,rdeg/180.0*M_PI);
        m.makeRotate(rdeg / 180.0 * M_PI, osg::Y_AXIS, pdeg / 180.0 * M_PI, osg::X_AXIS, hdeg / 180.0 * M_PI, osg::Z_AXIS);
        /* coCoord coord;
      coord = m;
      *h=coord.hpr[0];
      *p=coord.hpr[1];
      *r=coord.hpr[2];*/
        osg::Vec3 rv(0, radius, 0);
        float ld = tan(-left / 180.0 * M_PI) * radius;
        float rd = tan(right / 180.0 * M_PI) * radius;
        float td = tan(top / 180.0 * M_PI) * radius;
        float bd = tan(-bottom / 180.0 * M_PI) * radius;
        *hsize = ld + rd;
        *vsize = td + bd;
        rv[0] += rd - (*hsize / 2.0);
        rv[2] += td - (*vsize / 2.0);
        rv = osg::Matrix::transform3x3(rv, m);

        fprintf(stderr, "rv %f %f %f\n", rv[0], rv[1], rv[2]);
        *x = cx + rv[0];
        *y = cy + rv[1];
        *z = cz + rv[2];
    }

    return true;
}
