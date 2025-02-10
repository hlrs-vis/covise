/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvVIVEConfig.h"

#include <config/CoviseConfig.h>

#include <string.h>
#include <stdio.h>
#include <vsg/all.h>

using namespace covise;
using namespace vive;
vvVIVEConfig::~vvVIVEConfig()
{
}

bool vvVIVEConfig::getScreenConfigEntry(int pos, std::string &name, int *hsize, int *vsize, int *x, int *y, int *z, float *h, float *p, float *r)
{
    char str[200];
    sprintf(str, "VIVE.ScreenConfig.Screen:%d", pos);
    name = coCoviseConfig::getEntry("comment", str, "NoNameWindow");
    std::string frustum = coCoviseConfig::getEntry("frustum", str, "");
    if (frustum.length() == 0)
    {
        *hsize = coCoviseConfig::getInt("width", str, 2000);
        *vsize = coCoviseConfig::getInt("height", str, 1500);
        *x = coCoviseConfig::getInt("originX", str, 0);
        *y = coCoviseConfig::getInt("originY", str, 0);
        *z = coCoviseConfig::getInt("originZ", str, 0);
        *h = coCoviseConfig::getFloat("h", str, 0);
        *p = coCoviseConfig::getFloat("p", str, 0);
        *r = coCoviseConfig::getFloat("r", str, 0);
    }
    else
    {
        float radius = coCoviseConfig::getFloat("radius", str, 2800);
        float cx, cy, cz, left, right, bottom, top, hdeg, pdeg, rdeg;
        sscanf(frustum.c_str(), "%f;%f;%f;%f;%f;%f;%f;%f;%f;%f", &cx, &cy, &cz, &hdeg, &pdeg, &rdeg, &left, &right, &bottom, &top);
        *h = hdeg;
        *p = pdeg;
        *r = rdeg;
        vsg::dmat4 m;
        //MAKE_EULER_MAT(m,hdeg/180.0*M_PI,pdeg/180.0*M_PI,rdeg/180.0*M_PI);
        m = vsg::rotate(rdeg / 180.0 * M_PI, 0.0, 1.0, 0.0) * vsg::rotate(pdeg / 180.0 * M_PI, 1.0, 0.0, 0.0) * vsg::rotate(hdeg / 180.0 * M_PI, 0.0, 0.0, 1.0);
        /* coCoord coord;
      coord = m;
      *h=coord.hpr[0];
      *p=coord.hpr[1];
      *r=coord.hpr[2];*/
        vsg::dvec3 rv(0, radius, 0);
        double ld = tan((-left) / 180.0 * M_PI) * radius;
        double rd = tan(right / 180.0 * M_PI) * radius;
        double td = tan(top / 180.0 * M_PI) * radius;
        double bd = tan((-bottom) / 180.0 * M_PI) * radius;
        *hsize = (int)(ld + rd);
        *vsize = (int)(td + bd);
        rv[0] += (rd - (*hsize / 2.0));
        rv[2] += (td - (*vsize / 2.0));
        vsg::dmat3 m3;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                m3[r][c] = m[r][c];
        rv = m3* rv;
        //rv = vsg::dmat4::transform3x3(rv, m);

        fprintf(stderr, "rv %f %f %f\n", rv[0], rv[1], rv[2]);
        *x = (int)(cx + rv[0]);
        *y = (int)(cy + rv[1]);
        *z = (int)(cz + rv[2]);
    }

    return true;
}
