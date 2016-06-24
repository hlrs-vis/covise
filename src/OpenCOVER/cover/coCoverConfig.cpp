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
bool coCoverConfig::getWindowConfigEntry(int pos, std::string &name, int *pipeNum, int *ox, int *oy, int *sx, int *sy, bool *decoration, bool *stereo, bool *resize, bool *embedded, bool *pbuffer)
{
    char str[200];
    sprintf(str, "COVER.WindowConfig.Window:%d", pos);
    name = coCoviseConfig::getEntry("comment", str, "COVER");
    *pipeNum = coCoviseConfig::getInt("pipeIndex", str, 0);
    *ox = coCoviseConfig::getInt("left", str, 0);
    *oy = coCoviseConfig::getInt("top", str, 0);
    *sx = coCoviseConfig::getInt("width", str, 1024);
    *sy = coCoviseConfig::getInt("height", str, 768);
    bool have_bottom = false;
    coCoviseConfig::getInt("bottom", str, 0, &have_bottom);
    if (have_bottom)
        printf("bottom is ignored in %s, please use top\n", str);
    if (decoration)
        *decoration = coCoviseConfig::isOn("decoration", std::string(str), false);
    if (resize)
        *resize = coCoviseConfig::isOn("resize", str, true);
    //*visualID  = coCoviseConfig::getInt("visualID",str,-1);
    if (stereo)
        *stereo = coCoviseConfig::isOn("stereo", std::string(str), false);
    if (embedded)
        *embedded = coCoviseConfig::isOn("embedded", std::string(str), false);
    if (pbuffer)
        *pbuffer = coCoviseConfig::isOn("pbuffer", std::string(str), false);

    return true;
}

bool coCoverConfig::getScreenConfigEntry(int pos, std::string &name, int *hsize, int *vsize, int *x, int *y, int *z, float *h, float *p, float *r)
{
    char str[200];
    sprintf(str, "COVER.ScreenConfig.Screen:%d", pos);
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
        osg::Matrix m;
        //MAKE_EULER_MAT(m,hdeg/180.0*M_PI,pdeg/180.0*M_PI,rdeg/180.0*M_PI);
        m.makeRotate(rdeg / 180.0 * M_PI, osg::Y_AXIS, pdeg / 180.0 * M_PI, osg::X_AXIS, hdeg / 180.0 * M_PI, osg::Z_AXIS);
        /* coCoord coord;
      coord = m;
      *h=coord.hpr[0];
      *p=coord.hpr[1];
      *r=coord.hpr[2];*/
        osg::Vec3 rv(0, radius, 0);
        float ld = tan((-left) / 180.0 * M_PI) * radius;
        float rd = tan(right / 180.0 * M_PI) * radius;
        float td = tan(top / 180.0 * M_PI) * radius;
        float bd = tan((-bottom) / 180.0 * M_PI) * radius;
        *hsize = (int)(ld + rd);
        *vsize = (int)(td + bd);
        rv[0] += (rd - (*hsize / 2.0));
        rv[2] += (td - (*vsize / 2.0));
        rv = osg::Matrix::transform3x3(rv, m);

        fprintf(stderr, "rv %f %f %f\n", rv[0], rv[1], rv[2]);
        *x = (int)(cx + rv[0]);
        *y = (int)(cy + rv[1]);
        *z = (int)(cz + rv[2]);
    }

    return true;
}
