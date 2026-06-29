/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPLICATION_H
#define _APPLICATION_H

#include <appl/ApplInterface.h>
#include <util/coviseCompat.h>
#include <api/coModule.h>

using namespace covise;

class ShowUSG: public coModule
{

private:
    int compute(const char *);
    int get_color_rgb(float *r, float *g, float *b, const char *color);
    void genpolygons(char *GeometryN);

    coInputPort *pin_mesh, *pin_colors;
    coOutputPort *pout_geo;

    coStringParam *p_varName;
    coBooleanParam *p_varVisible;

public:
    ShowUSG(int argc, char *argv[]);
    ~ShowUSG();
};
#endif // _APPLICATION_H
