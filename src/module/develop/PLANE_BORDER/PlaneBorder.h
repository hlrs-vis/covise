/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUT_GEOMETRY_H
#define _CUT_GEOMETRY_H

#include <stdlib.h>
#include <stdio.h>

#include <api/coModule.h>
using namespace covise;

////// our class
class PlaneBorder : public coModule
{
private:
    /////////ports
    coInputPort *p_geo_in;
    coOutputPort *p_geo_out;

    coDoPolygons *in;
    int num_l, num_cl, num_ll;
    int *l_cl, *l_ll;
    float *l_x, *l_y, *l_z;

    bool *ready;

    float *x, *y, *z, sw;
    int *vl, *pl;

    int num_n, *vStart, *elemList;
    int f1[1000], f2[1000];
    int n_f1, n_f2;
    int i, j, k, t;
    int neigh;
    bool fin;
    bool new_line;

    virtual int compute();

    int matches(int *f1, int n_f1, int *f2, int n_f2);

public:
    PlaneBorder();
};
#endif
