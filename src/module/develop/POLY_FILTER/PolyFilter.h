/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUT_GEOMETRY_H
#define _CUT_GEOMETRY_H

#include <stdlib.h>
#include <stdio.h>

#include <api/coSimpleModule.h>
using namespace covise;

////// our class
class PolyFilter : public coSimpleModule
{
private:
    /////////ports
    coInputPort *p_geo_in;
    coOutputPort *p_geo_out;

    coDoPolygons *in;
    int num_l, num_cl, num_ll;
    int *l_cl, *l_ll;

    bool **ready;

    float *x, *y, *z, sw;
    int num_out, *vl, *pl;
    float *x_o, *y_o, *z_o;

    int num_n, *vStart, *elemList;
    int f1[10000], f2[10000];
    int n_f1, n_f2;
    int i, j, k, t;
    int neigh;
    bool fin;
    bool new_line;

    virtual int compute();

    int matches(int *f1, int n_f1, int *f2, int n_f2, int *res);
    void check();

public:
    PolyFilter();
};
#endif
