/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                    (C) 2000 VirCinity  **
 ** Description: axe-murder geometry                                       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Lars Frenzel                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  05.01.97  V0.1                                                  **
 **        18.10.2000 V1.0 new API, several data ports, triangle strips    **
 **                                               ( converted to polygons )**
 **			Sven Kufer 					  **
\**************************************************************************/

#define NUM_DATA_IN_PORTS 4 // number of data ports

#include <iostream.h>
#include "PolyFilter.h"

PolyFilter::PolyFilter()
    : coSimpleModule("cut something out of an object")
{
    // ports
    p_geo_in = addInputPort("geo_in", "Set_Polygons|Set_TriangleStrips", "geometry");
    p_geo_out = addOutputPort("geo_out", "Set_Polygons", "geometry");
}

int main(int argc, char *argv[])
{
    PolyFilter *application = new PolyFilter;
    application->start(argc, argv);
    return 0;
}

////// this is called whenever we have to do something

int PolyFilter::compute()
{
    // in objectsp_geo_in
    in = (coDoPolygons *)p_geo_in->getCurrentObject();

    num_l = 0, num_cl = 0, num_ll = 0;
    num_out = 0;
    l_cl = new int[in->getNumPolygons() * 10];
    l_ll = new int[in->getNumPolygons() * 10];

    x_o = new float[in->getNumPolygons() * 10];
    y_o = new float[in->getNumPolygons() * 10];
    z_o = new float[in->getNumPolygons() * 10];

    in->getAddresses(&x, &y, &z, &vl, &pl);

    check();

    //cout<< num_out << " " << num_cl << " " << num_ll << " " <<endl;

    /*coDoPolygons *out =  new coDoPolygons(p_geo_out->getObjName(), num_out, x_o, y_o, z_o,
                                   num_cl, l_cl,
                    num_ll, l_ll);      */
    coDoPolygons *out = new coDoPolygons(p_geo_out->getObjName(), in->getNumPoints(), x, y, z,
                                         num_cl, l_cl,
                                         num_ll, l_ll);

    delete[] x_o;
    delete[] y_o;
    delete[] z_o;

    delete[] l_cl;
    delete[] l_ll;

    p_geo_out->setCurrentObject(out);

    return CONTINUE_PIPELINE;
}

void PolyFilter::check()
{
    int match[1000];

    for (t = 0; t < in->getNumPolygons(); t++)
    {
        bool take_it = true;

        int next = pl[t + 1];
        if (t + 1 == in->getNumPolygons())
            next = in->getNumVertices();

        n_f2 = next - pl[t];
        for (k = 0; k < n_f2; k++)
        {
            f2[k] = vl[pl[t] + k];
        }

        for (i = 0; i < n_f2; i++)
        {
            f1[0] = vl[pl[t] + i];
            n_f1 = 1;
            if (matches(f1, n_f1, f2, n_f2, match) > 1)
                take_it = false;
        }

        if (take_it)
        {
            l_ll[num_ll++] = num_cl;

            for (k = 0; k < n_f2; k++)
            {
                l_cl[num_cl++] = vl[pl[t] + k]; // num_out;
                /*x_o[num_out] = x[vl[ pl[t]+k]];
                 y_o[num_out] = y[vl[ pl[t]+k]];
                 z_o[num_out] = z[vl[ pl[t]+k]];
                 num_out++;*/
            }
        }
        else
        {
            //cout << " remove " << t << endl;
        }
    }
}

int PolyFilter::matches(int *f1, int n_f1, int *f2, int n_f2, int *res)
{
    int i, j;
    int matches = 0;

    for (i = 0; i < n_f1; i++)
        for (j = 0; j < n_f2; j++)
            if (f1[i] == f2[j])
            {
                res[matches++] = f1[i];
            }
    //cout << "match: " << matches << endl << endl;
    return matches;
}
