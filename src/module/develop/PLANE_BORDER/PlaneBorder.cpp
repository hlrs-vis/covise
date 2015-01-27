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
#include "PlaneBorder.h"

PlaneBorder::PlaneBorder()
    : coModule("cut something out of an object")
{
    // ports
    p_geo_in = addInputPort("geo_in", "Set_Polygons|Set_TriangleStrips", "geometry");
    p_geo_out = addOutputPort("geo_out", "Set_Points", "geometry");
}

int main(int argc, char *argv[])
{
    PlaneBorder *application = new PlaneBorder;
    application->start(argc, argv);
    return 0;
}

////// this is called whenever we have to do something

int PlaneBorder::compute()
{
    // in objectsp_geo_in
    in = (coDoPolygons *)p_geo_in->getCurrentObject();

    num_l = 0, num_cl = 0, num_ll = 0;
    l_cl = new int[in->getNumPoints()];
    l_ll = new int[in->getNumPoints()];

    l_x = new float[in->getNumPoints()];
    l_y = new float[in->getNumPoints()];
    l_z = new float[in->getNumPoints()];

    ready = new bool[in->getNumPoints()];
    for (i = 0; i < in->getNumPoints(); i++)
        ready[i] = false;

    in->getAddresses(&x, &y, &z, &vl, &pl);

    in->getNeighborList(&num_n, &elemList, &vStart);

    fin = false;

    while (!fin)
    {
        for (i = 0; i < in->getNumPoints() && !ready[i]; i++)
            ;

        check(i);
    }

    coDoPoints *out = new coDoPoints(p_geo_out->getObjName(), num_out, x_o, y_o, z_o);
    //in->getNumVertices(), vl,
    //in-> getNumPolygons(), pl);
    out->addAttribute("vertexOrder", "2");

    delete[] x_o;
    delete[] y_o;
    delete[] z_o;

    delete[] l_cl;
    delete[] l_ll;
    delete[] ready;

    p_geo_out->setCurrentObject(out);
}

void PlaneBorder::check(int i)
{
    n_f1 = vStart[i + 1] - vStart[i];
    for (j = 0; j < n_f1; j++)
    {
        f1[j] = elemList[vStart[i] + j];
    }

    // find neighbor corner
    for (t = 0; t < n_f1; t++)
    {
        int next = pl[f1[t] + 1];
        if (f1[t] + 1 == in->getNumPolygons())
            next = in->getNumVertices();

        for (j = pl[f1[t]]; j < next && t < n_f1; j++)
        {
            if (vl[j] != i)
            {
                neigh = vl[j];

                n_f2 = vStart[neigh + 1] - vStart[neigh];
                for (k = 0; k < n_f2; k++)
                {
                    f2[k] = elemList[vStart[neigh] + k];
                }

                if (matches(f1, n_f1, f2, n_f2) == 1)
                {
                    x_o[num_out] = x[i];
                    y_o[num_out] = y[i];
                    z_o[num_out] = z[i];
                    num_out++;
                    //cout << "take" << endl;
                    t = n_f1;
                }
            }
        }
    }
}

int PlaneBorder::matches(int *f1, int n_f1, int *f2, int n_f2)
{
    int i, j;
    int matches = 0;

    /* for( i=0; i<n_f1; i++ )
     cout << f1[i] << " " ;
    cout << endl;

    for( i=0; i<n_f2; i++ )
     cout << f2[i] << " " ;
    cout << endl;
     */

    for (i = 0; i < n_f1; i++)
        for (j = 0; j < n_f2; j++)
            if (f1[i] == f2[j])
                matches++;
    //cout << "match: " << matches << endl << endl;
    return matches;
}
