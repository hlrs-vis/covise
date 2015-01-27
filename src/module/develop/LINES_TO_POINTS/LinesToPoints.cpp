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
#include "LinesToPoints.h"

LinesToPoints::LinesToPoints()
    : coSimpleModule("cut something out of an object")
{
    int i;
    char portname[32];

    // ports
    p_geo_in = addInputPort("geo_in", "coDoLines", "geometry");
    p_data_in = addInputPort("data_in", "coDoFloat", "data");
    p_geo_out = addOutputPort("geo_out", "coDoPoints", "geometry");
    p_data_out = addOutputPort("data_out", "coDoFloat", "data");
}

int main(int argc, char *argv[])
{
    LinesToPoints *application = new LinesToPoints;
    application->start(argc, argv);
    return 0;
}

////// this is called whenever we have to do something

int LinesToPoints::compute()
{
    // in objectsp_geo_in

    coDoLines *in = (coDoLines *)p_geo_in->getCurrentObject();

    float *x, *y, *z, sw, *p_x, *p_y, *p_z;
    int *vl, *pl;

    coDoFloat *data_in = (coDoFloat *)p_data_in->getCurrentObject();
    float *s_in, *s_out;
    data_in->getAddress(&s_in);

    coDoPoints *out = new coDoPoints(p_geo_out->getObjName(), in->getNumLines());
    out->getAddresses(&p_x, &p_y, &p_z);

    coDoFloat *data_out = new coDoFloat(p_data_out->getObjName(), in->getNumLines());
    data_out->getAddress(&s_out);

    in->getAddresses(&x, &y, &z, &vl, &pl);

    for (int i = 0; i < in->getNumLines(); i++)
    {
        p_x[i] = x[vl[pl[i]]];
        p_y[i] = y[vl[pl[i]]];
        p_z[i] = z[vl[pl[i]]];
        s_out[i] = s_in[vl[pl[i]]];
    }

    out->addAttribute("POINTSIZE", "2");

    p_geo_out->setCurrentObject(out);
    p_data_out->setCurrentObject(data_out);
    return CONTINUE_PIPELINE;
}
