/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description:  COVISE Reduce cell volume of unstructured grids          **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Oliver Heck                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  16.05.95  V1.0                                                  **
\**************************************************************************/

#include <stdio.h>
#include <iostream.h>
#include <math.h>
#include "ReduceCellVolume.h"
#include <stdlib.h>
//
// Covise include stuff
//

//  Shared memory data
coDoPolygons *Polygons = NULL;

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}

// =======================================================================
// START WORK HERE (main block)
// =======================================================================
void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//======================================================================
// Called before module exits
//======================================================================
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
    Covise::log_message(__LINE__, __FILE__, "Quitting now");
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
void Application::compute(void *)
{
    //      local variables
    int numelem, numconn, numcoord;
    int *el, *cl, *tl, i, j, ellistval = 0, cllistval = 0;
    int *tmp_cl, *tmp_el, no_p = 0, no_v = 0, no_pol = 0;

    float min_fact, max_fact, factor;
    float *x_in, *y_in, *z_in;
    float *tmp_x, *tmp_y, *tmp_z, sp[3];

    char *colorn, *GridIn, *GridOut;

    coDoUnstructuredGrid *tmp_grid;

    //	get input data object names
    GridIn = Covise::get_object_name("meshIn");

    //	get output data object names
    GridOut = Covise::get_object_name("meshOut");

    //	get parameters
    Covise::get_slider_param("factor", &min_fact, &max_fact, &factor);

    //	retrieve grid object from shared memory
    if (GridIn == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'meshIn'");
        return;
    }

    tmp_grid = new coDoUnstructuredGrid(GridIn);
    tmp_grid->getGridSize(&numelem, &numconn, &numcoord);
    tmp_grid->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
    //      el-Element List, cl-Verticy List

    tmp_grid->getTypeList(&tl); // tl-Type List

    //	set color in geometry
    if ((colorn = tmp_grid->getAttribute("COLOR")) == NULL)
    {
        colorn = new char[20];
        strcpy(colorn, "white");
    }

    //	Is there data in the array ?
    if (numelem == 0 || numconn == 0 || numcoord == 0)
    {
        Covise::sendWarning("WARNING: Data object 'meshIn' is empty");
    }

    //	Allocation of tmp arrays
    //	memory is allocated for hexahedra most of space is required herefore
    tmp_x = new float[numelem * 8];
    tmp_y = new float[numelem * 8];
    tmp_z = new float[numelem * 8];
    tmp_cl = new int[numelem * 6 * 4];
    tmp_el = new int[numelem * 6];

    for (i = 0; i < numelem; i++)
    {
        switch (tl[i])
        {
        case TYPE_HEXAGON:
        {
            //Computation for hexahedra
            // Finding the center
            sp[0] = (x_in[cl[el[i]]] + x_in[cl[el[i] + 1]] + x_in[cl[el[i] + 2]] + x_in[cl[el[i] + 3]] + x_in[cl[el[i] + 4]] + x_in[cl[el[i] + 5]] + x_in[cl[el[i] + 6]] + x_in[cl[el[i] + 7]]) / 8.;
            sp[1] = (y_in[cl[el[i]]] + y_in[cl[el[i] + 1]] + y_in[cl[el[i] + 2]] + y_in[cl[el[i] + 3]] + y_in[cl[el[i] + 4]] + y_in[cl[el[i] + 5]] + y_in[cl[el[i] + 6]] + y_in[cl[el[i] + 7]]) / 8.;
            sp[2] = (z_in[cl[el[i]]] + z_in[cl[el[i] + 1]] + z_in[cl[el[i] + 2]] + z_in[cl[el[i] + 3]] + z_in[cl[el[i] + 4]] + z_in[cl[el[i] + 5]] + z_in[cl[el[i] + 6]] + z_in[cl[el[i] + 7]]) / 8.;

            // Compute the new positions
            for (j = 0; j < 8; j++)
            {
                tmp_x[no_p + j] = sp[0] + (x_in[cl[el[i] + j]] - sp[0]) * factor;
                tmp_y[no_p + j] = sp[1] + (y_in[cl[el[i] + j]] - sp[1]) * factor;
                tmp_z[no_p + j] = sp[2] + (z_in[cl[el[i] + j]] - sp[2]) * factor;
            }
            no_p += 8;

            // Filling the temporary element list
            //		   ellistval = i*24;
            for (j = 0; j < 6; j++)
            {
                tmp_el[no_pol + j] = ellistval + j * 4;
            }
            no_pol += 6;
            ellistval += 24;

            // Filling the temporary verticy list
            tmp_cl[no_v] = cllistval;
            tmp_cl[no_v + 1] = cllistval + 3;
            tmp_cl[no_v + 2] = cllistval + 2;
            tmp_cl[no_v + 3] = cllistval + 1;

            tmp_cl[no_v + 4] = cllistval + 4;
            tmp_cl[no_v + 5] = cllistval + 5;
            tmp_cl[no_v + 6] = cllistval + 6;
            tmp_cl[no_v + 7] = cllistval + 7;

            tmp_cl[no_v + 8] = cllistval;
            tmp_cl[no_v + 9] = cllistval + 4;
            tmp_cl[no_v + 10] = cllistval + 7;
            tmp_cl[no_v + 11] = cllistval + 3;

            tmp_cl[no_v + 12] = cllistval;
            tmp_cl[no_v + 13] = cllistval + 1;
            tmp_cl[no_v + 14] = cllistval + 5;
            tmp_cl[no_v + 15] = cllistval + 4;

            tmp_cl[no_v + 16] = cllistval + 2;
            tmp_cl[no_v + 17] = cllistval + 3;
            tmp_cl[no_v + 18] = cllistval + 7;
            tmp_cl[no_v + 19] = cllistval + 6;

            tmp_cl[no_v + 20] = cllistval + 1;
            tmp_cl[no_v + 21] = cllistval + 2;
            tmp_cl[no_v + 22] = cllistval + 6;
            tmp_cl[no_v + 23] = cllistval + 5;

            no_v += 24;
            cllistval += 8;
        }
        break;

        case TYPE_TETRAHEDER:
        {
            //Computation for tetrahedra
            // Finding the center
            sp[0] = (x_in[cl[el[i]]] + x_in[cl[el[i] + 1]] + x_in[cl[el[i] + 2]] + x_in[cl[el[i] + 3]]) / 4.;
            sp[1] = (y_in[cl[el[i]]] + y_in[cl[el[i] + 1]] + y_in[cl[el[i] + 2]] + y_in[cl[el[i] + 3]]) / 4.;
            sp[2] = (z_in[cl[el[i]]] + z_in[cl[el[i] + 1]] + z_in[cl[el[i] + 2]] + z_in[cl[el[i] + 3]]) / 4.;

            // Compute the new positions
            for (j = 0; j < 4; j++)
            {
                tmp_x[no_p + j] = sp[0] + (x_in[cl[el[i] + j]] - sp[0]) * factor;
                tmp_y[no_p + j] = sp[1] + (y_in[cl[el[i] + j]] - sp[1]) * factor;
                tmp_z[no_p + j] = sp[2] + (z_in[cl[el[i] + j]] - sp[2]) * factor;
            }
            no_p += 4;

            // Filling the temporary element list
            ellistval = i * 12;
            for (j = 0; j < 4; j++)
            {
                tmp_el[no_pol + j] = ellistval + j * 3;
            }
            no_pol += 4;

            // Filling the temporary verticy list
            tmp_cl[no_v] = cllistval + 2;
            tmp_cl[no_v + 1] = cllistval + 1;
            tmp_cl[no_v + 2] = cllistval;

            tmp_cl[no_v + 3] = cllistval + 2;
            tmp_cl[no_v + 4] = cllistval + 3;
            tmp_cl[no_v + 5] = cllistval + 1;

            tmp_cl[no_v + 6] = cllistval + 1;
            tmp_cl[no_v + 7] = cllistval + 3;
            tmp_cl[no_v + 8] = cllistval;

            tmp_cl[no_v + 9] = cllistval + 3;
            tmp_cl[no_v + 10] = cllistval + 2;
            tmp_cl[no_v + 11] = cllistval;

            no_v += 12;
            cllistval += 4;
        }
        break;

        case TYPE_PRISM:
        {
            //Computation for prisma
            Covise::sendError("ERROR:at the moment unsupported grid type");
        }
        break;

        case TYPE_PYRAMID:
        {
            //Computation for pyramid
            // Finding the center
            sp[0] = (x_in[cl[el[i]]] + x_in[cl[el[i] + 1]] + x_in[cl[el[i] + 2]] + x_in[cl[el[i] + 3]] + x_in[cl[el[i] + 4]]) / 5.;
            sp[1] = (y_in[cl[el[i]]] + y_in[cl[el[i] + 1]] + y_in[cl[el[i] + 2]] + y_in[cl[el[i] + 3]] + x_in[cl[el[i] + 4]]) / 5.;
            sp[2] = (z_in[cl[el[i]]] + z_in[cl[el[i] + 1]] + z_in[cl[el[i] + 2]] + z_in[cl[el[i] + 3]] + x_in[cl[el[i] + 4]]) / 5.;

            // Compute the new positions
            for (j = 0; j < 5; j++)
            {
                tmp_x[no_p + j] = sp[0] + (x_in[cl[el[i] + j]] - sp[0]) * factor;
                tmp_y[no_p + j] = sp[1] + (y_in[cl[el[i] + j]] - sp[1]) * factor;
                tmp_z[no_p + j] = sp[2] + (z_in[cl[el[i] + j]] - sp[2]) * factor;
            }
            no_p += 5;

            // Filling the temporary element list
            for (j = 0; j < 5; j++)
            {
                tmp_el[no_pol + j] = ellistval + j * 3;
            }
            no_pol += 5;
            ellistval += 16;

            // Filling the temporary verticy list
            tmp_cl[no_v] = cllistval + 1;
            tmp_cl[no_v + 1] = cllistval + 4;
            tmp_cl[no_v + 2] = cllistval;

            tmp_cl[no_v + 3] = cllistval + 2;
            tmp_cl[no_v + 4] = cllistval + 4;
            tmp_cl[no_v + 5] = cllistval + 1;

            tmp_cl[no_v + 6] = cllistval + 3;
            tmp_cl[no_v + 7] = cllistval + 4;
            tmp_cl[no_v + 8] = cllistval + 2;

            tmp_cl[no_v + 9] = cllistval;
            tmp_cl[no_v + 10] = cllistval + 4;
            tmp_cl[no_v + 11] = cllistval + 3;

            tmp_cl[no_v + 12] = cllistval;
            tmp_cl[no_v + 13] = cllistval + 3;
            tmp_cl[no_v + 14] = cllistval + 2;
            tmp_cl[no_v + 15] = cllistval + 1;

            no_v += 16;
            cllistval += 5;
        }
        break;

        default:
        {
            Covise::sendError("ERROR: unsupported grid type");
        }
        break;
        }
    }

    //	Create Output
    //	create the polygons
    Polygons = new coDoPolygons(GridOut, no_p, tmp_x, tmp_y, tmp_z,
                                no_v, tmp_cl, no_pol, tmp_el);

    Polygons->addAttribute("COLOR", colorn);

    //	delete data (tmp-arrays)
    delete[] tmp_x;
    delete[] tmp_y;
    delete[] tmp_z;
    delete[] tmp_el;
    delete[] tmp_cl;

    delete Polygons;
    delete tmp_grid;
}
