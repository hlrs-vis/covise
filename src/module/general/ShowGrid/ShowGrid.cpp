/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE ShowGrid application module                       **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  27.01.95  V1.0                                                  **
\**************************************************************************/

#include "ShowGrid.h"
#include <do/coDoSet.h>

ShowGrid::ShowGrid(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Generate Grid-lines")
{
    p_meshIn = addInputPort("meshIn", "StructuredGrid|RectilinearGrid|UniformGrid|UnstructuredGrid|Polygons", "input mesh");
    p_lines = addOutputPort("lines", "Lines", "Grid Lines");
    p_points = addOutputPort("points", "Points", "Grid Points");

    p_options = addChoiceParam("options", "Display options");
    const char *choiceVal[] = {
        "all_lines", "hull", "three_sides_+++",
        "three_sides_++-", "three_sides_+-+",
        "three_sides_+--", "three_sides_-++",
        "three_sides_-+-", "three_sides_--+",
        "three_sides_---",
        "Bounding_box", "Edges", "Element"
    };
    p_options->setValue(13, choiceVal, 0);
    p_pos = addIntSliderParam("pos", "number of unstructured grid element");
    p_pos->setValue(0, 1, 0);
}

void findMax(const coDistributedObject *p_obj, int *no_max)
{
    if (p_obj && p_obj->objectOk())
    {
        if (p_obj->isType("SETELE"))
        {
            int no_eles, ele;
            const coDistributedObject *const *eleList = ((coDoSet *)(p_obj))->getAllElements(&no_eles);
            for (ele = 0; ele < no_eles; ++ele)
            {
                findMax(eleList[ele], no_max);
            }
        }
        else if (p_obj->isType("UNSGRD"))
        {
            int numelem, numconn, numcoord;
            ((coDoUnstructuredGrid *)(p_obj))->getGridSize(&numelem, &numconn, &numcoord);
            if (numelem > *no_max)
                *no_max = numelem;
        }
    }
}

// determine the maximum of the set consisting of the number
// of elements of the input unstructured grids
void ShowGrid::preHandleObjects(coInputPort **inPort)
{
    int no_max = -1;
    long pmin, pmax;
    const coDistributedObject *p_obj = inPort[0]->getCurrentObject();
    findMax(p_obj, &no_max);
    if (no_max >= 0)
    {
        p_pos->getValue(pmin, pmax, pos);
        if (pmin != 0)
            pmin = 0;
        pmax = no_max - 1;
        if (pos < pmin || pos > pmax)
            pos = 0;
        p_pos->setValue(pmin, pmax, pos);
    }
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
int ShowGrid::compute(const char *)
{

    int i /*,pmin,pmax*/;
    float f1, f2;
    poly_in = NULL;
    //   get parameter
    option = p_options->getValue();

    //   retrieve grid object from input port
    data_obj = p_meshIn->getCurrentObject();

    if (NULL != data_obj)
    {
        gtype = data_obj->getType();

        if (strcmp(gtype, "POLYGN") == 0)
        {
            poly_in = (const coDoPolygons *)data_obj;
            poly_in->getAddresses(&x_in, &y_in, &z_in, &cl, &pl);

            if ((colorn = poly_in->getAttribute("COLOR")) == NULL)
            {
                colorn = "red";
            }
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            s_grid_in = (coDoStructuredGrid *)data_obj;
            s_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
            s_grid_in->getAddresses(&x_in, &y_in, &z_in);

            if ((colorn = s_grid_in->getAttribute("COLOR")) == NULL)
            {
                colorn = "red";
            }
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            r_grid_in = (coDoRectilinearGrid *)data_obj;
            r_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
            r_grid_in->getAddresses(&x_in, &y_in, &z_in);

            if ((colorn = r_grid_in->getAttribute("COLOR")) == NULL)
            {
                colorn = "red";
            }
        }
        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            u_grid_in = (coDoUniformGrid *)data_obj;
            u_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
            if ((colorn = u_grid_in->getAttribute("COLOR")) == NULL)
            {
                colorn = "red";
            }
            x_in = new float[i_dim];
            y_in = new float[j_dim];
            z_in = new float[k_dim];
            for (i = 0; i < i_dim; i++)
                u_grid_in->getPointCoordinates(i, x_in + i, 0, &f1, 0, &f2);
            for (i = 0; i < j_dim; i++)
                u_grid_in->getPointCoordinates(0, &f1, i, y_in + i, 0, &f2);
            for (i = 0; i < k_dim; i++)
                u_grid_in->getPointCoordinates(0, &f1, 0, &f2, i, z_in + i);
        }
        else if (strcmp(gtype, "UNSGRD") == 0)
        {
            uns_grid_in = (coDoUnstructuredGrid *)data_obj;
            uns_grid_in->getGridSize(&numelem, &numconn, &numcoord);
            uns_grid_in->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
            uns_grid_in->getTypeList(&tl);
            if ((colorn = uns_grid_in->getAttribute("COLOR")) == NULL)
            {
                static const char *color[6] = { "red", "magenta", "blue", "cyan", "green", "yellow" };
                colorn = color[p_pos->getValue() % 6];
            }
        }

        // awe: removed set handling -> coSimpleModule

        else
        {
            sendError("ERROR: Data object 'meshIn' has wrong data type");
            return STOP_PIPELINE;
        }
    }
    else
    {
#ifndef TOLERANT
        sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
#endif
        return STOP_PIPELINE;
    }

    if ((i_dim == 0 && j_dim == 0 && k_dim == 0) && numelem == 0 && !poly_in)
    {
        sendWarning("WARNING: Data object 'meshIn' is empty");
    }

    LinesN = p_lines->getObjName();
    if (LinesN == NULL)
    {
        sendError("ERROR: Object name not correct for 'lines'");
        return STOP_PIPELINE;
    }
    PointsN = p_points->getObjName();
    if (PointsN == NULL)
    {
        sendError("ERROR: Object name not correct for 'points'");
        return STOP_PIPELINE;
    }

    genLinesAndPoints();

    p_lines->setCurrentObject(Lines);
    p_points->setCurrentObject(Points);

    return CONTINUE_PIPELINE;
}

void ShowGrid::genLinesAndPoints()
{
    if (strcmp(gtype, "POLYGN") == 0)
    {
        polygons();
    }
    else if (strcmp(gtype, "UNSGRD") == 0)
    {
        unsgrd_elem();
    }
    else if ((strcmp(gtype, "UNIGRD") == 0)
             || (strcmp(gtype, "RCTGRD") == 0))
    {
        switch (option)
        {
        case ALL_LINES:
            rct_all_lines();
            break;
        case HULL:
            rct_hull();
            break;
        case THREE_SIDES_PPP:
        case THREE_SIDES_PPN:
        case THREE_SIDES_PNP:
        case THREE_SIDES_PNN:
        case THREE_SIDES_NPP:
        case THREE_SIDES_NPN:
        case THREE_SIDES_NNP:
        case THREE_SIDES_NNN:
            rct_three_sides();
            break;
        case BOUND_BOX:
        case EDGES:
            rct_box();
            break;
        }
    }
    else // Structured Grid
    {
        switch (option)
        {
        case ALL_LINES:
            str_all_lines();
            break;
        case HULL:
            str_hull();
            break;
        case THREE_SIDES_PPP:
        case THREE_SIDES_PPN:
        case THREE_SIDES_PNP:
        case THREE_SIDES_PNN:
        case THREE_SIDES_NPP:
        case THREE_SIDES_NPN:
        case THREE_SIDES_NNP:
        case THREE_SIDES_NNN:
            str_three_sides();
            break;
        case BOUND_BOX:
            str_box();
            break;
        case EDGES:
            str_curv_box();
            break;
        case CELL:
        {
            long pmin, pmax;
            p_pos->getValue(pmin, pmax, pos);
            str_cell();
        }
        break;
        }
    }
}

//======================================================================
// Creates lines and points for the case:
//         grid: polygons
//======================================================================

void ShowGrid::polygons()
{
    coDoPolygons *poly_in = (coDoPolygons *)p_meshIn->getCurrentObject();
    int no_coord = poly_in->getNumPoints();
    int no_vert = poly_in->getNumVertices();
    int no_poly = poly_in->getNumPolygons();

    Lines = new coDoLines(LinesN, no_coord, no_vert + no_poly, no_poly);
    Lines->addAttribute("COLOR", colorn);
    float *x_c, *y_c, *z_c;
    int *v_l, *l_l;
    Lines->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);

    memcpy(x_c, x_in, sizeof(*x_c) * no_coord);
    memcpy(y_c, y_in, sizeof(*y_c) * no_coord);
    memcpy(z_c, z_in, sizeof(*z_c) * no_coord);

    int lc = 0;
    for (int i = 0; i < no_poly; i++)
    {
        l_l[i] = pl[i] + i;

        for (int j = pl[i]; j < (i < no_poly - 1 ? pl[i + 1] : no_vert); j++)
        {
            v_l[lc++] = cl[j];
        }
        v_l[lc++] = cl[pl[i]];
    }

    Points = new coDoPoints(PointsN, no_coord);
    //    Points->addAttribute("COLOR",colorn);
    Points->getAddresses(&x_c, &y_c, &z_c);

    memcpy(x_c, x_in, sizeof(*x_c) * no_coord);
    memcpy(y_c, y_in, sizeof(*y_c) * no_coord);
    memcpy(z_c, z_in, sizeof(*z_c) * no_coord);
}

//======================================================================
// Creates lines and points for the case:
//         grid: uniform or rectangular
//         option: all_lines
//======================================================================

void ShowGrid::rct_all_lines()
{
    int *v_l, *l_l, i, j, k, nl, n;
    float *x_c, *y_c, *z_c;

    // nl is the number of lines
    nl = i_dim * j_dim + j_dim * k_dim + k_dim * i_dim;

    // The objects are first created...
    Lines = new coDoLines(LinesN, 2 * nl, 2 * nl, nl);
    Points = new coDoPoints(PointsN, i_dim * j_dim * k_dim);

    Lines->addAttribute("COLOR", colorn);
    //    Points->addAttribute("COLOR",colorn);

    Lines->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
    // ... and now the information is worked out and supplied

    for (i = 0; i < nl; i++)
        l_l[i] = i * 2;

    n = 0;
    for (i = 0; i < i_dim; i++)
    {
        for (j = 0; j < j_dim; j++)
        {
            v_l[n] = n;
            v_l[n + 1] = n + 1;
            x_c[n] = x_in[i];
            y_c[n] = y_in[j];
            z_c[n] = z_in[0];
            x_c[n + 1] = x_in[i];
            y_c[n + 1] = y_in[j];
            z_c[n + 1] = z_in[k_dim - 1];
            n += 2;
        }
    }
    for (j = 0; j < j_dim; j++)
    {
        for (k = 0; k < k_dim; k++)
        {
            v_l[n] = n;
            v_l[n + 1] = n + 1;
            x_c[n] = x_in[0];
            y_c[n] = y_in[j];
            z_c[n] = z_in[k];
            x_c[n + 1] = x_in[i_dim - 1];
            y_c[n + 1] = y_in[j];
            z_c[n + 1] = z_in[k];
            n += 2;
        }
    }
    for (k = 0; k < k_dim; k++)
    {
        for (i = 0; i < i_dim; i++)
        {
            v_l[n] = n;
            v_l[n + 1] = n + 1;
            x_c[n] = x_in[i];
            y_c[n] = y_in[0];
            z_c[n] = z_in[k];
            x_c[n + 1] = x_in[i];
            y_c[n + 1] = y_in[j_dim - 1];
            z_c[n + 1] = z_in[k];
            n += 2;
        }
    }

    Points->getAddresses(&x_c, &y_c, &z_c);
    for (i = 0; i < i_dim; i++)
    {
        for (j = 0; j < j_dim; j++)
        {
            for (k = 0; k < k_dim; k++)
            {
                x_c[i * j_dim * k_dim + j * k_dim + k] = x_in[i];
                y_c[i * j_dim * k_dim + j * k_dim + k] = y_in[j];
                z_c[i * j_dim * k_dim + j * k_dim + k] = z_in[k];
            }
        }
    }

    if (strcmp(gtype, "UNIGRD") == 0)
    {
        delete[] x_in;
        delete[] y_in;
        delete[] z_in;
    }
}

//======================================================================
// Creates lines and points for the case:
//         grid: uniform or rectangular
//         option: hull
//======================================================================

void ShowGrid::rct_hull()
{
    int *v_l, *l_l, i, j, k, nl, n;
    float *x_c, *y_c, *z_c;

    // nl is the number of lines
    // danger: for points this variable nl is modified
    //         has the meaning of number of points (see below)
    nl = 4 * (i_dim + j_dim + k_dim - 3);

    // The line objects are first created...
    Lines = new coDoLines(LinesN, 2 * nl, 2 * nl, nl);
    Lines->addAttribute("COLOR", colorn);

    Lines->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
    // ... and now the information is worked out and supplied
    for (i = 0; i < nl; i++)
        l_l[i] = i * 2;

    for (i = 0; i < 2 * nl; i++)
        v_l[i] = i;

    n = 0;
    for (i = 0; i < (i_dim - 1); i++)
    {
        x_c[n] = x_in[i];
        y_c[n] = y_in[0];
        z_c[n] = z_in[0];
        x_c[n + 1] = x_in[i];
        y_c[n + 1] = y_in[0];
        z_c[n + 1] = z_in[k_dim - 1];
        x_c[n + 2] = x_in[i + 1];
        y_c[n + 2] = y_in[j_dim - 1];
        z_c[n + 2] = z_in[0];
        x_c[n + 3] = x_in[i + 1];
        y_c[n + 3] = y_in[j_dim - 1];
        z_c[n + 3] = z_in[k_dim - 1];
        n += 4;
    }
    for (j = 0; j < (j_dim - 1); j++)
    {
        x_c[n] = x_in[0];
        y_c[n] = y_in[j + 1];
        z_c[n] = z_in[0];
        x_c[n + 1] = x_in[0];
        y_c[n + 1] = y_in[j + 1];
        z_c[n + 1] = z_in[k_dim - 1];
        x_c[n + 2] = x_in[i_dim - 1];
        y_c[n + 2] = y_in[j];
        z_c[n + 2] = z_in[0];
        x_c[n + 3] = x_in[i_dim - 1];
        y_c[n + 3] = y_in[j];
        z_c[n + 3] = z_in[k_dim - 1];
        n += 4;
    }
    for (j = 0; j < (j_dim - 1); j++)
    {
        x_c[n] = x_in[0];
        y_c[n] = y_in[j];
        z_c[n] = z_in[0];
        x_c[n + 1] = x_in[i_dim - 1];
        y_c[n + 1] = y_in[j];
        z_c[n + 1] = z_in[0];
        x_c[n + 2] = x_in[0];
        y_c[n + 2] = y_in[j + 1];
        z_c[n + 2] = z_in[k_dim - 1];
        x_c[n + 3] = x_in[i_dim - 1];
        y_c[n + 3] = y_in[j + 1];
        z_c[n + 3] = z_in[k_dim - 1];
        n += 4;
    }
    for (k = 0; k < (k_dim - 1); k++)
    {
        x_c[n] = x_in[0];
        y_c[n] = y_in[0];
        z_c[n] = z_in[k + 1];
        x_c[n + 1] = x_in[i_dim - 1];
        y_c[n + 1] = y_in[0];
        z_c[n + 1] = z_in[k + 1];
        x_c[n + 2] = x_in[0];
        y_c[n + 2] = y_in[j_dim - 1];
        z_c[n + 2] = z_in[k];
        x_c[n + 3] = x_in[i_dim - 1];
        y_c[n + 3] = y_in[j_dim - 1];
        z_c[n + 3] = z_in[k];
        n += 4;
    }
    for (k = 0; k < (k_dim - 1); k++)
    {
        x_c[n] = x_in[0];
        y_c[n] = y_in[0];
        z_c[n] = z_in[k];
        x_c[n + 1] = x_in[0];
        y_c[n + 1] = y_in[j_dim - 1];
        z_c[n + 1] = z_in[k];
        x_c[n + 2] = x_in[i_dim - 1];
        y_c[n + 2] = y_in[0];
        z_c[n + 2] = z_in[k + 1];
        x_c[n + 3] = x_in[i_dim - 1];
        y_c[n + 3] = y_in[j_dim - 1];
        z_c[n + 3] = z_in[k + 1];
        n += 4;
    }
    for (i = 0; i < (i_dim - 1); i++)
    {
        x_c[n] = x_in[i + 1];
        y_c[n] = y_in[0];
        z_c[n] = z_in[0];
        x_c[n + 1] = x_in[i + 1];
        y_c[n + 1] = y_in[j_dim - 1];
        z_c[n + 1] = z_in[0];
        x_c[n + 2] = x_in[i];
        y_c[n + 2] = y_in[0];
        z_c[n + 2] = z_in[k_dim - 1];
        x_c[n + 3] = x_in[i];
        y_c[n + 3] = y_in[j_dim - 1];
        z_c[n + 3] = z_in[k_dim - 1];
        n += 4;
    }

    // Points
    nl = i_dim * j_dim + j_dim * k_dim + k_dim * i_dim;
    // The point objects are first created...
    Points = new coDoPoints(PointsN, 2 * nl);
    //         Points->addAttribute("COLOR",colorn);
    Points->getAddresses(&x_c, &y_c, &z_c);
    // ... and now the information is worked out and supplied
    n = 0;
    for (i = 0; i < i_dim; i++)
    {
        for (j = 0; j < j_dim; j++)
        {
            x_c[n] = x_in[i];
            y_c[n] = y_in[j];
            z_c[n] = z_in[0];
            x_c[n + 1] = x_in[i];
            y_c[n + 1] = y_in[j];
            z_c[n + 1] = z_in[k_dim - 1];
            n += 2;
        }
    }
    for (j = 0; j < j_dim; j++)
    {
        for (k = 0; k < k_dim; k++)
        {
            x_c[n] = x_in[0];
            y_c[n] = y_in[j];
            z_c[n] = z_in[k];
            x_c[n + 1] = x_in[i_dim - 1];
            y_c[n + 1] = y_in[j];
            z_c[n + 1] = z_in[k];
            n += 2;
        }
    }
    for (k = 0; k < k_dim; k++)
    {
        for (i = 0; i < i_dim; i++)
        {
            x_c[n] = x_in[i];
            y_c[n] = y_in[0];
            z_c[n] = z_in[k];
            x_c[n + 1] = x_in[i];
            y_c[n + 1] = y_in[j_dim - 1];
            z_c[n + 1] = z_in[k];
            n += 2;
        }
    }

    if (strcmp(gtype, "UNIGRD") == 0)
    {
        delete[] x_in;
        delete[] y_in;
        delete[] z_in;
    }
}

//======================================================================
// Creates lines and points for the case:
//         grid: uniform or rectangular
//         option: three_sides_???
//======================================================================

void ShowGrid::rct_three_sides()
{
    int *v_l, *l_l, i, j, k, nl, n, m;
    float *x_c, *y_c, *z_c;

    // nl is the number of lines
    // danger: for points this variable nl will be modified
    //         with the meaning of number of points (see below)
    nl = (2 * (i_dim + j_dim + k_dim)) - 3;
    Lines = new coDoLines(LinesN, 2 * nl, 2 * nl, nl);
    Lines->addAttribute("COLOR", colorn);

    Lines->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
    // ... and now the information is worked out and supplied

    for (i = 0; i < nl; i++)
    {
        l_l[i] = i * 2;
    }
    for (i = 0; i < 2 * nl; i++)
    {
        v_l[i] = i;
    }
    n = 0;
    m = 0;
    for (i = m ^ 1; i < i_dim; i++)
    {
        x_c[n] = x_in[ref_x(i)];
        y_c[n] = y_in[ref_y(m * (j_dim - 1))];
        z_c[n] = z_in[ref_z(0)];
        x_c[n + 1] = x_in[ref_x(i)];
        y_c[n + 1] = y_in[ref_y(m * (j_dim - 1))];
        z_c[n + 1] = z_in[ref_z(k_dim - 1)];
        n += 2;
    }
    for (j = 0; j < (j_dim - m); j++)
    {
        x_c[n] = x_in[ref_x(m * (i_dim - 1))];
        y_c[n] = y_in[ref_y(j)];
        z_c[n] = z_in[ref_z(0)];
        x_c[n + 1] = x_in[ref_x(m * (i_dim - 1))];
        y_c[n + 1] = y_in[ref_y(j)];
        z_c[n + 1] = z_in[ref_z(k_dim - 1)];
        n += 2;
    }
    for (j = m ^ 1; j < j_dim; j++)
    {
        x_c[n] = x_in[ref_x(0)];
        y_c[n] = y_in[ref_y(j)];
        z_c[n] = z_in[ref_z(m * (k_dim - 1))];
        x_c[n + 1] = x_in[ref_x(i_dim - 1)];
        y_c[n + 1] = y_in[ref_y(j)];
        z_c[n + 1] = z_in[ref_z(m * (k_dim - 1))];
        n += 2;
    }
    for (k = 0; k < (k_dim - m); k++)
    {
        x_c[n] = x_in[ref_x(0)];
        y_c[n] = y_in[ref_y(m * (j_dim - 1))];
        z_c[n] = z_in[ref_z(k)];
        x_c[n + 1] = x_in[ref_x(i_dim - 1)];
        y_c[n + 1] = y_in[ref_y(m * (j_dim - 1))];
        z_c[n + 1] = z_in[ref_z(k)];
        n += 2;
    }
    for (k = m ^ 1; k < k_dim; k++)
    {
        x_c[n] = x_in[ref_x(m * (i_dim - 1))];
        y_c[n] = y_in[ref_y(0)];
        z_c[n] = z_in[ref_z(k)];
        x_c[n + 1] = x_in[ref_x(m * (i_dim - 1))];
        y_c[n + 1] = y_in[ref_y(j_dim - 1)];
        z_c[n + 1] = z_in[ref_z(k)];
        n += 2;
    }
    for (i = 0; i < (i_dim - m); i++)
    {
        x_c[n] = x_in[ref_x(i)];
        y_c[n] = y_in[ref_y(0)];
        z_c[n] = z_in[ref_z(m * (k_dim - 1))];
        x_c[n + 1] = x_in[ref_x(i)];
        y_c[n + 1] = y_in[ref_y(j_dim - 1)];
        z_c[n + 1] = z_in[ref_z(m * (k_dim - 1))];
        n += 2;
    }

    // There remain the points

    nl = i_dim * j_dim + j_dim * k_dim + k_dim * i_dim;
    Points = new coDoPoints(PointsN, nl);
    // Points->addAttribute("COLOR",colorn);
    Points->getAddresses(&x_c, &y_c, &z_c);
    n = 0;
    for (i = 0; i < i_dim; i++)
    {
        for (j = 0; j < j_dim; j++)
        {
            x_c[n] = x_in[ref_x(i)];
            y_c[n] = y_in[ref_y(j)];
            z_c[n] = z_in[ref_z(0)];
            n++;
        }
    }
    for (j = 0; j < j_dim; j++)
    {
        for (k = 0; k < k_dim; k++)
        {
            x_c[n] = x_in[ref_x(0)];
            y_c[n] = y_in[ref_y(j)];
            z_c[n] = z_in[ref_z(k)];
            n++;
        }
    }
    for (k = 0; k < k_dim; k++)
    {
        for (i = 0; i < i_dim; i++)
        {
            x_c[n] = x_in[ref_x(i)];
            y_c[n] = y_in[ref_y(0)];
            z_c[n] = z_in[ref_z(k)];
            n++;
        }
    }
    if (strcmp(gtype, "UNIGRD") == 0)
    {
        delete[] x_in;
        delete[] y_in;
        delete[] z_in;
    }
}

//======================================================================
// Creates lines and points for the case:
//         grid: uniform or rectangular
//         option: bounding_box or edges
//======================================================================

void ShowGrid::rct_box()
{
    int *v_l, *l_l, i, j, k, nl, n;
    float *x_c, *y_c, *z_c;

    Lines = new coDoLines(LinesN, 8, 18, 6);
    Lines->addAttribute("COLOR", colorn);
    Lines->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);

    l_l[0] = 0;
    l_l[1] = 5;
    l_l[2] = 10;
    l_l[3] = 12;
    l_l[4] = 14;
    l_l[5] = 16;
    v_l[0] = 0;
    v_l[1] = 1;
    v_l[2] = 2;
    v_l[3] = 3;
    v_l[4] = 0;
    v_l[5] = 4;
    v_l[6] = 5;
    v_l[7] = 6;
    v_l[8] = 7;
    v_l[9] = 4;
    v_l[10] = 0;
    v_l[11] = 4;
    v_l[12] = 1;
    v_l[13] = 5;
    v_l[14] = 2;
    v_l[15] = 6;
    v_l[16] = 3;
    v_l[17] = 7;
    x_c[0] = x_in[0];
    y_c[0] = y_in[0];
    z_c[0] = z_in[0];
    x_c[1] = x_in[i_dim - 1];
    y_c[1] = y_in[0];
    z_c[1] = z_in[0];
    x_c[2] = x_in[i_dim - 1];
    y_c[2] = y_in[j_dim - 1];
    z_c[2] = z_in[0];
    x_c[3] = x_in[0];
    y_c[3] = y_in[j_dim - 1];
    z_c[3] = z_in[0];
    x_c[4] = x_in[0];
    y_c[4] = y_in[0];
    z_c[4] = z_in[k_dim - 1];
    x_c[5] = x_in[i_dim - 1];
    y_c[5] = y_in[0];
    z_c[5] = z_in[k_dim - 1];
    x_c[6] = x_in[i_dim - 1];
    y_c[6] = y_in[j_dim - 1];
    z_c[6] = z_in[k_dim - 1];
    x_c[7] = x_in[0];
    y_c[7] = y_in[j_dim - 1];
    z_c[7] = z_in[k_dim - 1];

    nl = 4 * (i_dim + j_dim + k_dim - 3);
    Points = new coDoPoints(PointsN, 2 * nl);
    // Points->addAttribute("COLOR",colorn);
    Points->getAddresses(&x_c, &y_c, &z_c);
    n = 0;
    for (i = 0; i < (i_dim - 1); i++)
    {
        x_c[n] = x_in[i];
        y_c[n] = y_in[0];
        z_c[n] = z_in[0];
        x_c[n + 1] = x_in[i];
        y_c[n + 1] = y_in[0];
        z_c[n + 1] = z_in[k_dim - 1];
        x_c[n + 2] = x_in[i + 1];
        y_c[n + 2] = y_in[j_dim - 1];
        z_c[n + 2] = z_in[0];
        x_c[n + 3] = x_in[i + 1];
        y_c[n + 3] = y_in[j_dim - 1];
        z_c[n + 3] = z_in[k_dim - 1];
        n += 4;
    }
    for (j = 0; j < (j_dim - 1); j++)
    {
        x_c[n] = x_in[0];
        y_c[n] = y_in[j + 1];
        z_c[n] = z_in[0];
        x_c[n + 1] = x_in[0];
        y_c[n + 1] = y_in[j + 1];
        z_c[n + 1] = z_in[k_dim - 1];
        x_c[n + 2] = x_in[i_dim - 1];
        y_c[n + 2] = y_in[j];
        z_c[n + 2] = z_in[0];
        x_c[n + 3] = x_in[i_dim - 1];
        y_c[n + 3] = y_in[j];
        z_c[n + 3] = z_in[k_dim - 1];
        n += 4;
    }
    for (j = 0; j < (j_dim - 1); j++)
    {
        x_c[n] = x_in[0];
        y_c[n] = y_in[j];
        z_c[n] = z_in[0];
        x_c[n + 1] = x_in[i_dim - 1];
        y_c[n + 1] = y_in[j];
        z_c[n + 1] = z_in[0];
        x_c[n + 2] = x_in[0];
        y_c[n + 2] = y_in[j + 1];
        z_c[n + 2] = z_in[k_dim - 1];
        x_c[n + 3] = x_in[i_dim - 1];
        y_c[n + 3] = y_in[j + 1];
        z_c[n + 3] = z_in[k_dim - 1];
        n += 4;
    }
    for (k = 0; k < (k_dim - 1); k++)
    {
        x_c[n] = x_in[0];
        y_c[n] = y_in[0];
        z_c[n] = z_in[k + 1];
        x_c[n + 1] = x_in[i_dim - 1];
        y_c[n + 1] = y_in[0];
        z_c[n + 1] = z_in[k + 1];
        x_c[n + 2] = x_in[0];
        y_c[n + 2] = y_in[j_dim - 1];
        z_c[n + 2] = z_in[k];
        x_c[n + 3] = x_in[i_dim - 1];
        y_c[n + 3] = y_in[j_dim - 1];
        z_c[n + 3] = z_in[k];
        n += 4;
    }
    for (k = 0; k < (k_dim - 1); k++)
    {
        x_c[n] = x_in[0];
        y_c[n] = y_in[0];
        z_c[n] = z_in[k];
        x_c[n + 1] = x_in[0];
        y_c[n + 1] = y_in[j_dim - 1];
        z_c[n + 1] = z_in[k];
        x_c[n + 2] = x_in[i_dim - 1];
        y_c[n + 2] = y_in[0];
        z_c[n + 2] = z_in[k + 1];
        x_c[n + 3] = x_in[i_dim - 1];
        y_c[n + 3] = y_in[j_dim - 1];
        z_c[n + 3] = z_in[k + 1];
        n += 4;
    }
    for (i = 0; i < (i_dim - 1); i++)
    {
        x_c[n] = x_in[i + 1];
        y_c[n] = y_in[0];
        z_c[n] = z_in[0];
        x_c[n + 1] = x_in[i + 1];
        y_c[n + 1] = y_in[j_dim - 1];
        z_c[n + 1] = z_in[0];
        x_c[n + 2] = x_in[i];
        y_c[n + 2] = y_in[0];
        z_c[n + 2] = z_in[k_dim - 1];
        x_c[n + 3] = x_in[i];
        y_c[n + 3] = y_in[j_dim - 1];
        z_c[n + 3] = z_in[k_dim - 1];
        n += 4;
    }

    if (strcmp(gtype, "UNIGRD") == 0)
    {
        delete[] x_in;
        delete[] y_in;
        delete[] z_in;
    }
}

//======================================================================
// Creates lines and points for the case:
//         grid: structured
//         option: all_lines
//======================================================================

void ShowGrid::str_all_lines()
{
    int *v_l, *l_l, i, j, k, n;

    v_l = new int[i_dim * j_dim * k_dim * 3];
    l_l = new int[i_dim * j_dim + j_dim * k_dim + k_dim * i_dim];

    n = 0;
    for (i = 0; i < i_dim; i++)
        for (j = 0; j < j_dim; j++)
            for (k = 0; k < k_dim; k++)
            {
                v_l[n] = i * j_dim * k_dim + j * k_dim + k;
                n++;
            }
    for (k = 0; k < k_dim; k++)
        for (i = 0; i < i_dim; i++)
            for (j = 0; j < j_dim; j++)
            {
                v_l[n] = i * j_dim * k_dim + j * k_dim + k;
                n++;
            }
    for (j = 0; j < j_dim; j++)
        for (k = 0; k < k_dim; k++)
            for (i = 0; i < i_dim; i++)
            {
                v_l[n] = i * j_dim * k_dim + j * k_dim + k;
                n++;
            }
    n = 0;
    for (i = 0; i < (i_dim * j_dim); i++)
    {
        l_l[i] = n;
        n += k_dim;
    }
    for (i = (i_dim * j_dim); i < (i_dim * j_dim) + (i_dim * k_dim); i++)
    {
        l_l[i] = n;
        n += j_dim;
    }
    for (i = (i_dim * k_dim) + (i_dim * j_dim); i < (i_dim * k_dim) + (j_dim * i_dim) + (k_dim * j_dim); i++)
    {
        l_l[i] = n;
        n += i_dim;
    }
    Lines = new coDoLines(LinesN, i_dim * j_dim * k_dim, x_in, y_in, z_in, i_dim * j_dim * k_dim * 3, v_l, i_dim * j_dim + j_dim * k_dim + k_dim * i_dim, l_l);
    Lines->addAttribute("COLOR", colorn);
    Points = new coDoPoints(PointsN, i_dim * j_dim * k_dim, x_in, y_in, z_in);
    // Points->addAttribute("COLOR",colorn);
    delete[] v_l;
    delete[] l_l;
}

//======================================================================
// Creates lines and points for the case:
//         grid: structured
//         option: hull
//======================================================================

void ShowGrid::str_hull()
{
    int *v_l, *l_l, i, j, k, n, m, nl, dim;
    float *x_c, *y_c, *z_c;

    nl = 2 * (i_dim * j_dim + j_dim * k_dim + k_dim * i_dim);
    Lines = new coDoLines(LinesN, nl, 2 * nl, 4 * (i_dim + j_dim + k_dim));
    Lines->addAttribute("COLOR", colorn);
    Lines->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);

    n = 0;
    m = 0;
    dim = i_dim * j_dim;
    for (i = 0; i < i_dim; i++)
    {
        l_l[m] = n;
        l_l[m + i_dim] = n + dim;
        m++;
        for (j = 0; j < j_dim; j++)
        {
            x_c[n] = x_in[i * j_dim * k_dim + j * k_dim];
            y_c[n] = y_in[i * j_dim * k_dim + j * k_dim];
            z_c[n] = z_in[i * j_dim * k_dim + j * k_dim];
            x_c[n + dim] = x_in[i * j_dim * k_dim + j * k_dim + k_dim - 1];
            y_c[n + dim] = y_in[i * j_dim * k_dim + j * k_dim + k_dim - 1];
            z_c[n + dim] = z_in[i * j_dim * k_dim + j * k_dim + k_dim - 1];
            v_l[n] = n;
            v_l[n + dim] = n + dim;
            n++;
        }
    }
    n += dim;
    m += i_dim;
    dim = j_dim * k_dim;
    for (j = 0; j < j_dim; j++)
    {
        l_l[m] = n;
        l_l[m + j_dim] = n + dim;
        m++;
        for (k = 0; k < k_dim; k++)
        {
            x_c[n] = x_in[j * k_dim + k];
            y_c[n] = y_in[j * k_dim + k];
            z_c[n] = z_in[j * k_dim + k];
            x_c[n + dim] = x_in[(i_dim - 1) * j_dim * k_dim + j * k_dim + k];
            y_c[n + dim] = y_in[(i_dim - 1) * j_dim * k_dim + j * k_dim + k];
            z_c[n + dim] = z_in[(i_dim - 1) * j_dim * k_dim + j * k_dim + k];
            v_l[n] = n;
            v_l[n + dim] = n + dim;
            n++;
        }
    }
    n += dim;
    m += j_dim;
    dim = k_dim * i_dim;
    for (k = 0; k < k_dim; k++)
    {
        l_l[m] = n;
        l_l[m + k_dim] = n + dim;
        m++;
        for (i = 0; i < i_dim; i++)
        {
            x_c[n] = x_in[i * j_dim * k_dim + k];
            y_c[n] = y_in[i * j_dim * k_dim + k];
            z_c[n] = z_in[i * j_dim * k_dim + k];
            x_c[n + dim] = x_in[i * j_dim * k_dim + (j_dim - 1) * k_dim + k];
            y_c[n + dim] = y_in[i * j_dim * k_dim + (j_dim - 1) * k_dim + k];
            z_c[n + dim] = z_in[i * j_dim * k_dim + (j_dim - 1) * k_dim + k];
            v_l[n] = n;
            v_l[n + dim] = n + dim;
            n++;
        }
    }
    n += dim;
    m += k_dim;
    dim = i_dim * j_dim;
    for (j = 0; j < j_dim; j++)
    {
        l_l[m] = n;
        l_l[m + j_dim] = n + dim;
        m++;
        for (i = 0; i < i_dim; i++)
        {
            v_l[n] = i * j_dim + j;
            v_l[n + dim] = i * j_dim + j + dim;
            n++;
        }
    }
    n += dim;
    m += j_dim;
    dim = j_dim * k_dim;
    for (k = 0; k < k_dim; k++)
    {
        l_l[m] = n;
        l_l[m + k_dim] = n + dim;
        m++;
        for (j = 0; j < j_dim; j++)
        {
            v_l[n] = j * k_dim + k + 2 * (i_dim * j_dim);
            v_l[n + dim] = j * k_dim + k + dim + 2 * (i_dim * j_dim);
            n++;
        }
    }
    n += dim;
    m += k_dim;
    dim = k_dim * i_dim;
    for (i = 0; i < i_dim; i++)
    {
        l_l[m] = n;
        l_l[m + i_dim] = n + dim;
        m++;
        for (k = 0; k < k_dim; k++)
        {
            v_l[n] = k * i_dim + i + 2 * (i_dim * j_dim + j_dim * k_dim);
            v_l[n + dim] = k * i_dim + i + 2 * (i_dim * j_dim + j_dim * k_dim) + dim;
            n++;
        }
    }
    n += dim;
    m += i_dim;
    Points = new coDoPoints(PointsN, nl, x_c, y_c, z_c);
    // Points->addAttribute("COLOR",colorn);
}

//======================================================================
// Creates lines and points for the case:
//         grid: structured
//         option: three_sides_???
//======================================================================

void ShowGrid::str_three_sides()
{
    int *v_l, *l_l, i, j, k, n, nl, m, dim;
    float *x_c, *y_c, *z_c;

    nl = i_dim * j_dim + j_dim * k_dim + k_dim * i_dim;
    Lines = new coDoLines(LinesN, nl, 2 * nl, 2 * (i_dim + j_dim + k_dim));
    Lines->addAttribute("COLOR", colorn);
    Lines->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);

    n = 0;
    m = 0;
    dim = 0;
    for (i = 0; i < i_dim; i++)
    {
        l_l[m] = n;
        m++;
        for (j = 0; j < j_dim; j++)
        {
            x_c[n] = x_in[ref_x(i) * j_dim * k_dim + ref_y(j) * k_dim + ref_z(dim * (k_dim - 1))];
            y_c[n] = y_in[ref_x(i) * j_dim * k_dim + ref_y(j) * k_dim + ref_z(dim * (k_dim - 1))];
            z_c[n] = z_in[ref_x(i) * j_dim * k_dim + ref_y(j) * k_dim + ref_z(dim * (k_dim - 1))];
            v_l[n] = n;
            n++;
        }
    }
    for (j = 0; j < j_dim; j++)
    {
        l_l[m] = n;
        m++;
        for (k = 0; k < k_dim; k++)
        {
            x_c[n] = x_in[ref_x(dim * (i_dim - 1)) * j_dim * k_dim + ref_y(j) * k_dim + ref_z(k)];
            y_c[n] = y_in[ref_x(dim * (i_dim - 1)) * j_dim * k_dim + ref_y(j) * k_dim + ref_z(k)];
            z_c[n] = z_in[ref_x(dim * (i_dim - 1)) * j_dim * k_dim + ref_y(j) * k_dim + ref_z(k)];
            v_l[n] = n;
            n++;
        }
    }
    for (k = 0; k < k_dim; k++)
    {
        l_l[m] = n;
        m++;
        for (i = 0; i < i_dim; i++)
        {
            x_c[n] = x_in[ref_x(i) * j_dim * k_dim + ref_y(dim * (j_dim - 1)) * k_dim + ref_z(k)];
            y_c[n] = y_in[ref_x(i) * j_dim * k_dim + ref_y(dim * (j_dim - 1)) * k_dim + ref_z(k)];
            z_c[n] = z_in[ref_x(i) * j_dim * k_dim + ref_y(dim * (j_dim - 1)) * k_dim + ref_z(k)];
            v_l[n] = n;
            n++;
        }
    }
    for (j = 0; j < j_dim; j++)
    {
        l_l[m] = n;
        m++;
        for (i = 0; i < i_dim; i++)
        {
            v_l[n] = ref_x(i) * j_dim + ref_y(j);
            n++;
        }
    }
    for (k = 0; k < k_dim; k++)
    {
        l_l[m] = n;
        m++;
        for (j = 0; j < j_dim; j++)
        {
            v_l[n] = ref_y(j) * k_dim + ref_z(k) + i_dim * j_dim;
            n++;
        }
    }
    for (i = 0; i < i_dim; i++)
    {
        l_l[m] = n;
        m++;
        for (k = 0; k < k_dim; k++)
        {
            v_l[n] = ref_z(k) * i_dim + ref_x(i) + i_dim * j_dim + j_dim * k_dim;
            n++;
        }
    }

    Points = new coDoPoints(PointsN, nl, x_c, y_c, z_c);
    // Points->addAttribute("COLOR",colorn);
}

//======================================================================
// Creates lines and points for the case:
//         grid: structured
//         option: bounding_box
//======================================================================

void ShowGrid::str_box()
{
    int *v_l, *l_l, i, j, k;
    float minx, maxx, miny, maxy, minz, maxz;
    float *x_c, *y_c, *z_c;

    x_c = new float[8];
    y_c = new float[8];
    z_c = new float[8];
    v_l = new int[18];
    l_l = new int[6];
    l_l[0] = 0;
    l_l[1] = 5;
    l_l[2] = 10;
    l_l[3] = 12;
    l_l[4] = 14;
    l_l[5] = 16;
    v_l[0] = 0;
    v_l[1] = 1;
    v_l[2] = 2;
    v_l[3] = 3;
    v_l[4] = 0;
    v_l[5] = 4;
    v_l[6] = 5;
    v_l[7] = 6;
    v_l[8] = 7;
    v_l[9] = 4;
    v_l[10] = 0;
    v_l[11] = 4;
    v_l[12] = 1;
    v_l[13] = 5;
    v_l[14] = 2;
    v_l[15] = 6;
    v_l[16] = 3;
    v_l[17] = 7;
    minx = maxx = x_in[0];
    miny = maxy = y_in[0];
    minz = maxz = z_in[0];
    for (i = 0; i < i_dim; i++)
    {
        for (j = 0; j < j_dim; j++)
        {
            for (k = 0; k < k_dim; k++)
            {
                if (x_in[i * j_dim * k_dim + j * k_dim + k] > maxx)
                    maxx = x_in[i * j_dim * k_dim + j * k_dim + k];
                if (x_in[i * j_dim * k_dim + j * k_dim + k] < minx)
                    minx = x_in[i * j_dim * k_dim + j * k_dim + k];
                if (y_in[i * j_dim * k_dim + j * k_dim + k] > maxy)
                    maxy = y_in[i * j_dim * k_dim + j * k_dim + k];
                if (y_in[i * j_dim * k_dim + j * k_dim + k] < miny)
                    miny = y_in[i * j_dim * k_dim + j * k_dim + k];
                if (z_in[i * j_dim * k_dim + j * k_dim + k] > maxz)
                    maxz = z_in[i * j_dim * k_dim + j * k_dim + k];
                if (z_in[i * j_dim * k_dim + j * k_dim + k] < minz)
                    minz = z_in[i * j_dim * k_dim + j * k_dim + k];
            }
        }
    }
    x_c[0] = minx;
    y_c[0] = miny;
    z_c[0] = minz;
    x_c[1] = maxx;
    y_c[1] = miny;
    z_c[1] = minz;
    x_c[2] = maxx;
    y_c[2] = maxy;
    z_c[2] = minz;
    x_c[3] = minx;
    y_c[3] = maxy;
    z_c[3] = minz;
    x_c[4] = minx;
    y_c[4] = miny;
    z_c[4] = maxz;
    x_c[5] = maxx;
    y_c[5] = miny;
    z_c[5] = maxz;
    x_c[6] = maxx;
    y_c[6] = maxy;
    z_c[6] = maxz;
    x_c[7] = minx;
    y_c[7] = maxy;
    z_c[7] = maxz;
    Lines = new coDoLines(LinesN, 8, x_c, y_c, z_c, 18, v_l, 6, l_l);
    Lines->addAttribute("COLOR", colorn);
    Points = new coDoPoints(PointsN, 8, x_c, y_c, z_c);
    // Points->addAttribute("COLOR",colorn);
    delete[] x_c;
    delete[] y_c;
    delete[] z_c;
    delete[] l_l;
    delete[] v_l;
}

//======================================================================
// Creates lines and points for the case:
//         grid: structured
//         option: edges
//======================================================================

void ShowGrid::str_curv_box()
{
    int num_of_points = 4 * (i_dim + j_dim + k_dim);
    int i, j, k;
    float *x_c, *y_c, *z_c;
    int *v_l;
    int *l_l;

    Lines = new coDoLines(LinesN, num_of_points, num_of_points, 12);
    Lines->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
    Lines->addAttribute("COLOR", colorn);

    for (i = 0; i < i_dim; ++i)
    {
        x_c[i] = x_in[index(i, 0, 0)];
        y_c[i] = y_in[index(i, 0, 0)];
        z_c[i] = z_in[index(i, 0, 0)];

        x_c[i + i_dim] = x_in[index(i, j_dim - 1, 0)];
        y_c[i + i_dim] = y_in[index(i, j_dim - 1, 0)];
        z_c[i + i_dim] = z_in[index(i, j_dim - 1, 0)];

        x_c[i + 2 * i_dim] = x_in[index(i, j_dim - 1, k_dim - 1)];
        y_c[i + 2 * i_dim] = y_in[index(i, j_dim - 1, k_dim - 1)];
        z_c[i + 2 * i_dim] = z_in[index(i, j_dim - 1, k_dim - 1)];

        x_c[i + 3 * i_dim] = x_in[index(i, 0, k_dim - 1)];
        y_c[i + 3 * i_dim] = y_in[index(i, 0, k_dim - 1)];
        z_c[i + 3 * i_dim] = z_in[index(i, 0, k_dim - 1)];
    }

    int l_offset = 4 * i_dim;

    for (j = 0; j < j_dim; ++j)
    {
        x_c[j + l_offset] = x_in[index(0, j, 0)];
        y_c[j + l_offset] = y_in[index(0, j, 0)];
        z_c[j + l_offset] = z_in[index(0, j, 0)];

        x_c[j + j_dim + l_offset] = x_in[index(0, j, k_dim - 1)];
        y_c[j + j_dim + l_offset] = y_in[index(0, j, k_dim - 1)];
        z_c[j + j_dim + l_offset] = z_in[index(0, j, k_dim - 1)];

        x_c[j + 2 * j_dim + l_offset] = x_in[index(i_dim - 1, j, k_dim - 1)];
        y_c[j + 2 * j_dim + l_offset] = y_in[index(i_dim - 1, j, k_dim - 1)];
        z_c[j + 2 * j_dim + l_offset] = z_in[index(i_dim - 1, j, k_dim - 1)];

        x_c[j + 3 * j_dim + l_offset] = x_in[index(i_dim - 1, j, 0)];
        y_c[j + 3 * j_dim + l_offset] = y_in[index(i_dim - 1, j, 0)];
        z_c[j + 3 * j_dim + l_offset] = z_in[index(i_dim - 1, j, 0)];
    }

    l_offset += 4 * j_dim;

    for (k = 0; k < k_dim; ++k)
    {
        x_c[k + l_offset] = x_in[index(0, 0, k)];
        y_c[k + l_offset] = y_in[index(0, 0, k)];
        z_c[k + l_offset] = z_in[index(0, 0, k)];

        x_c[k + k_dim + l_offset] = x_in[index(i_dim - 1, 0, k)];
        y_c[k + k_dim + l_offset] = y_in[index(i_dim - 1, 0, k)];
        z_c[k + k_dim + l_offset] = z_in[index(i_dim - 1, 0, k)];

        x_c[k + 2 * k_dim + l_offset] = x_in[index(i_dim - 1, j_dim - 1, k)];
        y_c[k + 2 * k_dim + l_offset] = y_in[index(i_dim - 1, j_dim - 1, k)];
        z_c[k + 2 * k_dim + l_offset] = z_in[index(i_dim - 1, j_dim - 1, k)];

        x_c[k + 3 * k_dim + l_offset] = x_in[index(0, j_dim - 1, k)];
        y_c[k + 3 * k_dim + l_offset] = y_in[index(0, j_dim - 1, k)];
        z_c[k + 3 * k_dim + l_offset] = z_in[index(0, j_dim - 1, k)];
    }

    Points = new coDoPoints(PointsN, num_of_points, x_c, y_c, z_c);
    // Points->addAttribute("COLOR",colorn);

    for (i = 0; i < num_of_points; ++i)
        v_l[i] = i;

    l_l[0] = 0;
    l_l[1] = i_dim;
    l_l[2] = 2 * i_dim;
    l_l[3] = 3 * i_dim;
    l_offset = 4 * i_dim;
    l_l[4] = l_offset;
    l_l[5] = l_offset + j_dim;
    l_l[6] = l_offset + 2 * j_dim;
    l_l[7] = l_offset + 3 * j_dim;
    l_offset += 4 * j_dim;
    l_l[8] = l_offset;
    l_l[9] = l_offset + k_dim;
    l_l[10] = l_offset + 2 * k_dim;
    l_l[11] = l_offset + 3 * k_dim;
}

//======================================================================
// Generates lines and points for the graphical representation
// of isolated elements of an unstructured grid (variable pos identifies
// the element at issue).
//======================================================================

void ShowGrid::unsgrd_elem()
{
    if (numelem == 0 || pos < 0 || pos >= numelem)
    {
        Lines = new coDoLines(LinesN, 0, 0, 0);
        Points = new coDoPoints(PointsN, 0);
        return;
    }
    switch (tl[pos])
    {
    case TYPE_POLYHEDRON:
    {
        int NumConnOfElement = uns_grid_in->getNumConnOfElement(pos);

        int *l_l = new int[NumConnOfElement];
        int *v_l = new int[NumConnOfElement];
        float *x_c = new float[NumConnOfElement];
        float *y_c = new float[NumConnOfElement];
        float *z_c = new float[NumConnOfElement];
        int oldV;
        int numLines = 1;
        l_l[0] = 0;
        oldV = cl[el[pos]];
        for (int i = 0; i < NumConnOfElement; i++)
        {
            x_c[i] = x_in[cl[el[pos] + i]];
            y_c[i] = y_in[cl[el[pos] + i]];
            z_c[i] = z_in[cl[el[pos] + i]];
            v_l[i] = i;
            if (i > l_l[numLines - 1])
            {
                if (oldV == cl[el[pos] + i])
                {
                    l_l[numLines] = i + 1;
                    numLines++;
                    if (i < NumConnOfElement - 1)
                        oldV = cl[el[pos] + i + 1];
                }
            }
        }
        numLines--;
        Lines = new coDoLines(LinesN, NumConnOfElement, x_c, y_c, z_c, NumConnOfElement, v_l, numLines, l_l);
        Lines->addAttribute("COLOR", colorn);
        Points = new coDoPoints(PointsN, NumConnOfElement, x_c, y_c, z_c);
    }
    break;
    case TYPE_HEXAGON:
    {
        int l_l[] = { 0, 5, 10, 12, 14, 16 };
        int v_l[] = { 0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 };
        float x_c[8], y_c[8], z_c[8];
        for (int i = 0; i < 8; i++)
        {
            x_c[i] = x_in[cl[el[pos] + i]];
            y_c[i] = y_in[cl[el[pos] + i]];
            z_c[i] = z_in[cl[el[pos] + i]];
        }
        Lines = new coDoLines(LinesN, 8, x_c, y_c, z_c, 18, v_l, 6, l_l);
        Lines->addAttribute("COLOR", colorn);
        Points = new coDoPoints(PointsN, 8, x_c, y_c, z_c);
        // Points->addAttribute("COLOR",colorn);
        break;
    }

    case TYPE_TETRAHEDER:
    {
        int l_l[] = { 0, 4, 7 };
        int v_l[] = { 0, 1, 2, 0, 0, 3, 1, 3, 2 };
        float x_c[4], y_c[4], z_c[4];

        for (int i = 0; i < 4; i++)
        {
            x_c[i] = x_in[cl[el[pos] + i]];
            y_c[i] = y_in[cl[el[pos] + i]];
            z_c[i] = z_in[cl[el[pos] + i]];
        }
        Lines = new coDoLines(LinesN, 4, x_c, y_c, z_c, 9, v_l, 3, l_l);
        Lines->addAttribute("COLOR", colorn);
        Points = new coDoPoints(PointsN, 4, x_c, y_c, z_c);
        // Points->addAttribute("COLOR",colorn);
        break;
    }

    case TYPE_PYRAMID:
    {
        int l_l[] = { 0, 5, 8 };
        int v_l[] = { 0, 1, 2, 3, 0, 0, 4, 1, 2, 4, 3 };
        float x_c[5], y_c[5], z_c[5];

        for (int i = 0; i < 5; i++)
        {
            x_c[i] = x_in[cl[el[pos] + i]];
            y_c[i] = y_in[cl[el[pos] + i]];
            z_c[i] = z_in[cl[el[pos] + i]];
        }
        Lines = new coDoLines(LinesN, 5, x_c, y_c, z_c, 11, v_l, 3, l_l);
        Lines->addAttribute("COLOR", colorn);
        Points = new coDoPoints(PointsN, 5, x_c, y_c, z_c);
        // Points->addAttribute("COLOR",colorn);
        break;
    }

    case TYPE_PRISM:
    {
        int l_l[] = { 0, 4, 8, 10, 12 };
        int v_l[] = { 0, 1, 2, 0, 3, 4, 5, 3, 0, 3, 1, 4, 2, 5 };
        float x_c[6], y_c[6], z_c[6];

        for (int i = 0; i < 6; i++)
        {
            x_c[i] = x_in[cl[el[pos] + i]];
            y_c[i] = y_in[cl[el[pos] + i]];
            z_c[i] = z_in[cl[el[pos] + i]];
        }
        Lines = new coDoLines(LinesN, 6, x_c, y_c, z_c, 14, v_l, 5, l_l);
        Lines->addAttribute("COLOR", colorn);
        Points = new coDoPoints(PointsN, 6, x_c, y_c, z_c);
        // Points->addAttribute("COLOR",colorn);
        break;
    }

    case TYPE_QUAD:
    {
        int l_l[] = { 0 };
        int v_l[] = { 0, 1, 2, 3, 0 };
        float x_c[4], y_c[4], z_c[4];

        for (int i = 0; i < 4; i++)
        {
            x_c[i] = x_in[cl[el[pos] + i]];
            y_c[i] = y_in[cl[el[pos] + i]];
            z_c[i] = z_in[cl[el[pos] + i]];
        }
        Lines = new coDoLines(LinesN, 4, x_c, y_c, z_c, 5, v_l, 1, l_l);
        Lines->addAttribute("COLOR", colorn);
        Points = new coDoPoints(PointsN, 4, x_c, y_c, z_c);
        // Points->addAttribute("COLOR",colorn);
        break;
    }

    case TYPE_TRIANGLE:
    {
        int l_l[] = { 0 };
        int v_l[] = { 0, 1, 2, 0 };
        float x_c[3], y_c[3], z_c[3];

        for (int i = 0; i < 3; i++)
        {
            x_c[i] = x_in[cl[el[pos] + i]];
            y_c[i] = y_in[cl[el[pos] + i]];
            z_c[i] = z_in[cl[el[pos] + i]];
        }
        Lines = new coDoLines(LinesN, 3, x_c, y_c, z_c, 4, v_l, 1, l_l);
        Lines->addAttribute("COLOR", colorn);
        Points = new coDoPoints(PointsN, 3, x_c, y_c, z_c);
        // Points->addAttribute("COLOR",colorn);
        break;
    }

    case TYPE_BAR:
    {
        int l_l[] = { 0 };
        int v_l[] = { 0, 1 };
        float x_c[2], y_c[2], z_c[2];

        for (int i = 0; i < 2; i++)
        {
            x_c[i] = x_in[cl[el[pos] + i]];
            y_c[i] = y_in[cl[el[pos] + i]];
            z_c[i] = z_in[cl[el[pos] + i]];
        }
        Lines = new coDoLines(LinesN, 2, x_c, y_c, z_c, 2, v_l, 1, l_l);
        Lines->addAttribute("COLOR", colorn);
        Points = new coDoPoints(PointsN, 2, x_c, y_c, z_c);
        // Points->addAttribute("COLOR",colorn);
        break;
    }
    case TYPE_POINT:
    {
        int l_l[] = { 0 };
        int v_l[] = { 0, 0 };
        float x_c[1], y_c[1], z_c[1];

        for (int i = 0; i < 1; i++)
        {
            x_c[i] = x_in[cl[el[pos] + i]];
            y_c[i] = y_in[cl[el[pos] + i]];
            z_c[i] = z_in[cl[el[pos] + i]];
        }
        Lines = new coDoLines(LinesN, 1, x_c, y_c, z_c, 2, v_l, 1, l_l);
        Lines->addAttribute("COLOR", colorn);
        Points = new coDoPoints(PointsN, 1, x_c, y_c, z_c);
        // Points->addAttribute("COLOR",colorn);
        break;
    }
    }
}

void ShowGrid::str_cell()
{
    int l_l[] = { 0, 5, 10, 12, 14, 16 };
    int v_l[] = { 0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 };
    float x_c[8], y_c[8], z_c[8];
    int p[8];

    if (pos > i_dim * j_dim * k_dim)
    {
        Lines = new coDoLines(LinesN, 0, 0, 0);
        Points = new coDoPoints(PointsN, 0);
        return;
    }

    p[0] = pos;
    p[1] = pos + 1;
    p[2] = pos + 1 + j_dim * k_dim;
    p[3] = pos + j_dim * k_dim;
    p[4] = pos + k_dim;
    p[5] = pos + k_dim + 1;
    p[6] = pos + k_dim + 1 + j_dim * k_dim;
    p[7] = pos + k_dim + j_dim * k_dim;

    for (int i = 0; i < 8; i++)
    {
        x_c[i] = x_in[p[i]];
        y_c[i] = y_in[p[i]];
        z_c[i] = z_in[p[i]];
    }
    Lines = new coDoLines(LinesN, 8, x_c, y_c, z_c, 18, v_l, 6, l_l);
    Lines->addAttribute("COLOR", colorn);
    Points = new coDoPoints(PointsN, 8, x_c, y_c, z_c);

    return;
}

MODULE_MAIN(Tools, ShowGrid)
