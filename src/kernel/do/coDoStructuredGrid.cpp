/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoStructuredGrid.h"
#include "covise_gridmethods.h"

/***********************************************************************\ 
 **                                                                     **
 **   Structured classes Routines                   Version: 1.1        **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of structured grids       **
 **                  in a distributed manner.                           **
 **                                                                     **
 **   Classes      : coDoUniformGrid, coDoRectilinearGrid,                **
 **                  coDoStructuredGrid                                  **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  26.05.93  Ver 1.1 shm-access restructured,         **
 **                                    recursive data-objects (simple   **
 **                                    version),                        **
 **                                    some new types added             **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

using namespace covise;

int coDoStructuredGrid::interpolateField(float *v_interp, const float *point,
                                         int *in_cell, int no_arrays, int array_dim,
                                         const float *const *velo)
{
    int ret = 0;
    if (x_disc < 2 || y_disc < 2 || z_disc < 2)
    {
        return -1;
    }
    float *x_in, *y_in, *z_in;
    float a, b, g;
    float amat[3][3], bmat[3][3];
    int cell[3];
    int no_idea = 0;

    if (in_cell[0] == -1 && in_cell[1] == -1 && in_cell[2] == -1)
    {
        no_idea = 1;
    }

    cell[0] = in_cell[0];
    cell[1] = in_cell[1];
    cell[2] = in_cell[2];
    int status = 0;
    getAddresses(&x_in, &y_in, &z_in);

    if (cell[0] < x_disc - 1 && cell[0] >= 0)
    {
        ++cell[0]; // OK, but cell3 requires a shift
    } // if we got meaningless numbers
    else
    {
        cell[0] = (cell[0] < 0) ? 1 : (x_disc - 1);
    }
    if (cell[1] < y_disc - 1 && cell[1] >= 0)
    {
        ++cell[1];
    }
    else
    {
        cell[1] = (cell[1] < 0) ? 1 : (y_disc - 1);
    }
    if (cell[2] < z_disc - 1 && cell[2] >= 0)
    {
        ++cell[2];
    }
    else
    {
        cell[2] = (cell[2] < 0) ? 1 : (z_disc - 1);
    }
    grid_methods::cell3(x_disc, y_disc, z_disc, x_in, y_in, z_in,
                        cell, cell + 1, cell + 2, &a, &b, &g, const_cast<float *>(point),
                        amat, bmat, &status);
    ret = -status;
    if (ret != 0 && no_idea && (x_disc > 3 || y_disc > 3 || z_disc > 3))
    {
        // try the opposite cell as starting point
        cell[0] = x_disc - 1;
        cell[1] = y_disc - 1;
        cell[2] = z_disc - 1;
        grid_methods::cell3(x_disc, y_disc, z_disc, x_in, y_in, z_in,
                            cell, cell + 1, cell + 2, &a, &b, &g, const_cast<float *>(point),
                            amat, bmat, &status);
        ret = -status;
    }
    // get the cell
    --cell[0];
    --cell[1];
    --cell[2];

    if (ret == 0)
    {
        in_cell[0] = cell[0];
        in_cell[1] = cell[1];
        in_cell[2] = cell[2];
    }

    if (!velo || !v_interp) // we do not want to interpolate, so we are done
    {
        return ret;
    }
    // transform a b g
    float fem_c[3];
    fem_c[0] = (a - 0.5f) * 2.0f;
    fem_c[1] = (b - 0.5f) * 2.0f;
    fem_c[2] = (g - 0.5f) * 2.0f;
    // a cell has 8 points
    float *velos = new float[no_arrays * array_dim * 8];
    int base, num = 0;
    int array_num, comp_num;
    const float *velo_array_num;

    // expand loop for better performance
    base = ((cell[0] + 0) * y_disc * z_disc + (cell[1] + 0) * z_disc + cell[2] + 0) * array_dim;
    for (array_num = 0; array_num < no_arrays; ++array_num)
    {
        velo_array_num = velo[array_num];
        for (comp_num = 0; comp_num < array_dim; ++comp_num)
        {
            velos[num] = velo_array_num[base + comp_num];
            ++num;
        }
    }
    base = ((cell[0] + 0) * y_disc * z_disc + (cell[1] + 0) * z_disc + cell[2] + 1) * array_dim;
    for (array_num = 0; array_num < no_arrays; ++array_num)
    {
        velo_array_num = velo[array_num];
        for (comp_num = 0; comp_num < array_dim; ++comp_num)
        {
            velos[num] = velo_array_num[base + comp_num];
            ++num;
        }
    }
    base = ((cell[0] + 0) * y_disc * z_disc + (cell[1] + 1) * z_disc + cell[2] + 0) * array_dim;
    for (array_num = 0; array_num < no_arrays; ++array_num)
    {
        velo_array_num = velo[array_num];
        for (comp_num = 0; comp_num < array_dim; ++comp_num)
        {
            velos[num] = velo_array_num[base + comp_num];
            ++num;
        }
    }
    base = ((cell[0] + 0) * y_disc * z_disc + (cell[1] + 1) * z_disc + cell[2] + 1) * array_dim;
    for (array_num = 0; array_num < no_arrays; ++array_num)
    {
        velo_array_num = velo[array_num];
        for (comp_num = 0; comp_num < array_dim; ++comp_num)
        {
            velos[num] = velo_array_num[base + comp_num];
            ++num;
        }
    }
    base = ((cell[0] + 1) * y_disc * z_disc + (cell[1] + 0) * z_disc + cell[2] + 0) * array_dim;
    for (array_num = 0; array_num < no_arrays; ++array_num)
    {
        velo_array_num = velo[array_num];
        for (comp_num = 0; comp_num < array_dim; ++comp_num)
        {
            velos[num] = velo_array_num[base + comp_num];
            ++num;
        }
    }
    base = ((cell[0] + 1) * y_disc * z_disc + (cell[1] + 0) * z_disc + cell[2] + 1) * array_dim;
    for (array_num = 0; array_num < no_arrays; ++array_num)
    {
        velo_array_num = velo[array_num];
        for (comp_num = 0; comp_num < array_dim; ++comp_num)
        {
            velos[num] = velo_array_num[base + comp_num];
            ++num;
        }
    }
    base = ((cell[0] + 1) * y_disc * z_disc + (cell[1] + 1) * z_disc + cell[2] + 0) * array_dim;
    for (array_num = 0; array_num < no_arrays; ++array_num)
    {
        velo_array_num = velo[array_num];
        for (comp_num = 0; comp_num < array_dim; ++comp_num)
        {
            velos[num] = velo_array_num[base + comp_num];
            ++num;
        }
    }
    base = ((cell[0] + 1) * y_disc * z_disc + (cell[1] + 1) * z_disc + cell[2] + 1) * array_dim;
    for (array_num = 0; array_num < no_arrays; ++array_num)
    {
        velo_array_num = velo[array_num];
        for (comp_num = 0; comp_num < array_dim; ++comp_num)
        {
            velos[num] = velo_array_num[base + comp_num];
            ++num;
        }
    }
    grid_methods::interpElem(fem_c, v_interp, no_arrays * array_dim, velos);
    delete[] velos;
    return ret;
}

coDistributedObject *coDoStructuredGrid::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;
    ret = new coDoStructuredGrid(coObjInfo(), arr);
    return ret;
}

int coDoStructuredGrid::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 6)
    {
        (*il)[0].description = "X Discretization";
        (*il)[1].description = "Y Discretization";
        (*il)[2].description = "Z Discretization";
        (*il)[3].description = "X Coordinates";
        (*il)[4].description = "Y Coordinates";
        (*il)[5].description = "Z Coordinates";
        return 6;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoStructuredGrid::coDoStructuredGrid(const coObjInfo &info, int x, int y, int z,
                                       float *xc, float *yc, float *zc)
    : coDoAbstractStructuredGrid(info, "STRGRD")
{

    char tmp_str[255];

    x_coord.set_length(x * y * z);
    y_coord.set_length(x * y * z);
    z_coord.set_length(x * y * z);
    sprintf(tmp_str, "set_length: %d  get_length: %d %d %d", x * y * z, x_coord.get_length(), y_coord.get_length(), z_coord.get_length());

    covise_data_list dl[6];
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&x_disc;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&y_disc;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&z_disc;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&x_coord;
    dl[4].type = FLOATSHMARRAY;
    dl[4].ptr = (void *)&y_coord;
    dl[5].type = FLOATSHMARRAY;
    dl[5].ptr = (void *)&z_coord;
    new_ok = store_shared_dl(6, dl) != 0;
    if (!new_ok)
        return;

    x_disc = x;
    y_disc = y;
    z_disc = z;

    int i;
    float *tmpx, *tmpy, *tmpz;
    getAddresses(&tmpx, &tmpy, &tmpz);
    i = x * y * z * sizeof(float);
    memcpy(tmpx, xc, i);
    memcpy(tmpy, yc, i);
    memcpy(tmpz, zc, i);
}

coDoStructuredGrid::coDoStructuredGrid(const coObjInfo &info, int x, int y, int z)
    : coDoAbstractStructuredGrid(info, "STRGRD")
{
    char tmp_str[255];

    x_coord.set_length(x * y * z);
    y_coord.set_length(x * y * z);
    z_coord.set_length(x * y * z);

    sprintf(tmp_str, "set_length: %d  get_length: %d %d %d", x * y * z, x_coord.get_length(), y_coord.get_length(), z_coord.get_length());

    covise_data_list dl[6];
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&x_disc;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&y_disc;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&z_disc;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&x_coord;
    dl[4].type = FLOATSHMARRAY;
    dl[4].ptr = (void *)&y_coord;
    dl[5].type = FLOATSHMARRAY;
    dl[5].ptr = (void *)&z_coord;
    new_ok = store_shared_dl(6, dl) != 0;
    if (!new_ok)
        return;

    x_disc = x;
    y_disc = y;
    z_disc = z;
}

coDoStructuredGrid::coDoStructuredGrid(const coObjInfo &info,
                                       coShmArray *arr)
    : coDoAbstractStructuredGrid(info, "STRGRD")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoStructuredGrid *coDoStructuredGrid::cloneObject(const coObjInfo &newinfo) const
{
    int n[3];
    getGridSize(&n[0], &n[1], &n[2]);
    float *c[3];
    getAddresses(&c[0], &c[1], &c[2]);
    return new coDoStructuredGrid(newinfo, n[0], n[1], n[2], c[0], c[1], c[2]);
}

int coDoStructuredGrid::rebuildFromShm()
{
    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    covise_data_list dl[6];
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&x_disc;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&y_disc;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&z_disc;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&x_coord;
    dl[4].type = FLOATSHMARRAY;
    dl[4].ptr = (void *)&y_coord;
    dl[5].type = FLOATSHMARRAY;
    dl[5].ptr = (void *)&z_coord;
    return restore_shared_dl(6, dl);
}
