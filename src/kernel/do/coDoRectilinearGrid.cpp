/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoRectilinearGrid.h"
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

int coDoRectilinearGrid::interpolateField(float *v_interp, const float *point,
                                          int *cell, int no_arrays, int array_dim,
                                          const float *const *velo)
{
    int ret = 0;
    float tmp;
    int fx = 0, fy = 0, fz = 0; // set to 1 if coordinates dimish as we advance in the array
    float *cell_p;
    float x_min, y_min, z_min;
    float x_max, y_max, z_max;
    float *x_start, *y_start, *z_start;

    getAddresses(&x_start, &y_start, &z_start);
    x_min = x_start[0];
    x_max = x_start[x_disc - 1];
    if (x_min > x_max)
    {
        fx = 1;
        tmp = x_min;
        x_min = x_max;
        x_max = tmp;
    }
    y_min = y_start[0];
    y_max = y_start[y_disc - 1];
    if (y_min > y_max)
    {
        fy = 1;
        tmp = y_min;
        y_min = y_max;
        y_max = tmp;
    }
    z_min = z_start[0];
    z_max = z_start[z_disc - 1];
    if (z_min > z_max)
    {
        fz = 1;
        tmp = z_min;
        z_min = z_max;
        z_max = tmp;
    }

    // assume that the coordinate arrays are monotone sequences
    if (point[0] < x_min || point[0] > x_max || point[1] < y_min || point[1] > y_max || point[2] < z_min || point[2] > z_max)
    {
        ret = -1; // not in domain
    }
    // use binary search to determine cell identifiers
    if (fx == 0)
    {
        cell_p = static_cast<float *>(bsearch(&point[0], x_start, x_disc - 1,
                                              sizeof(float), grid_methods::asc_compar_fp));
    }
    else
    {
        cell_p = static_cast<float *>(bsearch(&point[0], x_start, x_disc - 1,
                                              sizeof(float), grid_methods::desc_compar_fp));
    }
    if (cell_p)
    {
        cell[0] = (int)(cell_p - x_start);
    }
    else if (fx == 0)
    {
        cell[0] = (point[0] < x_min) ? 0 : x_disc - 2;
    }
    else
    {
        cell[0] = (point[0] > x_max) ? 0 : x_disc - 2;
    }

    if (fy == 0)
    {
        cell_p = static_cast<float *>(bsearch(&point[1], y_start, y_disc - 1,
                                              sizeof(float), grid_methods::asc_compar_fp));
    }
    else
    {
        cell_p = static_cast<float *>(bsearch(&point[1], y_start, y_disc - 1,
                                              sizeof(float), grid_methods::desc_compar_fp));
    }
    if (cell_p)
    {
        cell[1] = (int)(cell_p - y_start);
    }
    else if (fy == 0)
    {
        cell[1] = (point[1] < y_min) ? 0 : y_disc - 2;
    }
    else
    {
        cell[1] = (point[1] > y_max) ? 0 : y_disc - 2;
    }

    if (fz == 0)
    {
        cell_p = static_cast<float *>(bsearch(&point[2], z_start, z_disc - 1,
                                              sizeof(float), grid_methods::asc_compar_fp));
    }
    else
    {
        cell_p = static_cast<float *>(bsearch(&point[2], z_start, z_disc - 1,
                                              sizeof(float), grid_methods::desc_compar_fp));
    }
    if (cell_p)
    {
        cell[2] = (int)(cell_p - z_start);
    }
    else if (fz == 0)
    {
        cell[2] = (point[2] < y_min) ? 0 : z_disc - 2;
    }
    else
    {
        cell[2] = (point[2] > y_max) ? 0 : z_disc - 2;
    }

    if (!velo || !v_interp) // we do not want to interpolate, so we are done
    {
        return ret;
    }
    float fem_c[3];
    fem_c[0] = x_start[cell[0] + 1] != x_start[cell[0]] ? (point[0] - x_start[cell[0]]) / (x_start[cell[0] + 1] - x_start[cell[0]]) : 0.5f;
    fem_c[1] = y_start[cell[1] + 1] != y_start[cell[1]] ? (point[1] - y_start[cell[1]]) / (y_start[cell[1] + 1] - y_start[cell[1]]) : 0.5f;
    fem_c[2] = z_start[cell[2] + 1] != z_start[cell[2]] ? (point[2] - z_start[cell[2]]) / (z_start[cell[2] + 1] - z_start[cell[2]]) : 0.5f;
    fem_c[0] -= 0.5f;
    fem_c[1] -= 0.5f;
    fem_c[2] -= 0.5f;
    fem_c[0] += fem_c[0];
    fem_c[1] += fem_c[1];
    fem_c[2] += fem_c[2];
    // a cell has 8 points
    float *velos = new float[no_arrays * array_dim * 8];
    int i, j, k, base, num = 0;
    int array_num, comp_num;
    const float *velo_array_num;
    for (i = 0; i < 2; ++i)
    {
        for (j = 0; j < 2; ++j)
        {
            for (k = 0; k < 2; ++k)
            {
                base = ((cell[0] + i) * y_disc * z_disc + (cell[1] + j) * z_disc + cell[2] + k) * array_dim;
                for (array_num = 0; array_num < no_arrays; ++array_num)
                {
                    velo_array_num = velo[array_num];
                    for (comp_num = 0; comp_num < array_dim; ++comp_num)
                    {
                        velos[num] = velo_array_num[base + comp_num];
                        ++num;
                    }
                }
            }
        }
    }
    grid_methods::interpElem(fem_c, v_interp, no_arrays * array_dim, velos);
    delete[] velos;
    return ret;
}

coDistributedObject *coDoRectilinearGrid::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;
    ret = new coDoRectilinearGrid(coObjInfo(), arr);
    return ret;
}

int coDoRectilinearGrid::getObjInfo(int no, coDoInfo **il) const
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

coDoRectilinearGrid::coDoRectilinearGrid(const coObjInfo &info, int x, int y, int z,
                                         float *xc, float *yc, float *zc)
    : coDoAbstractStructuredGrid(info, "RCTGRD")
{

    x_coord.set_length(x);
    y_coord.set_length(y);
    z_coord.set_length(z);
#ifdef DEBUG
    cerr << "vor store_shared coDoRectilinearGrid\n";
#endif
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
    i = x_disc.get() * sizeof(float);
    memcpy(tmpx, xc, i);
    i = y_disc.get() * sizeof(float);
    memcpy(tmpy, yc, i);
    i = z_disc.get() * sizeof(float);
    memcpy(tmpz, zc, i);
}

coDoRectilinearGrid::coDoRectilinearGrid(const coObjInfo &info, int x, int y, int z)
    : coDoAbstractStructuredGrid(info, "RCTGRD")
{

    x_coord.set_length(x);
    y_coord.set_length(y);
    z_coord.set_length(z);
#ifdef DEBUG
    cerr << "vor store_shared coDoRectilinearGrid\n";
#endif
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

coDoRectilinearGrid::coDoRectilinearGrid(const coObjInfo &info, coShmArray *arr)
    : coDoAbstractStructuredGrid(info, "RCTGRD")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoRectilinearGrid *coDoRectilinearGrid::cloneObject(const coObjInfo &newinfo) const
{
    float *c[3];
    getAddresses(&c[0], &c[1], &c[2]);
    int n[3];
    getGridSize(&n[0], &n[1], &n[2]);
    return new coDoRectilinearGrid(newinfo, n[0], n[1], n[2], c[0], c[1], c[2]);
}

int coDoRectilinearGrid::rebuildFromShm()
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
