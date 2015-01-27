/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoUniformGrid.h"
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

coDistributedObject *coDoUniformGrid::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;
    ret = new coDoUniformGrid(coObjInfo(), arr);
    return ret;
}

void
coDoUniformGrid::SwapMinMax(int dimension)
{
    coFloatShm *min = NULL;
    coFloatShm *max = NULL;
    switch (dimension)
    {
    case 0:
        min = &x_min;
        max = &x_max;
        break;
    case 1:
        min = &y_min;
        max = &y_max;
        break;
    case 2:
        min = &z_min;
        max = &z_max;
        break;
    default:
        return;
    }
    //FIXME
    //For swapping ONE temp variable is enough, stupid!
    float tmp = *min;
    float tmp1 = *max;
    *min = tmp1;
    *max = tmp;
}

int coDoUniformGrid::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 9)
    {
        (*il)[0].description = "X Discretization";
        (*il)[1].description = "Y Discretization";
        (*il)[2].description = "Z Discretization";
        (*il)[3].description = "X minimum Value";
        (*il)[4].description = "X maximum Value";
        (*il)[5].description = "Y minimum Value";
        (*il)[6].description = "Y maximum Value";
        (*il)[7].description = "Z minimum Value";
        (*il)[8].description = "Z maximum Value";
        return 9;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoUniformGrid::coDoUniformGrid(const coObjInfo &info, coShmArray *arr)
    : coDoAbstractStructuredGrid(info, "UNIGRD")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoUniformGrid::coDoUniformGrid(const coObjInfo &info, int x, int y, int z,
                                 float xmin = 0, float xmax = 0, float ymin = 0,
                                 float ymax = 0, float zmin = 0, float zmax = 0)
    : coDoAbstractStructuredGrid(info, "UNIGRD")
{
#ifdef DEBUG
    cerr << "vor store_shared coDoRectilinearGrid\n";
#endif
    /*
       covise_data_list dl[9];
      dl[0].type = INTSHM  ; dl[0].ptr = (void *) &x_disc;
      dl[1].type = INTSHM  ; dl[1].ptr = (void *) &y_disc;
      dl[2].type = INTSHM  ; dl[2].ptr = (void *) &z_disc;
      dl[3].type = FLOATSHM; dl[3].ptr = (void *) &x_min ;
      dl[4].type = FLOATSHM; dl[4].ptr = (void *) &x_max ;
      dl[5].type = FLOATSHM; dl[5].ptr = (void *) &y_min ;
      dl[6].type = FLOATSHM; dl[6].ptr = (void *) &y_max ;
      dl[7].type = FLOATSHM; dl[7].ptr = (void *) &z_min ;
      dl[8].type = FLOATSHM; dl[8].ptr = (void *) &z_max ;
   */
    covise_data_list dl[] = {
        { INTSHM, &x_disc },
        { INTSHM, &y_disc },
        { INTSHM, &z_disc },
        { FLOATSHM, &x_min },
        { FLOATSHM, &x_max },
        { FLOATSHM, &y_min },
        { FLOATSHM, &y_max },
        { FLOATSHM, &z_min },
        { FLOATSHM, &z_max }
    };

    new_ok = store_shared_dl(9, dl) != 0;
    if (!new_ok)
        return;

    x_disc = x;
    y_disc = y;
    z_disc = z;
    x_min = xmin;
    x_max = xmax;
    y_min = ymin;
    y_max = ymax;
    z_min = zmin;
    z_max = zmax;
}

coDoUniformGrid::coDoUniformGrid(const coObjInfo &info, float dx, float dy, float dz,
                                 float xmin, float xmax, float ymin,
                                 float ymax, float zmin, float zmax)
    : coDoAbstractStructuredGrid(info, "UNIGRD")
{
#ifdef DEBUG
    cerr << "vor store_shared coDoRectilinearGrid\n";
#endif
    covise_data_list dl[9];
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&x_disc;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&y_disc;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&z_disc;
    dl[3].type = FLOATSHM;
    dl[3].ptr = (void *)&x_min;
    dl[4].type = FLOATSHM;
    dl[4].ptr = (void *)&x_max;
    dl[5].type = FLOATSHM;
    dl[5].ptr = (void *)&y_min;
    dl[6].type = FLOATSHM;
    dl[6].ptr = (void *)&y_max;
    dl[7].type = FLOATSHM;
    dl[7].ptr = (void *)&z_min;
    dl[8].type = FLOATSHM;
    dl[8].ptr = (void *)&z_max;
    new_ok = store_shared_dl(9, dl) != 0;
    if (!new_ok)
        return;

    x_disc = int((xmax - xmin) / dx + 1);
    y_disc = int((ymax - ymin) / dy + 1);
    z_disc = int((zmax - zmin) / dz + 1);
    x_min = xmin;
    x_max = xmax;
    y_min = ymin;
    y_max = ymax;
    z_min = zmin;
    z_max = zmax;
}

coDoUniformGrid *coDoUniformGrid::cloneObject(const coObjInfo &newinfo) const
{
    int n[3];
    float min[3], max[3];
    getGridSize(&n[0], &n[1], &n[2]);
    getMinMax(&min[0], &max[0], &min[1], &max[1], &min[2], &max[2]);
    return new coDoUniformGrid(newinfo,
                               n[0], n[1], n[2],
                               min[0], min[1], min[2],
                               max[0], max[1], max[2]);
}

int coDoUniformGrid::rebuildFromShm()
{
    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    covise_data_list dl[9];
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&x_disc;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&y_disc;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&z_disc;
    dl[3].type = FLOATSHM;
    dl[3].ptr = (void *)&x_min;
    dl[4].type = FLOATSHM;
    dl[4].ptr = (void *)&x_max;
    dl[5].type = FLOATSHM;
    dl[5].ptr = (void *)&y_min;
    dl[6].type = FLOATSHM;
    dl[6].ptr = (void *)&y_max;
    dl[7].type = FLOATSHM;
    dl[7].ptr = (void *)&z_min;
    dl[8].type = FLOATSHM;
    dl[8].ptr = (void *)&z_max;
    return restore_shared_dl(9, dl);
}

// extern "C" double floor(double);

// using namespace grid_methods;
// Interpolates fields of any nature given a point and an input field
// on the gitter nodes.
// return -1: point is not in domain
// return  0: is in domain, velo is interpolated in v_interp if this
//    pointer and velo are not NULL.
// Inputs: point, no_arrays, array_dim, velo.
// Outputs: v_interp, cell.

// For interpolation the function assumes that the input
// onsists of vertex-based data.
// velo is an array of $no_arrays pointers to float arrays,
// whose contents are grouped in groups of $array_dim floats.
// These groups are values for a point. This organisation
// is so complicated because of the unfelicitous fact that
// scalars and tensors are defined in a unique array and
// vectors in three. So if you get a scalar, you typically
// have no_arrays==1 and array_dim==1. If you get a vector,
// no_arrays==3 and array_dim==1. And if you get a tensor,
// then no_arrays==1 and array_dim==dimensionality of tensor type.
// The organisation of the output is as follows:
// cell is an array with 3 integer values determining the cell.
// v_interp contains no_arrays groups of array_dim floats:
// the caller is responsible for memory allocation.
int coDoUniformGrid::interpolateField(float *v_interp, const float *point,
                                      int *cell, int no_arrays, int array_dim,
                                      const float *const *velo)
{
    float dx, dy, dz;
    int ret = 0;

    if (point[0] < x_min || point[0] > x_max || point[1] < y_min || point[1] > y_max || point[2] < z_min || point[2] > z_max)
    {
        ret = -1; // not in domain
    }

    getDelta(&dx, &dy, &dz);
    cell[0] = (dx > 0) ? int(floor((point[0] - x_min) / dx)) : 0;
    cell[1] = (dy > 0) ? int(floor((point[1] - y_min) / dy)) : 0;
    cell[2] = (dz > 0) ? int(floor((point[2] - z_min) / dz)) : 0;
    if (cell[0] < 0)
        cell[0] = 0;
    if (cell[1] < 0)
        cell[1] = 0;
    if (cell[2] < 0)
        cell[2] = 0;
    if (cell[0] && cell[0] >= x_disc - 1)
        cell[0] = x_disc - 2;
    if (cell[1] && cell[1] >= y_disc - 1)
        cell[1] = y_disc - 2;
    if (cell[2] && cell[2] >= z_disc - 1)
        cell[2] = z_disc - 2;
    if (!velo || !v_interp) // we do not want to interpolate, so we are done
    {
        return ret;
    }
    float fem_c[3];
    fem_c[0] = (dx > 0) ? (point[0] - float(x_min) - dx * cell[0]) / dx : 0.5f;
    fem_c[1] = (dy > 0) ? (point[1] - float(y_min) - dy * cell[1]) / dy : 0.5f;
    fem_c[2] = (dz > 0) ? (point[2] - float(z_min) - dz * cell[2]) / dz : 0.5f;
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
