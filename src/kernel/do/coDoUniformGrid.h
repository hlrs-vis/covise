/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_UNIFORM_GRID_H
#define CO_DO_UNIFORM_GRID_H

#include "coDoAbstractStructuredGrid.h"

/*
 $Log:  $
 * Revision 1.2  1993/10/21  21:44:38  zrfg0125
 * bugs in type fixed
 *
 * Revision 1.1  93/09/25  20:51:34  zrhk0125
 * Initial revision
 *
*/

/***********************************************************************\ 
 **                                                                     **
 **   Structured class                              Version: 1.1        **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of a structured grid      **
 **                  and the data on it in a distributed manner.        **
 **                                                                     **
 **   Classes      : coDoGeometry, coDoStructuredGrid,                    **
 **                  DO_Scalar_3d_data, DO_Vector_3d_data,              **
 **                  DO_Solution                                        **
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
 **                  26.05.93  Ver 1.1 new Shm-Datatypes introduced     **
 **                                    redesign of rebuildFromShm     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/
namespace covise
{

class DOEXPORT coDoUniformGrid : public coDoAbstractStructuredGrid
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

private:
    coIntShm x_disc; // number of points in x-direction (X)
    coIntShm y_disc; // number of points in y-direction (Y)
    coIntShm z_disc; // number of points in z-direction (Z)
    coFloatShm x_min; // minimum x-value
    coFloatShm x_max; // maximum x-value
    coFloatShm y_min; // minimum y-value
    coFloatShm y_max; // maximum y-value
    coFloatShm z_min; // minimum z-value
    coFloatShm z_max; // maximum z-value

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoUniformGrid *cloneObject(const coObjInfo &newinfo) const;

public:
    /// SwapMinMax changes min and max of x(0),y(1) or z(3) directions
    void SwapMinMax(int dimension);

    coDoUniformGrid(const coObjInfo &info)
        : coDoAbstractStructuredGrid(info, "UNIGRD")
    {
        if (name)
        {
            if (getShmArray() != 0)
            {
                if (rebuildFromShm() == 0)
                {
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    }

    coDoUniformGrid(const coObjInfo &info, coShmArray *arr);
    coDoUniformGrid(const coObjInfo &info, int x, int y, int z,
                    float xmin, float xmax, float ymin,
                    float ymax, float zmin, float zmax);
    coDoUniformGrid(const coObjInfo &info,
                    float dx, float dy, float dz,
                    float xmin, float xmax, float ymin,
                    float ymax, float zmin, float zmax);
    virtual ~coDoUniformGrid()
    {
    }

    void getDelta(float *dx, float *dy, float *dz) const
    {
        *dx = ((float)x_max - (float)x_min) / (int)(x_disc - 1);
        *dy = ((float)y_max - (float)y_min) / (int)(y_disc - 1);
        *dz = ((float)z_max - (float)z_min) / (int)(z_disc - 1);
    }

    virtual void getGridSize(int *x, int *y, int *z) const
    {
        *x = x_disc;
        *y = y_disc;
        *z = z_disc;
    }

    void getMinMax(float *xmin, float *xmax,
                   float *ymin, float *ymax,
                   float *zmin, float *zmax) const
    {
        *xmin = x_min;
        *xmax = x_max;
        *ymin = y_min;
        *ymax = y_max;
        *zmin = z_min;
        *zmax = z_max;
    }

    virtual void getPointCoordinates(int i, float *x_c,
                                     int j, float *y_c,
                                     int k, float *z_c) const
    {
        if (x_disc > 1)
            *x_c = x_min + ((float)i / (float)(x_disc - 1.0)) * (x_max - x_min);
        else
            *x_c = x_min;
        if (y_disc > 1)
            *y_c = y_min + ((float)j / (float)(y_disc - 1.0)) * (y_max - y_min);
        else
            *y_c = y_min;
        if (z_disc > 1)
            *z_c = z_min + ((float)k / (float)(z_disc - 1.0)) * (z_max - z_min);
        else
            *z_c = z_min;
    }

    virtual int interpolateField(float *v_interp, const float *point,
                                 int *cell, int no_arrays, int array_dim,
                                 const float *const *velo);
};
}
#endif
