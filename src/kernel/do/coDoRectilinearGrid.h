/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_RECTILINEAR_GRID_H
#define CO_DO_RECTILINEAR_GRID_H

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

class DOEXPORT coDoRectilinearGrid : public coDoAbstractStructuredGrid
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);
    coIntShm x_disc; // number of points in x-direction (X)
    coIntShm y_disc; // number of points in y-direction (Y)
    coIntShm z_disc; // number of points in z-direction (Z)
    coFloatShmArray x_coord; // coordinates in x-direction (length X)
    coFloatShmArray y_coord; // coordinates in y-direction (length Y)
    coFloatShmArray z_coord; // coordinates in z-direction (length Z)

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoRectilinearGrid *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoRectilinearGrid(const coObjInfo &info)
        : coDoAbstractStructuredGrid(info, "RCTGRD")
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
    };
    coDoRectilinearGrid(const coObjInfo &info, coShmArray *arr);
    coDoRectilinearGrid(const coObjInfo &info, int x, int y, int z,
                        float *xc, float *yc, float *zc);
    coDoRectilinearGrid(const coObjInfo &info, int x, int y, int z);
    virtual ~coDoRectilinearGrid(){};
    virtual void getGridSize(int *x, int *y, int *z) const
    {
        *x = x_disc;
        *y = y_disc;
        *z = z_disc;
    };
    virtual void getPointCoordinates(int i, float *x_c,
                                     int j, float *y_c, int k, float *z_c) const
    {
        *x_c = x_coord[i];
        *y_c = y_coord[j];
        *z_c = z_coord[k];
    };
    void getAddresses(float **x_c, float **y_c, float **z_c) const
    {
        *x_c = (float *)x_coord.getDataPtr();
        *y_c = (float *)y_coord.getDataPtr();
        *z_c = (float *)z_coord.getDataPtr();
    };
    virtual int interpolateField(float *v_interp, const float *point,
                                 int *cell, int no_arrays, int array_dim,
                                 const float *const *velo);
};
}
#endif
