/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ABSTRACT_STRUCTURED_GRID_H
#define CO_ABSTRACT_STRUCTURED_GRID_H

#include "coDoGrid.h"

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

class DOEXPORT coDoAbstractStructuredGrid : public coDoGrid
{
public:
    coDoAbstractStructuredGrid(const coObjInfo &info, const char *t)
        : coDoGrid(info, t)
    {
    }
    int getNumPoints() const
    {
        int x, y, z;
        getGridSize(&x, &y, &z);
        return x * y * z;
    }
    virtual void getGridSize(int *x, int *y, int *z) const = 0;
    virtual void getPointCoordinates(int i, float *x_c, int j, float *y_c, int k, float *z_c) const = 0;
    virtual int interpolateField(float *v_interp, const float *point,
                                 int *cell, int no_arrays, int array_dim,
                                 const float *const *velo) = 0;
};
}
#endif
