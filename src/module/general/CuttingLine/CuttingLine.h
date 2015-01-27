/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CuttingLine_H
#define _CuttingLine_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE CuttingPlane CuttingLine module                   **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau                                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <do/coDoData.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoUniformGrid.h>

class CuttingLine : public coSimpleModule
{

private:
    // ports

    coInputPort *p_meshIn, *p_dataIn;
    coOutputPort *p_dataOut;

    // parameters

    coChoiceParam *param_plane;
    coIntSliderParam *param_ind_i, *param_ind_j, *param_ind_k;

    //  Routines
    void create_strgrid_data();
    void create_rectgrid_data();
    void create_unigrid_data();

    // variables

    //  Local data
    const char *DataOut;
    int npoint, direction, dim;
    long i_fro_min, i_fro_max; //  Boundaries (frontier)
    long j_fro_min, j_fro_max; //  Boundaries (frontier)
    long k_fro_min, k_fro_max; //  Boundaries (frontier)
    long i_index;
    long j_index;
    long k_index;
    int i_dim, j_dim, k_dim; //  Dim. of input object
    float xpos, ypos, zpos;
    float xmin, xmax, ymin, ymax, zmin, zmax;
    float *x_in, *x_out;
    float *y_in, *y_out;
    float *z_in, *z_out;
    float *s_in, *s_out;
    float *u_in, *u_out;
    float *v_in, *v_out;
    float *w_in, *w_out;
    char COLOR[8];
    char color[6];

    //  Shared memory data
    coDoVec2 *data_out;
    const coDoFloat *s_data_in;
    const coDoVec3 *v_data_in;
    const coDoStructuredGrid *s_grid_in;
    const coDoRectilinearGrid *r_grid_in;
    const coDoUniformGrid *u_grid_in;

public:
    virtual int compute(const char *port);
    CuttingLine(int argc, char *argv[]);
};
#endif // _CuttingLine_H
