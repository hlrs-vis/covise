/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _FILTERCROP_H
#define _FILTERCROP_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE FilterCrop application module                     **
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
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>

using namespace covise;

class FilterCrop : public coSimpleModule
{

private:
    // params & ports
    coInputPort *pMeshIn;
    coInputPort *pDataIn;
    coOutputPort *pMeshOut;
    coOutputPort *pDataOut;

    coIntSliderParam *pIMin, *pIMax;
    coIntSliderParam *pJMin, *pJMax;
    coIntSliderParam *pKMin, *pKMax;
    coIntScalarParam *pSample;

    // private member functions
    //
    virtual int compute(const char *port);

    // private data
    //
    //  Local data
    int npoint, plane, dim;
    int i_min, i_max;
    int j_min, j_max;
    int k_min, k_max;
    int sample;
    int i_dim, j_dim, k_dim; //  Dim. of input object
    int l_dim, m_dim, n_dim; //  Dim. of output object
    float x_min, x_max, y_min, y_max, z_min, z_max;
    float *x_in, *x_out;
    float *y_in, *y_out;
    float *z_in, *z_out;
    float *s_in, *s_out;
    float *u_in, *u_out;
    float *v_in, *v_out;
    float *w_in, *w_out;
    const char *dtype, *gtype;
    const char *GridIn, *GridOut, *DataIn, *DataOut;

    //  Shared memory data
    const coDoFloat *s_data_in;
    coDoFloat *s_data_out;
    const coDoVec3 *v_data_in;
    coDoVec3 *v_data_out;
    const coDoUniformGrid *u_grid_in;
    coDoUniformGrid *u_grid_out;
    const coDoRectilinearGrid *r_grid_in;
    coDoRectilinearGrid *r_grid_out;
    const coDoStructuredGrid *s_grid_in;
    coDoStructuredGrid *s_grid_out;

    //  Routines
    coDistributedObject *create_strgrid_plane();
    coDistributedObject *create_rectgrid_plane();
    coDistributedObject *create_unigrid_plane();
    coDistributedObject *create_scalar_plane();
    coDistributedObject *create_vector_plane();

public:
    FilterCrop(int argc, char *argv[]);
    virtual ~FilterCrop();

    //coDistributedObject **ComputeObject(coDistributedObject **, char **, int,int);
    coDistributedObject *HandleObjects(const coDistributedObject *mesh_object, char *Mesh_out_name);
};
#endif // _FILTERCROP_H
