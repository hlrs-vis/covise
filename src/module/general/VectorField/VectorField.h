/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VECT_FIELD_NEW_H
#define _VECT_FIELD_NEW_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE VectorField application module                    **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  U.Woessner, Sasha Cioringa                                    **
 **                                                                        **
 **                                                                        **
 ** Date:  27.09.94  V1.0                                                  **
 ** Date:  08.11.00                                                        **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoUnstructuredGrid.h>

#define S_U 1
#define S_V 2
#define S_DATA 3
#define on_the_bottom 1
#define on_the_middle 2

class VectField : public coSimpleModule
{
    COMODULE

private:
    virtual int compute(const char *port);

    // parameters

    coFloatSliderParam *p_scale;
    coChoiceParam *p_length;
    coChoiceParam *p_fasten;
    coIntScalarParam *p_num_sectors;
    coFloatParam *p_arrow_head_factor, *p_arrow_head_angle;

    // ports
    coInputPort *p_inPort1;
    coInputPort *p_inPort2;
    coInputPort *p_inPort3;
    coOutputPort *p_outPort1;
    coOutputPort *p_outPort2;

    // private data

    // data for arrows

    int num_sectors_;
    float arrow_head_factor_, arrow_head_angle_;

    int numc;
    int length_param, fasten_param;
    float scale;
    int i_dim, j_dim, k_dim;
    float *x_in, *x_c;
    float *y_in, *y_c;
    float *z_in, *z_c;
    float *s_in;
    float *u_in, *v_in, *w_in;
    int *l_l, *v_l;

    void fillRefLines(coDoMat3 *ur_data_in,
                      coDistributedObject **linesList,
                      int nume, int numv, int nump, int *l_l, int *v_l);

public:
    VectField(int argc, char *argv[]);
};
#endif
