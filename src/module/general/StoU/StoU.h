/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _STOU_NEW_H
#define _STOU_NEW_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE Stou application module                           **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner, Sasha Cioringa                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  15.02.95  V1.0                                                  **
 ** Date:  09.11.00                                                        **
\**************************************************************************/

#include <do/coDoUnstructuredGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoUniformGrid.h>
#include <api/coSimpleModule.h>
using namespace covise;
#define MAX_DATA_PORTS 4

class StoU : public coSimpleModule
{

private:
    virtual int compute(const char *port);

    // parameters

    coChoiceParam *p_option;

    // ports
    coInputPort **p_inPort;
    coOutputPort **p_outPort;

    // private data

public:
    StoU(int argc, char *argv[]);
    virtual ~StoU();
};

class Buffer
{
private:
    float *x_S, *y_S, *z_S;
    float *x_in, *x_out;
    float *y_in, *y_out;
    float *z_in, *z_out;
    float *s_in, *s_out;
    float *u_in, *u_out;
    float *v_in, *v_out;
    float *w_in, *w_out;
    int alloc;
    const coModule *module;

public:
    Buffer(const coModule *mod, coDoStructuredGrid *);
    Buffer(const coModule *mod, coDoRectilinearGrid *);
    Buffer(const coModule *mod, coDoUniformGrid *);
    ~Buffer();

    int i_dim, j_dim, k_dim;

    coDoUnstructuredGrid *create_tetrahedrons(const char *);
    coDoUnstructuredGrid *create_hexahedrons(const char *);
    coDoUnstructuredGrid *create_pyramids(const char *);
    coDoUnstructuredGrid *create_prisms(const char *);
};
#endif
