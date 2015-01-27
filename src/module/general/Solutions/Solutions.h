/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Calculating module for Solution-Data (Plot3D)             **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:                                                                  **
\**************************************************************************/

#ifndef _SOLUTIONS_H
#define _SOLUTIONS_H

// includes
#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>

// defines
#define _CALC_VELOCITY 0
#define _CALC_P_STATIC 1
#define _CALC_P_TOTAL 2
#define _CALC_M 3
#define _CALC_CP 4
#define _CALC_T_STATIC 5
#define _CALC_T_TOTAL 6
#define _CALC_MACH 7

#define _VECTOR_OUTPUT 1
#define _SCALAR_OUTPUT 2

class Solutions : public coSimpleModule
{

private:
    int compute(const char *port);

    //ports
    coInputPort *p_density, *p_momentum, *p_energy, *p_rhou, *p_rhov, *p_rhow;
    coOutputPort *p_solution;

    // parameters
    coChoiceParam *p_calctype;
    int calctype;
    coFloatParam *p_gamma, *p_cp, *p_Tref, *p_cref;
    coChoiceParam *p_T_or_c;

    // functions
    coDistributedObject *Calculate(const coDistributedObject *, const coDistributedObject *, const coDistributedObject *, const coDistributedObject *, const coDistributedObject *,
                                   const coDistributedObject *, const char *);
    int Get_Output_Type(int);
    void prepare(float density, float momentum_x, float momentum_y, float momentum_z,
                 float *u, float *v, float *w);

public:
    Solutions(int argc, char *argv[]);
};
#endif // _SOLUTIONS_H
