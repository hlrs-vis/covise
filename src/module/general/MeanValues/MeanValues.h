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

#ifndef _MeanValues_H
#define _MeanValues_H

// includes
#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>

// defines
#define MEAN_AVG 0
#define MEAN_ACCUM 1

#define _VECTOR_OUTPUT 1
#define _SCALAR_OUTPUT 2

class MeanValues : public coSimpleModule
{

private:
    int compute(const char *port);

    //ports
    coInputPort *p_mesh, *p_data;
    coOutputPort *p_solution, *p_numContrib;

    // parameters
    coChoiceParam *p_calctype;
    int calctype;
    // functions
    coDistributedObject * Calculate(const coDistributedObject *,
                                   const coDistributedObject *index_in, const char *, const char *nameNumContrib);
    int Get_Output_Type(int);

public:
    MeanValues(int argc, char *argv[]);
};
#endif // _MeanValues_H
