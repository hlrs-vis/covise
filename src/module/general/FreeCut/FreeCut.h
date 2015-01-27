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

#ifndef _FreeCut_H
#define _FreeCut_H

// includes
#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>

class FreeCut : public coSimpleModule
{

private:
    int compute(const char *port);

    //ports
    coInputPort *p_grid, *p_surface, *p_data;
    coOutputPort *p_surfaceOut, *p_dataOut;
    int numVertices;
    //bool myfunction (int i,int j);

public:
    FreeCut(int argc, char *argv[]);
};
#endif // _FreeCut_H
