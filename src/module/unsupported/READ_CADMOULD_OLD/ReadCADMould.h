/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READSAI_H
#define _READSAI_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Read in SAI Grids and Data
 **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Ralph Bruckschen                            **
 **                            Vircinity GmbH                              **
 **                            Nobelstr. 15                                **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  21.11.99  V0.1                                                  **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include <api/coModule.h>
using namespace covise;
#include "CadmouldData.h"

class ReadSAI : public coModule
{

private:
    //////////  member functions
    virtual int compute();
    virtual void param(const char *name);
    virtual void postInst();

    ////////// the data in- and output ports
    coFileBrowserParam *meshfile, *datafile;
    coIntScalarParam *no_of_steps;
    coChoiceParam *data_mapping;
    coOutputPort *Grid_Out_Port, *Fill_Level_Data_Out_Port, *Time_Data_Out_Port;
    coOutputPort *Animated_Grid_Out_Port, *Animated_Data_Out_Port, *Line_Out_Port;
    CadmouldData *dataset;

public:
    ReadSAI();
};
#endif // _READSAI_H
