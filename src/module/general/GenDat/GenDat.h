/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GenDat_H
#define _GenDat_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:Class for GenDat                                           **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  21.07.94  V1.0                                                  **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>

class GenDat : public coModule
{

private:
    int compute(const char *);

    coOutputPort *gridOut;
    coOutputPort *scalarOut;
    coOutputPort *vectorOut;

    coChoiceParam *coordType;
    coChoiceParam *coordRepr;
    coChoiceParam *coordRange;
    coChoiceParam *func;
    coChoiceParam *orient;
    coIntSliderParam *xSize, *ySize, *zSize;
    coFloatVectorParam *start, *end;
    coIntSliderParam *timestep;
    coColorParam *color;
    coStringParam *attrName, *attrValue;

public:
    GenDat(int argc, char *argv[]);
};
#endif // _GenDat_H
