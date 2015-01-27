/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef UPDATECHOICE_H
#define UPDATECHOICE_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Interpolation from Cell Data to Vertex Data               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Andreas Werner                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  05.01.97  V0.1                                                  **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <api/coModule.h>
using namespace covise;

class UpdateChoice : public coModule
{

private:
    //////////  member functions
    virtual void param(const char *name, bool inMapLoading);
    virtual void postInst();

    ////////// all our parameter ports
    coChoiceParam *choImm, *choCom;
    coStringParam *stringImm, *stringCom;

    ////////// the data in- and output ports
    coInputPort *inPortReq, *inPortNoReq;
    coOutputPort *outPort;

public:
    UpdateChoice(int argc, char *argv[]);
};
#endif
