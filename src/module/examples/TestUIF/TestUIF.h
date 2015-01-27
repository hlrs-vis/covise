/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _STAR_H
#define _STAR_H
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

class TestUIF : public coModule
{

private:
    //////////  member functions
    virtual int compute(const char *port);
    virtual void quit();
    virtual void param(const char *name, bool inMapLoading);
    virtual void postInst();

    ////////// all our parameter ports
    coBooleanParam *boolImm, *boolCom;
    coIntScalarParam *iScalImm, *iScalCom;
    coFloatParam *fScalImm, *fScalCom;
    coIntSliderParam *iSlidImm, *iSlidCom;
    coFloatSliderParam *fSlidImm, *fSlidCom;
    coIntVectorParam *iVectImm, *iVectCom;
    coFloatVectorParam *fVectImm, *fVectCom;
    coChoiceParam *choImm, *choCom, *choMaster;
    coStringParam *stringImm, *stringCom;
    coFileBrowserParam *browseImm;
    coTimerParam *timer;

    ////////// the data in- and output ports
    coOutputPort *outPort;

public:
    TestUIF(int argc, char *argv[]);
};
#endif // _READSTAR_H
