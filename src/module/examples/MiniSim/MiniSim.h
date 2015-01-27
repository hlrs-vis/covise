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

#ifndef YAC
#include <appl/ApplInterface.h>
#endif

using namespace covise;

#include <api/coSimLib.h>
#include <stdlib.h>
#include <stdio.h>

using namespace covise;

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace covise;

class MiniSim : public coSimLib
{
    COMODULE

private:
    //////////  member functions
    virtual int compute(const char *port);
    virtual int endIteration();

    // Create the mesh
    void dummyMesh();

    // current step number, -1 if not connected
    int stepNo;

    coBooleanParam *boolPara;
    coChoiceParam *choicePara;
#ifndef YAC
    coFloatSliderParam *val111, *val117, *val777;
#else
    coFloatParam *val111, *val117, *val777;
#endif
    coFloatParam *relax;
    coIntScalarParam *steps;
    coStringParam *dir;

    coOutputPort *mesh, *data;

public:
    MiniSim(int argc, char *argv[]);

    virtual ~MiniSim()
    {
    }

#ifdef YAC
    virtual void paramChanged(coParam *param);
#endif
};
#endif // _READSTAR_H
