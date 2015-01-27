/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READSIM_H
#define _READSIM_H

#ifndef YAC
#include <appl/ApplInterface.h>
using namespace covise;
#endif
#include <api/coSimLib.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#define CM_MAX 4
#define NUMDATA 2

class ReadSim : public coSimLib
{

private:
    //////////  member functions
    virtual int compute(const char *port);
    virtual void param(const char *, bool inMapLoading);
    virtual int endIteration();

    virtual char *ConnectionString(void);
    virtual const char *SimBatchString(void);
    virtual void PrepareSimStart();

    // Create the mesh
    void createMesh();

    // current step number, -1 if not connected
    int stepNo;

    // connections ...
    coChoiceParam *p_ConnectionMethod;
    coStringParam *p_User;
    coStringParam *p_Hostname;
    coStringParam *p_Port;

    // client simulation startup script (path&filename)
    coStringParam *p_StartScript;

    coBooleanParam *boolPara;
    coIntScalarParam *dimX;
    coIntScalarParam *dimY;
    coIntScalarParam *dimZ;
    coIntScalarParam *stopS;
    coIntScalarParam *sSteps;
    coIntScalarParam *numData;

    coOutputPort *mesh, *data0, *data1, *data2, *data3, *data4, *vector0;
    coOutputPort *dataPort[NUMDATA];

    char *s_ConnectionMethod[CM_MAX];

public:
    ReadSim(int argc, char *argv[]);

    virtual ~ReadSim()
    {
    }

#ifdef YAC
    virtual void paramChanged(coParam *param);
#endif
};
#endif // _READSIM_H
