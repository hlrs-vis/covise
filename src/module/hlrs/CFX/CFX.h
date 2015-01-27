/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _STAR_H
#define _STAR_H

#ifndef YAC
#include <appl/ApplInterface.h>
using namespace covise;
#endif
#include <api/coSimLibComm.h>
#include <api/coSimLib.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef WIN32
#define NUM_STARTUP_METHODS 5
#else
#define NUM_STARTUP_METHODS 3
#endif

class CFX : public coSimLib
{
    COMODULE

private:
    //////////  member functions
    virtual int compute(const char *port);
    virtual int endIteration();
    bool findAttribute(coDistributedObject *obj, const char *name, const char *scan, void *val);

    virtual int PrepareSimStart();
    //  virtual void param(const char *, bool inMapLoading);

    // current step number, -1 if not connected
    int stepNo;
    int numProc;

    coBooleanParam *boolPara;
#ifndef WIN32
    coBooleanParam *p_xterm;
#endif

    char *s_ConnectionMethod[2];
    char *s_MachineType[8];
    char *s_StartupMethod[NUM_STARTUP_METHODS];

#ifndef WIN32
    coChoiceParam *p_ConnectionMethod;
#endif
    coChoiceParam *p_StartupMethod;
    coChoiceParam *p_MachineType;
    coStringParam *p_Hostname;
#ifndef WIN32
    coStringParam *p_Hostlist;
#endif
    coStringParam *p_username;
    coStringParam *startScript;
    coStringParam *p_deffile;

    coIntScalarParam *updateInterval;
    coIntScalarParam *p_numProc;
    coIntScalarParam *p_revolutions;
    coIntScalarParam *p_maxIterations;

    coFloatParam *p_inletVelMulti;
    coFloatParam *p_incidenceAngular;

    coInputPort *p_grid, *p_boco;

    coOutputPort *p_inlet;
    coOutputPort *TEMP_FL1_1, *PRES_1, *VEL_FL1_1;
    coOutputPort *TEMP_FL1_2, *PRES_2, *VEL_FL1_2;
    coOutputPort *TEMP_FL1_3, *PRES_3, *VEL_FL1_3;
    coOutputPort *TEMP_FL1_4, *PRES_4, *VEL_FL1_4;
    coOutputPort *p_gridout;

    char *d_distGridName;

public:
    CFX(int argc, char *argv[]);

    virtual ~CFX()
    {
    }

#ifdef YAC
    virtual void paramChanged(coParam *param);
#endif
};

#endif
