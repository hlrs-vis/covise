/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _OPENFOAM_H
#define _OPENFOAM_H
/**************************************************************************\ 
 **                                                           (C)2010 HLRS **
 **                                                                        **
 ** Description: OpenFoam interface vai SimLib                             **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Date:  18.10.2010                                                      **
\**************************************************************************/

#ifndef YAC
#include <appl/ApplInterface.h>
#endif
#include <api/coSimLibComm.h>
#include <api/coSimLib.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif
//Connection methods
#ifndef WIN32
#define CM_MAX 7
#else
#define CM_MAX 8
#endif

using namespace covise;

class OpenFoam : public coSimLib
{
    COMODULE

private:
    //////////  member functions
    virtual int compute(const char *);
    virtual void param(const char *, bool);
    virtual int endIteration();

    void sendMesh();
    void sendBoundaryConditions();

    void createUserMenu();
    void prepareSimStart(int = 1);

    char *ConnectionString();
    const char *SimBatchString();

    // current step number, -1 if not connected
    int stepNo;

    // connections ...
    coChoiceParam *p_ConnectionMethod;
    char *s_ConnectionMethod[CM_MAX];
    coStringParam *p_User;
    coStringParam *p_Hostname;
    coStringParam *p_Port;
    // client simulation startup script (path&filename)
    coStringParam *p_StartScript;

    // simulation application, needed by flow_covise startup-script
    coChoiceParam *p_simApplication;

    //coStringParam       *dir;

    coInputPort *gridPort;
    coInputPort *bocoPort;
    std::string currentGridName;

    //coOutputPort        *mesh;
    coOutputPort *pDataOutputPort;
    coOutputPort *uDataOutputPort;
    //      coOutputPort        *TDataOutputPort;

    //Number of parallel regions
    coStringParam *p_ParRegs;
    int parRegs;

public:
    OpenFoam(int argc, char *argv[]);

    virtual ~OpenFoam()
    {
    }

#ifdef YAC
    virtual void paramChanged(coParam *param);
#endif
};
#endif // _READSTAR_H
