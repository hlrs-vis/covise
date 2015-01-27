/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <util/coviseCompat.h>
#include <do/coDoUniformGrid.h>
#include <config/CoviseConfig.h>

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <netdb.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#endif
#ifndef YAC
#include <appl/ApplInterface.h>
#endif
#include "ReadSim.h"

//// Constructor : set up User Interface//

#define VERBOSE
#undef MESSAGES

/*************************************************************
 *************************************************************
 **                                                         **
 **                  K o n s t r u k t o r                  **
 **                                                         **
 *************************************************************
 *************************************************************/

ReadSim::ReadSim(int argc, char *argv[])
    : coSimLib(argc, argv, argv[0], "Simulation coupling")
{
////////// set up default parameters
#ifndef YAC
    set_module_description("ReadSim Simulation");
#endif

    p_ConnectionMethod = addChoiceParam("ConnectionMethod", "ConnectionMethod");
    s_ConnectionMethod[0] = strdup("local");
    s_ConnectionMethod[1] = strdup("ssh");
    s_ConnectionMethod[2] = strdup("rdaemon");
    s_ConnectionMethod[3] = strdup("reattach");

    p_ConnectionMethod->setValue(CM_MAX, s_ConnectionMethod, 0);

    p_Hostname = addStringParam("Hostname", "Hostname");
    p_Hostname->setValue("localhost");

    p_Port = addStringParam("Port", "Port");
    p_Port->setValue(coCoviseConfig::getEntry("value", "Module.ReadSim.PORTS", "31500 31510").c_str());

    p_StartScript = addStringParam("StartupScriptOnClient", "Startup Script on Client");
    p_StartScript->setValue(coCoviseConfig::getEntry("value", "Module.ReadSim.StartUpScript", "~/bin/covise").c_str());

    p_User = addStringParam("User", "User");

    boolPara = addBooleanParam("pause", "PauseSimulation");

    dimX = addInt32Param("X", "DimensionInX");
    dimX->setValue(100);

    dimY = addInt32Param("Y", "DimensionInY");
    dimY->setValue(100);

    dimZ = addInt32Param("Z", "DimensionInZ");
    dimZ->setValue(100);

    sSteps = addInt32Param("steps", "Simulation-Steps");
    sSteps->setValue(500);

    // Output ports:

    mesh = addOutputPort("mesh", "StructuredGrid", "Data Output");
    vector0 = addOutputPort("vector0", "Vec3", "Vector data");

    char buf[1000];
    int i, max = 0;

    max = coCoviseConfig::getInt("Module.ReadSim.DATAPORTS", NUMDATA);

    for (i = 0; i < max; i++)
    {
        sprintf(buf, "data%d", i);
        dataPort[i] = addOutputPort(buf, "Float", buf);
        dataPort[i]->setInfo(buf);
    }

    numData = addInt32Param("numData", "Number of Float Data Outputs");
    numData->setValue(max);

    stopS = addInt32Param("stop", "Stop Simulation");
    stopS->setValue(0);

    stepNo = -1;
}

int ReadSim::compute(const char *port)
{
    (void)port;

    int reattach = !strcmp(s_ConnectionMethod[p_ConnectionMethod->getValue()], "reattach");
    if (reattach)
    {
        p_ConnectionMethod->setValue(0);
        stepNo = -1;
    }

    if (stepNo < 0)
    {

        PrepareSimStart();

        if (startSim(reattach))
            return -1;

        stepNo = 1;
    }

    // create mesh
    createMesh();

#ifndef YAC
    executeCommands();
#endif

    return SUCCESS;
}

/*************************************************************

 *************************************************************/

// create a Grid
void ReadSim::createMesh()
{
    int xDim, yDim, zDim;

    xDim = dimX->getValue();
    yDim = dimY->getValue();
    zDim = dimZ->getValue();

    coDoUniformGrid *grid = new coDoUniformGrid(mesh->getObjName(), xDim, yDim, zDim, 0, xDim - 1, 0, yDim - 1, 0, zDim - 1);

#ifdef YAC
    // set blockno/timestep
    grid->getHdr()->setBlock(0, 1);
    grid->getHdr()->setTime(-1, 0);
    grid->getHdr()->setRealTime(1.0);
#endif
    mesh->setCurrentObject(grid);
}

void ReadSim::param(const char *paramname, bool inMapLoading)
{
    int connMeth;
    (void)inMapLoading;

    if (!strcmp(paramname, p_ConnectionMethod->getName()))
    {

        connMeth = p_ConnectionMethod->getValue();
        //fprintf(stderr,"s_ConnectionMethod[connMeth]=%s\n",s_ConnectionMethod[connMeth]);
        if (!strcmp(s_ConnectionMethod[connMeth], "local"))
        {
            p_User->disable();
            p_Hostname->disable();
        }
        else
        {
            p_User->enable();
            p_Hostname->enable();
        }
    }
    else if (!strcmp(paramname, p_Port->getName()))
    {
        int min, max;
        int n = sscanf(p_Port->getValue(), "%d %d", &min, &max);

        if (!n)
            return;

        if (n == 1)
            setPorts(min, min);
        else
            setPorts(min, max);
    }

    numData->disable();
}

void ReadSim::PrepareSimStart()
{
    char *startshell = NULL;
    char simStart[1024];
    int connMeth;

    startshell = ConnectionString();

    connMeth = p_ConnectionMethod->getValue();

    setUserArg(0, startshell);

    const char *caseString = p_StartupSwitch->getActLabel();

    if (!strcmp(s_ConnectionMethod[connMeth], "rdaemon"))
    {
        sprintf(simStart, "\"%s start %s \" \" \" \"%s\" \" %s \"", SimBatchString(), caseString, p_Hostname->getValue(), p_User->getValue());
        //sprintf(simStart, "\"%s start %s %d\" \" \" \"%s\" \" %s \"",  SimBatchString(), caseString, p_Hostname->getValue(), p_User->getValue());
    }
    else if (!strcmp(s_ConnectionMethod[connMeth], "reattach"))
    {

        snprintf(simStart, sizeof(simStart), "()");
    }
    else
    {
        sprintf(simStart, "%s start %s ", SimBatchString(), caseString);
    }

    setUserArg(1, simStart);

    if (startshell)
        free(startshell);
}

const char *ReadSim::SimBatchString()
{
    //int local = 0;
    //int connMeth;

    const char *dp;

    //connMeth = p_ConnectionMethod->getValue();
    //local = !strcmp(s_ConnectionMethod[connMeth], "local");

    dp = p_StartScript->getValue();

    return (strdup(dp));
}

char *ReadSim::ConnectionString()
{
    int connMeth;
    char connStr[100];

    memset(connStr, 0, sizeof(connStr));

    connMeth = p_ConnectionMethod->getValue();

#ifndef WIN32
    if (!strcmp(s_ConnectionMethod[connMeth], "local") || !strcmp(s_ConnectionMethod[connMeth], "reattach"))
        *connStr = ' ';
    else
    {
        char user[50];

        memset(user, 0, sizeof(user));
        if (p_User->getValue() && *p_User->getValue())
            sprintf(user, "-l %s", p_User->getValue());
        sprintf(connStr, "%s %s %s", s_ConnectionMethod[connMeth],
                user, p_Hostname->getValue());
    }
#else
    sprintf(connStr, "%s", s_ConnectionMethod[connMeth]);
#endif

    return strdup(connStr);
}

int ReadSim::endIteration()
{

    return 1;
}

#ifdef YAC
void ReadSim::paramChanged(coParam *param)
{

    (void)param;
}
#endif

MODULE_MAIN(HLRS, ReadSim)
