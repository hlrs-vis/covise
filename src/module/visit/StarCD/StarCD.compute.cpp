/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "StarCD.h"
#include <star/File16.h>
#include "Mesh.h"
#include <api/coFeedback.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>

#define VERBOSE

// these are the sizes of the transferred blocks: it MUST be the same like ftn.
const int StarCD::ISETUP = 4 * (1 + MAX_REGIONS * (6 + MAX_SCALARS + MAX_UDATA));
const int StarCD::IPARAM = 4 * (1 + MAX_REGIONS * (8 + MAX_SCALARS + MAX_UDATA) + NUM_OUT_DATA);

///////////////////////////////////////////////////////////////////////////
// Startup of simulation
///////////////////////////////////////////////////////////////////////////

char *StarCD::activeRegionSettings()
{
    char *settings = new char[MAX_REGIONS * (MAX_SCALARS + MAX_UDATA + 20) * 128];
    *settings = '\0';
    char buffer[256];

    // Send all region definitions to script
    int reg;
    for (reg = 0; reg < flags.iconum; reg++)
    {
        if (flags.icoreg[reg] > 0)
        {
            sprintf(buffer, "REGION %d: %s\n", flags.icoreg[reg], choices[reg + 1]);
            strcat(settings, buffer);

            sprintf(buffer, "  LOCAL %f %f %f\n", p_euler[reg]->getValue(0),
                    p_euler[reg]->getValue(1),
                    p_euler[reg]->getValue(2));
            strcat(settings, buffer);

            if (flags.icovel[reg] > 0)
            {
                sprintf(buffer, "  VMAG  %s\n", p_vmag[reg]->getValString());
                strcat(settings, buffer);

                sprintf(buffer, "  VDIR  %f %f %f\n", p_v[reg]->getValue(0),
                        p_v[reg]->getValue(1),
                        p_v[reg]->getValue(2));
                strcat(settings, buffer);
            }
            if (flags.icot[reg] > 0)
            {
                sprintf(buffer, "  T    %s\n", p_t[reg]->getValString());
                strcat(settings, buffer);
            }
            if (flags.icop[reg] > 0)
            {
                sprintf(buffer, "  P    %s\n", p_p[reg]->getValString());
                strcat(settings, buffer);
            }
            if (flags.icotur[reg] == 1)
            {
                sprintf(buffer, "  K    %s\n", p_k[reg]->getValString());
                strcat(settings, buffer);
                sprintf(buffer, "  EPS  %s\n", p_eps[reg]->getValString());
                strcat(settings, buffer);
            }
            else if (flags.icotur[reg] == -1)
            {
                sprintf(buffer, "  TIN  %s\n", p_tin[reg]->getValString());
                strcat(settings, buffer);
                sprintf(buffer, "  TLEN %s\n", p_len[reg]->getValString());
                strcat(settings, buffer);
            }
            int sca;
            for (sca = 0; sca < MAX_SCALARS; sca++)
            {
                if (flags.icosca[reg][sca])
                {
                    sprintf(buffer, "  SCAL%d  %s\n", flags.icosca[reg][sca],
                            p_scal[reg][sca]->getValString());
                    strcat(settings, buffer);
                }
            }
            int userNo;
            for (userNo = 0; userNo < MAX_UDATA; userNo++)
            {
                if (flags.icousr[reg][userNo])
                {
                    sprintf(buffer, "  SCAL%d  %s\n", flags.icousr[reg][userNo],
                            p_user[reg][userNo]->getValString());
                    strcat(settings, buffer);
                }
            }
        }
    }
    return settings;
}

int StarCD::endIteration()
{

    cerr << "endIteration" << endl;
    return 0;
}

int StarCD::startupSimulation(const coDistributedObject *setupObj,
                              const coDistributedObject *commObj,
                              const char *settingsString)
{
#ifdef VERBOSE
    cerr << "startupSimulation(" << ((setupObj) ? "OBJ" : "NO_OBJ")
         << ")" << endl;
#endif

    // maybe we have to read the setup here:
    //   - when loaded from the Map
    //   - after we quit the simulation once (the user might have changed)
    //   - A setup object came from above module (e.g. HEXA)
    //   - A commObj wants to change out set-up
    // read setup file, but do NOT use .modified if existent
    if (!d_compDir || !d_case || setupObj || commObj)
        doSetup(setupObj, commObj, 0);

    // oops, there are REAL problems
    if (!d_compDir || !d_case)
    {
        sendError("Need directory and casein config file");
        return FAIL;
    }

    if (!d_user)
    {
#ifdef _WIN32
        d_user = getenv("USERNAME");
#else
        d_user = getlogin();
#endif
        sendWarning("USER not given in config '%s' : using %s",
                    p_setup->getValue(), d_user);
    }

    if (!d_host)
    {
        char *myname = new char[256];
        gethostname(myname, 255);
        myname[255] = '\0';
        d_host = myname;
        sendWarning("HOST not given in config '%s', using %s",
                    p_setup->getValue(), d_host);
    }

    setUserArg(0, d_compDir);
    setUserArg(1, d_case);
    setUserArg(2, d_user);
    setUserArg(3, p_numProc->getValString());

    setTargetHost(d_host);

    // if the user has his own args, these override ours...
    int i;
    for (i = 0; i <= 9; i++)
        if (d_usr[i])
            setUserArg(i, d_usr[i]);

    // Apply pre-start script if given
    if (d_script)
    {
        FILE *scriptInput = popen(d_script, "w");
        if (!scriptInput)
        {
            sendError("User-defined script failed");
            return FAIL;
        }
        sendInfo("Executing user script: %s", d_script);
        fprintf(scriptInput, "USER       %s\n", d_user);
        fprintf(scriptInput, "HOST       %s\n", d_host);
        fprintf(scriptInput, "COMPDIR    %s\n", d_compDir);
        fprintf(scriptInput, "MESHDIR    %s\n", d_meshDir);
        fprintf(scriptInput, "NPROC      %d\n", (int)p_numProc->getValue());
        fprintf(scriptInput, "CASE       %s\n", d_case);
        if (d_creator)
            fprintf(scriptInput, "CREATOR    %s\n", d_creator);
        fprintf(scriptInput, "STARCONFIG %s\n", p_setup->getValue());
        fprintf(scriptInput, "%s", d_commArg);

        if (settingsString)
            fprintf(scriptInput, "%s", settingsString);

        // check return value of script
        int scriptRetVal = pclose(scriptInput);
        if (scriptRetVal < 0)
        {
            if (errno <= 0)
            {
                sendError("Script '%s' returned %d", d_script, scriptRetVal);
                return FAIL;
            }
            else
            {
                sendInfo("Script '%s' : '%s'",
                         d_script, strerror(errno));
            }
        }

        // read setup file again: use .modified if existent
        doSetup(setupObj, commObj, 1);
    }

    // now really start up the simulation
    if (startSim())
    {
        sendError("Simulation did not come up");
        p_runSim->setValue(0);
        p_freeRun->setValue(0);
        return FAIL;
    }

    // this sends the flags block data to FORTRAN
    assert(ISETUP == sizeof(flags));
    if (ISETUP != sendBS_Data(&flags, ISETUP))
    {
        sendError("error sending setup data block");
        return FAIL;
    }
    else if (getVerboseLevel())
    {
        sendInfo("Sent setup data of size ISETUP=[%d]\n", ISETUP);
    }
    // get parallelisation info
    int32 numNodes;
    if (4 != recvBS_Data(&numNodes, 4))
    {
        sendError("error receiving node info");
        return FAIL;
    }
    else if (getVerboseLevel())
        sendInfo("Received block info from Simulation");

    // Retrieve Mesh from active mesh file
    // we MUST have the mesh now: startupParallel needs it
    // look in d_compDir if d_meshDir==NULL

    delete d_mesh;
    d_mesh = new StarMesh(d_compDir, d_case, d_meshDir, flags.iconum, flags.icoreg);
    switch (d_mesh->getStatus())
    {
    case -1:
    {
        sendError("Could not read StarCD mesh file");
        delete d_mesh;
        d_mesh = NULL;
        stopSimulation();
        return FAIL;
    }
    case 1:
    {
        Covise::sendWarning("some interaction BC have no inlet patches assigned");
        break;
    }
    }

    // this is a parallel case : prepare parallelisation
    if (numNodes > 0)
        if (startupParallel(numNodes))
            return FAIL;

    // make the Mapeditor reflect the REAL number of processors
    p_numProc->setValue(numNodes);

    // we cannot change it during the run
    p_numProc->disable();

    // Simulation is up, waiting for BOCO
    cerr << "STARTUPSIMULATION d_simstate: WAIT_BOCO" << endl;
    d_simState = WAIT_BOCO;

    sendInfo("Simulation connected to COVISE");

    return SUCCESS;
}

///////////////////////////////////////////////////////////////////////////
// Do the parallelisation stuff
///////////////////////////////////////////////////////////////////////////

int StarCD::startupParallel(int numNodes)
{
#ifdef VERBOSE
    cerr << "startupParallel(" << numNodes << ")" << endl;
#endif

    int i;

    // tell SimLib we go parallel
    coParallelInit(numNodes, NUM_OUT_DATA);

    // which output ports gather parallel cell data
    for (i = 0; i < NUM_OUT_DATA; i++)
        coParallelPort(p_data[i]->getName(), 1);

    // and now receive the cell data gathering tables
    int32 length;
    for (i = 0; i < numNodes; i++)
    {
        if (4 != recvBS_Data(&length, 4))
        {
            sendError("error receiving node mapping length");
            return FAIL;
        }
        int32 *map = new int32[length];
        if (4 * length != recvBS_Data(map, 4 * length))
        {
            sendError("error receiving node mapping data");
            return FAIL;
        }

        // convert ProStar to Covise numbering
        d_mesh->convertMap(map, length);

        // and set the local tables
        setParaMap(1, 0, i, length, map);
    }

    return SUCCESS;
}

///////////////////////////////////////////////////////////////////////////
// collect all port data into one block and send it
///////////////////////////////////////////////////////////////////////////

int StarCD::sendPortValues()
{
#ifdef VERBOSE
    cerr << "sendPortValues()" << endl;
#endif

    if (d_simState != WAIT_BOCO)
    {
        sendError("trying to send BOCO while StarCD doesn't wait for it");
        return FAIL;
    }

    int reg, scal;
    for (reg = 0; reg < flags.iconum; reg++) // for used regions: get only used components
    {
        // velocity components
        if (flags.icovel[reg])
            p_v[reg]->getValue(para.covvel[reg][0],
                               para.covvel[reg][1],
                               para.covvel[reg][2]);

        if (flags.icot[reg])
            para.covt[reg] = p_t[reg]->getValue() + 273.15;

        if (flags.icoden[reg])
            para.covden[reg] = p_den[reg]->getValue();

        if (flags.icotur[reg] == -1)
        {
            para.covtu1[reg] = p_k[reg]->getValue();
            para.covtu1[reg] = p_eps[reg]->getValue();
        }

        else if (flags.icotur[reg] == 1)
        {
            para.covtu1[reg] = p_tin[reg]->getValue();
            para.covtu1[reg] = p_len[reg]->getValue();
        }

        if (flags.icop[reg])
            para.covp[reg] = p_p[reg]->getValue();

        for (scal = 0; scal < MAX_SCALARS; scal++)
            if (flags.icosca[reg][scal])
                para.covsca[reg][scal] = p_scal[reg][scal]->getValue();

        for (scal = 0; scal < MAX_UDATA; scal++)
            if (flags.icousr[reg][scal])
                para.covusr[reg][scal] = p_user[reg][scal]->getValue();
    }

    // number of steps to go
    para.icostp = p_steps->getValue();

    // output data selection
    int i;
    for (i = 0; i < NUM_OUT_DATA; i++)
        para.icoout[i] = p_value[i]->getValue();

    // this sends the BC block data to FORTRAN
    assert(IPARAM == sizeof(para));
    if (IPARAM != sendBS_Data(&para, IPARAM))
    {
        sendError("error sending boco data block");
        return FAIL;
    }
    else
        sendInfo("Sent Boco: Simulation running...");
    cerr << "d_simstate: RUNNING" << endl;
    d_simState = RUNNING;

    return SUCCESS;
}

///////////////////////////////////////////////////////////////////////////
// Stop the simulation : only possible in WAIT_BOCO state
///////////////////////////////////////////////////////////////////////////
int StarCD::stopSimulation()
{
#ifdef VERBOSE
    cerr << "stopSimulation()" << endl;
#endif

    assert(d_simState == WAIT_BOCO);
    assert(IPARAM == sizeof(para));

    // send normal param bock with icostp=-999 as QUIT signal
    para.icostp = -999;
    if (IPARAM != sendBS_Data(&para, IPARAM))
        sendError("error sending boco data block");
    else
        sendInfo("Sent QUIT message to simulation");

    // whatever happened here - the module CAN disconnect
    d_simState = NOT_STARTED;
    p_quitSim->enable();
    p_quitSim->setValue(0);
    p_quitSim->disable();
    p_runSim->setValue(0);
    p_freeRun->setValue(0);
    resetSimLib();

    // make sure we re-read the setup at next start
    delete[] d_compDir;
    d_compDir = NULL;
    delete[] d_meshDir;
    d_meshDir = NULL;
    delete[] d_case;
    d_case = NULL;
    delete[] d_setupFileName;
    d_setupFileName = NULL;

    // now we can change the number of procs
    p_numProc->enable();

    // we mignt change the set-up now, so read mesh new next time
    delete d_mesh;
    d_mesh = NULL;

    return SUCCESS;
}

///////////////////////////////////////////////////////////////////////////
// Create the Feedback objects
void StarCD::createFeedbackObjects()
{
#ifdef VERBOSE
    cerr << "createFeedbackObjects()" << endl;
#endif

    // the region patches on the feedback port
    coDistributedObject **feedPatches
        = d_mesh->getBCPatches(p_feedback->getObjName());

    ////////////////////////////////////////////////////////////////////
    // add the current parameter calues as feedback parameters
    coDistributedObject **actPatch = feedPatches;

    int i = 0;
    while (*actPatch)
    {
        coFeedback feedback("StarCD");
        feedback.addString(choices[i + 1]);
        // we always add the parameter for the local coordinate systems
        feedback.addPara(p_euler[i]);

        // the others only if used
        if (flags.icovel[i]) // add VELOCITY+MAG
        {
            feedback.addPara(p_v[i]);
            feedback.addPara(p_vmag[i]);
        }
        if (flags.icot[i])
            feedback.addPara(p_t[i]);
        if (flags.icoden[i])
            feedback.addPara(p_den[i]);
        if (flags.icotur[i] > 0)
        {
            feedback.addPara(p_k[i]);
            feedback.addPara(p_eps[i]);
        }
        if (flags.icotur[i] < 0)
        {
            feedback.addPara(p_tin[i]);
            feedback.addPara(p_len[i]);
        }
        if (flags.icop[i])
            feedback.addPara(p_p[i]);

        int j;
        for (j = 0; j < MAX_SCALARS; j++)
            if (flags.icosca[i][j])
                feedback.addPara(p_scal[i][j]);

        for (j = 0; j < MAX_UDATA; j++)
            if (flags.icousr[i][j])
                feedback.addPara(p_user[i][j]);

        feedback.apply(*actPatch);

        actPatch++;
        i++;
    }

    // make the set
    coDoSet *feed = new coDoSet(p_feedback->getObjName(), feedPatches);
    for (i = 0; feedPatches[i] != NULL; i++)
        delete feedPatches[i];

    // Make COVER start the Plugin (if not done before) -> now in API
    //feed->addAttribute("MODULE","StarCDPlugin");

    // create the Feedback attributes for the feedback container object
    coFeedback feedback("StarCD");

    // First string: whether we use this setup first time or not
    if (d_useOldConfig)
        feedback.addString("OLD");
    else
    {
        feedback.addString("NEW");
        d_useOldConfig = 1;
    }
    // Index 0 is the Header label ("Regions")
    for (i = 1; i <= numRegions; i++)
        feedback.addString(choices[i]);

    //residualLabels = getResidualLabels();
    //feedback.addString(residualLabels);
    // add feedback for global parameters
    feedback.addPara(p_freeRun);
    feedback.addPara(p_runSim);
    feedback.addPara(p_steps);
    feedback.apply(feed);

    p_feedback->setCurrentObject(feed);
}

///////////////////////////////////////////////////////////////////////////
//
// Check, whether we have to reconfigure now
//
///////////////////////////////////////////////////////////////////////////

int StarCD::mustReconfigure()
{
    // configObj = Object von Grid Generator
    const coDistributedObject *configObj = p_configObj->getCurrentObject();

    // new config object ?
    if (configObj)
    {
        const char *configObjName = configObj->getName();
        if (!d_setupObjName || strcmp(d_setupObjName, configObjName))
            return 1;
    }

    // have a command object ? // any sequence of .starconfig commands,
    // e.g. NUMADDPART + ADDPART linse from ModifyAddPart
    const coDistributedObject *commObj = p_commObj->getCurrentObject();
    if (!commObj)
        return 0;

    // new command object ?
    const char *commObjName = commObj->getName();
    if (!d_commObjName || strcmp(d_commObjName, commObjName))
        return 1;

    return 0;
}

///////////////////////////////////////////////////////////////////////////
//
// Check, whether EXEC message should be ignored
//
///////////////////////////////////////////////////////////////////////////

bool StarCD::mustIgnoreExec()
{
    const coDistributedObject *configObj = p_configObj->getCurrentObject();
    const coDistributedObject *commObj = p_commObj->getCurrentObject();

    // we got NULL objects at port, although connected
    if ((d_setupObjName != NULL && configObj == NULL)
        || (d_commObjName != NULL && commObj == NULL))
    {
#ifdef VERBOSE
        cerr << "no exec: have name, but no object" << endl;
#endif
        return true;
    }
    if (commObj && commObj->getAttribute("NOEXEC"))
    {
#ifdef VERBOSE
        cerr << "no exec: NOEXEC attribute at comm object" << endl;
#endif
        return true;
    }

    if (configObj && configObj->getAttribute("NOEXEC"))
    {
#ifdef VERBOSE
        cerr << "no exec: NOEXEC attribute at config object" << endl;
#endif
        return true;
    }
    return false;
}

///////////////////////////////////////////////////////////////////////////
//
//        /####\  /####\  #\  /#  #####\  #    #   #####  ######
//        #    #  #    #  ##\/##  #    #  #    #     #    #
//        #       #    #  # ## #  #    #  #    #     #    #####
//        #       #    #  #    #  #####/  #    #     #    #
//        #    #  #    #  #    #  #       #    #     #    #
//        \####/  \####/  #    #  #       \####/     #    ######
//
///////////////////////////////////////////////////////////////////////////

int StarCD::compute(const char *)
{
    const coDistributedObject *configObj = p_configObj->getCurrentObject();
    const coDistributedObject *commObj = p_commObj->getCurrentObject();

    if (mustIgnoreExec())
        return STOP_PIPELINE;

#ifdef VERBOSE
    cerr << "COMPUTE: "
         << ((configObj) ? configObj->getName() : "without setup OBJ, ");
    switch (d_simState)
    {
    case NOT_STARTED:
        cerr << " : NOT_STARTED" << endl;
        break;
    case WAIT_BOCO:
        cerr << " : WAIT_BOCO" << endl;
        break;
    case RUNNING:
        cerr << " : RUNNING" << endl;
        break;
    case SHUTDOWN:
        cerr << " : SHUTDOWN" << endl;
        break;
    case RESTART:
        cerr << " : RESTART" << endl;
        break;
    }
#endif

    // if we are not connected, we'll want to do that now...
    if (d_simState == NOT_STARTED)
    {
        // if there is an error, we get out of here
        if (startupSimulation(configObj, commObj, NULL) == FAIL)
            return STOP_PIPELINE;
    }
    else
    {
        p_quitSim->enable(); // if we are running, we want to be able to stop
        configObj = NULL; // configObj has done its work for this turn
    }

    // A new config object OR -command came in, but sim can't handle it yet.
    // If the simulatuon requested EXEC, it has to shut down and re-configure
    // after sending its data
    if ((d_simState == RUNNING || d_simState == SHUTDOWN || d_simState == RESTART)
        && mustReconfigure()
        && !simRequestExec())
    {
        cerr << "d_simstate: RESTART" << endl;
        d_simState = RESTART;
        return STOP_PIPELINE; // don't care - we'll re-use the object next time
    }

    // if Simulation is not yet running, send BOCO now and return
    // .. autoexec when data arrives
    if (d_simState == WAIT_BOCO)
    {
        // re-configure NOW and immediately re-start if necessary
        if (mustReconfigure())
        {
            stopSimulation();
            char *oldSetup = activeRegionSettings();

            if (startupSimulation(configObj, commObj, oldSetup) == FAIL)
            {
                sendError("Start-up of new Star Simulation failed");
                delete[] oldSetup;
                return STOP_PIPELINE;
            }
            else
            {
                d_useOldConfig = 0;
                p_quitSim->enable(); // if we are running, we want to be able to stop
                delete[] oldSetup;
            }
        }
        sendPortValues(); // this will set d_simState to RUNNING if successful
        p_runSim->setValue(1); // make that button red
        return STOP_PIPELINE; // don't do anything behind now, no data yet
    }

    executeCommands();

    // put a grid on the grid output port
    p_mesh->setCurrentObject(d_mesh->getGrid(p_mesh->getObjName()));

    // if we stopped (not free-running): push out the RUN button
    if (!p_freeRun->getValue())
        p_runSim->setValue(0);

    //// Cover plugin information object is created here
    createFeedbackObjects();

    // from now on this config is OLD until a new config is done
    d_useOldConfig = 1;

    // attach the Plot-Attributes to the residual Data : ignore errors
    d_mesh->attachPlotAttrib(p_residual->getCurrentObject());

    // save old simulation status and set current
    SimStatus oldStatus = d_simState;
    cerr << "COMPUTE d_simstate: WAIT_BOCO " << oldStatus << endl;
    d_simState = WAIT_BOCO;

    // if we were in RUNNING mode and wanted to stop, we bail out here
    if (oldStatus == SHUTDOWN)
    {
        stopSimulation();
        return CONTINUE_PIPELINE;
    }

    // we have to restart: kill old and fire new
    else if (oldStatus == RESTART
             || (mustReconfigure()))
    {
        stopSimulation();
        char *oldSetup = activeRegionSettings();

        if (startupSimulation(configObj, commObj, oldSetup) == FAIL)
        {
            sendError("Start-up of new Star Simulation failed");
            delete[] oldSetup;
        }
        else
        {
            delete[] oldSetup;
            d_useOldConfig = 0;
            sendPortValues(); // this will set d_simState to RUNNING if successful
            p_runSim->setValue(1); // make that button red
        }

        // whether we startes or not: the current values do not reflect the
        // current state of the simulation, so don't use it
        return STOP_PIPELINE;
    }

    // if we are in free-running mode, we send new BOCO now, else Sim must wait
    if (p_freeRun->getValue())
        if (sendPortValues()) // this 'if' is error-checking
            return FAIL;

    return SUCCESS;
}
