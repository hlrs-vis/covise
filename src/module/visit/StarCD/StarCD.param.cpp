/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "StarCD.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

///////////////////////////////////////////////////////////////////////////
//
//                 #####     ##    #####     ##    #    #
//                 #    #   #  #   #    #   #  #   ##  ##
//                 #    #  #    #  #    #  #    #  # ## #
//                 #####   ######  #####   ######  #    #
//                 #       #    #  #   #   #    #  #    #
//                 #       #    #  #    #  #    #  #    #
//
///////////////////////////////////////////////////////////////////////////

void StarCD::param(const char *paraName, bool inMapLoading)
{
    float u, v, w, vmag;

    ////////////////////////////////////////////////////////////////////////
    // if the velocity is changed, we have to change the corresponding vmag
    if ((strstr(paraName, "vel") == paraName) && (!inMapLoading))
    {
        int num = atoi(paraName + 3); //
        p_v[num]->getValue(u, v, w);
        vmag = sqrt(u * u + v * v + w * w);
        p_vmag[num]->setValue(vmag);
    }

    ////////////////////////////////////////////////////////////////////////
    // if the vmag changed, scale corresponding uvw
    else if ((strstr(paraName, "vmag") == paraName) && (!inMapLoading))
    {
        int num = atoi(paraName + 4);
        p_v[num]->getValue(u, v, w);
        float scale = sqrt(u * u + v * v + w * w);
        vmag = p_vmag[num]->getValue();
        if ((scale == 0))
            sendWarning("cannot scale parameter %s: direction unknown", paraName);
        else
        {
            scale = vmag / scale;
            p_v[num]->setValue(u * scale, v * scale, w * scale);
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // If the setup file name changed
    else if ((strcmp(paraName, p_setup->getName()) == 0)
             && !(inMapLoading))
    {
        const char *newfile = p_setup->getValue();

        // If we press the 'setup' file twice, we don't really want it probably
        if (d_setupFileName && 0 == strcmp(newfile, d_setupFileName))
        {
            sendInfo("Already working with setup file %s", d_setupFileName);
            sendInfo("Use quit button to terminate simulation");
            return;
        }

        // If we are already working, we have to shut down
        switch (d_simState)
        {
        case RUNNING:
            d_simState = SHUTDOWN;
            sendInfo("shutting down simulation as soon as possible");
            break;

        case SHUTDOWN:
            sendInfo("still waiting for sim to shut down");
            break;

        case WAIT_BOCO:
            stopSimulation();
            break;
        default:
            ;
        }

        // we read and analyse the setup file right now, even if the
        // simulation has not stopped yet
        if (doSetup(NULL, NULL, 0))
            sendInfo("Problems with setup file %s", newfile);
        // ---> will be done again if setup objects come in, this is just in
        //      case we use a plain starConfig thing
    }

    ////////////////////////////////////////////////////////////////////////
    // If the user fires the simulation
    else if ((strcmp(paraName, p_runSim->getName()) == 0)
             && !(inMapLoading))
    {
        // if the simulation is not up yet, we'll do it right away
        if (d_simState == NOT_STARTED)
        {
            // changed: never start from here!

            // if we have an input config object, we can't fire from here: execute
            //if (d_setupObjName)
            //{
            selfExec();
            return;
            //}
            //else
            //   if (startupSimulation(NULL)==FAIL)
            //      return;                 // if there is an error, we get out of here
            //   else
            //      p_quitSim->enable();    // if we are running, we want to be able to stop
        }

        if (p_runSim->getValue())
            if (d_simState == WAIT_BOCO)
                sendPortValues(); // this will set d_simState to RUNNING if successful
            else
            {
                sendWarning("Simulation not waiting for new boco yet");
                p_runSim->setValue(0);
            }
        else
        {
            sendWarning("we cannot stop the simulation yet ... waiting for next boco");
            p_runSim->setValue(1);
            p_freeRun->setValue(0);
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // If the user wants to quit the simulation
    else if ((strcmp(paraName, p_quitSim->getName()) == 0)
             && !(inMapLoading))
    {
        if (!p_quitSim->getValue())
        {
            p_quitSim->setValue(0);
            p_quitSim->disable();
            switch (d_simState)
            {
            case NOT_STARTED:
                sendInfo("Simulation not yet running");
                break;

            case RUNNING:
                sendInfo("Stopping simulation as soon as possible: wait...");
                d_simState = SHUTDOWN;
                break;

            case SHUTDOWN:
                sendInfo("still waiting for sim to shut down");
                break;

            case WAIT_BOCO:
                stopSimulation();
                break;
            default:
                ;
            }
        }
    }
}
