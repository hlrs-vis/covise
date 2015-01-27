/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "StarCD.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

/////////////// Create output ports /////////////////////////////////////////

void StarCD::createOutPorts()
{
    int i;
    char buf[32], buf1[32];

    // the switch contains all output possibilities
    const char *switchName[MAX_SCALARS + 19] =

        //  1        2         3    4   5   6   7   8    9    --- Choice results
        {
          "---", "Velocity", "V-Mag", "U", "V", "W", "P", "TE", "ED",
          //  10    11  12     13      14    15                 --- Choice results
          "Tvis", "T", "Dens", "LamVis", "CP", "Conductivity",
          //  16    17     18                                   --- Choice results
          "Flux", "Void", "Volume"

        };

    for (i = 0; i < MAX_SCALARS; i++) // Scalar values in loop   --- result-19 = scalar
    {
        sprintf(buf, "Scalar %d", i + 1);
        switchName[i + 18] = strcpy(new char[strlen(buf) + 1], buf);
    }

    // mesh output port
    p_mesh = addOutputPort("mesh", "UnstructuredGrid", "Simulation grid");

    // Create one switch above the output port switches to get rid of them
    paraSwitch("output", "Show output selectors");
    paraCase("Hide output selectors");

    for (i = 0; i < NUM_OUT_DATA; i++)
    {
        // create description and name
        sprintf(buf, "Species for data port %d", i);
        sprintf(buf1, "out_%d", i);

        // the choice parameter
        p_value[i] = addChoiceParam(buf1, buf);
        p_value[i]->setValue(MAX_SCALARS + 18, switchName, 0);

        // the data port
        sprintf(buf, "data_%d", i);
        p_data[i] = addOutputPort(buf,
                                  "Float|Vec3",
                                  "Data output port");
    }
    paraEndCase();
    paraEndSwitch();

    // and the port for the residuals
    p_residual = addOutputPort("residual", "StepData|Float", "Residuals");

    // and the port for the feedback
    p_feedback = addOutputPort("feedback", "Polygons", "Feedback patches");
}

/////////////// Create Region parameters ////////////////////////////////////

void StarCD::createRegionParam()
{
    int i;
    char buf[32];
    p_region = paraSwitch("region", "Select inlet region");
    for (i = 0; i < MAX_REGIONS; i++)
    {
        // create description and name
        sprintf(buf, "Region %d", i);

        // case for the Region switching
        paraCase(buf);
        // the 'usual parameters

        sprintf(buf, "local%d", i);
        p_euler[i] = addFloatVectorParam(buf, "Local Euler angles");
        p_euler[i]->setActive(0);
        p_euler[i]->setValue(0.0, 0.0, 0.0);

        sprintf(buf, "vel%d", i);
        p_v[i] = addFloatVectorParam(buf, "Velocity");
        p_v[i]->setActive(0);
        p_v[i]->setValue(0.0, 0.0, 0.0);

        sprintf(buf, "vmag%d", i);
        p_vmag[i] = addFloatSliderParam(buf, "V-Magnitude");
        p_vmag[i]->setActive(0);
        p_vmag[i]->setValue(0.0, 1.01, 0.0);

        sprintf(buf, "t__%d", i);
        p_t[i] = addFloatSliderParam(buf, "Temperature");
        p_t[i]->setValue(0.0, 40.0, 20.0);
        p_t[i]->setActive(0);

        sprintf(buf, "den%d", i);
        p_den[i] = addFloatSliderParam(buf, "Density");
        p_den[i]->setActive(0);

        sprintf(buf, "p__%d", i);
        p_p[i] = addFloatSliderParam(buf, "Pressure");
        p_p[i]->setActive(0);

        sprintf(buf, "k__%d", i);
        p_k[i] = addFloatSliderParam(buf, "k");
        p_k[i]->setActive(0);

        sprintf(buf, "eps%d", i);
        p_eps[i] = addFloatSliderParam(buf, "Epsilon");
        p_eps[i]->setActive(0);

        sprintf(buf, "tInt%d", i);
        p_tin[i] = addFloatSliderParam(buf, "Turb. Intens");
        p_tin[i]->setActive(0);

        sprintf(buf, "tLen%d", i);
        p_len[i] = addFloatSliderParam(buf, "Turb. Length");
        p_len[i]->setActive(0);

        // multiple scalars
        int j;

        //sprintf(buf,"scal%d",i);
        //p_scalSw[i] = paraSwitch(buf,"Select scalar");
        //p_scalSw[i]->setActive(0);
        for (j = 0; j < MAX_SCALARS; j++)
        {
            sprintf(buf, "Scalar %d", j + 1);
            //paraCase(buf);

            sprintf(buf, "scal%d_%d", i, j + 1);
            p_scal[i][j] = addFloatSliderParam(buf, "Scalar");
            p_scal[i][j]->setActive(0);

            //paraEndCase();
        }
        //paraEndSwitch();

        // multipla user data
        //sprintf(buf,"user%d",i);
        //p_userSw[i] = paraSwitch(buf,"Select UserData");
        //p_userSw[i]->setActive(0);
        for (j = 0; j < MAX_SCALARS; j++)
        {
            sprintf(buf, "User %d", j + 1);
            //paraCase(buf);
            sprintf(buf, "user%d_%d", i, j + 1);
            p_user[i][j] = addFloatSliderParam(buf, "User Field");
            p_user[i][j]->setActive(0);

            //paraEndCase();
        }
        //paraEndSwitch();

        /// region case ends here
        paraEndCase();
    }
    paraEndSwitch();
}

/////////////////////////////////////////////////////////////////////////////
//
//      ####    ####   #    #   ####  #####  #####   #    #   ####  #####
//     #    #  #    #  ##   #  #        #    #    #  #    #  #    #   #
//     #       #    #  # #  #   ####    #    #    #  #    #  #        #
//     #       #    #  #  # #       #   #    #####   #    #  #        #
//     #    #  #    #  #   ##  #    #   #    #   #   #    #  #    #   #
//      ####    ####   #    #   ####    #    #    #   ####    ####    #
//
/////////////////////////////////////////////////////////////////////////////

void StarCD::createParam()
{
    char buf[MAXPATHLEN];
    /// ----- create all common parameter ports

    /// --- create all output ports
    createOutPorts();

    // give the file for the grid -> we'll need some more info from file 16
    // create a valid starting directory
    p_setup = addFileBrowserParam("setup", "Setup file");

    // the user may have a StarCD config dir
    const char *starconfig = getenv("STARCONFIG");
    if (starconfig)
    {
        strcpy(buf, starconfig);
        strcat(buf, "/dummy");
    }
    else
    {
        const char *covisedir = getenv("COVISEDIR");
        if (covisedir)
        {
            strcpy(buf, covisedir);
            strcat(buf, "/data/dummy");
        }
        else
            strcpy(buf, "./dummy");
    }

    p_setup->setValue(buf, "*.starconfig");

    /// the number of steps to go
    p_steps = addInt32Param("step", "Number of steps to go");
    p_steps->setValue(1);

    /// the boolean parameter decides if the simulation should re-exec
    p_freeRun = addBooleanParam("freeRun", "Execute again automagically");
    p_freeRun->setValue(0);

    /// the boolean parameter decides if the simulation should re-exec
    p_runSim = addBooleanParam("run_Simulation", "Simulation running now");
    p_runSim->setValue(0);

    /// the boolean parameter decides if the simulation should re-exec
    p_quitSim = addBooleanParam("quit_Simulation", "shut down simulation");
    p_quitSim->setValue(0);

    /// the number processors to use
    p_numProc = addInt32Param("numProc", "Number of Processors to use");
    p_numProc->setValue(4);

    /// --- create region parameter ports
    createRegionParam();

    /// --- we don't need input here: if we get anything, we set up a new case
    p_configObj = addInputPort("configObj", "Text",
                               "Configuration lines replace starconfig file");
    p_configObj->setRequired(0);

    /// --- we don't need input here: if we get anything, add it to script params
    p_commObj = addInputPort("scriptPara", "Text",
                             "Additional parameters for startup script");
    p_commObj->setRequired(0);
}

void StarCD::postInst()
{
    int i, j;

    // always show these parameters

    // This is the choich to switch the startup
    getStartupChoice()->show();

    p_setup->show();
    p_steps->show();
    p_freeRun->show();
    p_runSim->show();
    p_quitSim->show();
    p_quitSim->disable();
    p_numProc->show();

    // we now disable all, we'll enable them later WHATEVER the map tells !!
    for (i = 0; i < MAX_REGIONS; i++)
    {
        p_euler[i]->disable();
        p_euler[i]->hide();
        p_v[i]->disable();
        p_v[i]->hide();
        p_vmag[i]->disable();
        p_vmag[i]->hide();
        p_t[i]->disable();
        p_t[i]->hide();
        p_den[i]->disable();
        p_den[i]->hide();
        p_p[i]->disable();
        p_p[i]->hide();
        p_k[i]->disable();
        p_k[i]->hide();
        p_eps[i]->disable();
        p_eps[i]->hide();
        p_tin[i]->disable();
        p_tin[i]->hide();
        p_len[i]->disable();
        p_len[i]->hide();
        for (j = 0; j < MAX_SCALARS; j++)
        {
            p_scal[i][j]->disable();
            p_scal[i][j]->hide();
        }
        for (j = 0; j < MAX_UDATA; j++)
        {
            p_user[i][j]->disable();
            p_user[i][j]->hide();
        }
    }

    // make sure we only show the top switch and nothing else
    p_region->setValue(0);
}
