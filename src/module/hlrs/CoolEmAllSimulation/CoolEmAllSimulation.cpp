/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>

#include "CoolEmAllSimulation.h"

#include <config/CoviseConfig.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoIntArr.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

#define RAD(x) ((x)*M_PI / 180.0)
#define GRAD(x) ((x)*180.0 / M_PI)
#define USE_PLMXML
#define MAX_CUBES 27

CoolEmAllSimulation::CoolEmAllSimulation(int argc, char *argv[])
    : coModule(argc, argv, "CoolEmAllSimulation; starts shell script to run simulation of flow and heat inside one RECS module")
{

    //   fprintf(stderr, "CoolEmAllSimulation::CoolEmAllSimulation() Init of StartFile\n");
    debbURL = addStringParam("DEBBURL", "URL To DEBB repository");
    debbURL->setValue("none");
    experimentID = addStringParam("ExperimentID", "ExperimentID");
    experimentID->setValue("none");
    trialID = addStringParam("TrialID", "TrialID");
    trialID->setValue("none");
    experimentType = addInt32Param("ExperimentType", "experimentType");
    experimentType->setValue(1);
    databasePath = addStringParam("DatabasePath", "db path");
    databasePath->setValue("none");
    toSimulate = addStringParam("ToSimulate", "Object to simulate");
    toSimulate->setValue("none");
    startTime = addStringParam("StartTime", "experiment start Time");
    startTime->setValue("none");
    endTime = addStringParam("EndTime", "experiment end Time");
    endTime->setValue("none");
}

void CoolEmAllSimulation::postInst()
{
    //plmxmlFile->show();
    //projectString->show();
}

void CoolEmAllSimulation::param(const char *portname, bool)
{
}

void CoolEmAllSimulation::quit()
{
    // :-)
}

int CoolEmAllSimulation::compute(const char *)
{

    //std::string commandline = std::string("Autorun ") + plmxmlFile->getValue() + " " + projectString->getValue();
    //system(commandline.c_str());

    return SUCCESS;
}

MODULE_MAIN(Simulation, CoolEmAllSimulation)
