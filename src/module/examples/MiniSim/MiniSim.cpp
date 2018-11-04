/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <util/coviseCompat.h>
#include <do/coDoUnstructuredGrid.h>
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
#include "MiniSim.h"

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

MiniSim::MiniSim(int argc, char *argv[])
    : coSimLib(argc, argv, argv[0], "Simulation coupling demo")
{
////////// set up default parameters

#ifdef VERBOSE

    cerr << "##############################" << endl;
    cerr << "#####   MiniSim " << endl;
    cerr << "#####   PID =  " << getpid() << endl;
    cerr << "##############################" << endl;
#endif
#ifndef YAC
    set_module_description("MiniSim Simulation");
#endif
    boolPara = addBooleanParam("pause", "Pause simulation");

    choicePara = addChoiceParam("choice", "Test Choice param");
    static const char *choices[] = { "Choice 1", " -2- ", " --- 3 --- " };
    choicePara->setValue(3, choices, 1);

    //   paraSwitch("corner","Which corner to do");
    //   paraCase("Corner 1/1/1");

    val111 = addFloatSliderParam("Value111", "Value1 at coordinate 1,1,1");
    val111->setValue(0.0, 1.0, 0.5);
    //   paraEndCase();

    //   paraCase("Corner 1/1/7");
    val117 = addFloatSliderParam("Value117", "Value1 at coordinate 1,1,7");
    val117->setValue(0.0, 1.0, 0.5);
    //   paraEndCase();

    //   paraCase("Corner 7/7/7");
    val777 = addFloatSliderParam("Value777", "Value1 at coordinate 7,7,7");
    val777->setValue(0.0, 1.0, 0.5);

    //   paraEndCase();

    //   paraEndSwitch();

    relax = addFloatParam("relax", "Relaxation factor");
    relax->setValue(0.5);

    steps = addInt32Param("steps", "Number of steps per loop");
    steps->setValue(50);

    dir = addStringParam("dir", "Directory for run");

    dir->setValue(coCoviseConfig::getEntry("value", "Module.MiniSim.StartUpScript", "~/covise/src/application/examples/MiniSim/miniSim.sh").c_str());
    // Output ports:  3 vars

    mesh = addOutputPort("mesh", "UnstructuredGrid", "Data Output");
    data = addOutputPort("data", "Float", "Data Output");

    stepNo = -1;
}

int MiniSim::compute(const char *port)
{
    (void)port;
    const char *directory;

    if (stepNo < 0)
    {
        directory = dir->getValue();
        setUserArg(0, directory);
        startSim();
        stepNo = 1;
    }

    // create dummy-mesh
    dummyMesh();

#ifndef YAC
    executeCommands();
#endif

    return SUCCESS;
}

/*************************************************************
 *************************************************************
 *************************************************************
 **                                                         **
 **           A u x i l i a r y   r o ut i n e s            **
 **                                                         **
 *************************************************************
 *************************************************************
 *************************************************************/

// create a pseudo-grid
void MiniSim::dummyMesh()
{
    coDoUnstructuredGrid *grid = new coDoUnstructuredGrid(mesh->getObjName(), 6 * 6 * 6, 6 * 6 * 6 * 8, 7 * 7 * 7, 1);

    float *x, *y, *z;
    int *el, *conn, *tl;

    grid->getAddresses(&el, &conn, &x, &y, &z);
    grid->getTypeList(&tl);

    int i, j, k;
    int connNo = 0;
    for (k = 0; k < 6; k++)
    {
        for (j = 0; j < 6; j++)
        {
            for (i = 0; i < 6; i++)
            {
                int vertNo = i + 7 * j + 49 * k;
                *conn++ = vertNo;
                *conn++ = vertNo + 1;
                *conn++ = vertNo + 8;
                *conn++ = vertNo + 7;
                *conn++ = vertNo + 49;
                *conn++ = vertNo + 50;
                *conn++ = vertNo + 57;
                *conn++ = vertNo + 56;
                *el++ = connNo;
                *tl++ = TYPE_HEXAGON;
                connNo += 8;
            }
        }
    }

    for (k = 0; k < 7; k++)
    {
        for (j = 0; j < 7; j++)
        {
            for (i = 0; i < 7; i++)
            {
                *x++ = i / 6.0f;
                *y++ = j / 6.0f;
                *z++ = k / 6.0f;
            }
        }
    }
#ifdef YAC
    // set blockno/timestep
    grid->getHdr()->setBlock(0, 1);
    grid->getHdr()->setTime(-1, 0);
    grid->getHdr()->setRealTime(1.0);
#endif
    mesh->setCurrentObject(grid);
}

int MiniSim::endIteration()
{

    return 1;
}

#ifdef YAC
void MiniSim::paramChanged(coParam *param)
{

    (void)param;
}
#endif

MODULE_MAIN(Examples, MiniSim)
