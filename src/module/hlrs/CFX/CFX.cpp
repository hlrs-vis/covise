/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <config/CoviseConfig.h>
#include <math.h>
#include <util/coWristWatch.h>

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <netdb.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <pwd.h>
#endif

#ifndef YAC
#include <appl/ApplInterface.h>
#endif
#include "CFX.h"

//// Constructor : set up User Interface//

#define VERBOSE
#undef MESSAGES
coWristWatch ww_;
/*************************************************************
*************************************************************
**                                                         **
**                  K o n s t r u k t o r                  **
**                                                         **
*************************************************************
*************************************************************/

CFX::CFX(int argc, char *argv[])
    : coSimLib(argc, argv, argv[0], "Simulation coupling demo")
{
    ////////// set up default parameters

    d_distGridName = new char[1];
    d_distGridName[0] = '\0';
#ifdef VERBOSE

    cerr << "##############################" << endl;
    cerr << "#####   CFX " << endl;
    cerr << "#####   PID =  " << getpid() << endl;
    cerr << "##############################" << endl;
#endif
#ifndef YAC
    set_module_description("CFX Simulation");
#endif
#ifndef WIN32
    uid_t myuid;
    struct passwd *mypwd;
    // get user automatically
    myuid = getuid();
    while ((mypwd = getpwent()))
    {
        if (mypwd->pw_uid == myuid)
        {
            break;
        }
    }
#endif

    boolPara = addBooleanParam("pause", "Pause simulation");

#ifndef WIN32
    p_ConnectionMethod = addChoiceParam("Connection_Method", "ConnectionMethod");
    s_ConnectionMethod[0] = strdup("local");
    s_ConnectionMethod[1] = strdup("ssh");
    p_ConnectionMethod->setValue(2, s_ConnectionMethod, 0);
#endif

    p_StartupMethod = addChoiceParam("Startup_Method", "StartupMethod");
    s_StartupMethod[0] = strdup("serial");
#ifndef WIN32
    s_StartupMethod[1] = strdup("MPICH Local Parallel");
    s_StartupMethod[2] = strdup("MPICH Distributed Parallel");
    p_StartupMethod->setValue(NUM_STARTUP_METHODS, s_StartupMethod, 0);
#else
    s_StartupMethod[1] = strdup("PVM Local Parallel");
    s_StartupMethod[2] = strdup("PVM Distributed Parallel");
    s_StartupMethod[3] = strdup("MPICH2 Local Parallel for Windows");
    s_StartupMethod[4] = strdup("MPICH2 Distributed Parallel for Windows");
    p_StartupMethod->setValue(NUM_STARTUP_METHODS, s_StartupMethod, 3);
#endif

    p_MachineType = addChoiceParam("MachineType", "MachineType");
    s_MachineType[0] = strdup("radial");
    s_MachineType[1] = strdup("radial_machine");
    s_MachineType[2] = strdup("axial");
    s_MachineType[3] = strdup("axial_machine");
    s_MachineType[4] = strdup("complete_machine");
    s_MachineType[5] = strdup("rechenraum");
    s_MachineType[6] = strdup("surfacedemo");
    s_MachineType[7] = strdup("Definition_File");
    p_MachineType->setValue(8, s_MachineType, 0);

    p_inletVelMulti = addFloatParam("inletVelMulti", "Multiplier for inlet velocity");
    p_inletVelMulti->setValue(1.0f);

    p_incidenceAngular = addFloatParam("incidenceAngular", "angular of incidence");
    p_incidenceAngular->setValue(0.0f);

    p_maxIterations = addInt32Param("maxIterations", "max number of iterations");
    p_maxIterations->setValue(100);

#ifndef WIN32
    p_xterm = addBooleanParam("start_in_xterm", "Start sim in xterm?");
    p_xterm->setValue(0);
#endif

#ifndef WIN32
    p_Hostname = addStringParam("Hostname", "Hostname");
#else
    p_Hostname = addStringParam("Headnode", "Hostname of Cluster Headnode");
#endif
    p_Hostname->setValue("localhost");

    p_username = addStringParam("username", "username");
#ifndef _WIN32
    p_username->setValue(mypwd->pw_name);
#else
    p_username->setValue(getenv("USERNAME"));
#endif

#ifndef WIN32
    p_Hostlist = addStringParam("Hostlist", "Hosts for simulation");
    p_Hostlist->setValue("viscluster09*2");
#endif

    p_numProc = addInt32Param("numProc", "Number of Partitions");
    p_numProc->setValue(2);

    updateInterval = addInt32Param("updateInterval", "send simulation results every nth iteration");
    updateInterval->setValue(10);

    p_revolutions = addInt32Param("revolutions", "Number of Revolutions");
    p_revolutions->setValue(170);

    p_deffile = addStringParam("deffile", "def File to solve");
    p_deffile->setValue("0");

    startScript = addStringParam("start_script", "Path and name of start script on simulation host");
    startScript->setValue(coCoviseConfig::getEntry("value", "Module.CFX.StartUpScript", "~/covise/src/application/examples/CFX/CFX.sh").c_str());

    // Input ports : yet only parallel ones allowed
    p_grid = addInputPort("grid", "UnstructuredGrid", "Distributed Grid");
    p_boco = addInputPort("boco", "USR_FenflossBoco", "Boundary Conditions");

    // Output ports
    p_gridout = addOutputPort("gridout", "UnstructuredGrid", "the computational mesh");
    TEMP_FL1_1 = addOutputPort("TEMP_FL1_1", "Float", "TEMP Data Output Zone 1");
    PRES_1 = addOutputPort("PRES_1", "Float", "PRES Data Output Zone 1");
    VEL_FL1_1 = addOutputPort("VEL_FL1_1", "Vec3", "VEL Data Output Zone 1");

    TEMP_FL1_2 = addOutputPort("TEMP_FL1_2", "Float", "TEMP Data Output Zone 2");
    PRES_2 = addOutputPort("PRES_2", "Float", "PRES Data Output Zone 2");
    VEL_FL1_2 = addOutputPort("VEL_FL1_2", "Vec3", "VEL Data Output Zone 2");

    TEMP_FL1_3 = addOutputPort("TEMP_FL1_3", "Float", "TEMP Data Output Zone 3");
    PRES_3 = addOutputPort("PRES_3", "Float", "PRES Data Output Zone 3");
    VEL_FL1_3 = addOutputPort("VEL_FL1_3", "Vec3", "VEL Data Output Zone 3");

    TEMP_FL1_4 = addOutputPort("TEMP_FL1_4", "Float", "TEMP Data Output Zone 4");
    PRES_4 = addOutputPort("PRES_4", "Float", "PRES Data Output Zone 4");
    VEL_FL1_4 = addOutputPort("VEL_FL1_4", "Vec3", "VEL Data Output Zone 4");

    stepNo = -1;
}

int CFX::compute(const char *port)
{
    ww_.reset();
    (void)port;
    const coDoUnstructuredGrid *grid;
    const coDoSet *boco;

    //werden hier deklariert, da sie auch in StepNo==1 verwendet werden
    int *elem, *elem_machine;
    int *diricletIndex, *diricletIndex_machine;
    int *balance, *balance_machine;
    int *conn, *conn_machine;
    int *elem_out, *conn_out, *outTypeList, *inTypeList;
    int *wall, *wall_machine;

    float *x, *y, *z;
    float *x_out, *y_out, *z_out;
    float *x_machine, *y_machine, *z_machine;
    float *diricletVal, *diricletVal_machine;

    int numCoord, numCoord_machine;
    int numConn, numConn_machine;
    int numElem, numElem_machine;
    int numBalance, numBalance_machine;
    int numDiriclet, numDiriclet_machine;
    int numWall, numWall_machine;
    int colDiriclet, colWall, colBalance;
    int colDiricletVals = 5; // u,v,w,k,e
    int numberinletnodes, numberinletnodes_machine;

    int mtype;
    int numObj;
    int i, j;
    int numblades;
    float alpha, k;
    double PI = 3.14159265;
    const char *numberofblades;
    const coDistributedObject *const *setObj;

    grid = dynamic_cast<const coDoUnstructuredGrid *>(p_grid->getCurrentObject());
    const char *gridName = grid->getName();
    if (strcmp(gridName, d_distGridName) != 0)
    {

        // sim currently running
        if (stepNo >= 0)
        {

#ifndef WIN32
            //system("killall -KILL p_flow_4.8.2");
            sleep(5);
#endif
            resetSimLib();
            stepNo = -1;
        }
        delete[] d_distGridName;
        d_distGridName = strcpy(new char[strlen(gridName) + 1], gridName);
    }

    if (stepNo < 0)
    {
        ////////////  STARTING SIMULATION ////////////////
        if (PrepareSimStart() == -1)
            return STOP_PIPELINE;

        startSim();

        if (p_MachineType->getValue() == 0 || p_MachineType->getValue() == 2 || p_MachineType->getValue() == 5 || p_MachineType->getValue() == 6)
        /////// MachineType is radial or axial or rechenraum or surfacedemo
        {

            p_boco->setRequired(1);
            p_grid->setRequired(1);

            fprintf(stderr, "\nInputport wird ausgelesen\n");

            boco = dynamic_cast<const coDoSet *>(p_boco->getCurrentObject());

            if (!grid)
            {
                sendWarning("Grid for port %s not correctly received",
                            p_grid->getName());
                fprintf(stderr, "Set of old grid...\n");
            }

            if (!boco)
            {
                sendWarning("Boco for port %s not correctly received", p_boco->getName());
                return FAIL;
            }

            numblades = 1;

            fprintf(stderr, "Gitter wird eingelesen\n");
            grid->getAddresses(&elem, &conn, &x, &y, &z);
            grid->getGridSize(&numElem, &numConn, &numCoord);

            fprintf(stderr, "Gitter wird gesendet\n");
            mtype = p_MachineType->getValue();
            sendData(&mtype, sizeof(int));
            sendData(&numblades, sizeof(int));
            sendData(&numElem, sizeof(int));
            sendData(&numConn, sizeof(int));
            sendData(&numCoord, sizeof(int));

            sendData(elem, numElem * sizeof(int));
            sendData(conn, numConn * sizeof(int));

            sendData(x, numCoord * sizeof(float));
            sendData(y, numCoord * sizeof(float));
            sendData(z, numCoord * sizeof(float));

            setObj = boco->getAllElements(&numObj);

            ///////// Diriclet Nodes
            const coDoIntArr *diricletIndexObj = dynamic_cast<const coDoIntArr *>(setObj[3]);
            if (!diricletIndexObj)
            {
                sendWarning("illegal part type (3) in boco");
                return FAIL;
            }
            else
            {
                colDiriclet = diricletIndexObj->getDimension(0); //2
                numDiriclet = diricletIndexObj->getDimension(1);
                diricletIndex = diricletIndexObj->getAddress();
            }
            sendData(&colDiriclet, sizeof(int));
            sendData(&numDiriclet, sizeof(int));
            sendData(&colDiricletVals, sizeof(int));
            sendData(diricletIndex, (colDiriclet * numDiriclet) * sizeof(int));

            //	0,10,20,30,40....32 inletnodes
            numberinletnodes = (numDiriclet * colDiriclet) / (2 * colDiricletVals);
            int *inletnodes = (int *)malloc(numberinletnodes * sizeof(int));

            if (numberinletnodes == 0)
            {
                fprintf(stderr, "missing inlet boundary condition! Stopping. Please generate inlet boco.\n");
                sendError("missing inlet boundary condition! Stopping. Please generate inlet boco.");
                return STOP_PIPELINE;
            }

            for (int i = 0; i < numberinletnodes; i++)
            {
                inletnodes[i] = diricletIndex[10 * i];
            }

            ///////// Diriclet values
            const coDoFloat *diricletValObj = dynamic_cast<const coDoFloat *>(setObj[4]);
            float *diricletValnonshared = (float *)malloc((numDiriclet) * sizeof(float));

            if (!diricletValObj)
            {
                sendWarning("illegal part type (4) in boco");
                return FAIL;
            }
            else
            {
                float u, v, r, xr0, yr0, xu0, yu0, vu, vr, omega;
                int j, nodenr;
                diricletVal = diricletValObj->getAddress();

                omega = float(M_PI * p_revolutions->getValue() / 30.0f);

                for (j = 0; j < numberinletnodes; j++)
                {
                    if (mtype == 2)
                    {
                        if (diricletVal[j * 5 + 2] > 0) // nur bei axial so! bei radial checken wir vr
                        {
                            sendError("inlet velocity points in wrong direction. Stopping");
                            return STOP_PIPELINE;
                        }
                    }

                    u = diricletVal[j * 5 + 0]; // vx
                    v = diricletVal[j * 5 + 1]; // vy

                    nodenr = inletnodes[j] - 1;

                    r = sqrt((x[nodenr] * x[nodenr]) + (y[nodenr] * y[nodenr]));

                    xr0 = x[nodenr] / r;
                    yr0 = y[nodenr] / r;

                    xu0 = -yr0; //axial != radial
                    yu0 = xr0;

                    //lu = (xu0*u+yu0*v)/(xu0*xu0+yu0*yu0);
                    //lr = (xr0*u+yr0*v)/(xr0*xr0+yr0*yr0);
                    //vu = lu ;
                    //vr = lr ;

                    vr = u * xr0 + v * yr0;
                    vu = u * xu0 + v * yu0;

                    //diricletValnonshared[j*5+0] = u+omega*y[nodenr];
                    //diricletValnonshared[j*5+1] = v-omega*x[nodenr];

                    // 	       diricletValnonshared[j*5+0] = -x[nodenr]*xr0 -y[nodenr]*yr0;
                    // 	       diricletValnonshared[j*5+1] = -x[nodenr]*xu0 -y[nodenr]*yu0 -r;

                    diricletValnonshared[j * 5 + 0] = vr;
                    diricletValnonshared[j * 5 + 1] = vu - omega * r;

                    diricletValnonshared[j * 5 + 2] = diricletVal[j * 5 + 2];
                    diricletValnonshared[j * 5 + 3] = diricletVal[j * 5 + 3];
                    diricletValnonshared[j * 5 + 4] = diricletVal[j * 5 + 4];
                }
            }

            sendData(diricletValnonshared, (numDiriclet) * sizeof(float));

            ///////// Wall indices
            const coDoIntArr *wallObj = dynamic_cast<const coDoIntArr *>(setObj[5]);
            if (!wallObj)
            {
                sendWarning("illegal part type (5) in boco");
                return FAIL;
            }
            else
            {
                colWall = wallObj->getDimension(0); //spaltenanzahl  (7)
                numWall = wallObj->getDimension(1); //anzahl elemente (515)
                wall = wallObj->getAddress(); //intarray mit 3615
            }

            sendData(&colWall, sizeof(int));
            sendData(&numWall, sizeof(int));
            sendData(wall, (colWall * numWall) * sizeof(int));

            ///////// balance indices
            const coDoIntArr *balanceObj = dynamic_cast<const coDoIntArr *>(setObj[6]);
            if (!balanceObj)
            {
                sendWarning("illegal part type (6) in boco");
                return FAIL;
            }
            else
            {
                coDoIntArr *balanceObj = (coDoIntArr *)setObj[6];
                colBalance = balanceObj->getDimension(0);
                numBalance = balanceObj->getDimension(1);
                balance = balanceObj->getAddress();
            }
            sendData(&colBalance, sizeof(int));
            sendData(&numBalance, sizeof(int));
            sendData(balance, (colBalance * numBalance) * sizeof(int));

            // hier an der Stelle ist pre fertig
            // dann geht, weil Pre beendet wird automatisch der Socket zu
            // den muessen wir hier wieder aufmachen
            free(diricletValnonshared);
            free(inletnodes);

            int result = reAccept();
            printf("acceptServer %d\n", result);

            stepNo = 1;

        } // machinetype is radial or axial

        if (p_MachineType->getValue() == 1) //////// MachineType is radial_machine
        {

            fprintf(stderr, "\n!!!!!!! MachineType is radial_machine !!!!!!!\n");
            fprintf(stderr, "\nInputport wird eingelesen\n");

            grid = dynamic_cast<const coDoUnstructuredGrid *>(p_grid->getCurrentObject());
            boco = dynamic_cast<const coDoSet *>(p_boco->getCurrentObject());

            if (!grid)
            {
                sendWarning("Grid for port %s not correctly received",
                            p_grid->getName());
                fprintf(stderr, "Set of old grid...\n");
            }

            if (!boco)
            {
                sendWarning("Boco for port %s not correctly received", p_boco->getName());
                return FAIL;
            }

            fprintf(stderr, "Gitter wird eingelesen\n");
            grid->getAddresses(&elem, &conn, &x, &y, &z);
            grid->getGridSize(&numElem, &numConn, &numCoord);
            numberofblades = grid->getAttribute("NUMBER_OF_BLADES");
            if (numberofblades == NULL)
            {
                sendError("missing attribute NUMBER_OF_BLADES!");
                return STOP_PIPELINE;
            }
            numblades = atoi(numberofblades);
            fprintf(stderr, "Number of blades: %d\n", numblades);
            alpha = 360.0f / (float)numblades;
            fprintf(stderr, "alpha: %f\n", alpha);

            //	Gitter wird repliziert !

            grid->getTypeList(&inTypeList);

            coDoUnstructuredGrid *outgrid = new coDoUnstructuredGrid(p_gridout->getObjName(), numElem * numblades, numConn * numblades, numCoord * numblades, 1);
            outgrid->getAddresses(&elem_machine, &conn_machine, &x_machine, &y_machine, &z_machine);
            outgrid->getTypeList(&outTypeList);

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numElem; j++)
                {
                    outTypeList[j + numElem * i] = inTypeList[j];
                }
            }

            k = 0;
            for (i = 0; i < numblades; i++)
            {
                //          fprintf (stderr,"Drehwinkel: %f\n",k);
                for (j = 0; j < numCoord; j++)
                {
                    x_machine[numCoord * i + j] = float(x[j] * cos(k * PI / 180) - y[j] * sin(k * PI / 180));
                    y_machine[numCoord * i + j] = float(x[j] * sin(k * PI / 180) + y[j] * cos(k * PI / 180));
                    z_machine[numCoord * i + j] = z[j];
                }
                k = k + alpha;
            }

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numElem; j++)
                {
                    elem_machine[j + numElem * i] = elem[j] + numConn * i;
                }
            }

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numConn; j++)
                {
                    conn_machine[j + numConn * i] = conn[j] + numCoord * i;
                }
            }

            fprintf(stderr, "Gitter dem Outputport zuweisen.\n");
            p_gridout->setCurrentObject(outgrid);

            numElem_machine = numElem * numblades;
            numConn_machine = numConn * numblades;
            numCoord_machine = numCoord * numblades;

            mtype = p_MachineType->getValue();
            sendData(&mtype, sizeof(int));
            sendData(&numblades, sizeof(int));
            sendData(&numElem_machine, sizeof(int));
            sendData(&numConn_machine, sizeof(int));
            sendData(&numCoord_machine, sizeof(int));

            sendData(elem_machine, numElem_machine * sizeof(int));
            sendData(conn_machine, numConn_machine * sizeof(int));

            sendData(x_machine, numCoord_machine * sizeof(float));
            sendData(y_machine, numCoord_machine * sizeof(float));
            sendData(z_machine, numCoord_machine * sizeof(float));

            setObj = boco->getAllElements(&numObj);

            ///////// Diriclet Nodes
            const coDoIntArr *diricletIndexObj = dynamic_cast<const coDoIntArr *>(setObj[3]);
            if (!diricletIndexObj)
            {
                sendWarning("illegal part type (3) in boco");
                return FAIL;
            }
            else
            {
                colDiriclet = diricletIndexObj->getDimension(0); //2
                numDiriclet = diricletIndexObj->getDimension(1);
                diricletIndex = diricletIndexObj->getAddress();
            }
            numDiriclet_machine = numDiriclet * numblades;
            diricletIndex_machine = (int *)malloc(colDiriclet * numDiriclet_machine * sizeof(int));

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numDiriclet; j++) // 180
                {
                    diricletIndex_machine[2 * j + 0 + 2 * numDiriclet * i] = diricletIndex[2 * j + 0] + numCoord * i;
                    diricletIndex_machine[2 * j + 1 + 2 * numDiriclet * i] = diricletIndex[2 * j + 1];
                }
            }

            numberinletnodes = (numDiriclet * colDiriclet) / (2 * colDiricletVals); //Kanal
            numberinletnodes_machine = numberinletnodes * numblades;
            int *inletnodes = (int *)malloc(numberinletnodes_machine * sizeof(int));
            fprintf(stderr, "Covise: numberinletnodes_machine: %d\n", numberinletnodes_machine);
            for (i = 0; i < numberinletnodes_machine; i++)
            {
                inletnodes[i] = diricletIndex_machine[10 * i];
            }

            sendData(&colDiriclet, sizeof(int));
            sendData(&numDiriclet_machine, sizeof(int));
            sendData(&colDiricletVals, sizeof(int));
            sendData(diricletIndex_machine, (colDiriclet * numDiriclet_machine) * sizeof(int));

            ///////// Diriclet values
            const coDoFloat *diricletValObj = dynamic_cast<const coDoFloat *>(setObj[4]);
            if (!diricletValObj)
            {
                sendWarning("illegal part type (4) in boco");
                return FAIL;
            }
            else
            {
                diricletVal = diricletValObj->getAddress();
            }
            diricletVal_machine = (float *)malloc(numDiriclet_machine * sizeof(float));

            k = 0;
            float u, v, r, xr0, yr0, xu0, yu0, vu, vr, omega;
            int j, nodenr;
            omega = float(M_PI * p_revolutions->getValue() / 30.0f);

            for (i = 0; i < numblades; i++)
            {
                // 	     fprintf (stderr,"Drehwinkel: %f\n",k);
                for (j = 0; j < numberinletnodes; j++) //numberinletnodes des Kanals!!!!!!!
                {
                    diricletVal_machine[numDiriclet * i + j * 5 + 0] = float(diricletVal[j * 5 + 0] * cos(k * PI / 180) - diricletVal[j * 5 + 1] * sin(k * PI / 180));
                    diricletVal_machine[numDiriclet * i + j * 5 + 1] = float(diricletVal[j * 5 + 0] * sin(k * PI / 180) + diricletVal[j * 5 + 1] * cos(k * PI / 180));
                    diricletVal_machine[numDiriclet * i + j * 5 + 2] = diricletVal[j * 5 + 2];
                    diricletVal_machine[numDiriclet * i + j * 5 + 3] = diricletVal[j * 5 + 3];
                    diricletVal_machine[numDiriclet * i + j * 5 + 4] = diricletVal[j * 5 + 4];

                    nodenr = inletnodes[numberinletnodes * i + j] - 1;

                    u = diricletVal_machine[numDiriclet * i + j * 5 + 0]; // vx
                    v = diricletVal_machine[numDiriclet * i + j * 5 + 1]; // vy

                    r = sqrt((x_machine[nodenr] * x_machine[nodenr]) + (y_machine[nodenr] * y_machine[nodenr]));
                    xr0 = x_machine[nodenr] / r;
                    yr0 = y_machine[nodenr] / r;

                    xu0 = -yr0;
                    yu0 = xr0;

                    vr = u * xr0 + v * yr0;
                    vu = u * xu0 + v * yu0;

                    diricletVal_machine[numDiriclet * i + j * 5 + 0] = vr;
                    diricletVal_machine[numDiriclet * i + j * 5 + 1] = vu - omega * r;

                    // 	      diricletVal_machine[numDiriclet*i+j*5+0] = u;
                    // 	      diricletVal_machine[numDiriclet*i+j*5+1] = v;
                }
                k = k + alpha;
            }

            sendData(diricletVal_machine, (numDiriclet_machine) * sizeof(float));

            ///////// Wall indices
            const coDoIntArr *wallObj = dynamic_cast<const coDoIntArr *>(setObj[5]);
            if (!wallObj)
            {
                sendWarning("illegal part type (5) in boco");
                return FAIL;
            }
            else
            {
                colWall = wallObj->getDimension(0); //spaltenanzahl  (7)
                numWall = wallObj->getDimension(1); //anzahl elemente (515)
                wall = wallObj->getAddress(); //intarray mit 3615
            }
            wall_machine = (int *)malloc(colWall * numWall * numblades * sizeof(int));

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numWall; j++)
                {
                    wall_machine[7 * j + 0 + i * numWall * colWall] = wall[7 * j + 0] + numCoord * i;
                    wall_machine[7 * j + 1 + i * numWall * colWall] = wall[7 * j + 1] + numCoord * i;
                    wall_machine[7 * j + 2 + i * numWall * colWall] = wall[7 * j + 2] + numCoord * i;
                    wall_machine[7 * j + 3 + i * numWall * colWall] = wall[7 * j + 3] + numCoord * i;
                    wall_machine[7 * j + 4 + i * numWall * colWall] = wall[7 * j + 4] + numElem * i;
                    wall_machine[7 * j + 5 + i * numWall * colWall] = wall[7 * j + 5];
                    wall_machine[7 * j + 6 + i * numWall * colWall] = wall[7 * j + 6];
                }
            }

            numWall_machine = numWall * numblades;
            sendData(&colWall, sizeof(int));
            sendData(&numWall_machine, sizeof(int));
            sendData(wall_machine, (colWall * numWall_machine) * sizeof(int));

            ///////// balance indices
            const coDoIntArr *balanceObj = dynamic_cast<const coDoIntArr *>(setObj[6]);
            if (!balanceObj)
            {
                sendWarning("illegal part type (6) in boco");
                return FAIL;
            }
            else
            {
                coDoIntArr *balanceObj = (coDoIntArr *)setObj[6];
                colBalance = balanceObj->getDimension(0);
                numBalance = balanceObj->getDimension(1);
                balance = balanceObj->getAddress();
            }

            balance_machine = (int *)malloc(colBalance * numBalance * numblades * sizeof(int));

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numBalance; j++)
                {
                    balance_machine[7 * j + 0 + i * numBalance * colBalance] = balance[7 * j + 0] + numCoord * i;
                    balance_machine[7 * j + 1 + i * numBalance * colBalance] = balance[7 * j + 1] + numCoord * i;
                    balance_machine[7 * j + 2 + i * numBalance * colBalance] = balance[7 * j + 2] + numCoord * i;
                    balance_machine[7 * j + 3 + i * numBalance * colBalance] = balance[7 * j + 3] + numCoord * i;
                    balance_machine[7 * j + 4 + i * numBalance * colBalance] = balance[7 * j + 4] + numElem * i;
                    balance_machine[7 * j + 5 + i * numBalance * colBalance] = balance[7 * j + 5];
                    balance_machine[7 * j + 6 + i * numBalance * colBalance] = balance[7 * j + 6];
                }
            }

            numBalance_machine = numBalance * numblades;
            sendData(&colBalance, sizeof(int));
            sendData(&numBalance_machine, sizeof(int));
            sendData(balance_machine, (colBalance * numBalance_machine) * sizeof(int));

            // hier an der Stelle ist pre fertig
            // dann geht, weil Pre beendet wird automatisch der Socket zu
            // den muessen wir hier wieder aufmachen
            free(balance_machine);
            free(wall_machine);
            free(diricletVal_machine);
            free(diricletIndex_machine);
            free(inletnodes);

            int result = reAccept();
            printf("acceptServer %d\n", result);

            stepNo = 1;

        } // Machinetype radial_machine

        if ((p_MachineType->getValue() == 3) || (p_MachineType->getValue() == 4)) //////// MachineType is axial_machine or complete machine
        {

            p_boco->setRequired(1);
            p_grid->setRequired(1);

            fprintf(stderr, "\n!!!!!!! MachineType is axial_machine !!!!!!!\n");
            fprintf(stderr, "\nMachineType: %d\n", p_MachineType->getValue());

            grid = dynamic_cast<const coDoUnstructuredGrid *>(p_grid->getCurrentObject());
            boco = dynamic_cast<const coDoSet *>(p_boco->getCurrentObject());

            if (!grid)
            {
                sendWarning("Grid for port %s not correctly received",
                            p_grid->getName());
                fprintf(stderr, "Set of old grid...\n");
            }

            if (!boco)
            {
                sendWarning("Boco for port %s not correctly received", p_boco->getName());
                return FAIL;
            }

            fprintf(stderr, "Gitter wird eingelesen\n");
            grid->getAddresses(&elem, &conn, &x, &y, &z);
            grid->getGridSize(&numElem, &numConn, &numCoord);
            numberofblades = grid->getAttribute("NUMBER_OF_BLADES");
            if (numberofblades == NULL)
            {
                sendError("missing attribute NUMBER_OF_BLADES!");
                return STOP_PIPELINE;
            }
            numblades = atoi(numberofblades);

            //	numblades = 4;
            fprintf(stderr, "Number of blades: %d\n", numblades);
            alpha = 360.0f / (float)numblades;
            fprintf(stderr, "alpha: %f\n", alpha);

            //	Gitter wird repliziert !

            grid->getTypeList(&inTypeList);

            coDoUnstructuredGrid *outgrid = new coDoUnstructuredGrid(p_gridout->getObjName(), numElem * numblades, numConn * numblades, numCoord * numblades, 1);
            outgrid->getAddresses(&elem_machine, &conn_machine, &x_machine, &y_machine, &z_machine);
            outgrid->getTypeList(&outTypeList);

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numElem; j++)
                {
                    outTypeList[j + numElem * i] = inTypeList[j];
                }
            }

            k = 0;
            for (i = 0; i < numblades; i++)
            {
                //          fprintf (stderr,"Drehwinkel: %f\n",k);
                for (j = 0; j < numCoord; j++)
                {
                    x_machine[numCoord * i + j] = float(x[j] * cos(k * PI / 180) - y[j] * sin(k * PI / 180));
                    y_machine[numCoord * i + j] = float(x[j] * sin(k * PI / 180) + y[j] * cos(k * PI / 180));
                    z_machine[numCoord * i + j] = z[j];
                }
                k = k + alpha;
            }

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numElem; j++)
                {
                    elem_machine[j + numElem * i] = elem[j] + numConn * i;
                }
            }

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numConn; j++)
                {
                    conn_machine[j + numConn * i] = conn[j] + numCoord * i;
                }
            }

            p_gridout->setCurrentObject(outgrid);

            numElem_machine = numElem * numblades;
            numConn_machine = numConn * numblades;
            numCoord_machine = numCoord * numblades;

            fprintf(stderr, "Gitter wird gesendet\n");
            mtype = p_MachineType->getValue();
            sendData(&mtype, sizeof(int));
            sendData(&numblades, sizeof(int));
            sendData(&numElem_machine, sizeof(int));
            sendData(&numConn_machine, sizeof(int));
            sendData(&numCoord_machine, sizeof(int));

            sendData(elem_machine, numElem_machine * sizeof(int));
            sendData(conn_machine, numConn_machine * sizeof(int));
            sendData(x_machine, numCoord_machine * sizeof(float));
            sendData(y_machine, numCoord_machine * sizeof(float));
            sendData(z_machine, numCoord_machine * sizeof(float));

            setObj = boco->getAllElements(&numObj);

            ///////// Diriclet Nodes
            const coDoIntArr *diricletIndexObj = dynamic_cast<const coDoIntArr *>(setObj[3]);
            if (!diricletIndexObj)
            {
                sendWarning("illegal part type (3) in boco");
                return FAIL;
            }
            else
            {
                colDiriclet = diricletIndexObj->getDimension(0); //2
                numDiriclet = diricletIndexObj->getDimension(1);
                diricletIndex = diricletIndexObj->getAddress();
            }
            numDiriclet_machine = numDiriclet * numblades;
            diricletIndex_machine = (int *)malloc(colDiriclet * numDiriclet_machine * sizeof(int));

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numDiriclet; j++) // 180
                {
                    diricletIndex_machine[2 * j + 0 + 2 * numDiriclet * i] = diricletIndex[2 * j + 0] + numCoord * i;
                    diricletIndex_machine[2 * j + 1 + 2 * numDiriclet * i] = diricletIndex[2 * j + 1];
                }
            }

            numberinletnodes = (numDiriclet * colDiriclet) / (2 * colDiricletVals); //Kanal
            numberinletnodes_machine = numberinletnodes * numblades;
            int *inletnodes = (int *)malloc(numberinletnodes_machine * sizeof(int));
            for (i = 0; i < numberinletnodes_machine; i++)
            {
                inletnodes[i] = diricletIndex_machine[10 * i];
            }

            fprintf(stderr, "Covise: numberinletnodes_machine: %d\n", numberinletnodes_machine);

            sendData(&colDiriclet, sizeof(int));
            sendData(&numDiriclet_machine, sizeof(int));
            sendData(&colDiricletVals, sizeof(int));
            sendData(diricletIndex_machine, (colDiriclet * numDiriclet_machine) * sizeof(int));

            ///////// Diriclet values
            const coDoFloat *diricletValObj = dynamic_cast<const coDoFloat *>(setObj[4]);
            if (!diricletValObj)
            {
                sendWarning("illegal part type (4) in boco");
                return FAIL;
            }
            else
            {
                diricletVal = diricletValObj->getAddress();
            }
            diricletVal_machine = (float *)malloc(numDiriclet_machine * sizeof(float));

            k = 0;
            float u, v, r, xr0, yr0, xu0, yu0, vu, vr, omega;
            int j, nodenr;
            omega = float(M_PI * p_revolutions->getValue() / 30.0f);

            for (i = 0; i < numblades; i++)
            {

                // 	 fprintf (stderr,"Drehwinkel: %f\n",k);
                for (j = 0; j < numberinletnodes; j++) //numberinletnodes des Kanals!!!!!!!
                {

                    diricletVal_machine[numDiriclet * i + j * 5 + 0] = float(diricletVal[j * 5 + 0] * cos(k * PI / 180) - diricletVal[j * 5 + 1] * sin(k * PI / 180));
                    diricletVal_machine[numDiriclet * i + j * 5 + 1] = float(diricletVal[j * 5 + 0] * sin(k * PI / 180) + diricletVal[j * 5 + 1] * cos(k * PI / 180));
                    diricletVal_machine[numDiriclet * i + j * 5 + 2] = diricletVal[j * 5 + 2];
                    diricletVal_machine[numDiriclet * i + j * 5 + 3] = diricletVal[j * 5 + 3];
                    diricletVal_machine[numDiriclet * i + j * 5 + 4] = diricletVal[j * 5 + 4];

                    if (mtype == 2)
                    {
                        if (diricletVal[j * 5 + 2] > 0)
                        {
                            sendError("inlet velocity points in wrong direction. Stopping");
                            return STOP_PIPELINE;
                        }
                    }

                    u = diricletVal_machine[numDiriclet * i + j * 5 + 0]; // vx
                    v = diricletVal_machine[numDiriclet * i + j * 5 + 1]; // vy

                    nodenr = inletnodes[numberinletnodes * i + j] - 1;

                    //  	     fprintf (stderr,"numberinletnodes: %d\n",numberinletnodes);

                    r = sqrt((x_machine[nodenr] * x_machine[nodenr]) + (y_machine[nodenr] * y_machine[nodenr]));
                    xr0 = x_machine[nodenr] / r;
                    yr0 = y_machine[nodenr] / r;

                    xu0 = -yr0;
                    yu0 = xr0;

                    vr = u * xr0 + v * yr0;
                    vu = u * xu0 + v * yu0;

                    diricletVal_machine[numDiriclet * i + j * 5 + 0] = vr;
                    diricletVal_machine[numDiriclet * i + j * 5 + 1] = vu - omega * r;
                }
                k = k + alpha;
            }

            sendData(diricletVal_machine, (numDiriclet_machine) * sizeof(float));

            ///////// Wall indices
            const coDoIntArr *wallObj = dynamic_cast<const coDoIntArr *>(setObj[5]);
            if (!wallObj)
            {
                sendWarning("illegal part type (5) in boco");
                return FAIL;
            }
            else
            {
                colWall = wallObj->getDimension(0); //spaltenanzahl  (7)
                numWall = wallObj->getDimension(1); //anzahl elemente (515)
                wall = wallObj->getAddress(); //intarray mit 3615
            }
            wall_machine = (int *)malloc(colWall * numWall * numblades * sizeof(int));

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numWall; j++)
                {
                    wall_machine[7 * j + 0 + i * numWall * colWall] = wall[7 * j + 0] + numCoord * i;
                    wall_machine[7 * j + 1 + i * numWall * colWall] = wall[7 * j + 1] + numCoord * i;
                    wall_machine[7 * j + 2 + i * numWall * colWall] = wall[7 * j + 2] + numCoord * i;
                    wall_machine[7 * j + 3 + i * numWall * colWall] = wall[7 * j + 3] + numCoord * i;
                    wall_machine[7 * j + 4 + i * numWall * colWall] = wall[7 * j + 4] + numElem * i;
                    wall_machine[7 * j + 5 + i * numWall * colWall] = wall[7 * j + 5];
                    wall_machine[7 * j + 6 + i * numWall * colWall] = wall[7 * j + 6];
                }
            }

            numWall_machine = numWall * numblades;
            sendData(&colWall, sizeof(int));
            sendData(&numWall_machine, sizeof(int));
            sendData(wall_machine, (colWall * numWall_machine) * sizeof(int));

            ///////// balance indices
            const coDoIntArr *balanceObj = dynamic_cast<const coDoIntArr *>(setObj[6]);
            if (!balanceObj)
            {
                sendWarning("illegal part type (6) in boco");
                return FAIL;
            }
            else
            {
                coDoIntArr *balanceObj = (coDoIntArr *)setObj[6];
                colBalance = balanceObj->getDimension(0);
                numBalance = balanceObj->getDimension(1);
                balance = balanceObj->getAddress();
            }

            balance_machine = (int *)malloc(colBalance * numBalance * numblades * sizeof(int));

            for (i = 0; i < numblades; i++)
            {
                for (j = 0; j < numBalance; j++)
                {
                    balance_machine[7 * j + 0 + i * numBalance * colBalance] = balance[7 * j + 0] + numCoord * i;
                    balance_machine[7 * j + 1 + i * numBalance * colBalance] = balance[7 * j + 1] + numCoord * i;
                    balance_machine[7 * j + 2 + i * numBalance * colBalance] = balance[7 * j + 2] + numCoord * i;
                    balance_machine[7 * j + 3 + i * numBalance * colBalance] = balance[7 * j + 3] + numCoord * i;
                    balance_machine[7 * j + 4 + i * numBalance * colBalance] = balance[7 * j + 4] + numElem * i;
                    balance_machine[7 * j + 5 + i * numBalance * colBalance] = balance[7 * j + 5];
                    balance_machine[7 * j + 6 + i * numBalance * colBalance] = balance[7 * j + 6];
                }
            }

            numBalance_machine = numBalance * numblades;
            sendData(&colBalance, sizeof(int));
            sendData(&numBalance_machine, sizeof(int));
            sendData(balance_machine, (colBalance * numBalance_machine) * sizeof(int));

            free(balance_machine);
            free(wall_machine);
            free(diricletVal_machine);
            free(diricletIndex_machine);

            int result = reAccept();
            printf("acceptServer %d\n", result);
            stepNo = 1;

        } // if machineType is axial_machine

        if ((p_MachineType->getValue() == 7)) // Definition File
        {
            fprintf(stderr, "\n!!!!!!! MachineType is Definition File !!!!!!!\n");
            fprintf(stderr, "\nMachineType: %d\n", p_MachineType->getValue());

            //sendBS_Data(&numberinletnodes,sizeof(int));
            //int result = reAccept();
            stepNo = 1;
        }

    } // if stepno < 0

    if (stepNo == 1) /////////////////////Grid wird an den Outputport weitergegeben
    {

        if (p_MachineType->getValue() != 7)
        {
            fprintf(stderr, "\nInputport wird ausgelesen\n");

            grid = dynamic_cast<const coDoUnstructuredGrid *>(p_grid->getCurrentObject());
            boco = dynamic_cast<const coDoSet *>(p_boco->getCurrentObject());

            if (!grid)
            {
                sendWarning("Grid for port %s not correctly received",
                            p_grid->getName());
                fprintf(stderr, "Set of old grid...\n");
            }

            if (!boco)
            {
                sendWarning("Boco for port %s not correctly received", p_boco->getName());
                return FAIL;
            }

            if (p_MachineType->getValue() == 0 || p_MachineType->getValue() == 2 || p_MachineType->getValue() == 5 || p_MachineType->getValue() == 6) // MachineType is radial or axial or rechenraum
            {
                if (!((p_MachineType->getValue() == 5) || (p_MachineType->getValue() == 6)))
                {
                    numblades = 1;
                }
                else
                {
                    numblades = 0;
                }
                fprintf(stderr, "Gitter wird eingelesen\n");
                grid->getAddresses(&elem, &conn, &x, &y, &z);
                grid->getGridSize(&numElem, &numConn, &numCoord);
                grid->getTypeList(&inTypeList);
                coDoUnstructuredGrid *outgrid = new coDoUnstructuredGrid(p_gridout->getObjName(), numElem, numConn, numCoord, 1);
                outgrid->getAddresses(&elem_out, &conn_out, &x_out, &y_out, &z_out);
                outgrid->getTypeList(&outTypeList);
                for (j = 0; j < numElem; j++)
                {
                    outTypeList[j] = inTypeList[j];
                }
                for (j = 0; j < numCoord; j++)
                {
                    x_out[j] = x[j];
                    y_out[j] = y[j];
                    z_out[j] = z[j];
                }
                for (j = 0; j < numElem; j++)
                {
                    elem_out[j] = elem[j];
                }
                for (j = 0; j < numConn; j++)
                {
                    conn_out[j] = conn[j];
                }

                p_gridout->setCurrentObject(outgrid);
            }

            if (p_MachineType->getValue() == 1 || p_MachineType->getValue() == 3 || p_MachineType->getValue() == 4) // MachineType is radial_machine
            { // or axial_machine

                grid->getAddresses(&elem, &conn, &x, &y, &z);
                grid->getGridSize(&numElem, &numConn, &numCoord);
                if (p_MachineType->getValue() == 1)
                {
                    numberofblades = grid->getAttribute("NUMBER_OF_BLADES");
                    if (numberofblades == NULL)
                    {
                        sendError("missing attribute NUMBER_OF_BLADES!");
                        return STOP_PIPELINE;
                    }
                    numblades = atoi(numberofblades);
                }
                if (p_MachineType->getValue() == 3)
                {
                    numblades = 4;
                }
                alpha = 360.0f / numblades;

                //	Gitter wird repliziert !

                grid->getTypeList(&inTypeList);
                coDoUnstructuredGrid *outgrid = new coDoUnstructuredGrid(p_gridout->getObjName(), numElem * numblades, numConn * numblades, numCoord * numblades, 1);
                outgrid->getAddresses(&elem_machine, &conn_machine, &x_machine, &y_machine, &z_machine);
                outgrid->getTypeList(&outTypeList);
                numCoord_machine = numCoord * numblades;

                for (i = 0; i < numblades; i++)
                {
                    for (j = 0; j < numElem; j++)
                    {
                        outTypeList[j + numElem * i] = inTypeList[j];
                    }
                }

                k = 0;
                for (i = 0; i < numblades; i++)
                {
                    for (j = 0; j < numCoord; j++)
                    {
                        x_machine[numCoord * i + j] = float(x[j] * cos(k * PI / 180) - y[j] * sin(k * PI / 180));
                        y_machine[numCoord * i + j] = float(x[j] * sin(k * PI / 180) + y[j] * cos(k * PI / 180));
                        z_machine[numCoord * i + j] = z[j];
                    }
                    k = k + alpha;
                }

                for (i = 0; i < numblades; i++)
                {
                    for (j = 0; j < numElem; j++)
                    {
                        elem_machine[j + numElem * i] = elem[j] + numConn * i;
                    }
                }

                for (i = 0; i < numblades; i++)
                {
                    for (j = 0; j < numConn; j++)
                    {
                        conn_machine[j + numConn * i] = conn[j] + numCoord * i;
                    }
                }

                p_gridout->setCurrentObject(outgrid);

            } // if machineType is radial_machine

        } //if machineType is 1..5 no run for deffilemode

    } //if stepNo == 1

#ifndef YAC
    executeCommands();
#endif
    Covise::sendInfo("complete run of compute: %6.3f s\n", ww_.elapsed());
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

int CFX::PrepareSimStart()
{
    ww_.reset();
    char startString[1000];
    char locationString[1000] = "";
    char location[25];

    //	alphabet = new string [26];
    char alphabet[26] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                          'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };
#ifndef WIN32
    char *xterm;
    char *hosts;
#endif
    char *machineType;
    const char *startup;
    const char *numberofblades;
    const char *revolutions_runner = NULL;

    int numblades;
#ifndef WIN32
    int connMeth;
#endif
    int numProc;
    int revolutions;
    int maxIterations;
    int i;
    const coDoUnstructuredGrid *grid;

    fprintf(stderr, "CFX::PrepareSimStart\n");

    numProc = p_numProc->getValue();
#ifndef WIN32
    connMeth = p_ConnectionMethod->getValue();
    hosts = (char *)p_Hostlist->getValue();
#endif
    machineType = s_MachineType[p_MachineType->getValue()];
    revolutions = p_revolutions->getValue();
    maxIterations = p_maxIterations->getValue();

    //String Location mit Regions für session_template wird erstellt
    grid = dynamic_cast<const coDoUnstructuredGrid *>(p_grid->getCurrentObject());
    revolutions_runner = grid->getAttribute("REVOLUTIONS");
    if (revolutions_runner)
    {
        revolutions = atoi(revolutions_runner);
        p_revolutions->setValue(revolutions);
    }
    else
    {
        fprintf(stderr, "could not get REVOLUTIONS from Runner\n");
    }

    numberofblades = grid->getAttribute("NUMBER_OF_BLADES");
    if (numberofblades)
    {
        numblades = atoi(numberofblades);
    }
    else
    {
        numblades = 0;
    }
    //fprintf (stderr,"Number of blades for LocationString: %d\n",numblades);

    locationString[1] = '\0';
    for (i = 0; i < numblades; i++)
    {
        sprintf(location, "Primitive\\\\ 3D\\\\ %c,", alphabet[i]);
        // 	    fprintf (stderr,"location: %s\n",location);
        strcat(locationString, location);
    }
    locationString[strlen(locationString) - 1] = '\0';

// 	fprintf (stderr,"\nLocation string: %s\n",locationString);

// 	fprintf(stderr,"hosts: %s\n",hosts);

#ifndef WIN32
    char delims[] = ",";
    int numParts = 0;
    int anzahl = 0;
    char newhosts[500];
    newhosts[0] = '\0';

    char *iflocal = strtok(strdup(hosts), "*");
    char *hostfirst = strtok(strdup(hosts), ","); //zuerst wird vor dem Komma getrennt
    hostfirst = strtok(strdup(hostfirst), "*"); //und dann vor dem Stern falls einer vorhanden
    char *onehost = strtok(strdup(hosts), delims);
    char localh[100];

    gethostname(localh, 100);
    // 	fprintf(stderr,"gethostname: %s\n",localh);
    // 	fprintf(stderr,"localh: %s\n",localh);
    // 	fprintf(stderr,"hostfirst: %s\n",hostfirst);
    // 	fprintf(stderr,"p_Hostname: %s\n",p_Hostname->getValue());

    if (strcmp(p_Hostname->getValue(), hostfirst))
    {
        if (strcmp(localh, hostfirst))
        {
            fprintf(stderr, "\nUnable to find the master host in the host list: at least one partition must be assigned to the master host.\n\n");
            return -1;
        }
    }

    if (!strchr(hosts, ','))
    {
        if (!strcmp(p_Hostname->getValue(), "localhost"))
        {
            gethostname(localh, 100);
            // 			fprintf(stderr,"gethostname: %s\n",localh);
        }

        fprintf(stderr, "iflocal: %s\n", iflocal);
    }
#endif

    if (numProc == 1)
    {
        startup = "serial";
    }
    else
    {
        startup = s_StartupMethod[p_StartupMethod->getValue()];
    }
    fprintf(stderr, "startup: %s\n", startup);

#ifndef WIN32
    while (onehost != NULL)
    {
        // 		fprintf(stderr, "onehost is \"%s\"\n", onehost );
        if (!strchr(onehost, '*'))
        {
            if ((numProc - numParts) > 0)
            {
                strcat(newhosts, onehost);
                strcat(newhosts, "*1,");
                numParts++;
            }
            // 			fprintf(stderr,"numParts: %d\n",numParts);
        }
        else
        {
            anzahl = atoi(strchr(onehost, '*') + 1);

            if (numProc - numParts >= anzahl)
            {
                strcat(newhosts, onehost);
                strcat(newhosts, ",");
            }
            else
            {
                strncat(newhosts, onehost, strlen(onehost) - strlen(strchr(onehost, '*')));
                sprintf(newhosts, "%s*%d,", newhosts, numProc - numParts);
                strcat(newhosts, "\0");
            }
            numParts = numParts + anzahl;
            // 			fprintf(stderr,"numParts: %d\n",numParts);
        }
        onehost = strtok(NULL, delims);
    }

    newhosts[strlen(newhosts) - 1] = '\0';
    // 	fprintf(stderr, "newhost is \"%s\"\n", newhosts );

    if (numProc > numParts)
    {
        fprintf(stderr, "numProc has been reduced (hostlist too short) to %d \n", numParts);
        p_numProc->setValue(numParts);
    }
#endif

#ifndef WIN32
    if (p_xterm->getValue())
    {
        xterm = strdup("xterm -e");
    }
    else
    {
        xterm = strdup("\0");
    }
#endif

#ifndef WIN32
    if (connMeth == 0)
    {
#endif
        /* if (startup == "serial") 
      {
      strcat(newhosts,localh);
      strcat(newhosts, "\0");
      }*/
        fprintf(stderr, "p_ConnectionMethod is local\n");
#ifndef WIN32
        sprintf(startString, "%s %s numProc %d Hostlist %s MachineType %s startup \"%s\" revolutions %d deffile %s maxIterations %d locationString \\\\\\\"%s\\\\\\\" CO_SIMLIB_CONN", xterm, startScript->getValue(), numProc, newhosts, machineType, startup, revolutions, p_deffile->getValue(), maxIterations, locationString);
#else
    sprintf(startString, "%s numProc %d MachineType %s startup \\\"%s\\\" revolutions %d deffile %s maxIterations %d locationString \\\\\\\"%s\\\\\\\" CO_SIMLIB_CONN", startScript->getValue(), numProc, machineType, startup, revolutions, p_deffile->getValue(), maxIterations, locationString);
#endif
        fprintf(stderr, "startString: %s\n", startString);
        setUserArg(0, startString);
#ifndef WIN32
    }
    else if (connMeth == 1)
    {
        fprintf(stderr, "p_ConnectionMethod is ssh\n");
        sprintf(startString, "ssh -l %s %s %s %s numProc %d Hostlist %s MachineType %s startup \\\"%s\\\" revolutions %d deffile %s maxIterations %d locationString \\\\\\\"%s\\\\\\\" CO_SIMLIB_CONN", p_username->getValue(), p_Hostname->getValue(), xterm, startScript->getValue(), numProc, newhosts, machineType, startup, revolutions, p_deffile->getValue(), maxIterations, locationString);

        fprintf(stderr, "startString: %s\n", startString);
        setUserArg(0, startString);
    }
#endif
    Covise::sendInfo("complete run of PrepareSimStart: %6.3f s\n", ww_.elapsed());
    return 0;
}

int CFX::endIteration()
{

    return 1;
}

#ifdef YAC
void CFX::paramChanged(coParam *param)
{

    (void)param;
}
#endif

MODULE_MAIN(Simulation, CFX)
