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
#include <sys/types.h>

#ifndef YAC
#include <appl/ApplInterface.h>
#endif
#include "OpenFoam.h"

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

OpenFoam::OpenFoam(int argc, char *argv[])
    : coSimLib(argc, argv, argv[0], "OpenFoam coupling")
{
////////// set up default parameters

#ifdef VERBOSE

    cerr << "##############################" << endl;
    cerr << "#####   OpenFoam " << endl;
    cerr << "#####   PID =  " << getpid() << endl;
    cerr << "##############################" << endl;
#endif
#ifndef YAC
    set_module_description("OpenFoam Simulation");
#endif

    //dir = addStringParam("dir","Directory for run");
    //dir->setValue(coCoviseConfig::getEntry("value","Module.OpenFoam.StartUpScript", "$FOAM_RUN/gate/gate.sh").c_str());
    // Output ports:  3 vars
    createUserMenu();

    gridPort = addInputPort("grid", "UnstructuredGrid|USR_FoamMesh", "Distributed Grid");
    bocoPort = addInputPort("boco", "USR_FenflossBoco|USR_FoamBoco", "Boundary Conditions");

    //mesh=addOutputPort("mesh","UnstructuredGrid","Data Output");
    pDataOutputPort = addOutputPort("pData", "Float", "p Data Output");
    uDataOutputPort = addOutputPort("uData", "Vec3", "U Data Output");
    //   TDataOutputPort=addOutputPort("TData","Float","T Data Output");

    stepNo = -1;

    parRegs = 2;
}

void OpenFoam::param(const char *portname, bool)
{
    int connMeth;
    fprintf(stderr, "OpenFoam::param(): Entering OpenFoam::param(): %s\n", portname);
    // if spotpoint is changed, we have to copy the values in
    // the "transfer buffers"
    if (!strcmp(portname, p_ConnectionMethod->getName()))
    {
        fprintf(stderr, "p_ConnectionMethod changed ...\n");
        connMeth = p_ConnectionMethod->getValue();
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
    else if (!strcmp(portname, p_Port->getName()))
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
    else if (!strcmp(portname, p_ParRegs->getName()))
    {
        parRegs = atoi(p_ParRegs->getValue());
    }
}

int OpenFoam::compute(const char *) //port)
{
    //(void)port;
    //const char *directory;
    //const coDoUnstructuredGrid *grid = dynamic_cast<const coDoUnstructuredGrid*>(gridPort->getCurrentObject());
    const coDistributedObject *grid_DO = gridPort->getCurrentObject();

    if (stepNo < 0)
    {
        //directory = p_StartScript->getValue();
        //setUserArg(0,directory);
        prepareSimStart(parRegs);
        startSim();

        sendMesh();
        sendBoundaryConditions();

        if (grid_DO)
        {
            currentGridName = grid_DO->getName();
        }
        else
        {
            std::cerr << "OpenFoam::compute(): No grid, no name, returning!" << std::endl;
            return -1;
        }

        stepNo = 1;
    }
    else
    {
        if (grid_DO)
        {
            // we have a new grid input object
            std::string gridName = grid_DO->getName();
            if (gridName != currentGridName)
            {
                std::cout << "OpenFoam::compute(): resetting simLib!" << std::endl;

                resetSimLib();
                stepNo = -1;

                compute("");
            }
        }
    }

#ifndef YAC
    executeCommands();
#endif

    return SUCCESS;
}

/*************************************************************
 *************************************************************
 *************************************************************
 **                                                         **
 **           A u x i l i a r y   r o u t i n e s           **
 **                                                         **
 *************************************************************
 *************************************************************
 *************************************************************/

int OpenFoam::endIteration()
{
    return 1;
}

#ifdef YAC
void OpenFoam::paramChanged(coParam *param)
{
    (void)param;
}
#endif

void OpenFoam::sendMesh()
{
    const coDistributedObject *mesh = gridPort->getCurrentObject();

    const coDoUnstructuredGrid *grid;
    const coDoPolygons *faces;

    float *x, *y, *z;
    int *el, *conn, *tl, *fl;

    if (!mesh)
    {
        std::cerr << "OpenFoam::sendMesh(): No grid on port, nothing sent!" << std::endl;
        return;
    }

    //int numCoord;  USGNums[0]
    //int numElem;   USGNums[1]
    //int numConn;   USGNums[2]
    //machine type (=0: gate, =1: francis)  USGNums[5]

    int USGNums[10];

    USGNums[3] = atoi(mesh->getAttribute("number_of_blades"));

    if ((mesh->getAttribute("machinetype") != NULL) && (!strcmp(mesh->getAttribute("machinetype"), "francis")))
    {
        // francis: direct polyMesh constructor
        // -> we send the polyMesh, including face vertices, coordinates, owner and neighbour

        // contains internal mesh faces, without boundaries
        //   0. sizes
        //   1. internal faces, upper triangular order
        //   2. coordinate arrays (xyz)
        //   3. owner array
        //   4. neighbour array
        //   5. NULL

        // machine type
        USGNums[5] = 1; // = Francis

        // number of Hexaeder cells
        USGNums[6] = atoi(mesh->getAttribute("number_hex_cells"));
        USGNums[7] = atoi(mesh->getAttribute("number_boundary_faces"));
        int num_boundary_faces = USGNums[7];

        const coDoSet *mesh = dynamic_cast<const coDoSet *>(gridPort->getCurrentObject());
        if (!mesh)
        {
            std::cerr << "OpenFoam::sendMesh(): No mesh on port, nothing sent!" << std::endl;
            return;
        }

        int numSetObjects = mesh->getNumElements();
        if (numSetObjects != 5)
        {
            std::cerr << "OpenFoam::sendMesh(): Wrong number of elements in coDistributedObject (" << numSetObjects << "), resigning!" << std::endl;
            return;
        }

        // sizes
        const coDoIntArr *sizesArray = dynamic_cast<const coDoIntArr *>(mesh->getElement(0));
        if (!sizesArray)
        {
            std::cerr << "OpenFoam::sendMesh(): Illegal part type (0) of sizes in mesh, returning!";
            return;
        }
        int *sizes = sizesArray->getAddress();
        USGNums[0] = sizes[0]; // num3Dnodes
        USGNums[1] = sizes[1]; // num_internal_faces + num_boundary_faces
        USGNums[2] = sizes[2]; // num vertices (4*(num_internal_faces+num_boundary_faces))
        // int num_internal_faces = USGNums[1];
        // int num_faces = num_boundary_faces + num_internal_faces;
        int num_faces = USGNums[1];
        int num_internal_faces = num_faces - num_boundary_faces;
        //fprintf(stderr,"num_internal_faces = %d\n", num_internal_faces);
        //fprintf(stderr,"num_boundary_faces = %d\n", num_boundary_faces);
        //fprintf(stderr,"num_faces = %d\n", num_faces);

        //Number of parallel regions
        USGNums[4] = parRegs;

        sendBS_Data(USGNums, 8 * sizeof(int));

        // faces
        const coDoIntArr *facesArray = dynamic_cast<const coDoIntArr *>(mesh->getElement(1));
        int *faces = facesArray->getAddress();
        //fprintf(stderr,"OpenFoam: sending %d faces\n", 4*num_faces);
        sendBS_Data(faces, 4 * num_faces * sizeof(int));

        // coordinates
        const coDoFloat *coordinateArray = dynamic_cast<const coDoFloat *>(mesh->getElement(2));
        float *coordinates = coordinateArray->getAddress();
        //fprintf(stderr,"OpenFoam: sending %d coords\n", 3*USGNums[0]);
        sendBS_Data(coordinates, 3 * USGNums[0] * sizeof(float));

        // owner
        const coDoIntArr *ownerArray = dynamic_cast<const coDoIntArr *>(mesh->getElement(3));
        int *owner = ownerArray->getAddress();
        //fprintf(stderr,"sending owner, %d values\n", num_faces);
        //fprintf(stderr,"sender: owner[0] = %d\n", owner[0]);
        //fprintf(stderr,"sender: owner[%d] = %d\n", num_faces-1, owner[num_faces-1]);
        //fprintf(stderr,"OpenFoam: sending %d owner\n", num_faces);
        sendBS_Data(owner, num_faces * sizeof(int));

        // neighbour
        const coDoIntArr *neighbourArray = dynamic_cast<const coDoIntArr *>(mesh->getElement(4));
        int *neighbour = neighbourArray->getAddress();
        //fprintf(stderr,"OpenFoam: sending %d neighbour\n", num_internal_faces);
        //fprintf(stderr,"OpenFoam: neighbour[0] = %d\n", neighbour[0]);
        //fprintf(stderr,"OpenFoam: neighbour[%d] = %d\n", num_internal_faces-1, neighbour[num_internal_faces-1]);
        sendBS_Data(neighbour, num_internal_faces * sizeof(int));
    }
    else if (mesh->isType("UNSGRD"))
    {
        // gate: we use the shape mesh constructor for polyMesh in the simulation code

        // machine type
        USGNums[5] = 0; // = Gate

        grid = dynamic_cast<const coDoUnstructuredGrid *>(mesh);
        grid->getGridSize(USGNums + 1, USGNums + 2, USGNums);

        //Number of parallel regions
        USGNums[4] = parRegs;

        grid->getAddresses(&el, &conn, &x, &y, &z);
        grid->getTypeList(&tl);

        sendBS_Data(USGNums, 6 * sizeof(int));

        sendBS_Data(x, USGNums[0] * sizeof(float));
        sendBS_Data(y, USGNums[0] * sizeof(float));
        sendBS_Data(z, USGNums[0] * sizeof(float));

        sendBS_Data(el, USGNums[1] * sizeof(int));
        sendBS_Data(tl, USGNums[1] * sizeof(int));

        sendBS_Data(conn, USGNums[2] * sizeof(int));
    }
}

void OpenFoam::sendBoundaryConditions()
{
    fprintf(stderr, "entering OpenFoam::sendBoundaryConditions()\n");
    const coDoSet *boco = dynamic_cast<const coDoSet *>(bocoPort->getCurrentObject());
    if (!boco)
    {
        std::cerr << "OpenFoam::sendBoundaryConditions(): No bocos on port, nothing sent!" << std::endl;
        return;
    }

    int numSetObjects = boco->getNumElements();
    if (!((numSetObjects == 7) || (numSetObjects == 8) || (numSetObjects == 4)))
    {
        std::cerr << "OpenFoam::sendBoundaryConditions(): Wrong number of elements in coDistributedObject (" << numSetObjects << "), resigning!" << std::endl;
        return;
    }

    if ((boco->getAttribute("machinetype") != NULL) && (!strcmp(boco->getAttribute("machinetype"), "francis")))
    {
        // *******
        // Francis
        // *******

        // 1. number of patch faces / cells
        // data[0] = newmesh3d->num_blade_faces;
        // data[1] = newmesh3d->num_hubrot_faces;
        // data[2] = newmesh3d->num_hubnonrot_faces;
        // data[3] = newmesh3d->num_shroudrot_faces;
        // data[4] = newmesh3d->num_shroudnonrot_faces;
        // data[5] = newmesh3d->num_inlet_faces;
        // data[6] = newmesh3d->num_outlet_faces;
        // data[7] = newmesh3d->num_per1_faces;
        // data[8] = newmesh3d->num_per2_faces;
        // data[9] = newmesh3d->num3DElements;

        const coDoIntArr *numBCs = dynamic_cast<const coDoIntArr *>(boco->getElement(0));
        int n_numBCs = numBCs->getDimension(0); // should be 10
        int *data = numBCs->getAddress();

        sendBS_Data(data, n_numBCs * sizeof(int));
        int num_inlet_faces = data[5];
        int num3DElements = data[9];

        // boundary faces are already part of the mesh - sent in sendMesh()

        //   2. velocity at inlet (dirichlet bc)
        const coDoFloat *Uinlet = dynamic_cast<const coDoFloat *>(boco->getElement(1));
        float *Uinlet_values = Uinlet->getAddress();
        int n_Uinlet = 3 * num_inlet_faces;
        sendBS_Data(Uinlet_values, n_Uinlet * sizeof(float));

        //   3. U initialization (on cells)
        const coDoFloat *UVolume = dynamic_cast<const coDoFloat *>(boco->getElement(2));
        float *UVolume_values = UVolume->getAddress();
        int n_UVolume = 3 * num3DElements;
        sendBS_Data(UVolume_values, n_UVolume * sizeof(float));

        //   4. p initialization (on cells)
        const coDoFloat *pVolume = dynamic_cast<const coDoFloat *>(boco->getElement(3));
        float *pVolume_values = pVolume->getAddress();
        int n_pVolume = num3DElements;
        sendBS_Data(pVolume_values, n_pVolume * sizeof(float));
    }
    else
    {
        // *******
        // Gate
        // *******

        // Wall indices
        const coDoIntArr *wallArray = dynamic_cast<const coDoIntArr *>(boco->getElement(5));
        if (!wallArray)
        {
            std::cerr << "OpenFoam::sendBoundaryConditions(): Illegal part type (5) of wall indices in boco, returning!";
            return;
        }
        int sizeColumnWall = wallArray->getDimension(0);
        int numColumnsWall = wallArray->getDimension(1);
        int *wall = wallArray->getAddress();
        //DumpIntArr("wall", wall, numWall, colWall);

        // Balance indices
        const coDoIntArr *balanceArray = dynamic_cast<const coDoIntArr *>(boco->getElement(6));
        if (!balanceArray)
        {
            std::cerr << "OpenFoam::sendBoundaryConditions(): Illegal part type (6) of balance indices in boco, returning!";
            return;
        }
        int sizeColumnBalance = balanceArray->getDimension(0);
        int numColumnsBalance = balanceArray->getDimension(1);
        int *balance = balanceArray->getAddress();
        //DumpIntArr("balance", balance, numBalance, colBalance);

        int bocoNums[4] = { sizeColumnWall, numColumnsWall, sizeColumnBalance, numColumnsBalance };
        sendBS_Data(bocoNums, 4 * sizeof(int));

        sendBS_Data(wall, bocoNums[0] * bocoNums[1] * sizeof(int));

        sendBS_Data(balance, bocoNums[2] * bocoNums[3] * sizeof(int));

        //Initial values
        //
        const coDoIntArr *diricletNodes = dynamic_cast<const coDoIntArr *>(boco->getElement(3));
        int sizeColumnDirNodes = diricletNodes->getDimension(0);
        int sizeArrayDirValues = diricletNodes->getDimension(1);
        int *dirNodes = diricletNodes->getAddress();

        const coDoFloat *diricletValues = dynamic_cast<const coDoFloat *>(boco->getElement(4));
        float *dirValues = diricletValues->getAddress();

        int initValNums[2] = { sizeColumnDirNodes, sizeArrayDirValues };
        sendBS_Data(initValNums, 2 * sizeof(int));

        sendBS_Data(dirNodes, sizeColumnDirNodes * sizeArrayDirValues * sizeof(int));

        sendBS_Data(dirValues, sizeArrayDirValues * sizeof(int));
    }
}

void OpenFoam::createUserMenu()
{
#ifndef WIN32
    uid_t myuid;
    struct passwd *mypwd;
#endif
    p_ConnectionMethod = addChoiceParam("Connection_Method", "ConnectionMethod");
    s_ConnectionMethod[0] = strdup("local");
    s_ConnectionMethod[1] = strdup("ssh");
    s_ConnectionMethod[2] = strdup("rsh");
    s_ConnectionMethod[3] = strdup("rdaemon");
    s_ConnectionMethod[4] = strdup("echo");
    s_ConnectionMethod[5] = strdup("globus_gram");
    s_ConnectionMethod[6] = strdup("reattach");
#ifdef WIN32
    s_ConnectionMethod[7] = strdup("WMI");
#endif

    p_ConnectionMethod->setValue(CM_MAX, s_ConnectionMethod, 0);

    p_Hostname = addStringParam("Hostname", "Hostname");
    p_Hostname->setValue("localhost");

    p_Port = addStringParam("Port", "Port");
    p_Port->setValue(coCoviseConfig::getEntry("value", "Module.OpenFoam.PORTS", "31500 31510").c_str());

    p_StartScript = addStringParam("Startup_Script_on_Client", "Startup Script on Client");
    p_StartScript->setValue(coCoviseConfig::getEntry("value", "Module.OpenFoam.StartScript", "~/$FOAM_RUN/gate/gate.sh").c_str());

    p_ParRegs = addStringParam("Parallel_Regions", "Parallel_Regions");
    p_ParRegs->setValue("2");

#ifndef WIN32
    // get user automatically
    myuid = getuid();
    while ((mypwd = getpwent()))
    {
        if (mypwd->pw_uid == myuid)
        {
            fprintf(stderr, "You are ");
            fprintf(stderr, "%s, ", mypwd->pw_name);
            fprintf(stderr, "%d\n", mypwd->pw_uid);
            break;
        }
    }
#endif

    p_User = addStringParam("User", "User");
#ifndef WIN32
    p_User->setValue(mypwd->pw_name);
#else
    p_User->setValue(getenv("USERNAME"));
#endif
}

void OpenFoam::prepareSimStart(int numProc)
{
    char sNumNodes[30];
    char *startshell = NULL;
    char simStart[255];
    int connMeth;
    int simAppl = 0;

    fprintf(stderr, "OpenFoam::PrepareSimStart(%d)\n", numProc);
    sprintf(sNumNodes, "%d", numProc);
    startshell = ConnectionString();

    connMeth = p_ConnectionMethod->getValue();
    //simAppl = p_simApplication->getValue();
    simAppl = 0;
    const std::string foamApp = "simpleFoamCovise";
    const char *s_simApplication[1] = { foamApp.c_str() };

    setUserArg(0, startshell);

    const char *caseString = p_StartupSwitch->getActLabel();

    if (!strcmp(s_ConnectionMethod[connMeth], "echo"))
        sprintf(simStart, "echo %s@%s %s %s %d", p_User->getValue(), p_Hostname->getValue(), SimBatchString(), caseString, numProc);
#ifdef WIN32
    else if (!strcmp(s_ConnectionMethod[connMeth], "WMI"))
    {
        sprintf(simStart, "\"%s start %s %d\" \" \" \"%s\" \" %s \"", SimBatchString(), caseString, numProc, p_Hostname->getValue(), p_User->getValue());
        fprintf(stderr, "simStart='%s'\n", simStart);
    }
#endif
    else if (!strcmp(s_ConnectionMethod[connMeth], "rdaemon"))
    {
        sprintf(simStart, "\"%s start %s %d %s\" \" \" \"%s\" \" %s \"", SimBatchString(), caseString, numProc, s_simApplication[simAppl], p_Hostname->getValue(), p_User->getValue());
        //sprintf(simStart, "\"%s start %s %d\" \" \" \"%s\" \" %s \"",  SimBatchString(), caseString, numProc, p_Hostname->getValue(), p_User->getValue());

        fprintf(stderr, "\n\nsimStart='%s'\n\n", simStart);
    }
    else if (!strcmp(s_ConnectionMethod[connMeth], "globus_gram"))
    {
        std::string globusrun = coCoviseConfig::getEntry("value", "Module.Globus.GlobusRun", "/usr/local/globus-4.0.1/bin/globusrun-ws");
        std::string jobfactory = coCoviseConfig::getEntry("value", "Module.Globus.jobfactory", "/wsrf/services/ManagedJobFactoryService");

        printf("globusrun: [%s]\njobfactory: [%s]\n", globusrun.c_str(), jobfactory.c_str());

        //snprintf(simStart, sizeof(simStart), "%s -s -Ft PBS -F https://%s%s -submit -c %s start %s %d %s", globusrun, p_Hostname->getValue(), jobfactory, SimBatchString(), caseString, numProc, s_simApplication[simAppl]);
        snprintf(simStart, sizeof(simStart), "%s -s -Ft PBS -J -F https://%s%s -submit -c %s start %s %d %s", globusrun.c_str(), p_Hostname->getValue(), jobfactory.c_str(), SimBatchString(), caseString, numProc, s_simApplication[simAppl]);
        printf("simstart: [%s]\n", simStart);
    }
    else if (!strcmp(s_ConnectionMethod[connMeth], "reattach"))
    {
        snprintf(simStart, sizeof(simStart), "()"); // FIXME
    }
    // execProcessWMI: commandLine, workingdirectory, host, user, password
    else
    {
        //sprintf(simStart, "%s start %s %d %s", SimBatchString(), caseString, numProc, s_simApplication[simAppl]);
        sprintf(simStart, "%s", SimBatchString());
    }

    setUserArg(1, simStart);
    //fprintf(stderr, "\tPrepareSimStart: startshell=%s; simStart=%s\n",
    //            startshell, simStart);
    printf("\tPrepareSimStart: startshell=%s; simStart=%s\n", startshell, simStart);

    std::stringstream parRegsstream;
    parRegsstream << parRegs;
    std::string together = std::string(startshell) + std::string(" ") + std::string(simStart) + std::string(" ") + parRegsstream.str();
    setUserArg(0, together.c_str());

    if (startshell)
        free(startshell);
}

char *OpenFoam::ConnectionString()
{
    int connMeth;
    char connStr[100];

    memset(connStr, 0, sizeof(connStr));

    connMeth = p_ConnectionMethod->getValue();
#ifndef WIN32
    if (!strcmp(s_ConnectionMethod[connMeth], "local") || !strcmp(s_ConnectionMethod[connMeth], "globus_gram") || !strcmp(s_ConnectionMethod[connMeth], "reattach"))
        *connStr = ' ';
    else
    {
        char user[50];

        memset(user, 0, sizeof(user));
        if (p_User->getValue() && *p_User->getValue())
            sprintf(user, "-l %s", p_User->getValue());
        sprintf(connStr, "%s %s %s", s_ConnectionMethod[connMeth], user, p_Hostname->getValue());
    }
#else
    sprintf(connStr, "%s", s_ConnectionMethod[connMeth]);
#endif

    return strdup(connStr);
}

const char *OpenFoam::SimBatchString()
{
    //int local = 0;
    //int connMeth;

    const char *dp;

    //connMeth = p_ConnectionMethod->getValue();
    //local = !strcmp(s_ConnectionMethod[connMeth], "local");

    dp = p_StartScript->getValue();

    return (strdup(dp));
}

MODULE_MAIN(HLRS, OpenFoam)
