/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1999 RUS  **
 **                                                                        **
 ** Description: Read module for FLUVIS data format         	              **
 **                                                                        **
 ** Author: D. Rainer                                                      **
 **                                                                        **
 ** Date: August 99                                                        **
 **                                                                        **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadFluvisTS.h"
#define NUM_SCALAR_PORTS 3
#include <unistd.h>
void main(int argc, char *argv[])
{
    Application *application = new Application(argc, argv);
    application->run();
}

Application::Application(int argc, char *argv[])
{

    // this description appears in the module setup window
    Covise::set_module_description("Read FLUVIS data");

    // parameters
    Covise::add_port(PARIN, "elemPath", "Browser", "element file");
    Covise::set_port_default("elemPath", "./x.elem *");

    Covise::add_port(PARIN, "knotPath", "Browser", "knot file");
    Covise::set_port_default("knotPath", "./x.knot *");

    Covise::add_port(PARIN, "velPath", "Browser", "velocity file");
    Covise::set_port_default("velPath", "./x.velocity *");

    Covise::add_port(PARIN, "scalar0Path", "Browser", "1. scalar file");
    Covise::set_port_default("scalar0Path", "./x.density *");

    Covise::add_port(PARIN, "scalar1Path", "Browser", "2. scalar file");
    Covise::set_port_default("scalar1Path", "./x.pstatic *");

    Covise::add_port(PARIN, "scalar2Path", "Browser", "3. scalar file");
    Covise::set_port_default("scalar2Path", "./ptotal *");

    Covise::add_port(PARIN, "from_ts", "Scalar", "Start Timestep");
    Covise::set_port_default("from_ts", "1");

    Covise::add_port(PARIN, "to_ts", "Scalar", "End Timestep");
    Covise::set_port_default("to_ts", "1");

    // the output ports
    Covise::add_port(OUTPUT_PORT, "grid", "coDoUnstructuredGrid", "unstructured grid");
    Covise::add_port(OUTPUT_PORT, "velocity", "Vec3", "velocity data");
    Covise::add_port(OUTPUT_PORT, "density", "coDoFloat", "density data, e. g. temperature");
    Covise::add_port(OUTPUT_PORT, "pstatic", "coDoFloat", "pstatic data, e. g. pressure");
    Covise::add_port(OUTPUT_PORT, "ptotal", "coDoFloat", "ptotal data");

    // setup the connection to the controller and data manager
    Covise::init(argc, argv);

    // set the quit callback
    // this callback will be executed when the module gets a quit message
    // from the controller
    Covise::set_quit_callback(Application::quitCallback, this);

    // set the start callback
    // this callback will be executed when the module gets an execute message
    // from the controller
    Covise::set_start_callback(Application::computeCallback, this);
}

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

void Application::compute(void *)
{
    int i;
    char *knotFile, *elemFile, *velFile, *sc0File, *sc1File, *sc2File;
    char *knotPath, *elemPath, *velPath;
    char *scalar0Path, *scalar1Path, *scalar2Path;
    char *dir_base_name;
    char *gridObjName, *velObjName;
    char *densityObjName, *pstaticObjName, *ptotalObjName;
    char tmpGridName[1000], tmpVelName[1000], tmpDensityName[1000];
    char tmpPstaticName[1000], tmpPtotalName[1000];
    char tmpElemFileName[1000], tmpKnotFileName[1000];
    char tmpVelFileName[1000], tmpDensityFileName[1000];
    char tmppstaticFileName[1000], tmpptotalFileName[1000];

    coDoUnstructuredGrid *gridSubObj;
    coDoVec3 *velSubObj;
    coDoFloat *densSubObject;
    coDoFloat *pstatSubObject;
    coDoFloat *ptotSubObject;

    Covise::get_browser_param("knotPath", &knotPath);
    Covise::get_browser_param("elemPath", &elemPath);
    Covise::get_browser_param("velPath", &velPath);
    Covise::get_browser_param("scalar0Path", &scalar0Path);
    Covise::get_browser_param("scalar1Path", &scalar1Path);
    Covise::get_browser_param("scalar2Path", &scalar2Path);
    Covise::get_scalar_param("from_ts", &from_ts);
    Covise::get_scalar_param("to_ts", &to_ts);

    dir_base_name = extract_base_path(knotPath);
    elemFile = extract_filename(elemPath);
    knotFile = extract_filename(knotPath);
    velFile = extract_filename(velPath);
    sc0File = extract_filename(scalar0Path);
    sc1File = extract_filename(scalar1Path);
    sc2File = extract_filename(scalar2Path);

    gridObjName = Covise::get_object_name("grid");
    velObjName = Covise::get_object_name("velocity");
    densityObjName = Covise::get_object_name("density");
    pstaticObjName = Covise::get_object_name("pstatic");
    ptotalObjName = Covise::get_object_name("ptotal");

    coDoSet *GRID_Set = new coDoSet(gridObjName, SET_CREATE);
    coDoSet *VEL_Set = new coDoSet(velObjName, SET_CREATE);
    coDoSet *DENS_Set = new coDoSet(densityObjName, SET_CREATE);
    coDoSet *PSTAT_Set = new coDoSet(pstaticObjName, SET_CREATE);
    coDoSet *PTOT_Set = new coDoSet(ptotalObjName, SET_CREATE);

    for (i = from_ts; i <= to_ts; i++)
    {

        sprintf(tmpElemFileName, "%s%04d/%s", dir_base_name, i, elemFile);
        sprintf(tmpKnotFileName, "%s%04d/%s", dir_base_name, i, knotFile);
        sprintf(tmpGridName, "%s_%04d", gridObjName, i);
        gridSubObj = createGridObject(tmpElemFileName, tmpKnotFileName, tmpGridName);
        GRID_Set->addElement(gridSubObj);
        delete gridSubObj;

        sprintf(tmpVelFileName, "%s%04d/%s", dir_base_name, i, velFile);
        sprintf(tmpVelName, "%s_%04d", velObjName, i);
        velSubObj = createVelocityObject(tmpVelFileName, tmpVelName);
        VEL_Set->addElement(velSubObj);
        delete velSubObj;

        sprintf(tmpDensityFileName, "%s%04d/%s", dir_base_name, i, sc0File);
        sprintf(tmpDensityName, "%s_%04d", densityObjName, i);
        densSubObject = createScalarObject(tmpDensityFileName, tmpDensityName);
        DENS_Set->addElement(densSubObject);
        delete densSubObject;

        sprintf(tmppstaticFileName, "%s%04d/%s", dir_base_name, i, sc1File);
        sprintf(tmpPstaticName, "%s_%04d", pstaticObjName, i);
        pstatSubObject = createScalarObject(tmppstaticFileName, tmpPstaticName);
        PSTAT_Set->addElement(pstatSubObject);
        delete pstatSubObject;

        sprintf(tmpptotalFileName, "%s%04d/%s", dir_base_name, i, sc2File);
        sprintf(tmpPtotalName, "%s_%04d", ptotalObjName, i);
        ptotSubObject = createScalarObject(tmpptotalFileName, tmpPtotalName);
        PTOT_Set->addElement(ptotSubObject);
        delete ptotSubObject;
    }

    char attr[200];

    sprintf(attr, "%d %d", from_ts, to_ts);
    GRID_Set->addAttribute("TIMESTEP", attr);
    VEL_Set->addAttribute("TIMESTEP", attr);
    DENS_Set->addAttribute("TIMESTEP", attr);
    PSTAT_Set->addAttribute("TIMESTEP", attr);
    PTOT_Set->addAttribute("TIMESTEP", attr);

    delete GRID_Set;
    delete VEL_Set;
    delete DENS_Set;
    delete PSTAT_Set;
    delete PTOT_Set;
}

coDoUnstructuredGrid *
Application::createGridObject(char *elemPath, char *knotPath, char *gridObjectName)
{
    fprintf(stderr, "Application::createGridObject\n");
    int numElem;
    FILE *elemFp, *knotFp; // file pointers to elem and grid file

    //    char* gridObjectName; // name of the grid output object
    coDoUnstructuredGrid *gridObj; // covise grid object
    int *elemList, *vertexList, *typeList;
    float *xCoordList, *yCoordList, *zCoordList;
    char buf[300];

    int i, tmpi, v0, v1, v2, v3, v4, v5, v6, v7; // v0-v7: vertices for an hexagon element

    // open knot file
    if ((knotFp = Covise::fopen(knotPath, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file ");
        strcat(buf, knotPath);
        Covise::sendError(buf);
        fprintf(stderr, "\t%s\n", buf);
        return NULL;
    }

    // read the number of coordinates
    fgets(buf, 300, knotFp);
    sscanf(buf, "%d", &numCoord);
    fprintf(stderr, "\tknot file: %s\n", knotPath);
    fprintf(stderr, "\tnum coords: %d\n", numCoord);

    // open element file
    if ((elemFp = Covise::fopen(elemPath, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file ");
        strcat(buf, elemPath);
        Covise::sendError(buf);
        fprintf(stderr, "\t%s\n", buf);
        return NULL;
    }

    // read the no of elements
    fgets(buf, 300, elemFp);
    sscanf(buf, "%d", &numElem);
    fprintf(stderr, "\telem file: %s\n", elemPath);
    fprintf(stderr, "\tnum elems: %d\n", numElem);

    // get the name of the COVISE grid object
    if (gridObjectName != NULL)
    {
        gridObj = new coDoUnstructuredGrid(gridObjectName, numElem, numElem * 8, numCoord, 1);
        if (gridObj->objectOk())
        {
            gridObj->getAddresses(&elemList, &vertexList, &xCoordList, &yCoordList, &zCoordList);
            gridObj->getTypeList(&typeList);

            // read the coordinates
            fprintf(stderr, "reading the coordinates\n");
            for (i = 0; i < numCoord; i++)
            {
                fgets(buf, 300, knotFp);

                if (feof(knotFp))
                {
                    Covise::sendError("ERROR: unexpected end of file");
                    return NULL;
                }

                sscanf(buf, "%d%f%f%f", &tmpi, xCoordList, yCoordList, zCoordList);
                xCoordList++;
                yCoordList++;
                zCoordList++;
            }

            // read the elements
            fprintf(stderr, "reading the elements\n");
            for (i = 0; i < numElem; i++)
            {
                fgets(buf, 300, elemFp);

                if (feof(elemFp))
                {
                    Covise::sendError("ERROR: unexpected end of file");
                    return NULL;
                }
                sscanf(buf, "%d%d%d%d%d%d%d%d%d", &tmpi, &v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7);
                // convert FORTRAN INDICES to C INDICES
                vertexList[i * 8] = v0 - 1;
                vertexList[i * 8 + 1] = v1 - 1;
                vertexList[i * 8 + 2] = v2 - 1;
                vertexList[i * 8 + 3] = v3 - 1;
                vertexList[i * 8 + 4] = v4 - 1;
                vertexList[i * 8 + 5] = v5 - 1;
                vertexList[i * 8 + 6] = v6 - 1;
                vertexList[i * 8 + 7] = v7 - 1;
                elemList[i] = i * 8;
                typeList[i] = TYPE_HEXAGON;
            }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'gridObj' failed");
            return NULL;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'gridObj'");
        return NULL;
    }

    fclose(elemFp);
    fclose(knotFp);

    return gridObj;
}

coDoVec3 *
Application::createVelocityObject(char *velPath, char *velObjName)
{
    fprintf(stderr, "Application::createVelocityObject\n");

    FILE *velFp; // velicty file pointer
    char buf[300];
    coDoVec3 *velObj; // covise data object
    int i, tmpi;
    float *u, *v, *w;

    // open velocity file
    if ((velFp = Covise::fopen(velPath, "r")) == NULL)
    {
        Covise::sendInfo("No valid file for port velocity");
        return NULL;
    }

    // get the ouput object name
    if (velObjName != NULL)
    {
        velObj = new coDoVec3(velObjName, numCoord);
        if (velObj->objectOk())
        {
            velObj->getAddresses(&u, &v, &w);

            // read the coordinates
            fprintf(stderr, "reading the velocity data\n");
            for (i = 0; i < numCoord; i++)
            {
                fgets(buf, 300, velFp);

                if (feof(velFp))
                {
                    Covise::sendError("ERROR: unexpected end of file");
                    return NULL;
                }

                sscanf(buf, "%d%f%f%f", &tmpi, u, v, w);
                u++;
                v++;
                w++;
            }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'velObj' failed");
            return NULL;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'velObj'");
        return NULL;
    }

    fclose(velFp);

    return velObj;
}

coDoFloat *
Application::createScalarObject(char *scalarPath, char *scalarObjName)
{
    char parameterName[50]; // name of the parameter scalar*Path
    char portName[50]; // name of the port scalar*
    FILE *scalarFp; // file pointer
    char buf[300];
    coDoFloat *scalarObj; // scalar data object
    int i, tmpi;
    float *s; // scalar values

    if ((scalarFp = Covise::fopen(scalarPath, "r")) == NULL)
    {
        sprintf(buf, "No valid file for Scalar Object %d", scalarObjName);
        Covise::sendInfo(buf);
        return NULL;
    }

    if (scalarObjName != NULL)
    {
        scalarObj = new coDoFloat(scalarObjName, numCoord);
        if (scalarObj->objectOk())
        {
            scalarObj->getAddress(&s);

            // read the coordinates
            fprintf(stderr, "reading the scalar data %s\n", scalarObjName);
            for (i = 0; i < numCoord; i++)
            {
                fgets(buf, 300, scalarFp);

                if (feof(scalarFp))
                {
                    Covise::sendError("ERROR: unexpected end of file");
                    return NULL;
                }

                sscanf(buf, "%d%f", &tmpi, s);
                s++;
            }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'scalarObj' failed");
            return NULL;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'scalarObj'");
        return NULL;
    }

    fclose(scalarFp);

    return scalarObj;
}

char *Application::extract_base_path(char *path)
{
    int i;
    char *tmp;

    for (i = 0; path[i] != '\0'; i++)
        ;
    while (i > 0 && path[i] != '/')
        i--;
    i -= 4; // slash plus four digits
    if (i == 0)
        return NULL;

    tmp = new char[i + 1];
    strncpy(tmp, path, i);
    tmp[i] = '\0';

    cout << "extracted base path: " << tmp << endl;

    return tmp;
}

char *Application::extract_filename(char *path)
{
    int i;
    char *tmp;

    for (i = 0; path[i] != '\0'; i++)
        ;

    int max = i;

    while (i > 0 && path[i] != '/')
        i--;

    if (i == 0)
        return NULL;

    i++;

    tmp = new char[max - i + 1];
    strcpy(tmp, &path[i]);

    cout << "extracted filename: " << tmp << endl;

    return tmp;
}
