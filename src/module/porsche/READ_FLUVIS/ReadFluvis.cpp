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
 *\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadFluvis.h"
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
    Covise::set_port_default("scalar0Path", "./x.temp *");

    Covise::add_port(PARIN, "scalar1Path", "Browser", "2. scalar file");
    Covise::set_port_default("scalar1Path", "./x.pressure *");

    Covise::add_port(PARIN, "scalar2Path", "Browser", "3. scalar file");
    Covise::set_port_default("scalar2Path", "./scalar *");

    // the output ports
    Covise::add_port(OUTPUT_PORT, "grid", "coDoUnstructuredGrid", "unstructured grid");
    Covise::add_port(OUTPUT_PORT, "velocity", "Vec3", "velocity data");
    Covise::add_port(OUTPUT_PORT, "scalar0", "coDoFloat", "scalar data, e. g. temperature");
    Covise::add_port(OUTPUT_PORT, "scalar1", "coDoFloat", "scalar data, e. g. pressure");
    Covise::add_port(OUTPUT_PORT, "scalar2", "coDoFloat", "scalar data");

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
    int n;

    createGridObject();
    createVelocityObject();
    for (n = 0; n < NUM_SCALAR_PORTS; n++)
        createScalarObject(n);
}

void
Application::createGridObject()
{
    fprintf(stderr, "Application::createGridObject\n");
    int numElem;
    char *elemPath, *knotPath; // name of the element and knot file
    FILE *elemFp, *knotFp; // file pointers to elem and grid file

    char *gridObjectName; // name of the grid output object
    coDoUnstructuredGrid *gridObj; // covise grid object
    int *elemList, *vertexList, *typeList;
    float *xCoordList, *yCoordList, *zCoordList;
    char buf[300];

    int i, tmpi, v0, v1, v2, v3, v4, v5, v6, v7; // v0-v7: vertices for an hexagon element

    // open knot file
    Covise::get_browser_param("knotPath", &knotPath);
    if ((knotFp = Covise::fopen(knotPath, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file ");
        strcat(buf, knotPath);
        Covise::sendError(buf);
        fprintf(stderr, "\t%s\n", buf);
        return;
    }

    // read the number of coordinates
    fgets(buf, 300, knotFp);
    sscanf(buf, "%d", &numCoord);
    fprintf(stderr, "\tknot file: %s\n", knotPath);
    fprintf(stderr, "\tnum coords: %d\n", numCoord);

    // open element file
    Covise::get_browser_param("elemPath", &elemPath);
    if ((elemFp = Covise::fopen(elemPath, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file ");
        strcat(buf, elemPath);
        Covise::sendError(buf);
        fprintf(stderr, "\t%s\n", buf);
        return;
    }

    // read the no of elements
    fgets(buf, 300, elemFp);
    sscanf(buf, "%d", &numElem);
    fprintf(stderr, "\telem file: %s\n", elemPath);
    fprintf(stderr, "\tnum elems: %d\n", numElem);

    // get the name of the COVISE grid object
    gridObjectName = Covise::get_object_name("grid");
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
                    return;
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
                    return;
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
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'gridObj'");
        return;
    }

    fclose(elemFp);
    fclose(knotFp);

    delete gridObj;
}

void
Application::createVelocityObject()
{
    fprintf(stderr, "Application::createVelocityObject\n");

    char *velPath; // velocity file name
    FILE *velFp; // velicty file pointer
    char buf[300];
    char *velObjName; // name of the covise data object
    coDoVec3 *velObj; // covise data object
    int i, tmpi;
    float *u, *v, *w;

    // open velocity file
    Covise::get_browser_param("velPath", &velPath);
    if ((velFp = Covise::fopen(velPath, "r")) == NULL)
    {
        Covise::sendInfo("No valid file for port velocity");
        return;
    }

    // get the ouput object name
    velObjName = Covise::get_object_name("velocity");
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
                    return;
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
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'velObj'");
        return;
    }

    fclose(velFp);
    delete velObj;
}

void
Application::createScalarObject(int n)
{
    char *scalarPath; // name of the scalar file
    char parameterName[50]; // name of the parameter scalar*Path
    char portName[50]; // name of the port scalar*
    FILE *scalarFp; // file pointer
    char buf[300];
    char *scalarObjName; // name of the scalar data object
    coDoFloat *scalarObj; // scalar data object
    int i, tmpi;
    float *s; // scalar values

    // create the appropriate parameter name
    sprintf(parameterName, "scalar%dPath", n);

    // open scalar file
    Covise::get_browser_param(parameterName, &scalarPath);
    if ((scalarFp = Covise::fopen(scalarPath, "r")) == NULL)
    {
        sprintf(buf, "No valid file for port scalar%dPath", n);
        Covise::sendInfo(buf);
        return;
    }

    // create the appropriate port name
    sprintf(portName, "scalar%d", n);

    // get the ouput object name
    scalarObjName = Covise::get_object_name(portName);
    if (scalarObjName != NULL)
    {
        scalarObj = new coDoFloat(scalarObjName, numCoord);
        if (scalarObj->objectOk())
        {
            scalarObj->getAddress(&s);

            // read the coordinates
            fprintf(stderr, "reading the scalar %d data\n", n);
            for (i = 0; i < numCoord; i++)
            {
                fgets(buf, 300, scalarFp);

                if (feof(scalarFp))
                {
                    Covise::sendError("ERROR: unexpected end of file");
                    return;
                }

                sscanf(buf, "%d%f", &tmpi, s);
                s++;
            }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'scalarObj' failed");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'scalarObj'");
        return;
    }

    fclose(scalarFp);
    delete scalarObj;
}
