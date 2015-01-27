/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************************\
 **                                                                    (C)2009   **
 **                                                                              **
 ** Description: VTK data reading Module                                         **
 **              reads data in vtk format                                        **
 **              either a single file or a set of files (different timesteps)    **
 **                                                                              **
 ** Name:        ReadSIM                                                    **
 ** Category:    IO                                                       **
 **                                                                              **
 ** Author:      Julia Portl                                                     **
 **              Visualization and Numerical Geometry Group, IWR                 **
 **              Heidelberg University                                           **
 **                                                                              **
 ** History:     April 2009                                                      **

 Modified by Daniel Jungblut: G-CSC Frankfurt University
 October 2009



 **                                                                              **
 **                                                                              **
 \**********************************************************************************/

#include "ReadSIM2.h"

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

ReadSIM2::ReadSIM2(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Reads (a set) of VTK-files", false)
{

    GridOutPort = addOutputPort("Grid_Set", "UnstructuredGrid",
                                "Set of unstructured grids");

    ScalarOutPort = addOutputPort("ScalarData_Set", "Float",
                                  "Set of Scalar Data at vertex");

    fileBrowser = addFileBrowserParam("SIM_file", "Select sim file");
    fileBrowser->setValue("~/", "*.sim");

    maxTimeSteps = addInt32Param("maxTimesteps",
                                 "sets the maximum number of timesteps to process.");
    maxTimeSteps->setValue(5);

    processAllTimeSteps = addBooleanParam("ignoreMaxTimeSteps",
                                          "process all timesteps found in source file?");
}

int ReadSIM2::compute(const char *)
{

    char str[255];

    const char *fileName;
    fileName = fileBrowser->getValue();
    sendInfo("configured timesteps: %ld", maxTimeSteps->getValue());

    coDoUnstructuredGrid *gridObject = NULL;
    coDoUnstructuredGrid **timeOutputGrid = NULL;
    coDoSet *gridSet = NULL;
    const char *gridSetName = GridOutPort->getObjName();

    coDoFloat **scalarObject = NULL;
    coDoSet *scalarSet = NULL;
    const char *scalarSetName = ScalarOutPort->getObjName();

    FILE *fp;
    if ((fp = fopen(fileName, "rb")) == NULL)
    {

        sendError("Problems, while loading file %s", fileName);
        return STOP_PIPELINE;
    }

    int points = 0;
    int elements = 0;
    int corners = 0;
    int types = 0;
    int timesteps = 0;

    fscanf(fp, "%s %i", str, &points);
    fscanf(fp, "%s %i", str, &elements);
    fscanf(fp, "%s %i", str, &corners);
    fscanf(fp, "%s %i", str, &types);
    fscanf(fp, "%s %i", str, &timesteps);

    sendInfo("Points: %i\n", points);
    sendInfo("Elements: %i\n", elements);
    sendInfo("Corners: %i\n", corners);
    sendInfo("Types: %i\n", types);

    // prozessiere alle Zeitschritte (checkbox nicht angeklickt)
    if (processAllTimeSteps->getValue() == FALSE)
    {
        timesteps = maxTimeSteps->getValue();
    }

    sendInfo("Timesteps: %i\n", timesteps);

    // NULL TERMINATED:
    timeOutputGrid = new coDoUnstructuredGrid *[timesteps];
    timeOutputGrid[timesteps] = NULL;

    char gridName[255];
    sprintf(gridName, "%s%i", gridSetName, 0);

    gridObject = new coDoUnstructuredGrid(gridName, elements, corners, points,
                                          types);

    float *pointsX;
    float *pointsY;
    float *pointsZ;
    int *elementList;
    int *cornerList;
    int *typeList;

    gridObject->getAddresses(&elementList, &cornerList, &pointsX, &pointsY,
                             &pointsZ);
    gridObject->getTypeList(&typeList);

    // Ein Leerzeichen überspringen:
    fseek(fp, 1, SEEK_CUR);

    fread(pointsX, sizeof(float), points, fp);
    fread(pointsY, sizeof(float), points, fp);
    fread(pointsZ, sizeof(float), points, fp);

    sendInfo("Points loaded");

    printf("First point: %f %f %f\n", pointsX[0], pointsY[0], pointsZ[0]);

    fread(elementList, sizeof(int), elements, fp);

    sendInfo("Elements loaded");

    fread(cornerList, sizeof(int), corners, fp);

    sendInfo("Corners loaded");

    fread(typeList, sizeof(int), types, fp);

    sendInfo("Types loaded");

    for (int t = 0; t < timesteps; t++)
    {
        timeOutputGrid[t] = gridObject;
        if (t > 0)
        {
            gridObject->incRefCount();
        }
    }

    // Allokiere Null-terminiertes Feld:
    scalarObject = new coDoFloat *[timesteps + 1];
    scalarObject[timesteps] = NULL;

    float *data = new float[points];

    for (int t = 0; t < timesteps; t++)
    {

        char scalarName[255];
        sprintf(scalarName, "%s%i", scalarSetName, t);

        //		printf("%s\n", scalarName);

        fread(data, sizeof(float), points, fp);

        scalarObject[t] = new coDoFloat(scalarName, points, data);

        sendInfo("Timestep %i loaded", t);
    }

    delete[] data;

    fclose(fp);

    gridSet = new coDoSet(gridSetName, (coDistributedObject **)timeOutputGrid);

    scalarSet
        = new coDoSet(scalarSetName, (coDistributedObject **)scalarObject);

    char attribValue[32];
    sprintf(attribValue, "1 %d", timesteps);

    gridSet->addAttribute("TIMESTEP", attribValue);
    scalarSet->addAttribute("TIMESTEP", attribValue);

    // es macht keinen unterschied, ob man das zeitabhängige GitterSet an den Ausgang legt
    // oder direkt das einzelne GridObject.
    GridOutPort->setCurrentObject(gridObject); //????
    ScalarOutPort->setCurrentObject(scalarSet);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Testing, ReadSIM2)
