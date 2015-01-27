/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file BuilderGrid.h
 * a builder for grids.
 */

#include <do/coDoUnstructuredGrid.h>
#include "BuilderGrid.h" // a builder for grids.
#include "errorinfo.h" // a container for error data.
#include "PeriodicRotResults.h" // completes results/meshes to a full 360° rotation.
#include "PeriodicRotMesh.h" // completes meshes to a full 360° rotation.

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

BuilderGrid::BuilderGrid(OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
{
}

coDoSet *BuilderGrid::construct(const char *meshName,
                                bool scaleDisplacements,
                                float symmAngle,
                                int noOfSymmTimeSteps,
                                MeshDataTrans *meshDataTrans,
                                ResultsFileData *resFileData)
{
    bool isPeriodicRot = symmAngle != 0 && noOfSymmTimeSteps > 0;
    coDoSet *retval = NULL;
    if (isPeriodicRot)
    {
        PeriodicRotResults resFileDataRot(symmAngle, noOfSymmTimeSteps, resFileData, _outputHandler);
        PeriodicRotMesh meshDataTransRot(symmAngle, noOfSymmTimeSteps, meshDataTrans, _outputHandler);
        retval = constructMultipleMeshes(meshName, scaleDisplacements, &meshDataTransRot, &resFileDataRot, symmAngle, noOfSymmTimeSteps);
    }
    else if (meshDataTrans->getNoOfMeshes() == 1)
    {
        retval = constructSingleMesh(meshName, scaleDisplacements, meshDataTrans, resFileData, symmAngle, noOfSymmTimeSteps);
    }
    else
    {
        retval = constructMultipleMeshes(meshName, scaleDisplacements, meshDataTrans, resFileData, symmAngle, noOfSymmTimeSteps);
    }
    return retval;
}

/** 
 * returns all grids of all timesteps.
 */
coDoSet *BuilderGrid::constructSingleMesh(
    const char *meshName,
    bool scaleDisplacements,
    MeshDataTrans *meshDataTrans,
    ResultsFileData *resFileData,
    float symmAngle,
    int noOfSymmTimeSteps)
{
    // ---------------------  create Grid of timestep 0 ---------------------

    MeshDataStat *meshDataStat = meshDataTrans->getMeshDataStat(0);
    int noOfPoints = 0;
    int noOfElements = 0;
    int noOfVertices = 0;
    int *elementsArr = NULL;
    int *verticesArr = NULL;
    int *typesArr = NULL;
    float *xPointsArr = NULL;
    float *yPointsArr = NULL;
    float *zPointsArr = NULL;
    meshDataStat->getMeshData(&noOfElements, &noOfVertices, &noOfPoints,
                              &elementsArr, &verticesArr,
                              &xPointsArr, &yPointsArr, &zPointsArr,
                              &typesArr);

    std::ostringstream s;
    s << "no. of nodes: " << noOfPoints;
    s << ", no. of elements: " << noOfElements;
    std::string ss = s.str();
    _outputHandler->displayString(ss.c_str());

    // create grid of time step 0
    char buf[1000];
    sprintf(buf, "%s_Grid", meshName);
    coDoUnstructuredGrid *gridOfTimeStep0 = NULL;
    gridOfTimeStep0 = new coDoUnstructuredGrid(buf, noOfElements, noOfVertices, noOfPoints, elementsArr, verticesArr, xPointsArr, yPointsArr, zPointsArr, typesArr);

    // ----------------------------------------------------------------------

    int noOfTimeSteps = resFileData->getNoOfTimeSteps();
    if (noOfTimeSteps == 0)
    {
        noOfTimeSteps = 1;
    }

    // create following grids
    coDistributedObject **gridsOfTimestepsArr = new coDistributedObject *[noOfTimeSteps + 1];
    gridsOfTimestepsArr[0] = gridOfTimeStep0;
    int i;
    for (i = 1; i < noOfTimeSteps; i++)
    {
        coDoUnstructuredGrid *gridOfTimeStep = NULL;
        float *dx, *dy, *dz;
        if (resFileData->getDisplacements(i, &dx, &dy, &dz))
        {
            int n;
            for (n = 0; n < noOfPoints; n++) // displacements are relative to the original point, so add that to the displacements
            {
                if (scaleDisplacements)
                {
                    dx[n] = dx[n] * 1000 + xPointsArr[n];
                    dy[n] = dy[n] * 1000 + yPointsArr[n];
                    dz[n] = dz[n] * 1000 + zPointsArr[n];
                }
                else
                {
                    dx[n] += xPointsArr[n];
                    dy[n] += yPointsArr[n];
                    dz[n] += zPointsArr[n];
                }
            }
            sprintf(buf, "%s_%d_Grid", meshName, i);
            gridOfTimeStep = new coDoUnstructuredGrid(buf, noOfElements, noOfVertices, noOfPoints, elementsArr, verticesArr, dx, dy, dz, typesArr);
        }
        else
        {
            gridOfTimeStep = gridOfTimeStep0;
            gridOfTimeStep->incRefCount();
        }
        gridsOfTimestepsArr[i] = gridOfTimeStep;
    }
    gridsOfTimestepsArr[noOfTimeSteps] = NULL;
    coDoSet *gridSet = new coDoSet(meshName, gridsOfTimestepsArr);

    if (noOfTimeSteps > 1)
    {
        gridSet->addAttribute("TIMESTEP", "1 10");
        if (symmAngle > 0 && noOfSymmTimeSteps == 0)
        {
            int num = static_cast<int>(360.0 / symmAngle);
            char attrValue[200];
            sprintf(attrValue, "%d %f", num - 1, symmAngle);
            gridSet->addAttribute("MULTIROT", attrValue);
        }
    }
    delete[] gridsOfTimestepsArr;
    return gridSet;
}

/** 
 * returns all grids of all timesteps.
 * each time step has a different mesh.
 */
coDoSet *BuilderGrid::constructMultipleMeshes(
    const char *meshName,
    bool, //*scaleDisplacements,
    MeshDataTrans *meshDataTrans,
    ResultsFileData *resFileData,
    float symmAngle,
    int noOfSymmTimeSteps)
{
    char buf[1000];
    sprintf(buf, "%s_Grid", meshName);

    int noOfTimeStepsRes = resFileData->getNoOfTimeSteps();
    int noOfTimeStepsMesh = meshDataTrans->getNoOfMeshes();
    int noOfTimeSteps = min(noOfTimeStepsRes, noOfTimeStepsMesh);
    ASSERT(noOfTimeSteps > 0, _outputHandler);

    coDistributedObject **gridsOfTimestepsArr = new coDistributedObject *[noOfTimeSteps + 1];
    int i;
    for (i = 0; i < noOfTimeSteps; i++)
    {
        MeshDataStat *meshDataStat = meshDataTrans->getMeshDataStat(i);
        int noOfPoints = 0;
        int noOfElements = 0;
        int noOfVertices = 0;
        int *elementsArr = NULL;
        int *verticesArr = NULL;
        int *typesArr = NULL;
        float *xPointsArr = NULL;
        float *yPointsArr = NULL;
        float *zPointsArr = NULL;
        meshDataStat->getMeshData(&noOfElements, &noOfVertices, &noOfPoints,
                                  &elementsArr, &verticesArr,
                                  &xPointsArr, &yPointsArr, &zPointsArr,
                                  &typesArr);

        sprintf(buf, "%s_%d_Grid", meshName, i);
        gridsOfTimestepsArr[i] = new coDoUnstructuredGrid(buf, noOfElements, noOfVertices, noOfPoints,
                                                          elementsArr, verticesArr,
                                                          xPointsArr, yPointsArr, zPointsArr, typesArr);
    }
    gridsOfTimestepsArr[noOfTimeSteps] = NULL;
    coDoSet *gridSet = new coDoSet(meshName, gridsOfTimestepsArr);

    if (noOfTimeSteps > 1)
    {
        gridSet->addAttribute("TIMESTEP", "1 10");
        if (symmAngle > 0 && noOfSymmTimeSteps == 0)
        {
            int num = static_cast<int>(360.0 / symmAngle);
            char attrValue[200];
            sprintf(attrValue, "%d %f", num - 1, symmAngle);
            gridSet->addAttribute("MULTIROT", attrValue);
        }
    }
    delete[] gridsOfTimestepsArr;
    return gridSet;
}
