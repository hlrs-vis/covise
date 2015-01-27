/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file PeriodicRotMesh.h
 * completes meshes to a full 360° rotation.
 * rotation axís has to be z-axis.
 */

#include "PeriodicRotMesh.h" // completes results/meshes to a full 360° rotation.
#include "errorinfo.h" // a container for error data.
#include "coordconv.hxx" // conversion between coordinate systems.
#include "MeshDataStatBinary.h" // a container for mesh data/binary file format.
#include <math.h>
#include <cstdlib>

#define PI 3.14159265359

PeriodicRotMesh::PeriodicRotMesh(
    double symmAngleDeg,
    int noOfStepsPerBlock,
    MeshDataTrans *originalData,
    OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
    , _noOfStepsPerBlock(noOfStepsPerBlock)
    , _symmAngle((symmAngleDeg / 360) * 2 * PI)
    , _originalData(originalData)
    , _noOfBlocks(0)
    , _noOfTimeSteps(0)
{
    ASSERT(symmAngleDeg <= 180, _outputHandler)
    ASSERT(_noOfStepsPerBlock > 0, _outputHandler)
    ASSERT(fabs(((int)(360 / symmAngleDeg) - (360 / symmAngleDeg))) < 1e-6, _outputHandler);

    _noOfBlocks = (int)(2 * PI / _symmAngle);
    _noOfTimeSteps = _noOfBlocks * _noOfStepsPerBlock;
}

MeshDataStat *PeriodicRotMesh::getMeshDataStat(int timeStepNo)
{
    MeshDataStat *retval = NULL;
    if (timeStepNo < _noOfStepsPerBlock)
    {
        retval = _originalData->getMeshDataStat(timeStepNo);
    }
    else
    {
        int blockNo = timeStepNo / _noOfStepsPerBlock;
        double angle = blockNo * _symmAngle;
        retval = getRotatedMesh(timeStepNo % _noOfStepsPerBlock, angle);
    }
    return retval;
}

int PeriodicRotMesh::getOriginalTimeStepNo(int timeStepNo) const
{
    int retval = 0;
    if (timeStepNo < _noOfStepsPerBlock)
    {
        retval = timeStepNo;
    }
    else
    {
        retval = timeStepNo % _noOfStepsPerBlock;
    }
    return retval;
}

MeshDataStat *PeriodicRotMesh::getRotatedMesh(int timeStepNo,
                                              double angle)
{
    MeshDataStat *origDat = _originalData->getMeshDataStat(timeStepNo);
    int noOfPoints = 0;
    int noOfElements = 0;
    int noOfVertices = 0;
    int *elementsArr = NULL;
    int *verticesArr = NULL;
    int *typesArr = NULL;
    float *xPointsArr = NULL;
    float *yPointsArr = NULL;
    float *zPointsArr = NULL;
    origDat->getMeshData(&noOfElements, &noOfVertices, &noOfPoints,
                         &elementsArr, &verticesArr,
                         &xPointsArr, &yPointsArr, &zPointsArr,
                         &typesArr);
    // copy arrays
    float *xPointsCopy = new float[noOfPoints];
    float *yPointsCopy = new float[noOfPoints];
    float *zPointsCopy = new float[noOfPoints];
    int i;
    for (i = 0; i < noOfPoints; i++)
    {
        xPointsCopy[i] = xPointsArr[i];
        yPointsCopy[i] = yPointsArr[i];
        zPointsCopy[i] = zPointsArr[i];
    }
    //    int* verticesCopy = new int[noOfVertices];    // crashes if out of mem
    //    int* elementsCopy = new int[noOfElements];
    //    int* typesCopy = new int[noOfElements];
    int *verticesCopy = (int *)malloc(sizeof(int) * noOfVertices);
    int *elementsCopy = (int *)malloc(sizeof(int) * noOfElements);
    int *typesCopy = (int *)malloc(sizeof(int) * noOfElements);
    ASSERT0(verticesCopy != NULL && elementsCopy != NULL && typesCopy != NULL, "error: out of memory.", _outputHandler);

    for (i = 0; i < noOfVertices; i++)
    {
        verticesCopy[i] = verticesArr[i];
    }

    for (i = 0; i < noOfElements; i++)
    {
        elementsCopy[i] = elementsArr[i];
        typesCopy[i] = typesArr[i];
    }

    // rotate grid
    for (i = 0; i < noOfPoints; i++)
    {
        PointCC r = { xPointsCopy[i], yPointsCopy[i], zPointsCopy[i] };
        r = rotateVector(r, angle);
        xPointsCopy[i] = r.x;
        yPointsCopy[i] = r.y;
        zPointsCopy[i] = r.z;
    }

    // copy mapper array
    int maxNodeNoMesh = _originalData->getMaxNodeNo(timeStepNo);
    int *mesh2internal = new int[maxNodeNoMesh + 1];
    for (i = 0; i <= maxNodeNoMesh; i++)
    {
        mesh2internal[i] = origDat->getInternalNodeNo(i);
    }

    MeshDataStat *retval = new MeshDataStatBinary(noOfElements, noOfVertices, noOfPoints, elementsCopy,
                                                  verticesCopy, xPointsCopy, yPointsCopy, zPointsCopy,
                                                  typesCopy, mesh2internal, maxNodeNoMesh, _outputHandler);
    return retval;
}

PointCC PeriodicRotMesh::rotateVector(
    const PointCC &v,
    double angle) const
{
    double rho, phi, z;
    coordConv::kart2zy(v.x, v.y, v.z, rho, phi, z);

    PointCC retval = v;
    phi += angle;
    coordConv::zy2kart(rho, phi, z, retval.x, retval.y, retval.z);
    return retval;
}

// --------------------------------- trivial getter methods ---------------------------------

void PeriodicRotMesh::addMesh(MeshDataStat *)
{
    ERROR0("illegal function call.", _outputHandler);
}

int PeriodicRotMesh::getNoOfMeshes(void) const
{
    return _noOfTimeSteps;
}

int PeriodicRotMesh::getMaxNodeNo(int timeStepNo) const
{
    return _originalData->getMaxNodeNo(timeStepNo % _noOfStepsPerBlock);
}
