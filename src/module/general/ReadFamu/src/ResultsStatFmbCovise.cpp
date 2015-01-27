/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsStatFmbCovise.h
 * a container for results of a time step (=serialized famu results container).
 */

#include "ResultsStatFmbCovise.h" // a container for results of a time step (=serialized famu results container).
#include "errorinfo.h" // a container for error data.
#include <do/coDoData.h>

ResultsStatBinary::ResultsStatBinary(
    ObjectInputStream *archive,
    MeshDataStat *meshDataStat,
    NodeNoMapper *nodeNoMapperResFile,
    int maxNoOfDataTypes,
    OutputHandler *outputHandler,
    std::string bn)
    : ResultsStat(meshDataStat, nodeNoMapperResFile, outputHandler, bn)
{
    readDataTypes(archive, maxNoOfDataTypes);
}

ResultsStatBinary::~ResultsStatBinary()
{
}

bool ResultsStatBinary::readDataTypes(
    ObjectInputStream *archive,
    int maxNoOfDataTypes)
{
    int totalNoOfDataTypes = archive->readINT();
    ASSERT(totalNoOfDataTypes < 10000, _outputHandler);

    INT j;
    for (j = 0; j < totalNoOfDataTypes; j++)
    {
        ResultsFileData::EDimension dim = (ResultsFileData::EDimension)(archive->readINT());
        std::string dtName;
        coDistributedObject *newDt = NULL;
        switch (dim)
        {
        case ResultsFileData::SCALAR:
            readDataTypeScalar(&newDt, dtName, archive);
            break;
        case ResultsFileData::VECTOR:
            readDataTypeVector(&newDt, dtName, archive);
            break;
        default:
            ERROR0("not implemented yet.", _outputHandler);
        }
        if (j < maxNoOfDataTypes)
        {
            _dimension.push_back(dim);
            _dataObjects.push_back(newDt);
            _dataTypeNames.push_back(dtName);
            if (dtName == "Displacements")
            {
                _displacementNo = j;
            }
            _noOfDataTypes++;
        }
        else
        {
            delete newDt;
        }
    }
    return true;
}

void ResultsStatBinary::readDataTypeScalar(
    coDistributedObject **dt,
    std::string &dtName,
    ObjectInputStream *archive)
{
    dtName = archive->readString();

    INT_F *nodeNoArr_f = NULL;
    double *results_f = NULL;
    UINT_F noOfValues = 0;
    archive->readArray(&results_f, (UINT_F *)&noOfValues);
    archive->readArray(&nodeNoArr_f, (UINT_F *)&noOfValues);

    int noOfMeshNodes = _meshDataStat->getNoOfPoints();

    float *results = new float[noOfMeshNodes];
    int i;
    for (i = 0; i < noOfMeshNodes; i++)
    {
        results[i] = 0;
    }
    for (i = 0; i < noOfValues; i++)
    {
        int nodeNoRes = (int)nodeNoArr_f[i]; // internal node number of results file
        int nodeNoMesh = getNodeNoOfMesh(nodeNoRes); // internal node number of mesh file
        if (nodeNoMesh < noOfMeshNodes && nodeNoMesh != -1)
        {
            results[nodeNoMesh] = (float)results_f[i];
        }
        else
        {
            //            std::cout << "!";
        }
    }
    char *name = new char[baseName.length() + dtName.length() + 100];
    sprintf(name, "%s_%s_%d", baseName.c_str(), dtName.c_str(), dataObjectNum);
    dataObjectNum++;
    (*dt) = new coDoFloat(name, noOfMeshNodes, results);
    delete[] name;
    delete[] results;
    delete[] results_f;
    delete[] nodeNoArr_f;
}

void ResultsStatBinary::readDataTypeVector(
    coDistributedObject **dt,
    std::string &dtName,
    ObjectInputStream *archive)
{
    dtName += archive->readString();

    INT_F *nodeNoArr = NULL;
    PointCC *results = NULL;

    UINT_F noOfValues = 0;
    archive->readArray(&results, (UINT_F *)&noOfValues);
    archive->readArray(&nodeNoArr, (UINT_F *)&noOfValues);

    int noOfMeshNodes = _meshDataStat->getNoOfPoints();

    // Auskommentiert fuer Hlrs-Getriebe 17. April 2007
    //    noOfValues = noOfValues > noOfMeshNodes ? noOfMeshNodes : noOfValues;

    float *resultsX = new float[noOfMeshNodes];
    float *resultsY = new float[noOfMeshNodes];
    float *resultsZ = new float[noOfMeshNodes];
    int i;
    for (i = 0; i < noOfMeshNodes; i++)
    {
        resultsX[i] = 0;
        resultsY[i] = 0;
        resultsZ[i] = 0;
    }

#if 1
    int _debug_nodeNoResMin = 0;
    int _debug_nodeNoResMax = 0;

    int _debug_nodeNoMeshMin = 0;
    int _debug_nodeNoMeshMax = 0;
#endif

    for (i = 0; i < noOfValues; i++)
    {
        int nodeNoRes = (int)nodeNoArr[i]; // internal node number of results file
        int nodeNoMesh = getNodeNoOfMesh(nodeNoRes); // internal node number of mesh file

#if 1
        if (_debug_nodeNoResMin < nodeNoRes)
        {
            _debug_nodeNoResMin = nodeNoRes;
        }
        if (_debug_nodeNoResMax > nodeNoRes)
        {
            _debug_nodeNoResMax = nodeNoRes;
        }

        if (_debug_nodeNoMeshMin < nodeNoMesh)
        {
            _debug_nodeNoMeshMin = nodeNoMesh;
        }
        if (_debug_nodeNoMeshMax > nodeNoMesh)
        {
            _debug_nodeNoMeshMax = nodeNoMesh;
        }
#endif

        if (nodeNoMesh < noOfMeshNodes && nodeNoMesh != -1)
        {
            resultsX[nodeNoMesh] = (float)results[i].x;
            resultsY[nodeNoMesh] = (float)results[i].y;
            resultsZ[nodeNoMesh] = (float)results[i].z;
        }
    }
    char *name = new char[baseName.length() + dtName.length() + 100];
    sprintf(name, "%s_%s_%d", baseName.c_str(), dtName.c_str(), dataObjectNum);
    dataObjectNum++;
    (*dt) = new coDoVec3(name, noOfMeshNodes, resultsX, resultsY, resultsZ);
    delete[] name;
    delete[] resultsX;
    delete[] resultsY;
    delete[] resultsZ;
    delete[] results;
    delete[] nodeNoArr;
}
