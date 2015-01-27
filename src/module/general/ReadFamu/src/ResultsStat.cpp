/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsStat.h
 * a container for results of a time step.
 */

#include "ResultsStat.h" // a container for stationary mesh data.
#include "errorinfo.h" // a container for error data.
#include <do/coDoData.h>

int ResultsStat::dataObjectNum = 0;
ResultsStat::ResultsStat(MeshDataStat *meshDataStat,
                         NodeNoMapper *nodeNoMapperResFile,
                         OutputHandler *outputHandler,
                         std::string bn)
    : _outputHandler(outputHandler)
    , _meshDataStat(meshDataStat)
    , _nodeNoMapperResFile(nodeNoMapperResFile)
    , _noOfDataTypes(0)
    , _displacementNo(-1)
{
    baseName = bn;
}

ResultsStat::~ResultsStat()
{
    //_nodeNoMapperResFile->deleteInstance();   _nodeNoMapperResFile is deleted in derived classes
    //_dataObjects must not be deleted in destructor!
}

// maps node numbers from internal/results file --> internal/mesh file.
// returns -1 if mesh node no. is not available (happens when elements/nodes are missing in the mesh file)
int ResultsStat::getNodeNoOfMesh(int internalRes) const
{
    const int meshRes = _nodeNoMapperResFile->getMeshNodeNo(internalRes);
    const int internalMesh = _meshDataStat->getInternalNodeNo(meshRes); // returns -1 if meshNodeNo is too high.
    return internalMesh;
}

// -----------------------------------------------------------------------------------------------------------
//                                      debugging stuff
// -----------------------------------------------------------------------------------------------------------

void ResultsStat::debug(std::ofstream &f)
{
    f << "_noOfDataTypes = " << _noOfDataTypes << "\n";
    f << "_displacementNo = " << _noOfDataTypes << "\n";

    INT j;
    for (j = 0; j < _noOfDataTypes; j++)
    {
        if (_dimension[j] == ResultsFileData::SCALAR)
        {
            coDoFloat *dataObj = (coDoFloat *)_dataObjects[j];
            float *valuesArr;
            dataObj->getAddress(&valuesArr);
            int noOfPoints = dataObj->getNumPoints();

            f << "_dataTypeNames = " << _dataTypeNames[j] << "\n";
            INT k;
            for (k = 0; k < 5; k++)
            {
                f << k << "\t" << valuesArr[k] << "\n";
            }
            f << "...\n";
            int z = 0;
            for (k = 5; k < noOfPoints - 5; k++)
            {
                if (fabs(valuesArr[k]) > 1e-6 && z < 5)
                {
                    f << k << "\t" << valuesArr[k] << "\n";
                    z++;
                }
            }
            f << "...\n";
            for (k = noOfPoints - 5; k < noOfPoints; k++)
            {
                f << k << "\t" << valuesArr[k] << "\n";
            }
        }
        else
        {
            coDoVec3 *dataObj = (coDoVec3 *)_dataObjects[j];
            float *x, *y, *z;
            dataObj->getAddresses(&x, &y, &z);
            int noOfPoints = dataObj->getNumPoints();

            f << "_dataTypeNames = " << _dataTypeNames[j] << "\n";
            INT k;
            for (k = 0; k < 5; k++)
            {
                f << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
            }
            f << "...\n";
            int n = 0;
            for (k = 5; k < noOfPoints - 5; k++)
            {
                if (fabs(x[k]) > 1e-6 && n < 5)
                {
                    f << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
                    n++;
                }
            }
            f << "...\n";
            for (k = noOfPoints - 5; k < noOfPoints; k++)
            {
                f << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
            }
        }
    }
}

// --------------------------------------------------------------------------------------------------------------
//                                          getter methods
// --------------------------------------------------------------------------------------------------------------

std::string ResultsStat::getDataTypeName(int dataTypeNo)
{
    ASSERT(dataTypeNo < _noOfDataTypes, _outputHandler);
    std::string retval = _dataTypeNames[dataTypeNo];
    return retval;
}

int ResultsStat::getNoOfDataTypes(void) const
{
    return _noOfDataTypes;
}

coDistributedObject *ResultsStat::getDataObject(int dataTypeNo)
{
    ASSERT(dataTypeNo < _noOfDataTypes, _outputHandler);
    coDistributedObject *retval = _dataObjects[dataTypeNo];
    return retval;
}

bool ResultsStat::getDisplacements(float *dx[], float *dy[], float *dz[])
{
    bool retval = false;
    if (_displacementNo == -1)
    {
        retval = false;
    }
    else
    {
        *dx = *dy = *dz = NULL;
        coDoVec3 *dataObj = (coDoVec3 *)_dataObjects[_displacementNo];
        dataObj->getAddresses(dx, dy, dz);
        retval = true;
    }
    return retval;
}

ResultsFileData::EDimension ResultsStat::getDimension(int dataTypeNo)
{
    ASSERT(dataTypeNo < _noOfDataTypes, _outputHandler);
    ResultsFileData::EDimension retval;
    retval = _dimension[dataTypeNo];
    return retval;
}
