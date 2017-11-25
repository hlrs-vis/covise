/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshDataTransBinary.h
 * a container for mesh file data where every timestep has its own mesh/binary file format.
 */

#include "MeshDataTransBinary.h" // a container for mesh file data where every timestep has its own mesh/binary file format.
#include "ResultsFileParser.h" // a results file parser.
#include "errorinfo.h" // a container for error data.
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

MeshDataTransBinary::MeshDataTransBinary(
    ObjectInputStream *archive,
    const int &noOfTimeStepsToSkip,
    const int &noOfTimeStepsToParse,
    OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
{

    INT noOfMeshesTotal = (int)archive->readINT();
    ASSERT(noOfMeshesTotal > 0 && noOfMeshesTotal < MAXTIMESTEPS, _outputHandler);
    ASSERT0(noOfTimeStepsToParse > 0, "error: number of time steps to parse from mesh file must be greater zero!", _outputHandler);

    const int minTs = noOfTimeStepsToSkip;
    const int maxTs = min(noOfTimeStepsToParse + noOfTimeStepsToSkip, noOfMeshesTotal);

    int i;
    for (i = 0; i < noOfMeshesTotal; i++)
    {
        MeshDataStat *m = new MeshDataStatBinary(archive, outputHandler);
        if (i >= minTs && i < maxTs)
        {
            _meshData.push_back(m);
        }
    }
    ASSERT0(_meshData.size() > 0, "error reading results file.", _outputHandler);
    outputInfo();
}

MeshDataTransBinary::~MeshDataTransBinary()
{
    int i;
    for (i = 0; i < _meshData.size(); i++)
    {
        MeshDataStat *m = _meshData[i];
        delete m;
    }
}

void MeshDataTransBinary::addMesh(MeshDataStat * /*m*/)
{
    ERROR0("illegal function call.", _outputHandler);
    //_meshData.push_back(m);
}

int MeshDataTransBinary::getNoOfMeshes(void) const
{
    int retval = (int)_meshData.size();
    return retval;
}

MeshDataStat *MeshDataTransBinary::getMeshDataStat(int timeStepNo)
{
    // dodgy
    if (_meshData.size() == 1)
    {
        return _meshData[0]; // statischer Fall/"Displacements"
    }
    else
    {
        ASSERT0(timeStepNo < _meshData.size(), "illegal function call.", _outputHandler);
        return _meshData[timeStepNo];
    }
}

int MeshDataTransBinary::getMaxNodeNo(int timeStepNo) const
{
    ASSERT0(timeStepNo < _meshData.size(), "illegal function call.", _outputHandler);
    int retval = _meshData[timeStepNo]->getMaxNodeNo();
    return retval;
}

void MeshDataTransBinary::outputInfo(void) const
{
    std::ostringstream s;
    s << "mesh data: no. meshes: " << getNoOfMeshes() << ", max node no: ";
    int i;
    for (i = 0; i < getNoOfMeshes(); i++)
    {
        s << getMaxNodeNo(i) << "(" << i << "), ";
    }
    std::string ss = s.str();
    _outputHandler->displayString(ss.c_str());
}
