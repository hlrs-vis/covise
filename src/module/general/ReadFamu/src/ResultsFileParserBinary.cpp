/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileParserBinary.h
 * a results file parser for ASCII files.
 */

#include "ResultsFileParserBinary.h" // a results file parser for ASCII files.
#include "errorinfo.h" // a container for error data.
#include "objectinputstreambinary.hxx" // a stream that can be used for deserialization/binary files.
#include "errorinfo.h" // a container for error data.
#include <do/coDoData.h>

ResultsFileParserBinary::ResultsFileParserBinary(OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
    , _totalNoOfTimeSteps(0)
    , _noOfTimeSteps(0)
{
}

ResultsFileParserBinary::~ResultsFileParserBinary()
{
    INT i;
    for (i = 0; i < _results.size(); i++)
    {
        delete _results[i];
    }
}

void ResultsFileParserBinary::parseResultsFile(std::string filename,
                                               const int &noOfTimeStepsToSkip,
                                               const int &noOfTimeStepsToParse,
                                               MeshDataTrans *meshDataTrans,
                                               std::string baseName)
{
    ObjectInputStreamBinary instream(filename, _outputHandler);
    readTimeSteps(&instream, noOfTimeStepsToSkip, noOfTimeStepsToParse, meshDataTrans, baseName);
    // debug();
    outputInfo();
}

void ResultsFileParserBinary::outputInfo(void)
{
    std::ostringstream s;
    s << "results data: no. data: " << getNoOfTimeSteps();
    std::string ss = s.str();
    _outputHandler->displayString(ss.c_str());
    int i;
    for (i = 0; i < getNoOfTimeSteps(); i++)
    {
        std::ostringstream s;
        s << "time step " << i << ": no. values: ";
        int j;
        for (j = 0; j < getNoOfDataTypes(i); j++)
        {
            int noOfValues = 0;
            coDistributedObject *dat = getDataObject(j, i);
            ResultsFileData::EDimension dim = getDimension(i, j);
            if (dim == ResultsFileData::SCALAR)
            {
                noOfValues = ((coDoFloat *)dat)->getNumPoints();
            }
            else
            {
                noOfValues = ((coDoVec3 *)dat)->getNumPoints();
            }
            s << noOfValues << "(" << j << "), ";
        }
        ss = s.str();
        _outputHandler->displayString(ss.c_str());
    }
}

// --------------------------------------------------------------------------------------------------------------
//                                          getter methods
// --------------------------------------------------------------------------------------------------------------

int ResultsFileParserBinary::getNoOfTimeSteps(void) const
{
    return _noOfTimeSteps;
}

std::string ResultsFileParserBinary::getDataTypeName(int timeStepNo, int dataTypeNo)
{
    ASSERT(timeStepNo < _noOfTimeSteps, _outputHandler);
    return _results[timeStepNo]->getDataTypeName(dataTypeNo);
}

int ResultsFileParserBinary::getNoOfDataTypes(int timeStepNo) const
{
    ASSERT(timeStepNo < _noOfTimeSteps, _outputHandler);
    return _results[timeStepNo]->getNoOfDataTypes();
}

coDistributedObject *ResultsFileParserBinary::getDataObject(int dataTypeNo, int timeStepNo)
{
    ASSERT(timeStepNo < _noOfTimeSteps, _outputHandler);
    return _results[timeStepNo]->getDataObject(dataTypeNo);
}

bool ResultsFileParserBinary::getDisplacements(int timeStepNo,
                                               float *dx[], float *dy[], float *dz[])
{
    ASSERT(timeStepNo < _noOfTimeSteps, _outputHandler);
    return _results[timeStepNo]->getDisplacements(dx, dy, dz);
}

ResultsFileData::EDimension ResultsFileParserBinary::getDimension(int timeStepNo, int dataTypeNo)
{
    ASSERT(timeStepNo < _noOfTimeSteps, _outputHandler);
    return _results[timeStepNo]->getDimension(dataTypeNo);
}

// --------------------------------------------------------------------------------------------------------------
//                                          debugging
// --------------------------------------------------------------------------------------------------------------

void ResultsFileParserBinary::debug(void)
{
    std::ofstream fileHandle("content_binary.txt");
    fileHandle << "_noOfTimeSteps = " << _noOfTimeSteps << "\n";

    INT i;
    for (i = 0; i < _noOfTimeSteps; i++)
    {
        fileHandle << "----------------- time step " << i << " ----------------------\n";
        _results[i]->debug(fileHandle);
    }
    fileHandle.close();
}
