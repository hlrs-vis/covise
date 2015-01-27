/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileParserBinary.h
 * a results file parser for binary files.
 */

// #include "ResultsFileParserBinary.h"  // a results file parser for binary files.

#ifndef __resultsfileparserbinary_h__
#define __resultsfileparserbinary_h__

#include "ResultsFileParser.h" // a results file parser.
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include "objectinputstream.hxx" // a stream that can be used for deserialization.
#include "ResultsStat.h" // a container for stationary mesh data.
#include <vector>

/**
 * a results file parser for binary files.
 */
class ResultsFileParserBinary : public ResultsFileParser
{
public:
    ResultsFileParserBinary(OutputHandler *outputHandler);
    virtual ~ResultsFileParserBinary();

    virtual void parseResultsFile(std::string filename,
                                  const int &noOfTimeStepsToSkip,
                                  const int &noOfTimestepsToParse,
                                  MeshDataTrans *meshDataTrans,
                                  std::string baseName);

    // inherited getter methods
    virtual int getNoOfTimeSteps(void) const;
    virtual ResultsFileData::EDimension getDimension(int timeStepNo, int dataTypeNo);
    virtual std::string getDataTypeName(int timeStepNo, int dataTypeNo);
    virtual int getNoOfDataTypes(int timeStepNo) const;
    coDistributedObject *getDataObject(int dataTypeNo, int timeStepNo);

    virtual bool getDisplacements(int timeStepNo,
                                  float *dx[], float *dy[], float *dz[]);

    virtual void debug(void);

protected:
    OutputHandler *_outputHandler;

    int _totalNoOfTimeSteps;
    int _noOfTimeSteps;
    std::vector<std::string> _timeStepNames;

    std::vector<ResultsStat *> _results;

    virtual bool readTimeSteps(ObjectInputStream *archive,
                               int noOfTimeStepsToSkip,
                               int noOfTimeStepsToParse,
                               MeshDataTrans *meshDataTrans,
                               std::string baseName) = 0;

    void readNodesProblem(int *maxNodeNoGeoRes,
                          int *nodeNo2GeoRes[],
                          ObjectInputStream *archive) const;

    void outputInfo(void);
};

#endif
