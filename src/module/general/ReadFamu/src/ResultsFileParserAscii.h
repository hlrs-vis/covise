/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileParserAscii.h
 * a results file parser for ASCII files.
 */

// #include "ResultsFileParserAscii.h"  // a results file parser for ASCII files.

#ifndef __resultsfileparserascii_h__
#define __resultsfileparserascii_h__

#include "ResultsFileParser.h" // a results file parser.
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include <vector>

#define LINE_SIZE 1024

/**
 * a results file parser for ASCII files.
 */
class ResultsFileParserAscii : public ResultsFileParser
{
public:
    ResultsFileParserAscii(OutputHandler *outputHandler);
    virtual ~ResultsFileParserAscii(){};

    virtual void parseResultsFile(std::string filename,
                                  const int &noOfTimeStepToSkip,
                                  const int &noOfTimeStepsToParse,
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

private:
    OutputHandler *_outputHandler;
    std::vector<coDistributedObject *> _dataObjects[NUMRES];
    std::vector<ResultsFileData::EDimension> _dimension[NUMRES];
    std::vector<int> _displacementNo; // _displacementNos[i] indicates the displacement DataType no. of time step i; if no timestep availabe -> -1

    FILE *_file;
    char _line[LINE_SIZE];
    char *_c;

    int _maxNodeNo;
    int _noOfTimeSteps;
    int _currentDataTypeNo;

    char *_names[NUMRES]; // DataType names

    std::vector<float> _xData;
    std::vector<float> _yData;
    std::vector<float> _zData;

    // -------- stuff needed during parsing ---------
    void getLine();
    void pushBack(); // push back this _line (don´t read a new on next time)
    bool _pushedBack;

    bool _doSkip;
    int _skipped;

    coDistributedObject *readVec(const char *name, int timeStepNo);
    coDistributedObject *readScal(const char *name);
};

#endif
