/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileData.h
 * a container for results file data.
 */

//#include "ResultsFileData.h"  // a container for results file data.

#ifndef __resultsfiledata_h__
#define __resultsfiledata_h__

#include <api/coModule.h>
using namespace covise;

/**
 * a container for results file data.
 */
class ResultsFileData
{
public:
    ResultsFileData(){};
    virtual ~ResultsFileData(){};

    enum EDimension
    {
        SCALAR,
        VECTOR
    };

    virtual int getNoOfTimeSteps(void) const = 0;
    virtual int getNoOfDataTypes(int timeStepNo) const = 0; // ! number of DataTypes must be the same in all time steps !

    virtual EDimension getDimension(int timeStepNo, int dataTypeNo) = 0;
    virtual std::string getDataTypeName(int timeStepNo, int dataTypeNo) = 0;

    virtual bool getDisplacements(int timeStepNo,
                                  float *dx[], float *dy[], float *dz[]) = 0;

    virtual coDistributedObject *getDataObject(int dataTypeNo, int timeStepNo) = 0;
};

#endif
