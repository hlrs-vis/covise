/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileParserCovise.h
 * a results file parser for Covise *.cvr files.
 */

// #include "ResultsFileParserCovise.h"  // a results file parser for Covise *.cvr results files.

#ifndef __ResultsFileParserCovise_h__
#define __ResultsFileParserCovise_h__

#include "ResultsFileParserBinary.h" // a results file parser for binary files.

/**
 * a results file parser for Covise *.cvr files.
 */
class ResultsFileParserCovise : public ResultsFileParserBinary
{
public:
    ResultsFileParserCovise(OutputHandler *o)
        : ResultsFileParserBinary(o){};
    virtual ~ResultsFileParserCovise(){};

protected:
    virtual bool readTimeSteps(ObjectInputStream *archive,
                               int noOfTimeStepsToSkip,
                               int noOfTimeStepsToParse,
                               MeshDataTrans *meshDataTrans,
                               std::string bn);
};

#endif
