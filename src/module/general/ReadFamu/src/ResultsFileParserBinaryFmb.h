/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileParserBinaryFmb.h
 * a results file parser for Famu *.fmb results files.
 */

// #include "ResultsFileParserBinaryFmb.h"  // a results file parser for Famu *.fmb results files.

#ifndef __ResultsFileParserBinaryFmb_h__
#define __ResultsFileParserBinaryFmb_h__

#include "ResultsFileParserBinary.h" // a results file parser for binary files.

/**
 * a results file parser for Famu *.fmb results files.
 */
class ResultsFileParserBinaryFmb : public ResultsFileParserBinary
{
public:
    ResultsFileParserBinaryFmb(OutputHandler *o)
        : ResultsFileParserBinary(o){};
    virtual ~ResultsFileParserBinaryFmb(){};

protected:
    virtual bool readTimeSteps(ObjectInputStream *archive,
                               int noOfTimeStepsToSkip,
                               int noOfTimeStepsToParse,
                               MeshDataTrans *meshDataTrans,
                               std::string baseName);
};

#endif
