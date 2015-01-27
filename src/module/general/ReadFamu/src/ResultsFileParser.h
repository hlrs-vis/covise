/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileParser.h
 * a results file parser.
 */

//#include "ResultsFileParser.h"  // a results file parser.

#ifndef __resultsfileparser_h__
#define __resultsfileparser_h__

#include "ResultsFileData.h" // a container for results file data.
#include "MeshDataTrans.h" // a container for mesh file data where every timestep has its own mesh.
#include <api/coModule.h>
using namespace covise;

#define NUMRES 14
#define MAXTIMESTEPS 2048

/**
 * a results file parser.
 */
class ResultsFileParser : public ResultsFileData
{
public:
    ResultsFileParser(){};
    virtual ~ResultsFileParser(){};

    virtual void parseResultsFile(std::string filename,
                                  const int &noOfTimestepToSkip,
                                  const int &noOfTimestepsToParse,
                                  MeshDataTrans *meshDataTrans,
                                  std::string baseName) = 0;

    virtual void debug(void) = 0;
};

#endif
