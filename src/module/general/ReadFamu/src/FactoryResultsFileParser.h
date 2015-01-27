/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file factoryresultsfileconverter.hxx
 * a factory for results file parsers.
 */

// #include "FactoryResultsFileParser.h" // a factory for results file parsers.

#ifndef __FactoryResultsFileParser_h__
#define __FactoryResultsFileParser_h__

#include "ResultsFileParser.h" // a results file parser.
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include "MeshDataStat.h" // a container for stationary mesh data.

/** 
 * a factory for results file converters.
 */
class FactoryResultsFileParser
{
public:
    static FactoryResultsFileParser *getInstance(void);
    static void deleteInstance(void);

    virtual ResultsFileParser *create(std::string filename,
                                      MeshDataTrans *meshDataTrans,
                                      OutputHandler *outputHandler);

private:
    static FactoryResultsFileParser *_instance;
    FactoryResultsFileParser(void){};
    virtual ~FactoryResultsFileParser()
    {
        _instance = NULL;
    };
};

#endif // __factoryresultsfileconverter_hxx__
