/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file factoryresultsfileconverter.hxx
 * a factory for results file parsers.
 */

#include "FactoryResultsFileParser.h" // a factory for results file parsers.
#include "ResultsFileParserAscii.h" // a results file parser for ASCII files.
#include "ResultsFileParserBinaryFmb.h" // a results file parser for Famu *.fmb results files.
#include "ResultsFileParserCovise.h" // a results file parser for Covise *.cvr results files.
#include "errorinfo.h" // a container for error data.
#include "Tools.h" // some helpful tools.

ResultsFileParser *FactoryResultsFileParser::create(
    std::string filename,
    MeshDataTrans *meshDataTrans,
    OutputHandler *outputHandler)
{
    (void)meshDataTrans;
    std::string suffix = parserTools::getSuffix(filename);
    ResultsFileParser *retval = NULL;
    if (suffix == ".fma" || suffix == ".hma")
    {
        retval = new ResultsFileParserAscii(outputHandler);
    }
    else if (suffix == ".fmb")
    {
        retval = new ResultsFileParserBinaryFmb(outputHandler);
    }
    else if (suffix == ".cvr")
    {
        retval = new ResultsFileParserCovise(outputHandler);
    }
    else
    {
        ERROR0("unknown results file suffix.", outputHandler);
    }
    return retval;
}

// ----------------- singleton stuff --------------------
FactoryResultsFileParser *FactoryResultsFileParser::_instance = NULL;

FactoryResultsFileParser *FactoryResultsFileParser::getInstance()
{
    if (_instance == NULL)
    {
        _instance = new FactoryResultsFileParser();
    }
    return _instance;
}

void FactoryResultsFileParser::deleteInstance(void)
{
    if (_instance != NULL)
    {
        delete _instance;
        _instance = NULL;
    }
}
