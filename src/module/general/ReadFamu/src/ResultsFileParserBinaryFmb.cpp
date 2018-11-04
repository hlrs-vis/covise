/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileParserBinaryFmb.h
 * a results file parser for Famu *.fmb results files.
 */

#include "ResultsFileParserBinaryFmb.h" // a results file parser for binary files in combination with changing (dynamic) meshes.
#include "errorinfo.h" // a container for error data.
#include "objectinputstreambinary.hxx" // a stream that can be used for deserialization/binary files.
#include "NodeNoMapperFmb.h" // a manager for node numbers for Famu *.fmb files.
#include "ResultsStatFmbCovise.h" // a container for results of a time step (=serialized famu results container)
#include "ResultsStatFmbCovise.h" // a container for results of a time step (=serialized famu results container)
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

bool ResultsFileParserBinaryFmb::readTimeSteps(
    ObjectInputStream *archive,
    int noOfTimeStepsToSkip,
    int noOfTimeStepsToParse,
    MeshDataTrans *meshDataTrans,
    std::string baseName)
{

    NodeNoMapperFmb::delInstance();
    NodeNoMapperFmb *nodeNoMapper = NodeNoMapperFmb::createInstance(archive, _outputHandler);

    MeshDataStat *mdStat = meshDataTrans->getMeshDataStat(0);
    //const int noOfNodes = mdStat->getNoOfPoints();

    _timeStepNames.clear();
    _totalNoOfTimeSteps = (int)archive->readINT();
    const int minTs = noOfTimeStepsToSkip;
    const int maxTs = min(noOfTimeStepsToParse + noOfTimeStepsToSkip, _totalNoOfTimeSteps);
    int i;
    for (i = 0; i < maxTs; i++)
    {
        std::string name = archive->readString();
        ResultsStat *newResults = new ResultsStatFmb(archive, mdStat, nodeNoMapper, NUMRES, _outputHandler, baseName);

        if (i >= minTs && i < maxTs)
        {
            _timeStepNames.push_back(name);
            _results.push_back(newResults);
        }
    }
    _noOfTimeSteps = (int)_results.size();
    return true;
}
