/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileParserCovise.h
 * a results file parser for Covise *.cvr files.
 */

#include "ResultsFileParserCovise.h" // a results file parser for Famu *.fmb results files.
#include "ResultsStatFmbCovise.h" // a container for results of a time step (=serialized famu results container)
#include "errorinfo.h" // a container for error data.
#include "objectinputstreambinary.hxx" // a stream that can be used for deserialization/binary files.
#include "NodeNoMapperCovise.h" // a manager for node numbers for Covise *.cvr files.
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

bool ResultsFileParserCovise::readTimeSteps(
    ObjectInputStream *archive,
    int noOfTimeStepsToSkip,
    int noOfTimeStepsToParse,
    MeshDataTrans *meshDataTrans,
    std::string baseName)
{
    //MeshDataStat* mdStat = meshDataTrans->getMeshDataStat(0);
    //const int noOfNodes = mdStat->getNoOfPoints();

    _timeStepNames.clear();
    _totalNoOfTimeSteps = (int)archive->readINT();
    //    _totalNoOfTimeSteps = (int)archive->readINT();
    const int minTs = noOfTimeStepsToSkip;
    const int maxTs = min(noOfTimeStepsToParse + noOfTimeStepsToSkip, _totalNoOfTimeSteps);
    int i;
    for (i = 0; i < maxTs; i++)
    {
        NodeNoMapperCovise *nodeNoMapper = new NodeNoMapperCovise(archive, _outputHandler);

        //INT dummyNoOfSims =
        archive->readINT();
        //ASSERT(dummyNoOfSims==1, _outputHandler);      // due to serialized object of Famu's ResultsContainer
        std::string name = archive->readString();
        MeshDataStat *mdStat = meshDataTrans->getMeshDataStat(i);
        ResultsStat *newResults = new ResultsStatCovise(archive, mdStat, nodeNoMapper, NUMRES, _outputHandler, baseName);

        if (i >= minTs && i < maxTs)
        {
            _timeStepNames.push_back(name);
            _results.push_back(newResults);
        }
    }
    _noOfTimeSteps = (int)_results.size();
    return true;
}
