/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsStatFmbCovise.h
 * a container for results of a time step (=serialized famu results container)
 */

// #include "ResultsStatFmbCovise.h"  // a container for results of a time step (=serialized famu results container)

#ifndef __ResultsStatFmbCovise_h__
#define __ResultsStatFmbCovise_h__

#include "ResultsStat.h" // a container for stationary mesh data.
#include "NodeNoMapper.h" // a manager for node numbers (maps from internal to mesh file node numbers)
#include "NodeNoMapperFmb.h" // a manager for node numbers for Famu *.fmb files.

/**
 * (=serialized famu results container).
 */
class ResultsStatBinary : public ResultsStat
{
public:
    ResultsStatBinary(
        ObjectInputStream *archive,
        MeshDataStat *meshDataStat,
        NodeNoMapper *nodeNoMapperResFile,
        int maxNoOfDataTypes,
        OutputHandler *outputHandler,
        std::string bn);
    virtual ~ResultsStatBinary();

private:
    bool readDataTypes(ObjectInputStream *archive,
                       int maxNoOfDataTypes);

    void readDataTypeScalar(coDistributedObject **dt,
                            std::string &dtName,
                            ObjectInputStream *archive);

    void readDataTypeVector(coDistributedObject **dt,
                            std::string &dtName,
                            ObjectInputStream *archive);

    void outputInfo(void) const;
};

class ResultsStatFmb : public ResultsStatBinary
{
public:
    ResultsStatFmb(
        ObjectInputStream *archive,
        MeshDataStat *meshDataStat,
        NodeNoMapper *nodeNoMapperResFile,
        int maxNoOfDataTypes,
        OutputHandler *outputHandler,
        std::string bn)
        : ResultsStatBinary(archive, meshDataStat, nodeNoMapperResFile, maxNoOfDataTypes, outputHandler, bn){};

    virtual ~ResultsStatFmb()
    {
        NodeNoMapperFmb::delInstance();
    }; // all time steps share the same mapper
};

class ResultsStatCovise : public ResultsStatBinary
{
public:
    ResultsStatCovise(
        ObjectInputStream *archive,
        MeshDataStat *meshDataStat,
        NodeNoMapper *nodeNoMapperResFile,
        int maxNoOfDataTypes,
        OutputHandler *outputHandler,
        std::string bn)
        : ResultsStatBinary(archive, meshDataStat, nodeNoMapperResFile, maxNoOfDataTypes, outputHandler, bn){};

    virtual ~ResultsStatCovise()
    {
        _nodeNoMapperResFile->deleteInstance();
    }; // each time step has its own mapper
};

#endif
