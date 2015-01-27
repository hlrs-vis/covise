/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsStat.h
 * a container for results of a time step.
 */

// #include "ResultsStat.h"  // a container for stationary mesh data.

#ifndef __ResultsStat_h__
#define __ResultsStat_h__

#include <vector>
#include <string>
#include <api/coModule.h>
using namespace covise;

#include "objectinputstream.hxx" // a stream that can be used for deserialization.
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include "MeshDataStat.h" // a container for stationary mesh data.
#include "NodeNoMapper.h" // a manager for node numbers (maps from internal to mesh file node numbers)
#include "ResultsFileData.h" // a container for results file data.

class ResultsStat
{
public:
    ResultsStat(MeshDataStat *meshDataStat,
                NodeNoMapper *nodeNoMapperResFile,
                OutputHandler *outputHandler,
                std::string bn);
    virtual ~ResultsStat();

    virtual ResultsFileData::EDimension getDimension(int dataTypeNo);
    virtual std::string getDataTypeName(int dataTypeNo);
    virtual int getNoOfDataTypes(void) const;
    coDistributedObject *getDataObject(int dataTypeNo);

    virtual bool getDisplacements(float *dx[], float *dy[], float *dz[]);

    virtual void debug(std::ofstream &f);

protected:
    OutputHandler *_outputHandler;
    const MeshDataStat *_meshDataStat;
    NodeNoMapper *_nodeNoMapperResFile;

    int _noOfDataTypes;
    std::vector<coDistributedObject *> _dataObjects; // must not be deleted in destructor!
    std::vector<ResultsFileData::EDimension> _dimension;
    int _displacementNo; // if no timestep availabe -> -1

    std::vector<std::string> _dataTypeNames; // DataType names
    std::vector<std::string> _timeStepNames; // DataType names

    // maps node numbers from internal/results file --> internal/mesh file.
    // returns -1 if mesh node no. is not available (happens when elements/nodes are missing in the mesh file)
    int getNodeNoOfMesh(int nodeNoRes) const;
    std::string baseName;
    static int dataObjectNum;
};

#endif
