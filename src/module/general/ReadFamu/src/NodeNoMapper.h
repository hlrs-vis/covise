/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file NodeNoMapper.h
 * a manager for node numbers (maps from internal to mesh file node numbers)
 */

// #include "NodeNoMapper.h"  // a manager for node numbers (maps from internal to mesh file node numbers)

#ifndef __NodeNoMapper_h__
#define __NodeNoMapper_h__

#include "objectinputstream.hxx" // a stream that can be used for deserialization.
#include "OutputHandler.h" // an output handler for displaying information on the screen.

/**
 * a manager for node numbers (maps from internal to mesh file node numbers)
 */
class NodeNoMapper
{
public:
    NodeNoMapper(OutputHandler *outputHandler);

    int getInternalNodeNo(int meshNodeNo) const;
    int getMeshNodeNo(int internalNodeNo) const;
    int getNoOfNodes(void) const;
    int getMaxNodeNoMesh(void) const;

    virtual void deleteInstance(void) = 0; // must not be static!

protected:
    virtual ~NodeNoMapper(); // proteced destructor to force usage of deleteInstance()

    OutputHandler *_outputHandler;
    int _noOfNodes;
    int _maxNodeNoMesh;
    int *_internal2meshArr;
    int *_mesh2internalArr;

    virtual void readFromArchive(ObjectInputStream *archive) = 0;
};

#endif
