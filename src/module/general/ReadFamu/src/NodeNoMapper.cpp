/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file NodeNoMapper.h
 * a manager for node numbers (maps from internal to mesh file node numbers)
 */

#include "NodeNoMapper.h" // a manager for node numbers (maps from internal to mesh file node numbers)
#include "errorinfo.h" // a container for error data.

NodeNoMapper::NodeNoMapper(OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
    , _noOfNodes(0)
    , _maxNodeNoMesh(0)
    , _internal2meshArr(NULL)
    , _mesh2internalArr(NULL)
{
}

NodeNoMapper::~NodeNoMapper()
{
    DELETE_ARRAY(_internal2meshArr);
    DELETE_ARRAY(_mesh2internalArr);
}

// ----------------------------- getter methods -----------------------------

int NodeNoMapper::getInternalNodeNo(int meshNodeNo) const
{
    ASSERT(meshNodeNo <= _maxNodeNoMesh, _outputHandler);
    ASSERT(_mesh2internalArr, _outputHandler);
    return _mesh2internalArr[meshNodeNo];
}

int NodeNoMapper::getMeshNodeNo(int internalNodeNo) const
{
    ASSERT(internalNodeNo < _noOfNodes, _outputHandler);
    ASSERT(_internal2meshArr, _outputHandler);
    return _internal2meshArr[internalNodeNo];
}

int NodeNoMapper::getNoOfNodes(void) const
{
    return _noOfNodes;
}

int NodeNoMapper::getMaxNodeNoMesh(void) const
{
    return _maxNodeNoMesh;
}
