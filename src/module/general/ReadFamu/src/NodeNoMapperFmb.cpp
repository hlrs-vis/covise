/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file NodeNoMapperFmb.h
 * a manager for node numbers for Famu *.fmb files.
 */

#include "NodeNoMapperFmb.h" // a manager for node numbers for Famu *.fmb files.
#include "errorinfo.h" // a container for error data.

void NodeNoMapperFmb::readFromArchive(ObjectInputStream *archive)
{
    ASSERT(_internal2meshArr == NULL, _outputHandler);
    ASSERT(_mesh2internalArr == NULL, _outputHandler);

    _noOfNodes = (int)archive->readINT();
    _maxNodeNoMesh = (int)archive->readINT();

    PointCC *nodeCoordinatesArr = NULL;
    UINT_F *internal2meshTmp = NULL;
    UINT_F *mesh2internalTmp = NULL;
    UINT_F noOfNodesTmp = 0;
    archive->readArray(&nodeCoordinatesArr, &noOfNodesTmp);
    archive->readArray(&internal2meshTmp, &noOfNodesTmp);
    archive->readArray(&mesh2internalTmp, &noOfNodesTmp);
    delete[] nodeCoordinatesArr;

    _internal2meshArr = new int[_noOfNodes];
    _mesh2internalArr = new int[_noOfNodes];
    int i;
    for (i = 0; i < _noOfNodes; i++)
    {
        _internal2meshArr[i] = (int)internal2meshTmp[i];
        _mesh2internalArr[i] = (int)mesh2internalTmp[i];
    }
    delete[] internal2meshTmp;
    delete[] mesh2internalTmp;
}

// --------------- singleton stuff -----------------------------

NodeNoMapperFmb *NodeNoMapperFmb::_instance = NULL;
OutputHandler *NodeNoMapperFmb::_outputHandler = NULL;

NodeNoMapperFmb *NodeNoMapperFmb::getInstance(void)
{
    ASSERT(_instance != NULL, _outputHandler);
    return _instance;
}

NodeNoMapperFmb *NodeNoMapperFmb::createInstance(
    ObjectInputStream *archive,
    OutputHandler *outputHandler)
{
    ASSERT(_instance == NULL, outputHandler);
    _instance = new NodeNoMapperFmb(archive, outputHandler);
    return _instance;
}

void NodeNoMapperFmb::deleteInstance(void)
{
    DELETE_OBJ(_instance);
}

// same as deleteInstance() but static
void NodeNoMapperFmb::delInstance(void)
{
    DELETE_OBJ(_instance);
}

NodeNoMapperFmb::NodeNoMapperFmb(ObjectInputStream *archive,
                                 OutputHandler *outputHandler)
    : NodeNoMapper(outputHandler)
{
    readFromArchive(archive);
}

NodeNoMapperFmb::~NodeNoMapperFmb()
{
    _instance = NULL;
}
