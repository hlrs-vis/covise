/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file NodeNoMapperCovise.h
 * a manager for node numbers for Covise *.cvr files.
 */

#include "NodeNoMapperCovise.h" // a manager for node numbers for Covise *.cvr files.
#include "errorinfo.h" // a container for error data.
#include "Tools.h" // some helpful tools.

NodeNoMapperCovise::NodeNoMapperCovise(ObjectInputStream *archive,
                                       OutputHandler *outputHandler)
    : NodeNoMapper(outputHandler)
{
    readFromArchive(archive);
}

void NodeNoMapperCovise::readFromArchive(ObjectInputStream *archive)
{
    ASSERT(_internal2meshArr == NULL, _outputHandler);
    ASSERT(_mesh2internalArr == NULL, _outputHandler);

    _noOfNodes = (int)archive->readINT();
    _maxNodeNoMesh = (int)archive->readINT();

    // --- input & cast _internal2meshArr ---
    UINT_F *internal2meshTmp = NULL;
    UINT_F arrSize = 0;
    archive->readArray(&internal2meshTmp, &arrSize);
    ASSERT(arrSize == (UINT_F)_noOfNodes, _outputHandler);

    _internal2meshArr = new int[_noOfNodes];
    int i;
    for (i = 0; i < _noOfNodes; i++)
    {
        _internal2meshArr[i] = (int)internal2meshTmp[i];
    }
    delete[] internal2meshTmp;

    // --- input & cast _mesh2internalArr ---
    UINT_F *mesh2internalTmp = NULL;
    arrSize = 0;
    archive->readArray(&mesh2internalTmp, &arrSize);
    ASSERT(arrSize == (UINT_F)_maxNodeNoMesh + 1, _outputHandler);

    _mesh2internalArr = new int[_maxNodeNoMesh + 1];
    for (i = 0; i <= _maxNodeNoMesh; i++)
    {
        _mesh2internalArr[i] = (int)mesh2internalTmp[i];
    }
    //ArrayIo::writeArrayToDisk("_results_mesh2internalArr.txt",  _mesh2internalArr, _maxNodeNoMesh);
    delete[] mesh2internalTmp;
}
