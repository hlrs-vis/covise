/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshFileTransParserBinary.h
 * a mesh file parser for changing (transient) meshes in Famu binary format.
 */

#include "MeshFileTransParserBinary.h" // a mesh file parser for changing (transient) meshes in Famu binary format.
#include "objectinputstreambinary.hxx" // a stream that can be used for deserialization/binary files.
#include "MeshDataTransBinary.h" // a container for mesh file data where every timestep has its own mesh/binary file format.

#include "stdio.h"
#include <string.h>

using namespace covise;

MeshFileTransParserBinary::MeshFileTransParserBinary(OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
    , _meshDataTrans(NULL)
{
}

MeshFileTransParserBinary::~MeshFileTransParserBinary()
{
    delete _meshDataTrans;
}

void MeshFileTransParserBinary::parseMeshFile(
    std::string filename,
    coRestraint &sel,
    bool subdivide,
    const int &noOfTimeStepsToSkip,
    const int &noOfTimeStepsToParse)
{
    (void)sel;
    (void)subdivide;
    ObjectInputStreamBinary instream(filename, _outputHandler);
    _meshDataTrans = createMeshDataTrans(noOfTimeStepsToSkip, noOfTimeStepsToParse, &instream);
}

MeshDataTrans *MeshFileTransParserBinary::createMeshDataTrans(
    const int &noOfTimeStepsToSkip,
    const int &noOfTimeStepsToParse,
    ObjectInputStream *archive)
{
    MeshDataTrans *retval = NULL;
    retval = new MeshDataTransBinary(archive, noOfTimeStepsToSkip, noOfTimeStepsToParse, _outputHandler);
    return retval;
}

MeshDataTrans *MeshFileTransParserBinary::getMeshDataTrans(void)
{
    return _meshDataTrans;
}
