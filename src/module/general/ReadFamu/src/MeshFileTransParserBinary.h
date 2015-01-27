/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshFileTransParserBinary.h
 * a mesh file parser for changing (transient) meshes in Famu binary format.
 */

// #include "MeshFileTransParserBinary.h"  // a mesh file parser for changing (transient) meshes in Famu binary format.

#ifndef __MeshFileTransParserBinary_h__
#define __MeshFileTransParserBinary_h__

#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include "objectinputstream.hxx" // a stream that can be used for deserialization.
#include "MeshFileTransParser.h" // a mesh file parser.

/**
 * a mesh file parser for changing (transient) meshes in Famu binary format.
 */
class MeshFileTransParserBinary : public MeshFileTransParser
{
public:
    MeshFileTransParserBinary(OutputHandler *outputHandler);
    virtual ~MeshFileTransParserBinary();

    virtual void parseMeshFile(
        std::string filename,
        covise::coRestraint &sel,
        bool subdivide,
        const int &noOfTimeStepToSkip,
        const int &noOfTimeStepsToParse);

    virtual MeshDataTrans *getMeshDataTrans(void);

private:
    OutputHandler *_outputHandler;
    MeshDataTrans *_meshDataTrans;

    MeshDataTrans *createMeshDataTrans(
        const int &noOfTimeStepsToSkip,
        const int &noOfTimeStepsToParse,
        ObjectInputStream *archive);
};

#endif
