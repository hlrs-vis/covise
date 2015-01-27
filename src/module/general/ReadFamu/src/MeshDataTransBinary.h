/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshDataTransBinary.h
 * a container for mesh file data where every timestep has its own mesh/binary file format.
 */

//#include "MeshDataTransBinary.h"  // a container for mesh file data where every timestep has its own mesh/binary file format.

#ifndef __MeshDataTransBinary_h__
#define __MeshDataTransBinary_h__

#include "MeshDataTrans.h" // a container for mesh file data where every timestep has its own mesh.
#include "MeshDataStatBinary.h" // a container for mesh data/binary file format.
#include "objectinputstream.hxx" // a stream that can be used for deserialization.
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include <vector>
#include <string>

/**
 * a container for mesh file data where every timestep has its own mesh/binary file format.
 */
class MeshDataTransBinary : public MeshDataTrans
{
public:
    MeshDataTransBinary(ObjectInputStream *archive,
                        const int &noOfTimeStepsToSkip,
                        const int &noOfTimeStepsToParse,
                        OutputHandler *outputHandler);
    virtual ~MeshDataTransBinary();

    virtual void addMesh(MeshDataStat *m);

    virtual int getNoOfMeshes(void) const;

    virtual MeshDataStat *getMeshDataStat(int timeStepNo);

    virtual int getMaxNodeNo(int timeStepNo) const;

private:
    OutputHandler *_outputHandler;
    std::vector<MeshDataStat *> _meshData;

    void outputInfo(void) const;
};

#endif
