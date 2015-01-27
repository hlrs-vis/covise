/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file PeriodicRotMesh.h
 * completes meshes to a full 360° rotation.
 * rotation axís has to be z-axis.
 */

// #include "PeriodicRotMesh.h"  // completes meshes to a full 360° rotation.

#ifndef __PeriodicRotMesh_h__
#define __PeriodicRotMesh_h__

#include "MeshDataTrans.h" // a container for mesh file data where every timestep has its own mesh.
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include "os.h" // operating system dependent definitions

/**
 * completes meshes to a full 360° rotation.
 * rotation axís has to be z-axis.
 */
class PeriodicRotMesh : public MeshDataTrans
{
public:
    PeriodicRotMesh(double symmAngleDeg,
                    int noOfStepsPerBlock,
                    MeshDataTrans *originalData,
                    OutputHandler *outputHandler);
    virtual ~PeriodicRotMesh(){};

    // inherited getter methods
    virtual void addMesh(MeshDataStat *m);
    virtual int getNoOfMeshes(void) const;
    virtual MeshDataStat *getMeshDataStat(int timeStepNo);
    virtual int getMaxNodeNo(int timeStepNo) const;

protected:
    OutputHandler *_outputHandler;

    // original data
    int _noOfStepsPerBlock;
    double _symmAngle;
    MeshDataTrans *_originalData;

    // derived data
    int _noOfBlocks;
    int _noOfTimeSteps;

    MeshDataStat *getRotatedMesh(int timeStepNo,
                                 double angle);

    PointCC rotateVector(const PointCC &v,
                         double angle) const;

    int getOriginalTimeStepNo(int timeStepNo) const;
};

#endif
