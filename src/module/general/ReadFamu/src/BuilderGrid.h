/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file BuilderGrid.h
 * a builder for grids.
 */

//#include "BuilderGrid.h"  // a builder for grids.

#ifndef __BuilderGrid_h__
#define __BuilderGrid_h__

#include "MeshDataTrans.h" // a container for mesh file data where every timestep has its own mesh.
#include "ResultsFileData.h" // a container for results file data.
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include <do/coDoSet.h>
#include <util/coRestraint.h>
#include <api/coModule.h>
using namespace covise;

/**
 * a builder for grids.
 */
class BuilderGrid
{
public:
    BuilderGrid(OutputHandler *outputHandler);
    virtual ~BuilderGrid(){};

    coDoSet *construct(const char *meshName,
                       bool scaleDisplacements,
                       float symmAngle,
                       int noOfSymmTimeSteps,
                       MeshDataTrans *meshDataTrans,
                       ResultsFileData *resFileData);

private:
    OutputHandler *_outputHandler;

    // each time step has the same mesh and possibly displacements.
    coDoSet *constructSingleMesh(
        const char *meshName,
        bool scaleDisplacements,
        MeshDataTrans *meshDataTrans,
        ResultsFileData *resFileData,
        float symmAngle,
        int noOfSymmTimeSteps);

    // each time step has a different mesh.
    coDoSet *constructMultipleMeshes(
        const char *meshName,
        bool scaleDisplacements,
        MeshDataTrans *meshDataTrans,
        ResultsFileData *resFileData,
        float symmAngle,
        int noOfSymmTimeSteps);
};

#endif
