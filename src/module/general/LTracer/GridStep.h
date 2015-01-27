/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__GRIDSTEP_H)
#define __GRIDSTEP_H

class GridStep;

#include <api/coSimpleModule.h>
using namespace covise;
#include "Particle.h"
#include "GridBlock.h"
#include "Grid.h"
#include "GridBlockInfo.h"

class GridStep
{
protected:
    // we handle multiblock
    int numBlocks;
    GridBlock **block;
    // and we have to know our "owner"-object
    Grid *grid;
    int myStep;

    // in the case that we got a static grid with changing data, we need this information
    float *stepDuration;
    int numSteps;

    // caches
    int blockCache;
    int **fromBlockCache; // from another timestep timestep-cache
    int **fromBlockCache2; // inside this timestep
    void addFromBlockCache(int from, int to);
    void addFromBlockCache2(int from, int to);

    // parameter
    int needGradient, needRot;

public:
    // initialize the structure with the given objects and parameters
    //   g=grid, v=velocity, sS=saveSearch, bBa=boundingBoxAlgo, attA=attackAngle
    //   nG=needGradient, nR=needRotation, s=myStep
    GridStep(const coDistributedObject *g, const coDistributedObject *v, int sS,
             int bBa, float attA, int nG, int nR, Grid *parent, int s);
    ~GridStep();

    // returns the object-given duration of that step if provided, otherwise 0.0
    float getStepDuration(int s = 0);
    // check if its static grid with changing data (>1 if so)
    int getNumSteps();

    // return information about multiblock-grids
    int getNumBlocks();

    // and to get/start/init a new particle (returns NULL if invalid)
    Particle *startParticle(float x, float y, float z, int lsbCache = -1);
    Particle *startParticle(GridBlockInfo *sourceGBI);

    // used to get a new GBI (returns 0 if outside the grid/failure)
    //    iBlock gives number of block to ignore in the search...used
    //    during tracing if a particle leaves a specific block
    int initGBI(float x, float y, float z, GridBlockInfo *&gbi, int iBlock = -1, int fromBlock = -1);

    // returns myStep
    int stepNo()
    {
        return (myStep);
    };
};
#endif // __GRIDSTEP_H
