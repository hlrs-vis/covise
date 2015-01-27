/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__GRID_H)
#define __GRID_H

class Grid;

#include <api/coSimpleModule.h>
using namespace covise;
#include "GridStep.h"
#include "Particle.h"
#include "GridBlockInfo.h"

class Grid
{
protected:
    // this handles the timesteps
    int numSteps;
    GridStep **steps;
    float *stepDuration; // e.g. if the steps have an attribute giving each steps duration

    // the GBIs for the sources
    GridBlockInfo **sourceGBI;
    int numSources;
    int maxSources;

    // this flag is set (1) whenever we got static grid with changing data
    int dataChangesFlag;

    // initialize a new GBI with coordinates x,y,z in step s
    //    returns 0 if the point is not inside the grid
    int initGBI(int s, float x, float y, float z, GridBlockInfo *gbi);

    // symmetric boundary conditions
    int sbcFlag;
    float sbcMatrix[3][3];
    float sbcInvMatrix[3][3];

    // this will try to advance the given gbi by the given displacement and
    //  also try to apply SBC if possible
    int advanceGBI(GridBlockInfo *&gbi, float dx, float dy, float dz, GridBlockInfo *gbiCache = NULL);

public:
    // initialize the structure with the given objects and sS=saveSearch
    //    bBa=boundingBoxAlgorithm, attA=attackAngle, ng=needGradient, nR=needRotation
    Grid(const coDistributedObject *g, const coDistributedObject *v, int sS, int bBa, float attA,
         int nG, int nR);
    ~Grid();

    // return stepDuration (or 0.0 if none available)
    float getStepDuration(int s);
    int getNumSteps();

    // to get/start/init a new particle (returns NULL if invalid)
    //   s=step, sID=source-ID
    Particle *startParticle(int s, float x, float y, float z);
    Particle *startParticle(int sID);

    // handle the sources, quite usefull for VR-applications  (returns -1 if invalid)
    //   otherwise it returns a sourceID (sID) to be used with startParticle or
    //   moveSource (moveSource doesn't move a source out of bounds)
    int placeSource(int s, float x, float y, float z);
    void moveSource(int sID, float dx, float dy, float dz);

    // determine velocity at position GBI+[dx,dy,dz] (returns 1 if ok, 0 if
    //    position is outside the grid, 2 if attackAngle-condition met but would
    //    be outside the grid)
    //    [2]: will set (u,v,w) to a grid-boundary-parallel value representing the
    //      velocity at position (dx,dy,dz)=0.0 but with no directional components
    //      towards the grid-boundary.
    //    dc is returned and indicates the number of cells passed by displacing by [dx,dy,dz]
    int getVelocity(GridBlockInfo *&gbi, float dx, float dy,
                    float dz, float *u, float *v, float *w, float &dc, int sbc0 = 0, GridBlockInfo *gbiCache = NULL,
                    GridBlockInfo *gbi0 = NULL, int updateGBI = 0);
    // calculate velocity-gradient at current GBI (position)
    void getGradient(GridBlockInfo *gbi);
    // calculate velocity-rotation at current GBI (position)
    void getRotation(GridBlockInfo *gbi);

    // this will update gbiCur and gbiNext for the next integration-step
    //  NOTE that this doesn't mean 'next timestep' because it could be possible
    //  that there are multiple integration-steps in one timestep.
    //  (returns 1 if ok, 0 if position is outside the grid)
    int advanceGBI(GridBlockInfo *&gbiCur, GridBlockInfo *&gbiNext,
                   float dx, float dy, float dz, GridBlockInfo *gbiCache = NULL, GridBlockInfo *gbiNextCache = NULL);

    // try to find the "particle" defined by gbiCur in the next timestep and
    //  store it in gbiNext. returns 1 if ok, 0 if position is outside the grid.
    int getGBINextStep(GridBlockInfo *gbiCur, GridBlockInfo *&gbiNext);

    // allow access to the specified GridStep
    GridStep *getStep(int s);

    // initialize and use symmetric boundary conditions
    void useSBC(float axxis[3], float angle);
    void sbcApply(float *x, float *y, float *z);
    void sbcInvApply(float *x, float *y, float *z);
    void sbcVel(float *u, float *v, float *w, int sbcIdx);
};
#endif // __GRID_H
