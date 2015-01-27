/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__PARTICLE_H)
#define __PARTICLE_H

class Particle;

#include "GridBlockInfo.h"
#include <util/coviseCompat.h>

class Particle
{
protected:
    // the particle "knows" its current position and velocity etc.
    struct
    {
        float x, y, z;
        int sbcIdx;
    } pos;
    struct
    {
        float u, v, w;
    } vel;
    struct
    {
        float rx, ry, rz;
    } rot;
    struct
    {
        float g11, g12, g13, g21, g22, g23, g31, g32, g33;
    } grad;

    // stepSize from last numerical integration - step
    float lastH;

    // the structure holding the ParticlePath (thus pPstr)
    struct pPstr
    {
        float x, y, z, u, v, w;
        int sbcIdx;
    } *path;
    int numSteps, maxSteps;
    // and maybe we need the "gradient/rot-path"
    struct pGstr
    {
        float rx, ry, rz;
        float g11, g12, g13, g21, g22, g23, g31, g32, g33;
    } *pathRotGrad;
    int needGradient, needRot;

    // this flag is set if the last waypoint is outside the grid (thus the
    //   particle has left the grid through an outlet)
    int stoppedFlag;

    // this flag is set if a loop was detected by this->loopDetection()
    //   the loop is between   path[numSteps-1] and path[loopOffset]
    //    (besides that, stoppedFlag is also set)
    int loopDetectedFlag, loopOffset;

    // for simplification....adds the current pos/vel/grad to the path/s
    void addCurToPath();

    // get the velocity of the particle (by computing it from gbiCur/gbiNext)
    //   h = time inside gbiCur/gbiNext (thus  0 <= h <= dt )
    //  dt = duration of step gbiCur->gbiNext
    //  d* = displacement to gbiCur/gbiNext
    //  dc = reserved (yet to be implemented)
    int getVelocity(float h, float dt, float dx, float dy, float dz,
                    float *u, float *v, float *w, float *dc, int sbc0 = 0, GridBlockInfo *gbiCur0 = NULL, GridBlockInfo *gbiNext0 = NULL);

    // perform one integrational step of size h from gbi+[dx.dy.dz] with vel [u0,v0,w0]
    //    (returns 1 if ok, 0 if position is outside the grid,
    //     2 if attackAngle-condition met but would be outside the grid)
    int traceRK4(float u0, float v0, float w0, float *dx, float *dy, float *dz, float h,
                 float stepTime = 0.0, float stepDuration = 0.0);

public:
    // a Particle is bound to the Grid and has position/velocity (taken from gbC)
    //   (gbC=gbiCurrent, gbN=gbiNext/NULL)
    Particle(Grid *g, GridBlockInfo *gbC, GridBlockInfo *gbN, int nG, int nR);
    ~Particle();

    // returns wether the particle has stopped or not
    //   (and thus if tracing should continue or not)
    int hasStopped();

    // this will check if the particle is performing a closed loop and if so,
    //    the particle will stop now
    void loopDetection();

    // this may be used for postprocessing
    int getNumSteps();
    void getWayPoint(int i, int *sbcIdx, float *x, float *y, float *z, float *u, float *v,
                     float *w, float *rx = NULL, float *ry = NULL, float *rz = NULL,
                     float *g11 = NULL, float *g12 = NULL, float *g13 = NULL,
                     float *g21 = NULL, float *g22 = NULL, float *g23 = NULL,
                     float *g31 = NULL, float *g32 = NULL, float *g33 = NULL);

    // and the particle "knows" inside which Block it currently is
    //    and also holds all the detailed information the grid/block requires
    //    to perform the tracing
    GridBlockInfo *gbiCur, *gbiNext, *gbiCache, *gbiNextCache;
    Grid *grid;

    // trace this particle for a period of [dt]...note that dt is only used
    //    if not overridden by grid->stepDuration, dt is assumed to allways be
    //    exactly one timestep (if not given trough grid->stepDuration)
    void trace(float dt, float eps = 0.00001); // newest version, uses embedded RK5 with cash and karp parameters
    void oldTrace(float dt);
    void oldTrace2(float dt); // newer than oldTrace
    void oldTrace3(float dt); // newer than oldTrace2, uses 'doublestep rk4'
};
#endif // __PARTICLE_H
