/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__GRIDBLOCK_H)
#define __GRIDBLOCK_H

class GridBlock;

#include "Particle.h"
#include <api/coSimpleModule.h>
using namespace covise;
#include "GridStep.h"

class GridBlock
{
protected:
    // "owner"
    GridStep *step;
    Grid *grid;
    int myBlock;

    // parameters
    int saveSearch, boundingBoxAlgorithm;
    float attackAngle;

    // cache
    GridBlockInfo *gbiCache;

    // in case we have changing data on a static grid...
    int numSteps;
    bool isRotatingFlag;
    float rotSpeed;
    float rotAxis[3];

public:
    GridBlock(GridStep *parent, Grid *owner, int b);
    virtual ~GridBlock();

    bool isRotating()
    {
        return isRotatingFlag;
    }
    // returns myBlock
    int blockNo()
    {
        return (myBlock);
    };

    // a function to initialize this block from data (a covise-object)
    //   return 0 if ok, !0 on failure   (saveSearch/boundingBoxAlgo/attackAngle)
    //   (NOTE: d may be coDoSet which indicates a static grid with changing data)
    virtual int initialize(const coDistributedObject *g, const coDistributedObject *d, int sS,
                           int bBa, float attA)
    {
        (void)g;
        (void)d;
        (void)sS;
        (void)bBa;
        (void)attA;
        return 0;
    }

    // used to get a new GBI (returns 0 if outside the block/failure)
    //  (should use the boundingBox as a first step to do so)
    virtual int initGBI(float x, float y, float z, GridBlockInfo *&gbi)
    {
        (void)x;
        (void)y;
        (void)z;
        (void)gbi;
        return 0;
    }

    // determine velocity at position GBI+[dx,dy,dz] (returns 1 if ok, 0 if
    //    position is outside this block, 2 if attackAngle-condition met but would
    //    be outside this block)
    //    [2]: will set (u,v,w) to a grid-boundary-parallel value representing the
    //      velocity at position (dx,dy,dz)=0.0 but with no directional components
    //      towards the grid-boundary.
    //    dc is returned and indicates the number of cells passed by displacing by [dx,dy,dz]
    virtual int getVelocity(GridBlockInfo *gbi, float dx, float dy,
                            float dz, float *u, float *v, float *w, float &dc)
    {
        (void)gbi;
        (void)dx;
        (void)dy;
        (void)dz;
        (void)u;
        (void)v;
        (void)w;
        (void)dc;
        return 0;
    }

    // try to displace the given gbi by dx/dy/dz. returns 1 if ok, 0 if
    //    position is outside this block, note that in any case dx/dy/dz must
    //    be added to x,y,z
    virtual int advanceGBI(GridBlockInfo *gbi, float dx, float dy, float dz)
    {
        (void)gbi;
        (void)dx;
        (void)dy;
        (void)dz;
        return 0;
    }

    // calculate wether the line from O to P intersects with the triangle A,B,C
    //   The point of intersection is:  A+alpha*(B-A)+beta*(C-A)
    //   Returns: 0 on success, 1 if parallel, 2 if the intersection is behind the ray,
    //            3 if the intersection is outside the polygon
    int intersectionTest(float *A, float *B, float *C, float *O, float *P,
                         float *alpha, float *beta, float *t);

    // compute the volume of the tetrahedra defined through the given points
    float tetraVolume(float *p0, float *p1, float *p2, float *p3);

    // calculates via which side we leave a tetrahedra to reach point px
    //   the sides are: 0=0-1-2, 1=0-1-3, 2=0-2-3, 3=1-2-3
    int tetraTrace(float *p0, float *p1, float *p2, float *p3, float *px);

    // this is for debugging...generates a tetrahedra on the debug-output port
    //  (only if maximum number of tetrahedra not exceeded or maybe disabled)
    void debugTetra(float p0[3], float p1[3], float p2[3], float p3[3]);
    void debugTetra(float x, float y, float z, float dl);

    // which increment do we propose for numerical integration at the given position ?
    //  (return 0.0 if not available)
    float proposeIncrement(GridBlockInfo *gbi)
    {
        (void)gbi;
        return 0.0;
    }

    void setRotation(float speed, float rax, float ray, float raz);
};
#endif // __GRIDBLOCK_H
