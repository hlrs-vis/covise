/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__USG_H)
#define __USG_H

class USGBlock;
class USGBlockInfo;

#include "GridBlock.h"
#include "Particle.h"

class USGBlockInfo : public GridBlockInfo
{
public:
    USGBlockInfo()
    {
        type = _GBI_USG;
        return;
    };
    ~USGBlockInfo(){};

    // current cell
    int cell;

    // id of current tetrahedra we are in
    int tetra;
    int decomposeOrder;

    // assignment operator
    // override
    //virtual USGBlockInfo& operator= (const USGBlockInfo& s)
    //{
    //   // hack or is it valid ?!....see C++ FAQ 218
    //   *this = (USGBlockInfo&)s;
    //   return( *this );
    //};
    // overload
    USGBlockInfo &operator=(const USGBlockInfo &s)
    {
        GridBlockInfo::operator=(s);
        cell = s.cell;
        tetra = s.tetra;
        decomposeOrder = s.decomposeOrder;
        return (*this);
    };

    // copy constructor
    USGBlockInfo(const USGBlockInfo &s)
        : GridBlockInfo(s)
    {
        cell = s.cell;
        tetra = s.tetra;
        decomposeOrder = s.decomposeOrder;
        return;
    };

    // "virtual constructor"
    virtual USGBlockInfo *createSimilar()
    {
        return (new USGBlockInfo());
    };
    virtual USGBlockInfo *createCopy()
    {
        return (new USGBlockInfo(*this));
    };
};

class USGBlock : public GridBlock
{
protected:
    // Grid+Data
    coDoUnstructuredGrid *usGrid;
    float *xCoord, *yCoord, *zCoord;
    float *uData, *vData, *wData;
    float **aUData, **aVData, **aWData;
    int numElem, numConn, numPoints;
    int *elemList, *connList, *typeList;
    int numNeighbors, *neighborList, *neighborIndexList;
    int val[3]; // vertices of the side, the particle left the current cell

    // boundingbox
    float xMin, xMax, yMin, yMax, zMin, zMax;
    float searchSizeX, searchSizeY, searchSizeZ;

    // decomposition
    int numTetra; // number of tetrahedra the current tetraConnList holds
    int tetraConnList[24]; // max. 24/4 = 6 tetrahedra at the moment, subject to change
    void decomposeElem(int e, int d);

    // determines which decompositionOrder to use to get a tetrahedra on v1,v2,v3
    int getDecomposition(int e, int v1, int v2, int v3);

    // try to find the element(e)+tetrahedron(t) which contains x/y/z
    //    returns 1 on success, 0 if outside the block, -1 on success and e left unchanged
    int findNextElem(float x, float y, float z, int &e, int &t, int &dc);
    int findNextElem_old(float x, float y, float z, int &e, int &t, int &dc);

    // check if the given point lies inside the given element(e) and use decomposition-scheme(d)
    //    will return the idx of the tetrahedra inside (e) or -1 if not found/outside the element
    int isInsideCell(float x, float y, float z, int e, int d);
    int isInsideCell(float x, float y, float z, int e, int vd0, int vd1);

    // compute the volume of the given tetraheder
    float tetraVol(float p0[3], float p1[3], float p2[3], float p3[3]);

    // check if the given point is inside the given tetrahedra (0=inside, 1=outside)
    int isInsideTetra(int tc0, int tc1, int tc2, int tc3, float x, float y, float z, float &u, float &v, float &w);

    // compute the centroid of the given element
    void getCentroid(int e, float &x, float &y, float &z);

public:
    USGBlock(GridStep *parent, Grid *owner, int b);
    virtual ~USGBlock();

    // a function to initialize this block from data (a covise-object)
    //   return 0 if ok, !0 on failure   (saveSearch/boundingBoxAlgo/attackAngle)
    //   (NOTE: d may be coDoSet which indicates a static grid with changing data)
    int initialize(coDistributedObject *g, coDistributedObject *d, int sS,
                   int bBa, float attA);

    // used to get a new GBI (returns 0 if outside the block/failure)
    //  (should use the boundingBox as a first step to do so)
    int initGBI(float x, float y, float z, GridBlockInfo *&gbi);

    // determine velocity at position GBI+[dx,dy,dz] (returns 1 if ok, 0 if
    //    position is outside this block, 2 if attackAngle-condition met but would
    //    be outside this block)
    //    [2]: will set (u,v,w) to a grid-boundary-parallel value representing the
    //      velocity at position (dx,dy,dz)=0.0 but with no directional components
    //      towards the grid-boundary.
    int getVelocity(GridBlockInfo *gbi, float dx, float dy,
                    float dz, float *u, float *v, float *w, float &dc);

    // try to displace the given gbi by dx/dy/dz. returns 1 if ok, 0 if
    //    position is outside this block, note that in any case dx/dy/dz must
    //    be added to x,y,z
    int advanceGBI(GridBlockInfo *gbi, float dx, float dy, float dz);
};
#endif // __USG_H
