/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__STRCT_H)
#define __STRCT_H

class StrctBlock;
class StrctBlockInfo;

#include "GridBlock.h"
#include "Particle.h"

class StrctBlockInfo : public GridBlockInfo
{
public:
    StrctBlockInfo()
    {
        type = _GBI_STRCT;
        return;
    };
    ~StrctBlockInfo(){};

    // lower-left cell corner
    int i, j, k;

    // id of current tetrahedra we are in
    int tetra;

    // for getVelocity speed-up
    int iTmp, jTmp, kTmp, tTmp;

    // hold the 8 corner-values of the cell (iTmp,jTmp,kTmp)
    //  these will be updated in StrctBlock::getVelocity
    //  (tetrahedra-decomposition is achieved via a lookup-table inside the functions)
    float A[3], B[3], C[3], D[3], E[3], F[3], G[3], H[3];

    // assignment operator
    // override
    virtual StrctBlockInfo &operator=(const GridBlockInfo &s)
    {
        // hack or is it valid ?!....see C++ FAQ 218
        *this = (StrctBlockInfo &)s;
        return (*this);
    };
    // overload
    StrctBlockInfo &operator=(const StrctBlockInfo &s)
    {
        GridBlockInfo::operator=(s);
        //cerr << "rbi::ao" << endl;
        i = s.i;
        j = s.j;
        k = s.k;
        tetra = s.tetra;
        //o = s.o;
        iTmp = s.iTmp;
        jTmp = s.jTmp;
        kTmp = s.kTmp;
        tTmp = s.tTmp;
        int l;
        for (l = 0; l < 3; l++)
        {
            A[l] = s.A[l];
            B[l] = s.B[l];
            C[l] = s.C[l];
            D[l] = s.D[l];
            E[l] = s.E[l];
            F[l] = s.F[l];
            G[l] = s.G[l];
            H[l] = s.H[l];
        }
        return (*this);
    };

    // copy constructor
    StrctBlockInfo(const StrctBlockInfo &s)
        : GridBlockInfo(s)
    {
        //cerr << "rbi::cc" << endl;
        i = s.i;
        j = s.j;
        k = s.k;
        tetra = s.tetra;
        //o = s.o;
        iTmp = s.iTmp;
        jTmp = s.jTmp;
        kTmp = s.kTmp;
        tTmp = s.tTmp;
        int l;
        for (l = 0; l < 3; l++)
        {
            A[l] = s.A[l];
            B[l] = s.B[l];
            C[l] = s.C[l];
            D[l] = s.D[l];
            E[l] = s.E[l];
            F[l] = s.F[l];
            G[l] = s.G[l];
            H[l] = s.H[l];
        }
        return;
    };

    // "virtual constructor"
    virtual StrctBlockInfo *createSimilar()
    {
        return (new StrctBlockInfo());
    };
    virtual StrctBlockInfo *createCopy()
    {
        return (new StrctBlockInfo(*this));
    };
};

class StrctBlock : public GridBlock
{
protected:
    // Grid+Data
    int iDim, jDim, kDim;
    float *xCoord, *yCoord, *zCoord;
    float *uData, *vData, *wData;
    float **aUData, **aVData, **aWData;

    // 'cache'
    float wg_, w0_, w1_, w2_, w3_;
    int d_;

    // boundingbox
    float xMin, xMax, yMin, yMax, zMin, zMax;

    // to jump in the data-array (also in the coordinates)
    int iStep, jStep, kStep;
    // a speedup-table to get coordinates/data corresponding to a cell(node)
    //   computed in initialize() and used e.g. in bruteForceSearch
    int cornerTbl[8];
    // and a table for the tetrahedra-decomposition (A=0, B=1) into
    //   5 tetrahedra (with 4 vertices each, of course)
    int tetraTable[2][5][4];
    // table for finding the neighbours ;)
    int neighbourTable[2][5][4][4];

    // returns 1 on success, 0 if outside the block (i,j,k,t hold last cell inside the grid which
    //   the particle passes),
    //   -1 if success and cell not changed (tetra may have changed though)
    int findNextCell(float x, float y, float z, int &i, int &j, int &k, int &t, float &dc);

    // perform brute-force search (1 on success, 0 on failure)
    float searchSizeX, searchSizeY, searchSizeZ;
    int bruteForceSearch(float x, float y, float z, int &i, int &j, int &k, int &t);

    // returns tetrahedral-decomposition-style (0 or 1), so that the decomposition
    //    is regular
    int decompositionOrder(int i, int j, int k);

public:
    StrctBlock(GridStep *parent, Grid *owner, int b);
    virtual ~StrctBlock();

    // a function to initialize this block from data (a covise-object)
    //   return 0 if ok, !0 on failure   (saveSearch/boundingBoxAlgo/attackAngle)
    //   (NOTE: d may be coDoSet which indicates a static grid with changing data)
    int initialize(coDistributedObject *g, coDistributedObject *d, int sS,
                   int bBa, float attA);

    // used to get a new GBI (returns 0 if outside the block/failure)
    //  (should use the boundingBox as first step to do so)
    int initGBI(float x, float y, float z, GridBlockInfo *&gbi);

    // determine velocity at position GBI+[dx,dy,dz] (returns 1 if ok, 0 if
    //    position is outside this block, 2 if attackAngle-condition met but would
    //    be outside this block)
    int getVelocity(GridBlockInfo *gbi, float dx, float dy,
                    float dz, float *u, float *v, float *w, float &dc);

    // try to displace the given gbi by dx/dy/dz. returns 1 if ok, 0 if
    //    position is outside this block
    int advanceGBI(GridBlockInfo *gbi, float dx, float dy, float dz);

    // which increment do we propose for numerical integration at the given position ?
    //  (return 0.0 if not available)
    float proposeIncrement(GridBlockInfo *gbi);
};
#endif // __STRCT_H
