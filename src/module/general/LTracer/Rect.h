/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__RECT_H)
#define __RECT_H

class RectBlock;
class RectBlockInfo;

#include "GridBlock.h"
#include "Particle.h"

class RectBlockInfo : public GridBlockInfo
{
public:
    RectBlockInfo()
    {
        type = _GBI_RECT;
        return;
    };
    ~RectBlockInfo(){};

    // lower-left cell corner (+offset in array for speedup)
    int i, j, k;
    int o;

    // for getVelocity speed-up
    int iTmp, jTmp, kTmp;

    // hold the 8 corner-values of the cell (iTmp,jTmp,kTmp)
    //  these will be updated in RectBlock::getVelocity
    float A0, A1, A2, B0, B1, B2, C0, C1, C2, D0, D1, D2;
    float E0, E1, E2, F0, F1, F2, G0, G1, G2, H0, H1, H2;

    // assignment operator
    // override
    virtual RectBlockInfo &operator=(const GridBlockInfo &s)
    {
        // hack or is it valid ?!....see C++ FAQ 218
        *this = (RectBlockInfo &)s;
        return (*this);
    };
    // overload
    RectBlockInfo &operator=(const RectBlockInfo &s)
    {
        GridBlockInfo::operator=(s);
        //cerr << "rbi::ao" << endl;
        i = s.i;
        j = s.j;
        k = s.k;
        o = s.o;
        iTmp = s.iTmp;
        jTmp = s.jTmp;
        kTmp = s.kTmp;
        A0 = s.A0;
        A1 = s.A1;
        A2 = s.A2;
        B0 = s.B0;
        B1 = s.B1;
        B2 = s.B2;
        C0 = s.C0;
        C1 = s.C1;
        C2 = s.C2;
        D0 = s.D0;
        D1 = s.D1;
        D2 = s.D2;
        E0 = s.E0;
        E1 = s.E1;
        E2 = s.E2;
        F0 = s.F0;
        F1 = s.F1;
        F2 = s.F2;
        G0 = s.G0;
        G1 = s.G1;
        G2 = s.G2;
        H0 = s.H0;
        H1 = s.H1;
        H2 = s.H2;
        return (*this);
    };

    // copy constructor
    RectBlockInfo(const RectBlockInfo &s)
        : GridBlockInfo(s)
    {
        //cerr << "rbi::cc" << endl;
        i = s.i;
        j = s.j;
        k = s.k;
        o = s.o;
        iTmp = s.iTmp;
        jTmp = s.jTmp;
        kTmp = s.kTmp;
        A0 = s.A0;
        A1 = s.A1;
        A2 = s.A2;
        B0 = s.B0;
        B1 = s.B1;
        B2 = s.B2;
        C0 = s.C0;
        C1 = s.C1;
        C2 = s.C2;
        D0 = s.D0;
        D1 = s.D1;
        D2 = s.D2;
        E0 = s.E0;
        E1 = s.E1;
        E2 = s.E2;
        F0 = s.F0;
        F1 = s.F1;
        F2 = s.F2;
        G0 = s.G0;
        G1 = s.G1;
        G2 = s.G2;
        H0 = s.H0;
        H1 = s.H1;
        H2 = s.H2;
        return;
    };

    // "virtual constructor"
    virtual GridBlockInfo *createSimilar()
    {
        return (new RectBlockInfo());
    };
    virtual GridBlockInfo *createCopy()
    {
        return (new RectBlockInfo(*this));
    };
};

class RectBlock : public GridBlock
{
protected:
    // Grid+Data
    int iDim, jDim, kDim;
    float *xCoord, *yCoord, *zCoord;
    float *uData, *vData, *wData;
    float **aUData, **aVData, **aWData;

    // boundingbox
    float xMin, xMax, yMin, yMax, zMin, zMax;

    // for code simplification
    int i0, j0, k0, iInc, jInc, kInc;
    // to jump in the data-array
    int iStep, jStep, kStep;

    // returns 1 on success, 0 if outside the block (i,j,k hold last cell inside the grid which
    //   the particle passes),
    //   -1 if success and cell not changed
    int findNextCell(float x, float y, float z, int &i, int &j, int &k);
    bool isect(float *pos, float *vec, float *normal, float *hitPoint);

public:
    RectBlock(GridStep *parent, Grid *owner, int b);
    virtual ~RectBlock();

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
};
#endif // __RECT_H
