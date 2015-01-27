/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__GRIDBLOCKINFO_H)
#define __GRIDBLOCKINFO_H

class GridBlockInfo;

#include "Grid.h"
#include "GridBlock.h"
#include "GridStep.h"

#define _GBI_UNDEFINED 0
#define _GBI_RECT 1
#define _GBI_STRCT 2
#define _GBI_USG 3

class GridBlockInfo
{
protected:
public:
    GridBlockInfo();
    virtual ~GridBlockInfo();

    // pointer to the main-Grid-object
    Grid *grid;

    // step/block id/object this GBI refers to
    int stepNo, blockNo;
    GridBlock *block;
    GridStep *step;

    // actual position/velocity at the GBI (position)
    float x, y, z;
    int sbcIdx;
    float u, v, w;
    float rx, ry, rz;
    float g11, g12, g13, g21, g22, g23, g31, g32, g33;

    // used as a temporary block-cache (-1 if uninitialized, otherwise
    //  blockNo that was used for a getVelocity call that left the current block)
    int tmpBlock;

    // used to identify the grid-type
    int type;

    // assignment operator
    virtual GridBlockInfo &operator=(const GridBlockInfo &s)
    {
        //cerr << "gbi::ao" << endl;
        x = s.x;
        y = s.y;
        z = s.z;
        sbcIdx = s.sbcIdx;
        u = s.u;
        v = s.v;
        w = s.w;
        rx = s.rx;
        ry = s.ry;
        rz = s.rz;
        g11 = s.g11;
        g12 = s.g12;
        g13 = s.g13;
        g21 = s.g21;
        g22 = s.g22;
        g23 = s.g23;
        g31 = s.g31;
        g32 = s.g32;
        g33 = s.g33;
        stepNo = s.stepNo;
        blockNo = s.blockNo;
        step = s.step;
        block = s.block;
        grid = s.grid;
        type = s.type;
        return (*this);
    };

    // copy constructor
    GridBlockInfo(const GridBlockInfo &s)
    {
        //cerr << "gbi:cc" << endl;
        x = s.x;
        y = s.y;
        z = s.z;
        sbcIdx = s.sbcIdx;
        u = s.u;
        v = s.v;
        w = s.w;
        rx = s.rx;
        ry = s.ry;
        rz = s.rz;
        g11 = s.g11;
        g12 = s.g12;
        g13 = s.g13;
        g21 = s.g21;
        g22 = s.g22;
        g23 = s.g23;
        g31 = s.g31;
        g32 = s.g32;
        g33 = s.g33;
        stepNo = s.stepNo;
        blockNo = s.blockNo;
        step = s.step;
        block = s.block;
        grid = s.grid;
        type = s.type;
        return;
    };

    // "virtual constructor"
    virtual GridBlockInfo *createSimilar() = 0;
    virtual GridBlockInfo *createCopy() = 0;
};
#endif // __GRIDBLOCKINFO_H
