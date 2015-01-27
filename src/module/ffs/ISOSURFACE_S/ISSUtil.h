/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__ISSUTIL_H)
#define __ISSUTIL_H

#include "ISSGrid.h"

class ISS3DSlice;

class ISS3DSlice
{
protected:
    ISSGrid *grid;

    int numI, numJ, numK;
    float *x1, *y1, *z1, *d1; // data for 2D slice 1 (lower slice)
    float *x2, *y2, *z2, *d2; // data for 2D slice 2 (upper slice)
    int *vl1, *vl2, *vlInter; // vertexlists for slice 1, 2 and for 3D
    int curUpper; // gives number of current upper slice (1 or 2)

    int curSlice;

public:
    ISS3DSlice(ISSGrid *g);
    ~ISS3DSlice();

    // handle current slice
    void handleSlice(int *vertList, int *numVert, int *stripList, int *numStrips,
                     float *xCoord, float *yCoord, float *zCoord, float threshold);
    void handle2DSlice(int *vl, float *d, float th);

    // go one slice up
    void sliceUp();
};
#endif // __ISSUTIL_H
