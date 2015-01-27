/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BB_H
#define BB_H

#include <values.h>

#include <vector>
#include <cutil.h>
#include <cutil_math.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

enum ELEMENTSIDE
{
    FRONT = 0,
    RIGHT = 1,
    BACK = 2,
    LEFT = 3,
    DOWN = 4,
    UP = 5
};
enum WALLTYPE
{
    WALL = 1,
    PERIODIC = 2
};

/*
  type   WALLTYPE
    side ELEMENTSIDE
          partition
| t|side| p|       element index         |
 00 0000 00 0000 0000 0000 0000 0000 0000 


const unsigned int ELEMENTTYPE       = 0x03 << 30;
const unsigned int ELEMENTSIDE       = 0x0F << 26;
const unsigned int ELEMENTPARTITION  = 0x03 << 24;
const unsigned int ELEMENTINDEX      = 0xFFFFFF;

enum ELEMENTSIDE { NONE = 0; FRONT = 1 << 26, RIGHT = 2 << 26, BACK = 3 << 26,
                   LEFT = 4 << 26, DOWN = 5 << 26, UP = 6 << 26 };

*/

class BB
{
public:
    BB()
    {
        minx = FLT_MAX;
        miny = FLT_MAX;
        minz = FLT_MAX;
        maxx = -FLT_MAX;
        maxy = -FLT_MAX;
        maxz = -FLT_MAX;
    }

    float minx;
    float miny;
    float minz;
    float maxx;
    float maxy;
    float maxz;
    int left, right;
    int cells;
};

struct usg
{

    int numElements;
    int numCorners;
    int numPoints;

    int *elementList;
    int *typeList;
    int *cornerList;
    float *x, *y, *z;

    float *vx, *vy, *vz;

    float *boundingSpheres;

    // TODO: merge faceType & wallVertexMap into neighbors
    // wall or periodic boundary
    unsigned char *faceType;
    // mapping from usg vertex to wall vertex
    int *wallVertexMap;

    // mapping from usg element face to wall face
    int *wallElementMap;

    // mapping from inlet wall to usg element
    int numInletElements;
    int *inletElementMap;

    BB *flat;
    int *cells;
    int *neighbors;

    unsigned int numFlat;
    unsigned int numCells;
};

struct poly
{

    int numPolygons;
    int numCorners;
    int numPoints;

    int *polygonList;
    int *cornerList;
    float *x, *y, *z;
    int *neighbors;
};
#endif
