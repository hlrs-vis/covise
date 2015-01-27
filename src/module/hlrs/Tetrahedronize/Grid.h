/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__TETRAHEDRONIZE_GRID_H)
#define __TETRAHEDRONIZE_GRID_H

// includes
#include <appl/ApplInterface.h>
using namespace covise;
#include <do/coDoUnstructuredGrid.h>

class Grid
{
private:
    // object names
    char *gridObjName, *dataObjName;

    // objects
    const coDoUnstructuredGrid *gridObjIn;

    // flags, if elements have been processed
    char *elementProcessed;

    // information on how the sides of an element
    // have to be triangulated
    //      0   not yet processed/unknown
    //     -1   doesn't matter
    //     -2   same as neighbor-element
    // we look on the side from infront of the side and start
    // walking around the side clockwise, starting at the lowest
    // vertex-id
    //      1   split on the diagonal from start to second
    //      2   split on the diagonal from first to third
    //
    // the []-elements are different, depending on the elementType:
    //     HEXAHEDRON:
    //        0  -  1234  (top)
    //        1  -  3784  (right)
    //        2  -  5876  (bottom)
    //        3  -  1562  (left)
    //        4  -  1458  (front)  // ARGH, should be 1485
    //        5  -  2673  (rear)
    //     PYRAMID:
    //        0  -  1234
    //     PRISM:
    //        0  -  1364  (front)
    //        1  -  2563  (right)
    //        2  -  1452  (left)
    //
    char *elementSide[6];

    // flag if we should try to generate a regular grid
    int regulateFlag;

    // compute the size of actual grid if created out of the given grid
    // and return number of additional points required
    int computeNewSize(Grid *grid);

    // perform the transformation
    coDistributedObject **irregularTransformation(Grid *grid);
    void regularTransformation(Grid *grid);

    // process the given element, reqd. by regularTransformation
    void processElement(int no);

    // do the given vertices have the same neighbor-cell ?
    // (returns id of cell or -1 if none)
    int haveSameNeighbor(int el, int c0, int c1, int c2 = -1, int c3 = -1);
    int onWhichSide; // if the neighbor shares all 4 coordinates,
    // then this value represents the side of
    // the neighbor (see elementSide-description)

    // if the two elements share a quad, then return side of element el
    // that does so
    int findSharedSide(int el, int c0, int c1, int c2, int c3);

    // check if the two given quads are the same
    int areIdentical(int pl1[4], int pl2[4]);

    // this will return -1, 1 or 2 (see elementSide-description above)
    int analyzeQuad(Grid *grid, int el, int v0, int v1, int v2, int v3);

    // get coordId of first coord on the given side of the element
    int getStart(int el, int side);

    // process a single side of a hexahedron in 2nd step
    void handleHexaSide(int el, int s, int c0, int c1, int c2, int c3, int os, int sFlag, Grid *grid);

    void handlePrismSide(int el, int s, int c0, int c1, int c2, int c3, Grid *grid);
    void splitPyra(int el, Grid *grid);

    void splitHexa(int el, Grid *grid, int doFlag = 1);
    void splitPrism(int el, Grid *grid, int doFlag = 1);

    int haveToSplitPrism(int s0, int s1, int s2);

    // get new triangulation and vertices, so that the hexagon may
    // be cut into two prisms by cutting along the c0-c2 diagonal
    void createIdealHexagon(int el, Grid *grid, int &c0, int &c1, int &c2, int &c3, int &c4, int &c5, int &c6, int &c7, int &s1, int &s3, int &s4, int &s5);

    // check if point lies inside tetrahedra and get data at that point
    int isInsideGetData(int c0, int c1, int c2, int c3, float xP, float yP, float zP, float &uP, float &vP, float &wP);
    float tetraVolume(float p0[3], float p1[3], float p2[3], float p3[3]);

    // this is for debugging only
    void analyzeSelf(Grid *grid, int &numSides, int &okSides, int &numHexa, int &numHexaOk);

public:
    // the grid itself
    float *xCoord, *yCoord, *zCoord;
    int numElem, numConn, numPoints;
    int *elemList, *connList, *typeList;

    // the neighborlist
    int *neighborList, *neighborIndexList, numNeighbors;

    // and the data
    float *uData, *vData, *wData;
    int scalarFlag, vectorFlag;

    // initialize a grid out of COVISE-Objects
    Grid(const coDistributedObject *gridIn, const coDistributedObject *dataIn);

    // prepare the grid to be created out of another one
    Grid(char *gridName, char *dataName);

    // clean up
    ~Grid();

    // build neighborlist
    void computeNeighborList();

    // transform the given grid into tetrahedrons
    coDistributedObject **tetrahedronize(Grid *oldGrid, int regulate);
};
#endif // __TETRAHEDRONIZE_GRID_H
