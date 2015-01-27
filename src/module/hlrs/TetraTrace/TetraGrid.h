/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__TETRAGRID_H)
#define __TETRAGRID_H

#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

using namespace covise;

class TetraGrid
{
private:
    // the grid-data
    float *xCoord, *yCoord, *zCoord;
    int numElem, numConn, numPoints;
    int *elemList, *connList;
    int BOX_INCREASE_FACTOR;

    // the neighborlist
    float *neighborList;
    int neighborListComplete;

    // the volumes
    float *volumeList;

    // the velocity-field
    float *velU, *velV, *velW;

    // dimensions of box used for fast searching of a cell containing
    // a given point
    float edgeDim;

    // stuff for time-neighbor-stuff

    int *lastTimeNeighbor;

    int TNCache[1000][2];
    int numTNCache, TNCacheIndex;
    int searchMode;

    // compute volume of a tetrahedron with the given vertices
    float tetraVolume(float p0[3], float p1[3], float p2[3], float p3[3]);

    // compute intersection with one single side (spanned by c0, c1, c2)
    float computeSingleSide(float x, float y, float z, int c0, int c1, int c2, float u, float v, float w);

    // via which side has the given cell been left by the trace starting
    // at the given position with the given velocity. We also compute
    // the time (dt) after which we pass that side
    int leftWhichSide(int el, float &x, float &y, float &z, float u, float v, float w, float &dt);

    // the following function is used by leftWhichSide to find out
    // via which side we left the cell after what time (s,t)
    void analyzeCase(float t0, float t1, float t2, float t3,
                     float &t, int &s,
                     float &x, float &y, float &z,
                     int c0, int c1, int c2, int c3);

public:
    TetraGrid(const coDoUnstructuredGrid *grid, const coDoVec3 *velocity,
              const coDoFloat *volume, const coDoFloat *neighbor, int smode, int numTraces);
    ~TetraGrid();

    // is the given point in the given cell ?
    //   (return -1 if not inside cell or return el if inside cell)
    int isInCell(int el, float x, float y, float z);
    int isInCell(int el, float x, float y, float z, float &u, float &v, float &w);

    // find actual cell by starting at the given cell or perform an
    // overall search if the given cell is -1
    //   (if no cell is found we return -1)
    int findActCell(float &x, float &y, float &z, float &u, float &v, float &w, int el = -1);

    // return volume of the given element
    float getVolume(int el)
    {
        return (volumeList[el]);
    };

    // get timeneighbor
    int getTimeNeighbor(int el, TetraGrid *nGrid, int traceNum);
    /*
      {
         return( (int)neighborList[(el*5)+4] );
      };*/

    // for precomputing
    coDoFloat *buildNewNeighborList(char *name, TetraGrid *toGrid);
};
#endif // __TETRAGRID_H
