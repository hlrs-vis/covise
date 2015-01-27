/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_OCTTREE_H
#define CO_DO_OCTTREE_H

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS OctTree
//
//  This class manages an octtree structure for cell location in unstructured grids
//
//  Initial version: 2001-12-10 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#include "coDoBasisTree.h"

namespace covise
{

/**
 * Derived class from coDoBasisTree,
 * used for cell location in unstructured grids.
 */
class DOEXPORT coDoOctTree : public coDoBasisTree
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

public:
    /** Constructor
       * @param n object name
       */
    coDoOctTree(const coObjInfo &info);

    /** Constructor
       * @param n object name
       * @param arr pointer to coShmArray
       */
    coDoOctTree(const coObjInfo &info, coShmArray *arr);

    /** Constructor for deserialization
       * @param info object name
       * @param cellListSize number of cellList items
       * @param macroCellListSize number of macroCellList items
       * @param cellBBoxesSize number of cellBBoxes items
       * @param gridBBoxSize number of gridBBox items
       */
    coDoOctTree(const coObjInfo &info,
                int cellListSize,
                int macroCellListSize,
                int cellBBoxesSize,
                int gridBBoxSize)
        : coDoBasisTree(info, "OCTREE", "OCT-TREE", cellListSize, macroCellListSize, cellBBoxesSize, gridBBoxSize)
    {
    }

    /** Constructor
       * @param n object name
       * @param nelem number of elements in the grid
       * @param nconn length of the grid connectivity list
       * @param ncoord number of points in the grid
       * @param el array with "pointers" to the connectivity list for each element
       * @param conn connectivity list
       * @param x_c X coordinates of the grid points
       * @param y_c Y coordinates of the grid points
       * @param z_c Z coordinates of the grid points
       */
    coDoOctTree(const coObjInfo &info, int nelem, int nconn, int ncoord,
                int *el, int *conn, float *x_c, float *y_c, float *z_c,
                int normal_size = NORMAL_SIZE,
                int max_no_levels = MAX_NO_LEVELS,
                int min_small_enough = MIN_SMALL_ENOUGH,
                int crit_level = CRIT_LEVEL,
                int limit_fX = INT_MAX,
                int limit_fY = INT_MAX,
                int limit_fZ = INT_MAX);

    /** IsInBBox: returns 1 if a point is in the bounding box of a cell
       * @param cell integer cell identifier
       * @param no_e number of cells
       * @param point array to 3 point coordinates
       * @return 1 if the point is in the bounding box of the cell, 0 otherwise
       */
    int IsInBBox(int cell, int no_e, const float *point) const;

    /// Destructor
    virtual ~coDoOctTree();

protected:
    coDoOctTree *cloneObject(const coObjInfo &newinfo) const;

private:
};
}
#endif
