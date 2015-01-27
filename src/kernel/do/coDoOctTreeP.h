/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_OCTTREE_P_H
#define CO_DO_OCTTREE_P_H

#include "coDoBasisTree.h"
#include <vector>

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
namespace covise
{

/**
 * Derived class from coDoBasisTree,
 * used for cell location in unstructured grids.
 */
class DOEXPORT coDoOctTreeP : public coDoBasisTree
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

public:
    /** Constructor
       * @param n object name
       */
    coDoOctTreeP(const coObjInfo &info);

    /** Constructor
       * @param n object name
       * @param arr pointer to coShmArray
       */
    coDoOctTreeP(const coObjInfo &info, coShmArray *arr);

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
    coDoOctTreeP(const coObjInfo &info, int nelem, int nconn, int ncoord,
                 int *el, int *conn, float *x_c, float *y_c, float *z_c,
                 int normal_size = NORMAL_SIZE,
                 int max_no_levels = MAX_NO_LEVELS,
                 int min_small_enough = MIN_SMALL_ENOUGH,
                 int crit_level = CRIT_LEVEL,
                 int limit_fX = INT_MAX,
                 int limit_fY = INT_MAX,
                 int limit_fZ = INT_MAX);

    /** LoadCellPopulations delivers arrays of cells in near-by octtrees
       * @ param oct_trees_cells array of 9 pointers to infinite arrays of cells
       * @ param point
       */
    void LoadCellPopulations(std::vector<int> **oct_trees_cells, const float *point,
                             int search_level);

    /// Destructor
    virtual ~coDoOctTreeP();

    void setInfo(int nelem_l,
                 int nconn_l,
                 int ncoord_l,
                 int *el,
                 int *conn,
                 float *x_c,
                 float *y_c,
                 float *z_c);

protected:
    coDoOctTreeP *cloneObject(const coObjInfo &newinfo) const;

private:
};
}
#endif
