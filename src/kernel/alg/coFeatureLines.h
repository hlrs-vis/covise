/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS coFeatureLines
//
//  coFeatureLines contains utilities for feature line extraction
//
//  Initial version: 21.05.2004 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _COVISE_FEATURE_LINES_H_
#define _COVISE_FEATURE_LINES_H_

#include "coMiniGrid.h"
#include "MagmaUtils.h"

namespace covise
{

class ALGEXPORT coFeatureLines
{
public:
    // cutPoly separates triangles whose normals span an angle whose cosinus
    // is smaller than argument coseno. connList and {x,y,z}coord are used
    // for input and output. xn,yn,zn are the normals per cell.
    // !!!! At the moment this function is limited to triangular grids !!!!
    // If this condition is not satisfied by your grid, you may want to
    // use function Triangulate.
    static void cutPoly(float coseno,
                        const vector<int> &elemList,
                        // connectivity list (input and output)
                        vector<int> &connList,
                        // coordinate points (input and output)
                        vector<float> &xcoord,
                        vector<float> &ycoord,
                        vector<float> &zcoord,
                        // input normals (per cell)
                        const vector<float> &xn,
                        const vector<float> &yn,
                        const vector<float> &zn,
                        // feature lines are returned in these arrays
                        vector<int> &ll,
                        vector<int> &cl,
                        vector<float> &lx,
                        vector<float> &ly,
                        vector<float> &lz,
                        // domain lines are returned in these arrays
                        vector<int> &dll,
                        vector<int> &dcl,
                        vector<float> &dlx,
                        vector<float> &dly,
                        vector<float> &dlz);
    // Triangulate triangulates a grid of polygons, which are assumed
    // to be convex.
    // In tri_conn_list you get a connectivity list describing
    // the triangulated grid, which reuses the same coordinate lists: x,y,z
    static void
    Triangulate(vector<int> &tri_conn_list, // tri_conn_list is the output.
                vector<int> &tri_codes, // for each triangle we get the
                // label of the polygon it is part of.
                int num_poly, // number of original polygons.
                int num_conn, // length of the connectivity list.
                const int *poly_list,
                const int *conn_list,
                const float *x, const float *y, const float *z);

private:
    // DifferentOrientation returns true when two cells (cell and other cell)
    // are 'almost' parallel oriented. xn,yn,zn are normals per cell.
    static bool
    DifferentOrientation(float coseno,
                         int elem,
                         int other_cell,
                         const vector<float> &xn,
                         const vector<float> &yn,
                         const vector<float> &zn);
    // SplitNode performs the splitting of a grid referred
    // to in the comments to cutPoly around
    // a node 'node'. Nodal and cell neighbourhoods are required
    // as defined by class MagmaUtils. mini_kanten is the set
    // of feature edges in the whole grid separating cells
    // which are NOT 'almost' parallel.
    // A mini-grid is calculated, as the set of cells around node 'node'
    // and the RainAlgorithm is calculated is calculated in this grid
    // restricting mini_kanten to the edges of mini-grid. Node 'node'
    // is then duplicated for each 'extra'-subdomain of the rain
    // algorithm. So, in successive calls of SplitNode, the grid
    // gets separated.
    // You may think of possible alternatives, for instance, not
    // dupplicating a node for a subdomain, when you may find a cell
    // in the subdomain at issue and another in another subdomain
    // for which DifferentOrientation returns false..
    static void
    SplitNode(int node,
              const vector<int> &elemList,
              vector<int> &connList,
              vector<float> &xcoord,
              vector<float> &ycoord,
              vector<float> &zcoord,
              const vector<int> &nodal_number_neigh,
              const vector<int> &nodal_start_neigh,
              const vector<int> &nodal_neighbours,
              const vector<int> &elem_start_neigh,
              const vector<int> &elem_number_neigh,
              const vector<int> &elem_neighbours,
              const vector<MagmaUtils::Edge> &edge_neighbours,
              const coMiniGrid::Border &mini_kanten);
    // MaxAngleVertex returns the vertex of a polygon
    // with the largest angle
    static int
    MaxAngleVertex(int num_conn,
                   //                  const int *start_conn,
                   const float *x,
                   const float *y,
                   const float *z);
};
}
#endif
