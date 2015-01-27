/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS MagmaUtils
//
//  Neighbourhood calculation for nodes and cells.
//  You get also a function which calculates the border (lines)
//  of a polygon. And other functions, which extend a surface beyond
//  its borders, eventually with data.
//
//  Initial version: 29.02.2004 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _MAGMA_UTILS_H_
#define _MAGMA_UTILS_H_

#include "covise/covise.h"

namespace covise
{

class ALGEXPORT MagmaUtils
{
public:
    typedef std::pair<int, int> Edge;
    // NodeNeighbours calculates in neighbours the neighbours
    // cells for each node. start_neigh is used to get the starting
    // position in neighbours of the neighbours of a node, and number_neigh
    // is used to get the number of cell neighbours.
    // num_conn gives for each cell the number of vertices,
    // this may seem redundant with el, but it is not, when you are
    // using a subgrid.
    static void NodeNeighbours(const vector<int> &el,
                               const vector<int> &num_conn,
                               const vector<int> &cl,
                               int num_coord, // this is in most cases redundant...
                               // but not always!!!
                               vector<int> &start_neigh,
                               vector<int> &number_neigh,
                               vector<int> &neighbours);
    // CellNeighbours calculates in elem_neighbours the neighbours
    // cells for each cell. Two cells are neighbours in this context
    // not when they have one common node, but when they have one common edge.
    // For the meaning of elem_start_neigh, elem_number_neigh and
    // elem_neighbours, you may read the comments for NodeNeighbours.
    // edge_neighbours is as long as elem_neighbours. For each
    // neighbourhood relationship, a (the) common edge is kept in this array.
    // In general, you will have to use NodeNeighbours prior
    // calling CellNeighbours.
    static void CellNeighbours(const vector<int> &el,
                               const vector<int> &num_conn,
                               const vector<int> &cl,
                               // these fields come from NodeNeighbours
                               const vector<int> &nodal_start_neigh,
                               const vector<int> &nodal_number_neigh,
                               const vector<int> &nodal_neighbours, // elements touching a node
                               vector<int> &elem_start_neigh,
                               vector<int> &elem_number_neigh,
                               vector<int> &elem_neighbours,
                               vector<Edge> &edge_neighbours);
    // The output of DomainLines are two arrays: border_els, the
    // cells lying at the border, and border_edges, the edges defining
    // the border. You have to call CellNeighbours before using
    // this function.
    static void DomainLines(const vector<int> &el,
                            const vector<int> &num_conn,
                            const vector<int> &cl,
                            const vector<int> &elem_start_neigh,
                            const vector<int> &elem_number_neigh,
                            const vector<Edge> &edge_neighbours,
                            vector<int> &border_els,
                            vector<Edge> &border_edges);
    // the following two functions work at the moment
    // only for triangles!!!!
    // You are likely to append clExt and {x,y,z}Ext to cl and x,y,z
    // after calling these functions in order to get the whole
    // extended geometry
    static void ExtendBorder(float length, const vector<int> &border_els,
                             const vector<Edge> &border_edges,
                             const vector<int> &cl,
                             vector<int> &clExt,
                             const vector<float> &x,
                             const vector<float> &y, const vector<float> &z,
                             vector<float> &xExt,
                             vector<float> &yExt, vector<float> &zExt);
    static void ExtendBorderAndData(float length, const vector<int> &border_els,
                                    const vector<Edge> &border_edges,
                                    const vector<int> &cl,
                                    vector<int> &clExt,
                                    const vector<float> &x,
                                    const vector<float> &y, const vector<float> &z,
                                    const vector<float> &InData,
                                    vector<float> &xExt,
                                    vector<float> &yExt, vector<float> &zExt,
                                    vector<float> &OutData); // extended data
};
}
#endif
