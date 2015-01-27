/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS EdgeCollapseBasis
//
//  EdgeCollapseBasis defines the interface for the simplification
//  of a grid of polygons
//
//  Initial version: end of 2003
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes: 19.05.2004, SL: CheckDirection is added to support
//                          a better preservation of border lines

#ifndef _EDGE_COLLAPSE_BASIS_H_
#define _EDGE_COLLAPSE_BASIS_H_

#include "util/coviseCompat.h"

#include "VertexContainer.h"
#include "TriangleContainer.h"
#include "EdgeContainer.h"

class PQ;

class EdgeCollapseBasis
{
public:
    /// Constructor: the meaning of the parameters is trivial
    /// except the last three. These were included
    /// in order to investigate the impact on performance
    /// of the container type choices in the algorithm
    EdgeCollapseBasis(const vector<float> &x_c,
                      const vector<float> &y_c,
                      const vector<float> &z_c,
                      const vector<int> &conn_list,
                      const vector<float> &data_c,
                      const vector<float> &normals_c,
                      VertexContainer::TYPE vertCType,
                      TriangleContainer::TYPE triCType,
                      EdgeContainer::TYPE edgeCType);

    /// here is the meat of the calculation
    virtual int EdgeContraction(int num_max) = 0;
    /// destructor
    virtual ~EdgeCollapseBasis();
    /// This function is called to get the output
    void LeftEntities(vector<int> &leftTriangles,
                      vector<float> &leftVertexX,
                      vector<float> &leftVertexY,
                      vector<float> &leftVertexZ,
                      vector<float> &leftData,
                      vector<float> &leftNormals) const;
    // for debugging purposes
    bool PQ_OK() const;

protected:
    int CheckDirection(const Vertex *, const Vertex *, const Edge *) const;
    VertexContainer *_vertexList;
    TriangleContainer *_triangleList;
    EdgeContainer *_edgeSet;
    PQ *_pq;

private:
};
#endif
