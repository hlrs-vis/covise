/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Vertex
//
//  This class represents a vertex in a polygon grid
//
//  Initial version: end of 2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _SS_VERTEX_H_
#define _SS_VERTEX_H_

#include "util/coviseCompat.h"
//#include <set>

using std::set;

class Point;
class Triangle;
class Edge;
class Q;

class Vertex
{
public:
    /// constructor: vertex label, coordinates, data dimensionality,
    /// data array, normal vector
    Vertex(int label, float x, float y, float z, int no_data, const float *data,
           const float *normal);
    /// copy constructor
    Vertex(const Vertex &rhs);
    /// destructor
    virtual ~Vertex();

    /// UpdatePoint: when an edge is collapsed, a vertex is moved
    /// to a new point. Point includes here geoometry and data.
    /// The normal vector is interpolated with that of the other
    /// vertex of the collapsed edge (v1)
    void UpdatePoint(const Point *point, const Vertex *v1);

    /// point returns the Point of this vertex (coordinates + data)
    const Point *point() const;
    /// QBound returns a point to the cost function associated with
    /// the boundary
    const Q *QBound() const;
    /// QFull returns the full associated cost function
    const Q *QFull() const;
    /// MakeBoundary accumulates cost functions from boundary nodes
    void MakeBoundary();
    /// MakeQFull works out _q_full from _q_bound and the
    /// cost functions of associated triangles
    void MakeQFull();
    /// AccumulateQFull adds a new cost contribution to _q_full
    void AccumulateQFull(const Q *q);
    /// AccumulateBoundary adds a new cost contribution to _q_bound
    void AccumulateBoundary(const Q *q);

    /// add_tr inserts a new associated triangle to _tr_set
    void add_tr(Triangle *tr);
    /// add_edge inserts a new associated edge to _edge_set
    void add_edge(const Edge *edge);
    /// erase_edge deletes a no longer existing edge from _edge_set
    void erase_edge(const Edge *edge);
    /// erase tags this Vertex as no longer existing by setting _label to -1
    void erase();
    /// IsInEdgeSet tests whether an edge is associated to this Vertex
    bool IsInEdgeSet(const Edge *edge) const;

    /// ValenceTooHigh tests whether the valence of this vertex (number
    /// of associated edges) is greater than max_valence
    bool ValenceTooHigh(int max_valence) const;

    /// label returns _label
    int label() const;

    bool operator==(const Vertex &rhs) const;
    bool operator<(const Vertex &rhs) const;

    pair<set<const Edge *>::const_iterator, set<const Edge *>::const_iterator> edge_interval() const;
    pair<set<const Triangle *>::const_iterator, set<const Triangle *>::const_iterator> triangle_interval() const;

    /// BoundaryEdges outputs the set of associated boundary edges.
    void BoundaryEdges(set<const Edge *> &boundaryEdges) const;

    /// normal returns an array with the normal vector
    const float *normal() const;

    friend ostream &operator<<(ostream &, const Vertex &);

    void print(std::string pre);

protected:
private:
    int _label;
    Point *_point;
    set<const Triangle *> _tr_set;
    set<const Edge *> _edge_set;
    Q *_q_bound;
    Q *_q_full;
    float *_normal;
};

struct VertexCompare
{
    bool operator()(const Vertex *a, const Vertex *b)
    {
        return (*a) < (*b);
    }
};

#endif
