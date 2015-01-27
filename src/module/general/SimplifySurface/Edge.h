/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Edge
//
//  This class represents an edge in a polygon grid
//
//  Initial version: end of 2003
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes: 19.05.2004, SL: CheckDirection is added to support
//                          a better preservation of border lines
#ifndef _SS_EDGE_H_
#define _SS_EDGE_H_
#include "util/coviseCompat.h"

class Vertex;
class Triangle;
class Point;
class Q;

class Edge
{
public:
    // an edge is constructed when looping over triangles
    // an triangle edges
    Edge(const Vertex *v0, const Vertex *v1, const Triangle *tr);
    Edge(const Edge &rhs);
    virtual ~Edge();
    // we are using sets od Edges
    bool operator==(const Edge &rhs) const;
    bool operator<(const Edge &rhs) const;
    operator size_t() const;

    enum QCalc
    {
        DO_NOT_ADD_TRIANGLES,
        ADD_TRIANGLES
    };
    // ADD_TRIANGLES is used when the calculation is initialised,
    // afterwards, the cost function (_q) is changed incrementally
    // when another edge is collapsed, and the edge at issue
    // inherits some extra cost.
    void new_cost(QCalc calc);
    // inherit an extra cost function, q
    void AddQ(const Q &q);
    // get cost optimising the cost function
    float cost() const;

    // (un)mark as boundary edge
    void boundary(bool);
    // get if this is a boundary edge
    bool boundary() const;

    // v0, v1 get the associated nodes
    const Vertex *v0() const;
    const Vertex *v1() const;

    // popt gets the optimal new position
    // of the associated vertices if this edge is to be
    // collapsed
    const Point *popt() const;
    // gets cost associated with the boundary
    const Q *QBound() const;
    // called when a nera-by edge is collapsed and a vertex is
    // moved
    void UpdateVertex(const Vertex *vold, const Vertex *vnew);
    friend ostream &operator<<(ostream &out, const Edge &edge);
    // tests if the orientation changes too much.
    // this is important for border edges.
    bool CheckDirection(const Vertex *, const Point *) const;
    void print(std::string pre);

protected:
private:
    const Vertex *minV() const;
    const Vertex *maxV() const;
    const Triangle *_tr;
    const Vertex *_v0;
    const Vertex *_v1;
    Point *_popt;
    Q *_q;
    Q *_q_bound;
    float _cost;
    bool _boundary;
    float _direction[3]; // original direction
};

struct EdgeCompare
{
    bool operator()(const Edge *a, const Edge *b)
    {
        return (*a) < (*b);
    }
};

#endif
