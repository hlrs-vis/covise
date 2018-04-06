/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Triangle
//
//  This class represents a triangle in a polygon grid
//
//  Initial version: end of 2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _SS_TRIANGLE_H_
#define _SS_TRIANGLE_H_

class Vertex;
class Q;
class Point;

#include "util/coviseCompat.h"

class Triangle
{
public:
    /// constructor using triangle vertices
    Triangle(const Vertex *v0, const Vertex *v1, const Vertex *v2);
    /// copy constructor
    Triangle(const Triangle &rhs);
    /// destructor
    virtual ~Triangle();
    /// CheckSide tests whether the normal of a triangle
    /// changes too much when vertex v is moved to point p
    bool CheckSide(const Vertex *v, const Point *point) const;
    /// UpdateVertex updates one of the pointers _v0, _v1, _v2
    /// (the one equal to old), and sets it to neu. This is relevant
    /// when an edge is collapsed, and a vertex has to be substituted by
    /// the other vertex of the edge
    void UpdateVertex(Vertex *old, Vertex *neu);
    /// Sum performs the sum of all cost functions in an interval
    // of triangles given by two iterators (when using vector containers)
    static Q *Sum(vector<const Triangle *>::iterator start,
                  vector<const Triangle *>::iterator ende);
    /// Sum performs the sum of all cost functions in an interval
    // of triangles given by two iterators (when using set containers)
    static Q *Sum(set<const Triangle *>::iterator start,
                  set<const Triangle *>::iterator ende);
    /// QBound returns the boundary-associated cost for a boundary
    /// edge va-vb of this triangle
    Q *QBound(const Vertex *va, const Vertex *vb) const;
    /// operator[] returns one of the three _v0, _v1, _v2 for i==0,1,2
    Vertex *operator[](int i);
    /// operator[] has a const version
    const Vertex *operator[](int i) const;
    friend ostream &operator<<(ostream &, const Triangle &);
    /// invisible marks this triangle as invisible, when it
    /// disappears because of an edge having been collapsed
    void invisible();
    /// visible returns true if this triangle has not been collapsed
    bool visible() const;
    void print(std::string pre);
    bool operator==(const Triangle &rhs) const;
    bool operator<(const Triangle &rhs) const;

protected:
private:
    bool _invisible;
    Vertex *_v0;
    Vertex *_v1;
    Vertex *_v2;
    Q *_q;
};

struct TriangleCompare
{
    bool operator()(const Triangle *a, const Triangle *b) const
    {
        return (*a) < (*b);
    }
};

#endif
