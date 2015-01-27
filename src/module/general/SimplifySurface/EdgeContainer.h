/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _EDGE_CONTAINER_H_
#define _EDGE_CONTAINER_H_

#include <util/coviseCompat.h>
#include <alg/unordered_set.h>
#include "Edge.h"

class Triangle;
class Vertex;
class Edge;
class PQ;

class EdgeContainer
{
public:
    enum TYPE
    {
        HASHED_SET
    };
    static EdgeContainer *NewEdgeContainer(TYPE, size_t no_tr);
    virtual pair<Edge *, bool> insert(const Vertex *v0, const Vertex *v1, const Triangle *tr) = 0;
    virtual pair<Edge *, bool> insert(const Edge &) = 0;
    virtual int size() const = 0;
    virtual void ComputeCost(PQ *) = 0;
    virtual ~EdgeContainer();
    virtual void erase(Edge *) = 0;
    virtual void print(std::string pre) = 0;
};

#ifdef HASH_NAMESPACE
namespace HASH_NAMESPACE
{
#endif
#ifdef HASH_NAMESPACE2
namespace HASH_NAMESPACE2
{
#endif
    template <>
    struct hash<Edge>
    {
        size_t operator()(const Edge &edge) const
        {
            //return size_t(edge.v0() + edge.v1()->label()*_dimension);
            return (size_t)edge;
        }
    };
#ifdef HASH_NAMESPACE2
}
#endif
#ifdef HASH_NAMESPACE
}
#endif

class EdgeHashedSet : public EdgeContainer
{
public:
    EdgeHashedSet(size_t no_tr);
    virtual ~EdgeHashedSet();
    virtual pair<Edge *, bool> insert(const Vertex *v0, const Vertex *v1, const Triangle *tr);
    virtual pair<Edge *, bool> insert(const Edge &);
    virtual int size() const;
    virtual void ComputeCost(PQ *);
    virtual void erase(Edge *);
    virtual void print(std::string pre);

protected:
private:
    typedef unordered_set<Edge> EdgeSet;
    EdgeSet _edgeSet;
};
#endif
