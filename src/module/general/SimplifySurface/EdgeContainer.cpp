/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "EdgeContainer.h"
#include "PQ.h"
#include "Vertex.h"
#include "Point.h"
#define SIZE_FACTOR 2

EdgeContainer *
EdgeContainer::NewEdgeContainer(TYPE type, size_t no_tr)
{
    switch (type)
    {
    case HASHED_SET:
        return (new EdgeHashedSet(no_tr));
    }
    return NULL;
}

EdgeContainer::~EdgeContainer()
{
}

EdgeHashedSet::EdgeHashedSet(size_t no_tr)
{
#if !defined(_WIN32) && !defined(__hpux) && !defined(CO_ia64icc) && !defined(__GNUC__)
    _edgeSet.resize(no_tr * SIZE_FACTOR);
#else
    (void)no_tr;
#endif
}

EdgeHashedSet::~EdgeHashedSet()
{
}

pair<Edge *, bool>
EdgeHashedSet::insert(const Vertex *v0, const Vertex *v1, const Triangle *tr)
{
    pair<EdgeSet::iterator, bool> insert_ret = _edgeSet.insert(Edge(v0, v1, tr));
    pair<Edge *, bool> ret;
    ret.first = const_cast<Edge *>(&(*(insert_ret.first)));
    ret.second = insert_ret.second;
    return ret;
}

pair<Edge *, bool>
EdgeHashedSet::insert(const Edge &edge)
{
    pair<EdgeSet::iterator, bool> insert_ret = _edgeSet.insert(edge);
    pair<Edge *, bool> ret;
    ret.first = const_cast<Edge *>(&(*(insert_ret.first)));
    ret.second = insert_ret.second;
    return ret;
}

int
EdgeHashedSet::size() const
{
    return int(_edgeSet.size());
}

void
EdgeHashedSet::ComputeCost(PQ *pq)
{
    for (EdgeSet::iterator edge_it = _edgeSet.begin(); edge_it != _edgeSet.end(); ++edge_it)
    {
        Edge *edge_p = const_cast<Edge *>(&(*edge_it));
        // the triangle assoc. q-contribution
        // is calculated here
        edge_p->new_cost(Edge::ADD_TRIANGLES);
        pq->push(edge_p);
    }
}

void
EdgeHashedSet::erase(Edge *edge)
{
    _edgeSet.erase(*edge);
}

void EdgeHashedSet::print(std::string pre)
{
    std::cout << pre << "EdgeContainer" << std::endl;

    for (EdgeSet::iterator edge_it = _edgeSet.begin(); edge_it != _edgeSet.end(); ++edge_it)
    {
        Edge *edge_p = const_cast<Edge *>(&(*edge_it));
        edge_p->print(pre + " ");
    }
}
