/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SS_PQ_H_
#define _SS_PQ_H_

// we are not using priority_queue because
// it does not provide all the required additional functionality:
// reorder, remove
#include <util/coviseCompat.h>
#include <alg/unordered_set.h>

class Edge;

struct hash_edge_pointer
{
    size_t operator()(Edge *edge) const
    {
        return size_t(edge);
    }
};

class PQ
{
public:
    PQ(int size);
    virtual ~PQ();
    void push(Edge *);
    Edge *top();
    void pop();
    void reorder(Edge *);
    void remove(Edge *);

    bool OK() const; // just for debugging purposes
    void print(std::string pre);

    friend ostream &operator<<(ostream &out, const PQ &pq);

private:
    bool Ordered(int smaller, int greater);
    void Swap(int pos0, int pos1);
    void AdjustHeapDownwards(int element);
    Edge **_edgeList;

#if defined(WIN32) || defined(__hpux) || defined(CO_ia64icc)
    typedef unordered_map<Edge *, int> EdgePositionMap;
#else
    typedef map<Edge *, int> EdgePositionMap;
#endif

    EdgePositionMap _edgePosition;
};
#endif
