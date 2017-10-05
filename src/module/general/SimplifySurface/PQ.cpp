/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "PQ.h"
#include "Edge.h"

PQ::PQ(int size)
{
    _edgeList = new Edge *[size];
}

PQ::~PQ()
{
    delete[] _edgeList;
}

bool
PQ::OK() const
{
    int length = int(_edgePosition.size());
    if (length == 0)
    {
        return true;
    }
    float costTop = _edgeList[0]->cost();
    int i;
    for (i = 1; i < length; ++i)
    {
        if (costTop > _edgeList[i]->cost())
        {
            return false;
        }
    }
    return true;
}

Edge *
PQ::top()
{
    if (_edgePosition.size() == 0)
    {
        return NULL;
    }
    return _edgeList[0];
}

void
PQ::push(Edge *edge)
{
    int N = int(_edgePosition.size());
    _edgePosition.insert(pair<Edge *, int>(edge, N));
    _edgeList[N] = edge;
    int test_node = N;
    while (test_node > 0)
    {
        int parent_node = (test_node + 1) / 2 - 1;
        if (!Ordered(parent_node, test_node))
        {
            Swap(parent_node, test_node);
            test_node = parent_node;
        }
        else
        {
            break;
        }
    }
}

void
PQ::AdjustHeapDownwards(int element)
{
    int N = int(_edgePosition.size());
    int test_node = element;
    while (1)
    {
        int child;
        if ((test_node + 1) * 2 - 1 > N - 1) // no children
        {
            break;
        }
        if ((test_node + 1) * 2 > N - 1) // only one children (see 3 lines above)
        {
            child = (test_node + 1) * 2 - 1;
        }
        else if (Ordered((test_node + 1) * 2 - 1, (test_node + 1) * 2))
        {
            child = (test_node + 1) * 2 - 1;
        }
        else
        {
            child = (test_node + 1) * 2;
        }
        if (Ordered(test_node, child))
        {
            break;
        }
        else
        {
            Swap(test_node, child);
            test_node = child;
        }
    }
}

void
PQ::pop()
{
    int N = int(_edgePosition.size());
    if (N == 0)
    {
        return;
    }
    Swap(0, N - 1);
    _edgePosition.erase(_edgeList[N - 1]);
    AdjustHeapDownwards(0);
}

void
PQ::reorder(Edge *edge)
{
    // is it in the list?
    // else return
    EdgePositionMap::iterator it = _edgePosition.find(edge);
    if (it == _edgePosition.end())
    {
        return;
    }
    int test_node = it->second;
    int father = (test_node + 1) / 2 - 1;
    // is it greater than the father? -> make it float upwards
    if (test_node > 0 && !Ordered(father, test_node))
    {
        do
        {
            Swap(father, test_node);
            test_node = father;
            father = (test_node + 1) / 2 - 1;
        } while (test_node > 0 && !Ordered(father, test_node));
    }
    // else is it smaller than any of the children? -> make it sink
    else
    {
        AdjustHeapDownwards(test_node);
    }
}

void
PQ::Swap(int pos0, int pos1)
{
    Edge *edge0 = _edgeList[pos0];
    Edge *edge1 = _edgeList[pos1];
    _edgeList[pos0] = edge1;
    _edgeList[pos1] = edge0;
    _edgePosition[edge0] = pos1;
    _edgePosition[edge1] = pos0;
}

bool
PQ::Ordered(int smaller, int greater)
{
    Edge *EGreater = _edgeList[greater];
    Edge *ESmaller = _edgeList[smaller];
    return !(ESmaller->cost() > EGreater->cost());
}

void
PQ::remove(Edge *edge)
{
    // is it in the list?
    // else return
    EdgePositionMap::iterator it = _edgePosition.find(edge);
    if (it == _edgePosition.end())
    {
        return;
    }
    int N = int(_edgePosition.size());
    Edge *reorderEdge = _edgeList[N - 1];
    Swap(it->second, N - 1);
    _edgePosition.erase(edge);
    reorder(reorderEdge); // here an extra info might be helpful
}

void PQ::print(std::string pre)
{
    std::cout << pre << "PQ" << std::endl;

    int size = int(_edgePosition.size());
    for (int i = 0; i < size; ++i)
    {
        _edgeList[i]->print(pre + " ");
    }

    EdgePositionMap::iterator it = _edgePosition.begin();
    while (it != _edgePosition.end())
    {
        it->first->print(pre + " ");
        std::cout << pre << " Position " << it->second << std::endl;
        ++it;
    }
}

ostream &
operator<<(ostream &out, const PQ &pq)
{
    int size = int(pq._edgePosition.size());
    int i;
    for (i = 0; i < size; ++i)
    {
        out << *(pq._edgeList[i]);
    }
    return out;
}
