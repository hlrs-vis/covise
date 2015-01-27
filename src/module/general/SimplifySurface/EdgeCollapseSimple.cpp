/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "EdgeCollapseSimple.h"
#include "PQ.h"
#include "Point.h"
#include "Vertex.h"
#include "Edge.h"
#include "Triangle.h"
#ifdef WIN32
#include <iterator>
#endif
#include "util/coviseCompat.h"

//#include <algorithm>

using namespace std;

EdgeCollapseSimple::EdgeCollapseSimple(const vector<float> &x_c,
                                       const vector<float> &y_c, const vector<float> &z_c,
                                       const vector<int> &conn_list,
                                       const vector<float> &data_c,
                                       const vector<float> &normals_c,
                                       VertexContainer::TYPE vertCType,
                                       TriangleContainer::TYPE triCType,
                                       EdgeContainer::TYPE edgeCType)
    : EdgeCollapseBasis(x_c, y_c, z_c, conn_list, data_c, normals_c,
                        vertCType, triCType, edgeCType)
{
    _vertexList->MakeQFull();
}

EdgeCollapseSimple::~EdgeCollapseSimple()
{
}

int
EdgeCollapseSimple::EdgeContraction(int num_max)
{
    int ret = 0;
    Edge *theEdge = _pq->top();
    if (!theEdge)
    {
        return -1;
    }
    const Vertex *v0 = theEdge->v0();
    const Vertex *v1 = theEdge->v1();
    if (v0->ValenceTooHigh(num_max) || v1->ValenceTooHigh(num_max))
    {
        _pq->pop();
        return 0;
    }
    Vertex *modifiable_v0 = const_cast<Vertex *>(v0);
    Vertex *modifiable_v1 = const_cast<Vertex *>(v1);

    // work out triangle set differences for the two vertices
    vector<const Triangle *> diff_tr_set_0, diff_tr_set_1;
    std::insert_iterator<vector<const Triangle *> > res_0_ins(diff_tr_set_0, diff_tr_set_0.begin());
    pair<set<const Triangle *>::const_iterator, set<const Triangle *>::const_iterator>
        tr0_inter = v0->triangle_interval();
    pair<set<const Triangle *>::const_iterator, set<const Triangle *>::const_iterator>
        tr1_inter = v1->triangle_interval();
    set_difference(tr0_inter.first, tr0_inter.second,
                   tr1_inter.first, tr1_inter.second, res_0_ins);

    std::insert_iterator<vector<const Triangle *> > res_1_ins(diff_tr_set_1, diff_tr_set_1.begin());
    set_difference(tr1_inter.first, tr1_inter.second,
                   tr0_inter.first, tr0_inter.second, res_1_ins);

    // check geometric coordinate change for all these triangles
    vector<const Triangle *>::const_iterator tr_p_it;
    for (tr_p_it = diff_tr_set_0.begin(); tr_p_it != diff_tr_set_0.end();
         ++tr_p_it)
    {
        if ((*tr_p_it)->visible() && !(*tr_p_it)->CheckSide(v0, theEdge->popt()))
        {
            _pq->pop();
            return 0;
        }
    }
    for (tr_p_it = diff_tr_set_1.begin(); tr_p_it != diff_tr_set_1.end();
         ++tr_p_it)
    {
        if ((*tr_p_it)->visible() && !(*tr_p_it)->CheckSide(v1, theEdge->popt()))
        {
            _pq->pop();
            return 0;
        }
    }

    // we also want to test, if boundary edges change
    // their orientation too much. In that case, we
    // do not collapse the edge
    if (CheckDirection(v0, v1, theEdge) == 0)
    {
        _pq->pop();
        return 0;
    }

    // make common triangles invisible
    vector<const Triangle *> intersec_tr;
    insert_iterator<vector<const Triangle *> > intersec_tr_ins(intersec_tr, intersec_tr.begin());
    set_intersection(tr0_inter.first, tr0_inter.second,
                     tr1_inter.first, tr1_inter.second, intersec_tr_ins);
    for (tr_p_it = intersec_tr.begin(); tr_p_it != intersec_tr.end(); ++tr_p_it)
    {
        Triangle *modifiable_triangle = const_cast<Triangle *>(*tr_p_it);
        if ((*tr_p_it)->visible())
        {
            ++ret;
        }
        modifiable_triangle->invisible();
    }

    // update point
    modifiable_v0->UpdatePoint(theEdge->popt(), v1);

    // update v0.edge_set cost
    // list<Triangle *> diff_tr_set_0,diff_tr_set_1;
    const Q *q1 = v1->QFull();
    if (q1 != NULL)
    {
        pair<set<const Edge *>::const_iterator, set<const Edge *>::const_iterator> edge0_interval = v0->edge_interval();
        set<const Edge *, EdgeCompare> sorted_edge0_interval;
        for (set<const Edge *>::const_iterator edge0_it = edge0_interval.first; edge0_it != edge0_interval.second;
             ++edge0_it)
        {
            sorted_edge0_interval.insert(*edge0_it);
        }

        for (set<const Edge *, EdgeCompare>::const_iterator edge0_it = sorted_edge0_interval.begin(); edge0_it != sorted_edge0_interval.end();
             ++edge0_it)
        {
            if (*edge0_it != theEdge)
            {
                Edge *modifiable_edge0_it = (Edge *)(*edge0_it);
                if (q1 != NULL)
                {
                    modifiable_edge0_it->AddQ(*q1);
                }
                modifiable_edge0_it->new_cost(Edge::DO_NOT_ADD_TRIANGLES);
                _pq->reorder(modifiable_edge0_it);
            }
        }
        modifiable_v0->AccumulateQFull(q1);
    }

    // eliminate or accumulate new edges from v1
    const Q *q0 = v0->QFull();
    pair<set<const Edge *>::const_iterator, set<const Edge *>::const_iterator> edge1_interval = v1->edge_interval();
    set<const Edge *, EdgeCompare> sorted_edge1_interval;
    for (set<const Edge *>::const_iterator edge1_it = edge1_interval.first; edge1_it != edge1_interval.second;
         ++edge1_it)
    {
        sorted_edge1_interval.insert(*edge1_it);
    }
    for (set<const Edge *, EdgeCompare>::const_iterator edge1_it = sorted_edge1_interval.begin(); edge1_it != sorted_edge1_interval.end();
         ++edge1_it)
    {
        if (*edge1_it == theEdge)
        {
            continue;
        }
        Edge edge1_it_copy(*(*edge1_it));
        edge1_it_copy.UpdateVertex(v1, v0);
        Edge *modifiable_edge = (Edge *)(*edge1_it);
        if (!v0->IsInEdgeSet(&edge1_it_copy)) // not found, it is added below
        {
            // this instruction would be  potentially dangerous
            // because we would be manipulating an object in set _edgeSet
            // and thus violating the ordering of this container...
            // !!!!!!!!!!!!!!(*edge1_it)->UpdateVertex(v1,v0);
            // ... that is why we have to redress this before
            // proceeding any further
            pair<Edge *, bool> out = _edgeSet->insert(edge1_it_copy);
            assert(out.second);
            Edge &new_edge = *(out.first);

            // eliminate *edge1_it from the vertex
            // different from v0 in edge1_it_copy
            Vertex *avertex = const_cast<Vertex *>(edge1_it_copy.v0());
            if (avertex == v0)
            {
                avertex = const_cast<Vertex *>(edge1_it_copy.v1());
            }

            assert(avertex != v0 && avertex != v1);
            avertex->erase_edge(*edge1_it);

            _pq->remove(modifiable_edge);
            _edgeSet->erase(modifiable_edge);

            modifiable_v0->add_edge(&new_edge);
            avertex->add_edge(&new_edge);

            if (q0 != NULL)
            {
                new_edge.AddQ(*q0);
                new_edge.new_cost(Edge::DO_NOT_ADD_TRIANGLES);
            }
            _pq->push(&new_edge);
        }
        else // repeated
        {
            _pq->remove(modifiable_edge);
            // *edge1_it will be erased, prevent dangling pointers
            Vertex *avertex = const_cast<Vertex *>((*edge1_it)->v0());
            if (avertex == v1)
            {
                avertex = const_cast<Vertex *>((*edge1_it)->v1());
            }
            avertex->erase_edge(modifiable_edge);
            _edgeSet->erase(modifiable_edge); // *edge1_it  is erased here
        }
    }

    // do not remove redundant triangles
    for (tr_p_it = diff_tr_set_1.begin(); tr_p_it != diff_tr_set_1.end();
         ++tr_p_it)
    {
        Triangle *modifiable_triangle = const_cast<Triangle *>(*tr_p_it);
        modifiable_triangle->UpdateVertex(modifiable_v1, modifiable_v0);
        modifiable_v0->add_tr(modifiable_triangle);
    }

    modifiable_v0->erase_edge(theEdge);
    _pq->remove(theEdge);
    _edgeSet->erase(theEdge);
    _vertexList->erase(modifiable_v1);
    return ret;
}
