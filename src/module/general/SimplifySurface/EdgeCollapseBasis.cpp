/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "EdgeCollapseBasis.h"
#include "Vertex.h"
#include "PQ.h"

EdgeCollapseBasis::EdgeCollapseBasis(const vector<float> &x_c,
                                     const vector<float> &y_c,
                                     const vector<float> &z_c,
                                     const vector<int> &conn_list,
                                     const vector<float> &data_c,
                                     const vector<float> &normals_c,
                                     VertexContainer::TYPE vertCType,
                                     TriangleContainer::TYPE triCType,
                                     EdgeContainer::TYPE edgeCType)
{
    int no_vertex = (int)x_c.size();
    _vertexList = VertexContainer::NewVertexContainer(vertCType);
    _triangleList = TriangleContainer::NewTriangleContainer(triCType);
    _edgeSet = EdgeContainer::NewEdgeContainer(edgeCType, conn_list.size() / 3);

    int no_tri = int(conn_list.size()) / 3;
    int vertex, tri;
    // set up _vertexList
    _vertexList->reserve(no_vertex);
    int no_data_per_vertex = 0;
    if (no_vertex > 0)
    {
        no_data_per_vertex = int(data_c.size() / no_vertex);
    }
    float *data = new float[no_data_per_vertex];
    float normals[3];
    for (vertex = 0; vertex < no_vertex; ++vertex)
    {
        int i;
        int base = vertex * no_data_per_vertex;
        for (i = 0; i < no_data_per_vertex; ++i)
        {
            data[i] = data_c[base + i];
        }
        if (normals_c.size() > 0)
        {
            int base_n = vertex * 3;
            normals[0] = normals_c[base_n];
            normals[1] = normals_c[base_n + 1];
            normals[2] = normals_c[base_n + 2];
            _vertexList->insert(vertex, x_c[vertex], y_c[vertex], z_c[vertex],
                                no_data_per_vertex, data, normals);
        }
        else
        {
            _vertexList->insert(vertex, x_c[vertex], y_c[vertex], z_c[vertex],
                                no_data_per_vertex, data, NULL);
        }
    }
    delete[] data;
    // set up _triangleList
    _triangleList->reserve(no_tri);
    for (tri = 0; tri < no_tri; ++tri)
    {
        Vertex *v0 = NULL;
        Vertex *v1 = NULL;
        Vertex *v2 = NULL;
        const Triangle *tr = _triangleList->insert(
            v0 = _vertexList->operator[](conn_list[3 * tri]),
            v1 = _vertexList->operator[](conn_list[3 * tri + 1]),
            v2 = _vertexList->operator[](conn_list[3 * tri + 2]));
        Triangle *mod_tr = const_cast<Triangle *>(tr);
        v0->add_tr(mod_tr);
        v1->add_tr(mod_tr);
        v2->add_tr(mod_tr);

        pair<Edge *, bool> out = _edgeSet->insert(v0, v1, tr);
        // if v0, v1 already present,
        // set boundary to false
        if (out.second)
        {
            v0->add_edge(out.first);
            v1->add_edge(out.first);
            out.first->boundary(true);
        }
        else // the edge was already created and has more than 1 triangle
        {
            // out.first->add_q(tr->q());
            out.first->boundary(false);
        }

        out = _edgeSet->insert(v1, v2, tr);
        // if v1, v2 already present,
        // set boundary to false
        if (out.second)
        {
            v1->add_edge(out.first);
            v2->add_edge(out.first);
            out.first->boundary(true);
        }
        else // the edge was already created and has more than 1 triangle
        {
            // out.first->add_q(tr->q());
            out.first->boundary(false);
        }

        out = _edgeSet->insert(v2, v0, tr);
        // if v1, v2 already present,
        // set boundary to false
        if (out.second)
        {
            v2->add_edge(out.first);
            v0->add_edge(out.first);
            out.first->boundary(true);
        }
        else // the edge was already created and has more than 1 triangle
        {
            // out.first->add_q(tr->q());
            out.first->boundary(false);
        }
    }

    // preserve boundary and set simplification order
    _pq = new PQ(_edgeSet->size());
    _edgeSet->ComputeCost(_pq); // _edgeSet does not move its elements

    // accumulate _q_bound from boundary edges for each vertex
    _vertexList->MakeBoundary();
}

EdgeCollapseBasis::~EdgeCollapseBasis()
{
    delete _vertexList;
    delete _triangleList;
    delete _edgeSet;
    delete _pq;
}

void
EdgeCollapseBasis::LeftEntities(vector<int> &leftTriangles,
                                vector<float> &leftVertexX,
                                vector<float> &leftVertexY,
                                vector<float> &leftVertexZ,
                                vector<float> &leftData,
                                vector<float> &leftNormals) const
{
    leftTriangles.clear();
    leftVertexX.clear();
    leftVertexY.clear();
    leftVertexZ.clear();
    leftData.clear();
    leftNormals.clear();

    int vert, mark_max = -1;

    _triangleList->MaxLabel(mark_max);
    ++mark_max;

    int *mark = new int[mark_max];
    for (vert = 0; vert < mark_max; ++vert)
    {
        mark[vert] = -1;
    }

    _triangleList->SetMarks(mark, mark_max);

    _triangleList->SetConnectivities(leftTriangles, mark);
    _vertexList->SetCoordinates(leftVertexX, leftVertexY, leftVertexZ,
                                leftData, leftNormals,
                                mark, mark_max);
    delete[] mark;
}

bool
EdgeCollapseBasis::PQ_OK() const
{
    if (_pq)
    {
        return _pq->OK();
    }
    return true;
}

int
EdgeCollapseBasis::CheckDirection(const Vertex *v0, const Vertex *v1,
                                  const Edge *theEdge) const
{
    set<const Edge *> boundaryEdges0, boundaryEdges1;
    v0->BoundaryEdges(boundaryEdges0);
    boundaryEdges0.erase(theEdge);
    v1->BoundaryEdges(boundaryEdges1);
    boundaryEdges1.erase(theEdge);
    set<const Edge *>::iterator bedge_it = boundaryEdges0.begin();
    set<const Edge *>::iterator bedge_it_end = boundaryEdges0.end();
    for (; bedge_it != bedge_it_end; ++bedge_it)
    {
        const Edge *this_bedge = *bedge_it;
        if (!this_bedge->CheckDirection(v0, theEdge->popt()))
        {
            return 0;
        }
    }
    bedge_it = boundaryEdges1.begin();
    bedge_it_end = boundaryEdges1.end();
    for (; bedge_it != bedge_it_end; ++bedge_it)
    {
        const Edge *this_bedge = *bedge_it;
        if (!this_bedge->CheckDirection(v1, theEdge->popt()))
        {
            return 0;
        }
    }
    return 1;
}
