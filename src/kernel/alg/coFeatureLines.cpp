/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coFeatureLines.h"
#include "MagmaUtils.h"
#include "coMiniGrid.h"
#include "RainAlgorithm.h"

using namespace covise;

// this function produces 'seams' in the grid for
// a correct interpolation of the normals
void
coFeatureLines::cutPoly(float coseno,
                        const vector<int> &elemList,
                        vector<int> &connList,
                        vector<float> &xcoord, vector<float> &ycoord, vector<float> &zcoord,
                        const vector<float> &xn,
                        const vector<float> &yn,
                        const vector<float> &zn,
                        vector<int> &ll,
                        vector<int> &cl,
                        vector<float> &lx,
                        vector<float> &ly,
                        vector<float> &lz,
                        vector<int> &dll,
                        vector<int> &dcl,
                        vector<float> &dlx,
                        vector<float> &dly,
                        vector<float> &dlz)
{
    // number of vertices per cell: required by the rain algorith
    vector<int> num_conn(elemList.size(), 3);
    vector<int> nodal_start_neigh;
    vector<int> nodal_number_neigh;
    vector<int> nodal_neighbours; // cells touching a node
    MagmaUtils::NodeNeighbours(elemList, num_conn, connList, (int)xcoord.size(),
                               nodal_start_neigh, nodal_number_neigh, nodal_neighbours);
    vector<int> elem_start_neigh;
    vector<int> elem_number_neigh;
    vector<int> elem_neighbours; // neighbour cells of a cell
    vector<MagmaUtils::Edge> edge_neighbours; //this is as long as elem_neighbours
    MagmaUtils::CellNeighbours(elemList, num_conn, connList,
                               nodal_start_neigh,
                               nodal_number_neigh,
                               nodal_neighbours, // elements touching a node
                               elem_start_neigh, elem_number_neigh, elem_neighbours, edge_neighbours);

    // we may calculate now the domain lines
    {
        vector<int> border_els;
        vector<MagmaUtils::Edge> border_edges;
        MagmaUtils::DomainLines(elemList, num_conn, connList,
                                elem_start_neigh, elem_number_neigh, edge_neighbours,
                                border_els, border_edges);
        vector<int> l_dll;
        vector<int> l_dcl;
        vector<MagmaUtils::Edge>::iterator border_edges_it = border_edges.begin();
        vector<MagmaUtils::Edge>::iterator border_edges_end = border_edges.end();
        int count = 0;
        map<int, int> global2domlocal;
        for (; border_edges_it != border_edges_end; ++border_edges_it)
        {
            l_dll.push_back((int)l_dcl.size());
            map<int, int>::iterator global2domlocal_it = global2domlocal.find(border_edges_it->first);
            int node0 = -1;
            if (global2domlocal_it == global2domlocal.end())
            {
                global2domlocal.insert(map<int, int>::value_type(
                    border_edges_it->first, count));
                node0 = count;
                ++count;
            }
            else
            {
                node0 = global2domlocal_it->second;
            }
            int node1 = -1;
            global2domlocal_it = global2domlocal.find(border_edges_it->second);
            if (global2domlocal_it == global2domlocal.end())
            {
                global2domlocal.insert(map<int, int>::value_type(
                    border_edges_it->second, count));
                node1 = count;
                ++count;
            }
            else
            {
                node1 = global2domlocal_it->second;
            }
            l_dcl.push_back(node0);
            l_dcl.push_back(node1);
        }
        vector<float> l_dlx(count);
        vector<float> l_dly(count);
        vector<float> l_dlz(count);
        map<int, int>::iterator global2domlocal_it = global2domlocal.begin();
        map<int, int>::iterator global2domlocal_end = global2domlocal.end();
        for (; global2domlocal_it != global2domlocal_end; ++global2domlocal_it)
        {
            l_dlx[global2domlocal_it->second] = xcoord[global2domlocal_it->first];
            l_dly[global2domlocal_it->second] = ycoord[global2domlocal_it->first];
            l_dlz[global2domlocal_it->second] = zcoord[global2domlocal_it->first];
        }
        l_dll.swap(dll);
        l_dcl.swap(dcl);
        l_dlx.swap(dlx);
        l_dly.swap(dly);
        l_dlz.swap(dlz);
    }

    // feature edges
    typedef unordered_set<coTriEdge, HashCoTriEdge> setType;
    setType set_kanten;
    int elem;
    for (elem = 0; elem < elem_start_neigh.size(); ++elem) // loop over cells
    {
        int num_elem_neigh = elem_number_neigh[elem];
        int e_n = elem_start_neigh[elem];
        int e_counter = 0;
        // loop over neighbours
        for (; e_counter < num_elem_neigh; ++e_counter, ++e_n)
        {
            int other_cell = elem_neighbours[e_n];
            if (other_cell < elem)
                continue;
            const MagmaUtils::Edge &thisEdge = edge_neighbours[e_n];
            // we test whether elements elem and other_cell have
            // different orientations
            if (DifferentOrientation(coseno, elem, other_cell, xn, yn, zn))
            {
                set_kanten.insert(coTriEdge(thisEdge.first, thisEdge.second));
            }
        }
    }
    // feature lines may be processed now
    {
        map<int, int> global2lines;
        setType::iterator triedge_it = set_kanten.begin();
        setType::iterator triedge_it_end = set_kanten.end();
        int counter = 0;
        for (; triedge_it != triedge_it_end; ++triedge_it)
        {
            int node0 = triedge_it->getMin();
            if (global2lines.find(node0) == global2lines.end())
            {
                global2lines.insert(map<int, int>::value_type(node0, counter));
                ++counter;
            }
            int node1 = triedge_it->getMax();
            if (global2lines.find(node1) == global2lines.end())
            {
                global2lines.insert(map<int, int>::value_type(node1, counter));
                ++counter;
            }
        }
        vector<int> l_ll(set_kanten.size());
        vector<int> l_cl(2 * set_kanten.size());
        vector<float> l_lx(global2lines.size());
        vector<float> l_ly(global2lines.size());
        vector<float> l_lz(global2lines.size());

        triedge_it = set_kanten.begin();
        counter = 0;
        for (; triedge_it != triedge_it_end; ++triedge_it, ++counter)
        {
            l_ll[counter] = 2 * counter;
            l_cl[2 * counter] = global2lines[triedge_it->getMin()];
            l_cl[2 * counter + 1] = global2lines[triedge_it->getMax()];
        }
        map<int, int>::iterator global2lines_it = global2lines.begin();
        map<int, int>::iterator global2lines_it_end = global2lines.end();
        for (; global2lines_it != global2lines_it_end; ++global2lines_it)
        {
            l_lx[global2lines_it->second] = xcoord[global2lines_it->first];
            l_ly[global2lines_it->second] = ycoord[global2lines_it->first];
            l_lz[global2lines_it->second] = zcoord[global2lines_it->first];
        }
        l_ll.swap(ll);
        l_cl.swap(cl);
        l_lx.swap(lx);
        l_ly.swap(ly);
        l_lz.swap(lz);
    }
    vector<int> mark_processed_feature_nodes(xcoord.size(), 0);
    // loop (indirectly through feature_edges) over nodes in feature_edges
    vector<int> l_connList(connList);
    vector<float> l_xcoord(xcoord);
    vector<float> l_ycoord(ycoord);
    vector<float> l_zcoord(zcoord);
    setType::iterator set_kanten_it = set_kanten.begin();
    setType::iterator set_kanten_end = set_kanten.end();
    coMiniGrid::Border mini_kanten(set_kanten);
    for (; set_kanten_it != set_kanten_end; ++set_kanten_it)
    {
        int node0 = set_kanten_it->getMin();
        int node1 = set_kanten_it->getMax();
        // here we split these nodes in SplitNode
        // see below
        if (mark_processed_feature_nodes[node0] == 0)
        {
            mark_processed_feature_nodes[node0] = 1;
            SplitNode(node0, elemList, l_connList, l_xcoord, l_ycoord, l_zcoord,
                      nodal_number_neigh, nodal_start_neigh, nodal_neighbours,
                      elem_start_neigh, elem_number_neigh, elem_neighbours,
                      edge_neighbours, mini_kanten);
        }
        if (mark_processed_feature_nodes[node1] == 0)
        {
            mark_processed_feature_nodes[node1] = 1;
            SplitNode(node1, elemList, l_connList, l_xcoord, l_ycoord, l_zcoord,
                      nodal_number_neigh, nodal_start_neigh, nodal_neighbours,
                      elem_start_neigh, elem_number_neigh, elem_neighbours,
                      edge_neighbours, mini_kanten);
        }
    }
    l_connList.swap(connList);
    l_xcoord.swap(xcoord);
    l_ycoord.swap(ycoord);
    l_zcoord.swap(zcoord);
}

bool
coFeatureLines::DifferentOrientation(float coseno,
                                     int elem,
                                     int other_cell,
                                     const vector<float> &xn, // normals are cell-based
                                     const vector<float> &yn,
                                     const vector<float> &zn)
{
    float v0[3], v1[3];
    v0[0] = xn[elem];
    v0[1] = yn[elem];
    v0[2] = zn[elem];
    v1[0] = xn[other_cell];
    v1[1] = yn[other_cell];
    v1[2] = zn[other_cell];
    float len_0 = sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
    float len_1 = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    float scal = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
    return (coseno * len_0 * len_1 >= scal);
}

// SplitNode creates the minigrid around a node
// and divides it in domains according to the rain
// algorithm applied to the minigrid with the borders
// taken from mini_kanten. For each 'extra'-domain,
// a copy of the node is created and the connectivity
// of the cells in the domain at issue is corrected
// os as to use the new copy of the node
void
coFeatureLines::SplitNode(int node,
                          const vector<int> &elemList,
                          vector<int> &connList,
                          vector<float> &xcoord,
                          vector<float> &ycoord,
                          vector<float> &zcoord,
                          const vector<int> &nodal_number_neigh,
                          const vector<int> &nodal_start_neigh,
                          const vector<int> &nodal_neighbours,
                          const vector<int> &elem_start_neigh,
                          const vector<int> &elem_number_neigh,
                          const vector<int> &elem_neighbours, // neighbour cells of a cell
                          const vector<MagmaUtils::Edge> &edge_neighbours,
                          const coMiniGrid::Border &mini_kanten)
{
    // find minigrid around this node
    int num_nodes_minigrid = nodal_number_neigh[node];
    int num_start_minigrid = nodal_start_neigh[node];
    vector<int> minigrid_cells(nodal_neighbours.begin() + num_start_minigrid,
                               nodal_neighbours.begin() + num_start_minigrid + num_nodes_minigrid);
    // we work always with triangles
    vector<int> minigrid_numconn(minigrid_cells.size(), 3);
    // apply rainAlgorithm to this minigrid with inherited feature_edges.
    // feature_edges are a subset of set_kanten, but we can use
    // set_kanten (mini_kanten) without problems
    coMiniGrid miniGrid(minigrid_cells,
                        elem_start_neigh,
                        elem_number_neigh,
                        elem_neighbours, // neighbour cells of a cell
                        edge_neighbours //this is as long as elem_neighbours
                        );
    vector<int> tags;
    RainAlgorithm<coMiniGrid>(miniGrid, mini_kanten, tags);
    // enhance connectivity of cells in miniGrid, according
    // to the tags
    map<int, int> tag2newnode;
    int mini_cell;
    for (mini_cell = 0; mini_cell < tags.size(); ++mini_cell)
    {
        // tag 0 reuses available node
        if (tags[mini_cell] == 0)
            continue;
        if (tag2newnode.find(tags[mini_cell]) == tag2newnode.end())
        {
            // insert new coordinates
            tag2newnode.insert(map<int, int>::value_type(tags[mini_cell],
                                                         (int)xcoord.size()));
            float x = xcoord[node];
            float y = ycoord[node];
            float z = zcoord[node];
            xcoord.push_back(x);
            ycoord.push_back(y);
            zcoord.push_back(z);
        }
        int new_coord = tag2newnode[tags[mini_cell]];
        // enhance connectivity
        int cell = minigrid_cells[mini_cell];
        int init_conn = elemList[cell];
        int vertex;
        for (vertex = 0; vertex < 3; ++vertex)
        {
            if (connList[init_conn + vertex] == node)
            {
                connList[init_conn + vertex] = new_coord;
            }
        }
    }
}

void
coFeatureLines::Triangulate(vector<int> &tri_conn_list,
                            vector<int> &tri_codes,
                            int num_poly, int num_conn,
                            const int *poly_list,
                            const int *conn_list,
                            const float *x, const float *y, const float *z)
{

    // do nothing if we have empty polygons
    if ((num_poly == 0) && (num_conn == 0))
        return;

    // a number of 100 for the size of an individual polygon
    // in the mesh seems to be resonable
    const int polyVertNum = 100;
    float xn[polyVertNum];
    float yn[polyVertNum];
    float zn[polyVertNum];

    vector<int> l_tri_conn_list;
    int poly;
    int l_num_conn, max_angle_vertex, new_triangle;
    int endConn = 0;
    const int *start_conn;
    for (poly = 0; poly < num_poly - 1; ++poly)
    {
        endConn = poly_list[poly + 1];
        l_num_conn = endConn - poly_list[poly];
        start_conn = conn_list + poly_list[poly];
        int ii;
        for (ii = 0; ii < l_num_conn; ++ii)
        {
            xn[ii] = x[start_conn[ii]];
            yn[ii] = y[start_conn[ii]];
            zn[ii] = z[start_conn[ii]];
        }

        max_angle_vertex = MaxAngleVertex(l_num_conn, xn, yn, zn);

        for (new_triangle = 0; new_triangle < l_num_conn - 2; ++new_triangle)
        {
            if (start_conn[max_angle_vertex] != start_conn[(max_angle_vertex + new_triangle + 1) % l_num_conn] && start_conn[(max_angle_vertex + new_triangle + 1) % l_num_conn] != start_conn[(max_angle_vertex + new_triangle + 2) % l_num_conn] && start_conn[(max_angle_vertex + new_triangle + 2) % l_num_conn] != start_conn[max_angle_vertex])
            {
                tri_codes.push_back(poly);
                l_tri_conn_list.push_back(start_conn[max_angle_vertex]);
                l_tri_conn_list.push_back(start_conn[(max_angle_vertex + new_triangle + 1) % l_num_conn]);
                l_tri_conn_list.push_back(start_conn[(max_angle_vertex + new_triangle + 2) % l_num_conn]);
            }
        }
    }
    // and now the same for the last polygon...
    int num_last_conn = num_conn - poly_list[poly];
    start_conn = conn_list + poly_list[poly];
    int ii;
    for (ii = 0; ii < num_last_conn; ++ii)
    {
        xn[ii] = x[start_conn[ii]];
        yn[ii] = y[start_conn[ii]];
        zn[ii] = z[start_conn[ii]];
    }

    max_angle_vertex = MaxAngleVertex(num_last_conn, x, y, z);

    for (new_triangle = 0; new_triangle < num_last_conn - 2; ++new_triangle)
    {
        if (start_conn[max_angle_vertex] != start_conn[(max_angle_vertex + new_triangle + 1) % num_last_conn] && start_conn[(max_angle_vertex + new_triangle + 1) % num_last_conn] != start_conn[(max_angle_vertex + new_triangle + 2) % num_last_conn] && start_conn[(max_angle_vertex + new_triangle + 2) % num_last_conn] != start_conn[max_angle_vertex])
        {
            tri_codes.push_back(num_poly - 1);
            l_tri_conn_list.push_back(start_conn[max_angle_vertex]);
            l_tri_conn_list.push_back(start_conn[(max_angle_vertex + new_triangle + 1) % num_last_conn]);
            l_tri_conn_list.push_back(start_conn[(max_angle_vertex + new_triangle + 2) % num_last_conn]);
        }
    }
    l_tri_conn_list.swap(tri_conn_list);
}

int
coFeatureLines::MaxAngleVertex(int num_conn,
                               const float *x,
                               const float *y,
                               const float *z)
{
    int ret = 0;
    float min_cos = 1.0;
    int vertex = 0;
    for (; vertex < num_conn; ++vertex)
    {
        int next = (vertex + 1) % num_conn;
        int prev = (vertex + num_conn - 1) % num_conn;
        float p_next[3];
        float p_prev[3];
        p_next[0] = x[next] - x[vertex];
        p_prev[0] = x[prev] - x[vertex];
        p_next[1] = y[next] - y[vertex];
        p_prev[1] = y[prev] - y[vertex];
        p_next[2] = z[next] - z[vertex];
        p_prev[2] = z[prev] - z[vertex];
        float scal_prod = p_next[0] * p_prev[0] + p_next[1] * p_prev[1] + p_next[2] * p_prev[2];
        float len_next = sqrt(p_next[0] * p_next[0] + p_next[1] * p_next[1] + p_next[2] * p_next[2]);
        float len_prev = sqrt(p_prev[0] * p_prev[0] + p_prev[1] * p_prev[1] + p_prev[2] * p_prev[2]);
        if (len_next == 0.0 || len_prev == 0.0)
        {
            continue;
        }
        float this_cos = scal_prod / (len_next * len_prev);
        if (this_cos < min_cos)
        {
            min_cos = this_cos;
            ret = vertex;
        }
    }
    return ret;
}
