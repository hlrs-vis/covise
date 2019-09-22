/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MagmaUtils.h"
#include <numeric>
#if defined(__INTEL_COMPILER)
#include <algorithm>
#endif
#include <functional>

#include <iterator>

using namespace covise;

void
MagmaUtils::NodeNeighbours(const vector<int> &el,
                           const vector<int> &num_conn,
                           const vector<int> &cl,
                           int num_coords,
                           vector<int> &start_neigh,
                           vector<int> &number_neigh,
                           vector<int> &neighbours)
{
    // first count number of neighbours...
    vector<int> local_number_neigh(num_coords, 0);
    unsigned int elem;
    for (elem = 0; elem < el.size(); ++elem)
    {
        int conn;
        int end_conn = el[elem] + num_conn[elem];
        for (conn = el[elem]; conn < end_conn; ++conn)
        {
            ++(local_number_neigh[cl[conn]]);
        }
    }
    // now go for start_neigh...
    vector<int> local_start_neigh(num_coords, 0);
    if (num_coords > 0)
    {
        std::partial_sum(local_number_neigh.begin(), local_number_neigh.end() - 1,
                         local_start_neigh.begin() + 1);
    }
    // it only remains neighbours
    int total_neighbours = std::accumulate(local_number_neigh.begin(),
                                           local_number_neigh.end(), 0);
    vector<int> local_neighbours(total_neighbours, -1);
    for (elem = 0; elem < el.size(); ++elem)
    {
        int conn;
        int end_conn = el[elem] + num_conn[elem];
        for (conn = el[elem]; conn < end_conn; ++conn)
        {
            int coord = cl[conn];
            vector<int>::iterator start_it = local_neighbours.begin() + local_start_neigh[coord];
            vector<int>::iterator new_neigh_it = std::find(start_it, local_neighbours.end(), -1);
            (*new_neigh_it) = elem;
        }
    }
    // swap results
    local_number_neigh.swap(number_neigh);
    local_start_neigh.swap(start_neigh);
    local_neighbours.swap(neighbours);
}

void
MagmaUtils::CellNeighbours(const vector<int> &el,
                           const vector<int> &num_conn,
                           const vector<int> &cl,
                           // these fields come from NodeNeighbours
                           const vector<int> &nodal_start_neigh,
                           const vector<int> &nodal_number_neigh,
                           const vector<int> &nodal_neighbours, // elements touching a node
                           vector<int> &elem_start_neigh,
                           vector<int> &elem_number_neigh,
                           vector<int> &elem_neighbours,
                           vector<Edge> &edge_neighbours)
{
    vector<int> local_elem_start_neigh, local_elem_number_neigh, local_elem_neighbours;
    vector<Edge> local_edge_neighbours;
    // for each cell, loop over edges and calculate neighbour cells
    // of those edges, calculate intersection and subtract the cell at issue
    unsigned int elem;
    for (elem = 0; elem < el.size(); ++elem)
    {
        int conn;
        int end_conn = el[elem] + num_conn[elem];
        int previous_size = (int)local_elem_neighbours.size();
        local_elem_start_neigh.push_back(previous_size);
        for (conn = el[elem]; conn < end_conn; ++conn) // all edges up to the last
        {
            int node0 = cl[conn];
            int numele0 = nodal_number_neigh[node0];
            int node1 = ((conn < end_conn - 1) ? cl[conn + 1] : cl[el[elem]]);
            int numele1 = nodal_number_neigh[node1];
            vector<int> common_cells;
            std::set_intersection(nodal_neighbours.begin() + nodal_start_neigh[node0],
                                  nodal_neighbours.begin() + nodal_start_neigh[node0] + numele0,
                                  nodal_neighbours.begin() + nodal_start_neigh[node1],
                                  nodal_neighbours.begin() + nodal_start_neigh[node1] + numele1,
                                  std::back_inserter(common_cells));
            std::remove_copy_if(common_cells.begin(), common_cells.end(),
                                std::back_inserter(local_elem_neighbours),
                                std::bind(std::equal_to<int>(), std::placeholders::_1, elem));
            std::fill_n(std::back_inserter(local_edge_neighbours),
                        common_cells.size() - 1, (node0 < node1) ? Edge(node0, node1) : Edge(node1, node0));
        }
        // now we may compare sizes...
        local_elem_number_neigh.push_back((int)local_elem_neighbours.size() - previous_size);
    }
    // swap results
    local_elem_start_neigh.swap(elem_start_neigh);
    local_elem_number_neigh.swap(elem_number_neigh);
    local_elem_neighbours.swap(elem_neighbours);
    local_edge_neighbours.swap(edge_neighbours);
}

void
MagmaUtils::DomainLines(const vector<int> &el,
                        const vector<int> &num_conn,
                        const vector<int> &cl,
                        const vector<int> &elem_start_neigh,
                        const vector<int> &elem_number_neigh,
                        const vector<Edge> &edge_neighbours,
                        vector<int> &border_els,
                        vector<Edge> &border_edges)
{
    vector<int> local_border_els;
    vector<Edge> local_border_edges;
    // the idea is simple: loop over elements, get
    // edge_meighbours and compare this set with the set of all
    // edges seen by this cell, compute the difference and we are
    // done with it: the difference goes to border_edges and the
    // element at issue to border_els
    for (unsigned int cell = 0; cell < el.size(); ++cell)
    {
        int start = elem_start_neigh[cell];
        int num_neigh = elem_number_neigh[cell];
        vector<Edge>::const_iterator edge_begin = edge_neighbours.begin() + start;
        vector<Edge>::const_iterator edge_end = edge_neighbours.begin() + start + num_neigh;

        int num_edges = num_conn[cell];
        vector<Edge> all_edges;
        all_edges.reserve(num_edges);
        int end_conn = el[cell] + num_conn[cell];
        for (int conn = el[cell]; conn < end_conn; ++conn) // all edges up to the last
        {
            int node0 = cl[conn];
            int node1 = ((conn < end_conn - 1) ? cl[conn + 1] : cl[el[cell]]);
            all_edges.push_back((node0 < node1) ? Edge(node0, node1) : Edge(node1, node0));
        }
        // the sequel might be accelerated making the most of a trivial order relationship...
        //      int previous_size = local_border_edges.size();
        vector<Edge>::iterator edge_it = all_edges.begin();
        for (; edge_it != all_edges.end(); ++edge_it)
        {
            if (std::find(edge_begin, edge_end, *edge_it) == edge_end)
            {
                local_border_els.push_back(cell);
                local_border_edges.push_back(*edge_it);
            }
        }
    }
    local_border_els.swap(border_els);
    local_border_edges.swap(border_edges);
}

void
MagmaUtils::ExtendBorder(float length, const vector<int> &border_els,
                         const vector<Edge> &border_edges,
                         const vector<int> &cl,
                         vector<int> &clExt,
                         const vector<float> &x,
                         const vector<float> &y,
                         const vector<float> &z,
                         vector<float> &xExt,
                         vector<float> &yExt,
                         vector<float> &zExt)
{
    // produce future node labels and associate them
    // to border node labels
    int num_nodes = (int)x.size();
    map<int, int> border_new;
    map<int, vector<float> > node_normals;
    for (unsigned int segment = 0; segment < border_edges.size(); ++segment)
    {
        int node = border_edges[segment].first;
        if (border_new.find(node) == border_new.end())
        {
            border_new[node] = num_nodes;
            vector<float> normalini(3, 0.0);
            node_normals[node] = normalini;
            ++num_nodes;
        }
        node = border_edges[segment].second;
        if (border_new.find(node) == border_new.end())
        {
            border_new[node] = num_nodes;
            vector<float> normalini(3, 0.0);
            node_normals[node] = normalini;
            ++num_nodes;
        }
    }
    clExt.clear();
    xExt.clear();
    yExt.clear();
    zExt.clear();
    // now work out for each segment a normal vector
    vector<float> tri_normals;
    assert(border_els.size() == border_edges.size());
    for (unsigned int triangle = 0; triangle < border_els.size(); ++triangle)
    {
        int node0 = border_edges[triangle].first;
        int node1 = border_edges[triangle].second;
        // identify node2!!!
        int triangle_label = border_els[triangle];
        int node2 = cl[3 * triangle_label];
        if (node2 == node0 || node2 == node1)
            node2 = cl[3 * triangle_label + 1];
        if (node2 == node0 || node2 == node1)
            node2 = cl[3 * triangle_label + 2];
        assert(node2 != node0 && node2 != node1); // yes, too radical...
        float p0[3] = { 0.0, 0.0, 0.0 };
        float p1[3] = { 0.0, 0.0, 0.0 };
        float p2[3] = { 0.0, 0.0, 0.0 };
        p0[0] = x[node0];
        p0[1] = y[node0];
        p0[2] = z[node0];
        p1[0] = x[node1];
        p1[1] = y[node1];
        p1[2] = z[node1];
        p2[0] = x[node2];
        p2[1] = y[node2];
        p2[2] = z[node2];
        float s1[3] = { 0.0, 0.0, 0.0 };
        float s2[3] = { 0.0, 0.0, 0.0 };
        s1[0] = p1[0] - p0[0];
        s1[1] = p1[1] - p0[1];
        s1[2] = p1[2] - p0[2];
        s2[0] = p2[0] - p0[0];
        s2[1] = p2[1] - p0[1];
        s2[2] = p2[2] - p0[2];
        // normalise s1...
        float len1 = sqrt(s1[0] * s1[0] + s1[1] * s1[1] + s1[2] * s1[2]);
        if (len1 > 0.0)
            len1 = 1.0f / len1;
        s1[0] *= len1;
        s1[1] *= len1;
        s1[2] *= len1;
        // the normal!!!
        float n[3] = { 0.0, 0.0, 0.0 };
        float scal = s1[0] * s2[0] + s1[1] * s2[1] + s1[2] * s2[2];
        n[0] = scal * s1[0] - s2[0];
        n[1] = scal * s1[1] - s2[1];
        n[2] = scal * s1[2] - s2[2];
        // normalise n...
        float lenn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (lenn > 0.0)
            lenn = 1.0f / lenn;
        n[0] *= lenn;
        n[1] *= lenn;
        n[2] *= lenn;
        tri_normals.push_back(n[0]);
        tri_normals.push_back(n[1]);
        tri_normals.push_back(n[2]);
    }
    // ok now we fill the map relating border nodes and added normals
    // loop over triangles (segments) adding tri_normals to node_normals for 2 nodes
    for (unsigned int triseg = 0; triseg < border_edges.size(); ++triseg)
    {
        int node0 = border_edges[triseg].first;
        int node1 = border_edges[triseg].second;
        vector<float> &nor0 = node_normals[node0];
        vector<float> &nor1 = node_normals[node1];
        nor0[0] += tri_normals[3 * triseg];
        nor0[1] += tri_normals[3 * triseg + 1];
        nor0[2] += tri_normals[3 * triseg + 2];
        nor1[0] += tri_normals[3 * triseg];
        nor1[1] += tri_normals[3 * triseg + 1];
        nor1[2] += tri_normals[3 * triseg + 2];
    }
    map<int, vector<float> >::iterator mapnit = node_normals.begin();
    map<int, vector<float> >::iterator mapend = node_normals.end();
    for (; mapnit != mapend; ++mapnit)
    {
        vector<float> &normal = (*mapnit).second;
        float lenn = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (lenn > 0.0)
            lenn = length / lenn;
        normal[0] *= lenn;
        normal[1] *= lenn;
        normal[2] *= lenn;
    }
    // now we may start the extension : clExt, xExt....
    vector<Edge>::const_iterator edge_it = border_edges.begin();
    int tri_counter = 0;
    for (; edge_it != border_edges.end(); ++edge_it, ++tri_counter)
    {
        int node0 = edge_it->first;
        int node1 = edge_it->second;
        int trinode0 = cl[3 * border_els[tri_counter]];
        int trinode1 = cl[3 * border_els[tri_counter] + 1];
        int trinode2 = cl[3 * border_els[tri_counter] + 2];
        if ((node1 == trinode0 && node0 == trinode1)
            || (node1 == trinode1 && node0 == trinode2)
            || (node1 == trinode2 && node0 == trinode0))
        {
            int tmp = node1;
            node1 = node0;
            node0 = tmp;
        }
        int node0p = border_new[node0];
        int node1p = border_new[node1];
        clExt.push_back(node0); // extra conn
        clExt.push_back(node0p);
        clExt.push_back(node1);
        clExt.push_back(node1);
        clExt.push_back(node0p);
        clExt.push_back(node1p);
    }
    //mapnit = node_normals.begin();
    mapnit = node_normals.begin();
    mapend = node_normals.end();
    int old_num_nodes = (int)x.size();
    xExt.resize(node_normals.size(), 0.0);
    yExt.resize(node_normals.size(), 0.0);
    zExt.resize(node_normals.size(), 0.0);
    for (; mapnit != mapend; ++mapnit)
    {
        // get also the old node label...
        int nodelab = (*mapnit).first;
        const vector<float> &normal = (*mapnit).second;
        int new_label = border_new[nodelab] - old_num_nodes;
        xExt[new_label] = x[nodelab] + normal[0];
        yExt[new_label] = y[nodelab] + normal[1];
        zExt[new_label] = z[nodelab] + normal[2];
    }
}

// Eilige copy 'nd paste Erweiterung für Geometrie+Daten
void
MagmaUtils::ExtendBorderAndData(float length, const vector<int> &border_els,
                                const vector<Edge> &border_edges,
                                const vector<int> &cl,
                                vector<int> &clExt,
                                const vector<float> &x,
                                const vector<float> &y,
                                const vector<float> &z,
                                const vector<float> &InData,
                                vector<float> &xExt,
                                vector<float> &yExt,
                                vector<float> &zExt,
                                vector<float> &OutData)
{
    // produce future node labels and associate them
    // to border node labels
    int num_nodes = (int)x.size();
    map<int, int> border_new;
    map<int, vector<float> > node_normals;
    for (unsigned int segment = 0; segment < border_edges.size(); ++segment)
    {
        int node = border_edges[segment].first;
        if (border_new.find(node) == border_new.end())
        {
            border_new[node] = num_nodes;
            vector<float> normalini(3, 0.0);
            node_normals[node] = normalini;
            ++num_nodes;
        }
        node = border_edges[segment].second;
        if (border_new.find(node) == border_new.end())
        {
            border_new[node] = num_nodes;
            vector<float> normalini(3, 0.0);
            node_normals[node] = normalini;
            ++num_nodes;
        }
    }
    clExt.clear();
    xExt.clear();
    yExt.clear();
    zExt.clear();
    OutData.clear();
    // now work out for each segment a normal vector
    vector<float> tri_normals;
    assert(border_els.size() == border_edges.size());
    for (unsigned int triangle = 0; triangle < border_els.size(); ++triangle)
    {
        int node0 = border_edges[triangle].first;
        int node1 = border_edges[triangle].second;
        // identify node2!!!
        int triangle_label = border_els[triangle];
        int node2 = cl[3 * triangle_label];
        if (node2 == node0 || node2 == node1)
            node2 = cl[3 * triangle_label + 1];
        if (node2 == node0 || node2 == node1)
            node2 = cl[3 * triangle_label + 2];
        assert(node2 != node0 && node2 != node1); // yes, too radical...
        float p0[3] = { 0.0, 0.0, 0.0 };
        float p1[3] = { 0.0, 0.0, 0.0 };
        float p2[3] = { 0.0, 0.0, 0.0 };
        p0[0] = x[node0];
        p0[1] = y[node0];
        p0[2] = z[node0];
        p1[0] = x[node1];
        p1[1] = y[node1];
        p1[2] = z[node1];
        p2[0] = x[node2];
        p2[1] = y[node2];
        p2[2] = z[node2];
        float s1[3] = { 0.0, 0.0, 0.0 };
        float s2[3] = { 0.0, 0.0, 0.0 };
        s1[0] = p1[0] - p0[0];
        s1[1] = p1[1] - p0[1];
        s1[2] = p1[2] - p0[2];
        s2[0] = p2[0] - p0[0];
        s2[1] = p2[1] - p0[1];
        s2[2] = p2[2] - p0[2];
        // normalise s1...
        float len1 = sqrt(s1[0] * s1[0] + s1[1] * s1[1] + s1[2] * s1[2]);
        if (len1 > 0.0)
            len1 = 1.0f / len1;
        s1[0] *= len1;
        s1[1] *= len1;
        s1[2] *= len1;
        // the normal!!!
        float n[3] = { 0.0, 0.0, 0.0 };
        float scal = s1[0] * s2[0] + s1[1] * s2[1] + s1[2] * s2[2];
        n[0] = scal * s1[0] - s2[0];
        n[1] = scal * s1[1] - s2[1];
        n[2] = scal * s1[2] - s2[2];
        // normalise n...
        float lenn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (lenn > 0.0)
            lenn = 1.0f / lenn;
        n[0] *= lenn;
        n[1] *= lenn;
        n[2] *= lenn;
        tri_normals.push_back(n[0]);
        tri_normals.push_back(n[1]);
        tri_normals.push_back(n[2]);
    }
    // ok now we fill the map relating border nodes and added normals
    // loop over triangles (segments) adding tri_normals to node_normals for 2 nodes
    for (unsigned int triseg = 0; triseg < border_edges.size(); ++triseg)
    {
        int node0 = border_edges[triseg].first;
        int node1 = border_edges[triseg].second;
        vector<float> &nor0 = node_normals[node0];
        vector<float> &nor1 = node_normals[node1];
        nor0[0] += tri_normals[3 * triseg];
        nor0[1] += tri_normals[3 * triseg + 1];
        nor0[2] += tri_normals[3 * triseg + 2];
        nor1[0] += tri_normals[3 * triseg];
        nor1[1] += tri_normals[3 * triseg + 1];
        nor1[2] += tri_normals[3 * triseg + 2];
    }
    map<int, vector<float> >::iterator mapnit = node_normals.begin();
    map<int, vector<float> >::iterator mapend = node_normals.end();
    for (; mapnit != mapend; ++mapnit)
    {
        vector<float> &normal = (*mapnit).second;
        float lenn = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (lenn > 0.0)
            lenn = length / lenn;
        normal[0] *= lenn;
        normal[1] *= lenn;
        normal[2] *= lenn;
    }
    // now we may start the extension : clExt, xExt....
    vector<Edge>::const_iterator edge_it = border_edges.begin();
    int tri_counter = 0;
    for (; edge_it != border_edges.end(); ++edge_it, ++tri_counter)
    {
        int node0 = edge_it->first;
        int node1 = edge_it->second;
        int trinode0 = cl[3 * border_els[tri_counter]];
        int trinode1 = cl[3 * border_els[tri_counter] + 1];
        int trinode2 = cl[3 * border_els[tri_counter] + 2];
        if ((node1 == trinode0 && node0 == trinode1)
            || (node1 == trinode1 && node0 == trinode2)
            || (node1 == trinode2 && node0 == trinode0))
        {
            int tmp = node1;
            node1 = node0;
            node0 = tmp;
        }
        int node0p = border_new[node0];
        int node1p = border_new[node1];
        clExt.push_back(node0); // extra conn
        clExt.push_back(node0p);
        clExt.push_back(node1);
        clExt.push_back(node1);
        clExt.push_back(node0p);
        clExt.push_back(node1p);
    }
    //mapnit = node_normals.begin();
    mapnit = node_normals.begin();
    mapend = node_normals.end();
    int old_num_nodes = (int)x.size();
    xExt.resize(node_normals.size(), 0.0);
    yExt.resize(node_normals.size(), 0.0);
    zExt.resize(node_normals.size(), 0.0);
    OutData.resize(node_normals.size(), 0.0);
    for (; mapnit != mapend; ++mapnit)
    {
        // get also the old node label...
        int nodelab = (*mapnit).first;
        const vector<float> &normal = (*mapnit).second;
        int new_label = border_new[nodelab] - old_num_nodes;
        xExt[new_label] = x[nodelab] + normal[0];
        yExt[new_label] = y[nodelab] + normal[1];
        zExt[new_label] = z[nodelab] + normal[2];
        OutData[new_label] = InData[nodelab];
    }
}
