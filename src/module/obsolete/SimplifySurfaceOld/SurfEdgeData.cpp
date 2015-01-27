/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface edge collapse class                       **
 **                      for Data-Dependent Surface Reduction Methods      **
 **                                                                        **
 **                                                                        **
 **                             (C) 1998                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  April 1998  V1.0                                                **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "SurfEdgeData.h"
#include <util/coviseCompat.h>

coDistributedObject **ScalarDataEdgeCollapse::createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name)
{
    coDistributedObject **DO_Return = new coDistributedObject *[3];
    coDoPolygons *polygons_out;
    coDoFloat *data_out = 0;
    coDoVec3 *normals_out;
    int *vl, *pl, count, j;
    int *index;
    float *co_x, *co_y, *co_z;
    float *dt = 0;
    float *no_x = NULL, *no_y = NULL, *no_z = NULL;

    if (red_points == 0)
        return NULL;

    co_x = new float[red_points];
    co_y = new float[red_points];
    co_z = new float[red_points];
    if (norm_x)
    {
        no_x = new float[red_points];
        no_y = new float[red_points];
        no_z = new float[red_points];
    }
    if (data)
        dt = new float[red_points];
    vl = new int[red_tri * 3];
    pl = new int[red_tri];
    index = new int[num_points];

    count = 0;
    for (j = 0; j < num_points; j++)
        if (is_removed[j])
            index[j] = -1;
        else
            index[j] = count++;

    if (count != red_points)
    {
        Covise::sendError("ERROR: non-consistent number of points!");
        return NULL;
    }

    count = 0;
    for (j = 0; j < num_triangles; j++)
    {
        if (tri_list[j] != -1)
        {
            pl[count] = count * 3;
            if ((vertex_list[tri_list[j]] == -1) || (vertex_list[tri_list[j] + 1] == -1) || (vertex_list[tri_list[j] + 2] == -1))
            {
                Covise::sendError("ERROR: no entry in vertex_list for triangle");
                return NULL;
            }
            vl[count * 3] = index[vertex_list[tri_list[j]]];
            vl[count * 3 + 1] = index[vertex_list[tri_list[j] + 1]];
            vl[count * 3 + 2] = index[vertex_list[tri_list[j] + 2]];

            co_x[vl[count * 3]] = coords_x[vertex_list[tri_list[j]]];
            co_x[vl[count * 3 + 1]] = coords_x[vertex_list[tri_list[j] + 1]];
            co_x[vl[count * 3 + 2]] = coords_x[vertex_list[tri_list[j] + 2]];

            co_y[vl[count * 3]] = coords_y[vertex_list[tri_list[j]]];
            co_y[vl[count * 3 + 1]] = coords_y[vertex_list[tri_list[j] + 1]];
            co_y[vl[count * 3 + 2]] = coords_y[vertex_list[tri_list[j] + 2]];

            co_z[vl[count * 3]] = coords_z[vertex_list[tri_list[j]]];
            co_z[vl[count * 3 + 1]] = coords_z[vertex_list[tri_list[j] + 1]];
            co_z[vl[count * 3 + 2]] = coords_z[vertex_list[tri_list[j] + 2]];

            if (norm_x)
            {
                no_x[vl[count * 3]] = norm_x[vertex_list[tri_list[j]]];
                no_x[vl[count * 3 + 1]] = norm_x[vertex_list[tri_list[j] + 1]];
                no_x[vl[count * 3 + 2]] = norm_x[vertex_list[tri_list[j] + 2]];

                no_y[vl[count * 3]] = norm_y[vertex_list[tri_list[j]]];
                no_y[vl[count * 3 + 1]] = norm_y[vertex_list[tri_list[j] + 1]];
                no_y[vl[count * 3 + 2]] = norm_y[vertex_list[tri_list[j] + 2]];

                no_z[vl[count * 3]] = norm_z[vertex_list[tri_list[j]]];
                no_z[vl[count * 3 + 1]] = norm_z[vertex_list[tri_list[j] + 1]];
                no_z[vl[count * 3 + 2]] = norm_z[vertex_list[tri_list[j] + 2]];
            }
            if (data)
            {
                dt[vl[count * 3]] = data[vertex_list[tri_list[j]]];
                dt[vl[count * 3 + 1]] = data[vertex_list[tri_list[j] + 1]];
                dt[vl[count * 3 + 2]] = data[vertex_list[tri_list[j] + 2]];
            }

            count++;
        }
    }

    if (count != red_tri)
    {
        Covise::sendError("ERROR: Creation of output objects failed (non-consistent number of triangles)");
        return NULL;
    }

    //  num_vertices=vertex-vertice_list;

    polygons_out = new coDoPolygons(Triangle_name, red_points, co_x, co_y, co_z, 3 * red_tri, vl, red_tri, pl);

    if (!polygons_out->objectOk())
    {
        Covise::sendError("ERROR: Creation of geometry object 'meshOut' failed");
        return NULL;
    }

    if (data)
    {
        data_out = new coDoFloat(Data_name, red_points, dt);

        if (!data_out->objectOk())
        {
            Covise::sendError("ERROR: Creation of geometry object 'dataOut' failed");
            return NULL;
        }
    }

    if (norm_x)
    {
        normals_out = new coDoVec3(Normals_name, red_points, no_x, no_y, no_z);
        if (!normals_out->objectOk())
        {
            Covise::sendError("ERROR: Creation of geometry object 'normalsOut' failed");
            return NULL;
        }
    }
    else
        normals_out = NULL;

    DO_Return[0] = polygons_out;
    DO_Return[1] = data_out;
    DO_Return[2] = normals_out;

    delete[] vl;
    delete[] pl;
    delete[] index;
    delete[] co_x;
    delete[] co_y;
    delete[] co_z;
    delete[] no_x;
    delete[] no_y;
    delete[] no_z;
    delete[] dt;

    return DO_Return;
}

coDistributedObject **VectorDataEdgeCollapse::createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name)
{
    coDistributedObject **DO_Return = new coDistributedObject *[3];
    coDoPolygons *polygons_out;
    coDoVec3 *data_out;
    coDoVec3 *normals_out;
    int *vl, *pl, count, j;
    int *index;
    float *co_x, *co_y, *co_z;
    float *dt_x, *dt_y, *dt_z;
    float *no_x = NULL, *no_y = NULL, *no_z = NULL;

    if (red_points == 0)
        return NULL;

    co_x = new float[red_points];
    co_y = new float[red_points];
    co_z = new float[red_points];
    if (norm_x)
    {
        no_x = new float[red_points];
        no_y = new float[red_points];
        no_z = new float[red_points];
    }
    dt_x = new float[red_points];
    dt_y = new float[red_points];
    dt_z = new float[red_points];

    vl = new int[red_tri * 3];
    pl = new int[red_tri];
    index = new int[num_points];

    count = 0;
    for (j = 0; j < num_points; j++)
        if (is_removed[j])
            index[j] = -1;
        else
            index[j] = count++;

    if (count != red_points)
    {
        Covise::sendError("ERROR: non-consistent number of points!");
        return NULL;
    }

    count = 0;
    for (j = 0; j < num_triangles; j++)
    {
        if (tri_list[j] != -1)
        {
            pl[count] = count * 3;
            if ((vertex_list[tri_list[j]] == -1) || (vertex_list[tri_list[j] + 1] == -1) || (vertex_list[tri_list[j] + 2] == -1))
            {
                Covise::sendError("ERROR: no entry in vertex_list for triangle");
                return NULL;
            }
            vl[count * 3] = index[vertex_list[tri_list[j]]];
            vl[count * 3 + 1] = index[vertex_list[tri_list[j] + 1]];
            vl[count * 3 + 2] = index[vertex_list[tri_list[j] + 2]];

            co_x[vl[count * 3]] = coords_x[vertex_list[tri_list[j]]];
            co_x[vl[count * 3 + 1]] = coords_x[vertex_list[tri_list[j] + 1]];
            co_x[vl[count * 3 + 2]] = coords_x[vertex_list[tri_list[j] + 2]];

            co_y[vl[count * 3]] = coords_y[vertex_list[tri_list[j]]];
            co_y[vl[count * 3 + 1]] = coords_y[vertex_list[tri_list[j] + 1]];
            co_y[vl[count * 3 + 2]] = coords_y[vertex_list[tri_list[j] + 2]];

            co_z[vl[count * 3]] = coords_z[vertex_list[tri_list[j]]];
            co_z[vl[count * 3 + 1]] = coords_z[vertex_list[tri_list[j] + 1]];
            co_z[vl[count * 3 + 2]] = coords_z[vertex_list[tri_list[j] + 2]];

            if (norm_x)
            {
                no_x[vl[count * 3]] = norm_x[vertex_list[tri_list[j]]];
                no_x[vl[count * 3 + 1]] = norm_x[vertex_list[tri_list[j] + 1]];
                no_x[vl[count * 3 + 2]] = norm_x[vertex_list[tri_list[j] + 2]];

                no_y[vl[count * 3]] = norm_y[vertex_list[tri_list[j]]];
                no_y[vl[count * 3 + 1]] = norm_y[vertex_list[tri_list[j] + 1]];
                no_y[vl[count * 3 + 2]] = norm_y[vertex_list[tri_list[j] + 2]];

                no_z[vl[count * 3]] = norm_z[vertex_list[tri_list[j]]];
                no_z[vl[count * 3 + 1]] = norm_z[vertex_list[tri_list[j] + 1]];
                no_z[vl[count * 3 + 2]] = norm_z[vertex_list[tri_list[j] + 2]];
            }
            dt_x[vl[count * 3]] = data_u[vertex_list[tri_list[j]]];
            dt_x[vl[count * 3 + 1]] = data_u[vertex_list[tri_list[j] + 1]];
            dt_x[vl[count * 3 + 2]] = data_u[vertex_list[tri_list[j] + 2]];

            dt_y[vl[count * 3]] = data_v[vertex_list[tri_list[j]]];
            dt_y[vl[count * 3 + 1]] = data_v[vertex_list[tri_list[j] + 1]];
            dt_y[vl[count * 3 + 2]] = data_v[vertex_list[tri_list[j] + 2]];

            dt_z[vl[count * 3]] = data_w[vertex_list[tri_list[j]]];
            dt_z[vl[count * 3 + 1]] = data_w[vertex_list[tri_list[j] + 1]];
            dt_z[vl[count * 3 + 2]] = data_w[vertex_list[tri_list[j] + 2]];

            count++;
        }
    }

    if (count != red_tri)
    {
        Covise::sendError("ERROR: non-consistent number of triangles!");
        return NULL;
    }

    //  num_vertices=vertex-vertice_list;

    polygons_out = new coDoPolygons(Triangle_name, red_points, co_x, co_y, co_z, 3 * red_tri, vl, red_tri, pl);

    if (!polygons_out->objectOk())
    {
        Covise::sendError("ERROR: creation of geometry object 'meshOut' failed");
        return NULL;
    }

    data_out = new coDoVec3(Data_name, red_points, dt_x, dt_y, dt_z);

    if (!data_out->objectOk())
    {
        Covise::sendError("ERROR: creation of geometry object 'dataOut' failed");
        return NULL;
    }

    if (norm_x)
    {
        normals_out = new coDoVec3(Normals_name, red_points, no_x, no_y, no_z);

        if (!normals_out->objectOk())
        {
            Covise::sendError("ERROR: creation of geometry object 'normalsOut' failed");
            return NULL;
        }
    }
    else
        normals_out = NULL;

    DO_Return[0] = polygons_out;
    DO_Return[1] = data_out;
    DO_Return[2] = normals_out;

    delete[] vl;
    delete[] pl;
    delete[] index;
    delete[] co_x;
    delete[] co_y;
    delete[] co_z;
    delete[] no_x;
    delete[] no_y;
    delete[] no_z;
    delete[] dt_x;
    delete[] dt_y;
    delete[] dt_z;

    return DO_Return;
}

/////////////////////////////////////////////////////////////////////////////
// Functions for Scalar Data Supplied Surfaces                             //
/////////////////////////////////////////////////////////////////////////////

float ScalarDataEdgeCollapse::L1_gradient(int v, int *points, int num)
{
    int i;
    double max, x, norm;

    // compute the maximum over all surrounding points of
    //      || data(i) - data||_1
    //   ---------------------------
    //   ||coords(i) - coords(v)||_1

    max = 0.0;
    for (i = 0; i < num; i++)
    {
        x = fabs((double)data[points[i]] - (double)data[v]);
        norm = fabs((double)coords_x[points[i]] - (double)coords_x[v]);
        norm += fabs((double)coords_y[points[i]] - (double)coords_y[v]);
        norm += fabs((double)coords_z[points[i]] - (double)coords_z[v]);

        if (norm > 5E-02)
            x /= norm;
        else
            x = 0.0;

        if (x > max)
            max = x;
    }
    return max;
}

float ScalarDataEdgeCollapse::Data_deviation(int v, int *points, int num)
{
    int i;
    double max, x;

    // compute the maximum over all surrounding points of
    //      || data(i) - data||_1

    max = 0.0;
    for (i = 0; i < num; i++)
    {
        x = fabs((double)data[points[i]] - (double)data[v]);
        if (x > max)
            max = x;
    }
    return max;
}

float ScalarDataEdgeCollapse::Zalgaller_gradient(int v, int *points, int num)
{
    int i;
    double diff[MAXTRI];
    double grad[3];
    double Y[MAXTRI][MAXTRI];

    // approximate the gradient vector by a least square approximation
    //      || data(i) - data||_1
    //   ---------------------------        right hand side, i = 0,...,num-1
    //   ||coords(i) - coords(v)||_2
    // scalar products of edge direction and gradient build left hand side

    for (i = 0; i < num; i++)
    {
        diff[i] = fabs((double)data[points[i]] - (double)data[v]);
        Y[i][0] = ((double)coords_x[points[i]] - (double)coords_x[v]);
        Y[i][1] = ((double)coords_y[points[i]] - (double)coords_y[v]);
        Y[i][2] = ((double)coords_z[points[i]] - (double)coords_z[v]);
    }

    if (!Least_Square(Y, diff, grad, num, 3))
    { // system of equations is underdetermined
        // add condition: (normal(v),grad(v)) = 1
        num++;
        diff[num - 1] = 0.0;
        Y[num - 1][0] = norm_x[v];
        Y[num - 1][1] = norm_y[v];
        Y[num - 1][2] = norm_z[v];
        if (!Least_Square(Y, diff, grad, num, 3))
        {
            grad[0] = 0.0;
            grad[1] = 0.0;
            grad[2] = 0.0;
        }
    }

    return fabs(grad[0]) + fabs(grad[1]) + fabs(grad[2]);
}

float ScalarDataEdgeCollapse::compute_gradient(int v, int *points, int num, float norm)
{
    float x;

    if (!data)
        return 0.0;

    switch (which_grad)
    {
    case 1:
        x = L1_gradient(v, points, num);
        break;
    case 2:
        x = Zalgaller_gradient(v, points, num);
        break;
    case 3:
        x = Data_deviation(v, points, num);
        break;
    default:
        x = L1_gradient(v, points, num);
    };
    if (norm > 1E-06)
        x /= norm;
    return x;
}

float ScalarDataEdgeCollapse::compute_weight(int v1, int v2)
{
    float length;
    //////////////////////////////////////////////////////////////////////////
    // weight is square of edge length + sum of gradients in both endpoints //
    //////////////////////////////////////////////////////////////////////////
    length = (coords_x[v1] - coords_x[v2]) * (coords_x[v1] - coords_x[v2]);
    length += (coords_y[v1] - coords_y[v2]) * (coords_y[v1] - coords_y[v2]);
    length += (coords_z[v1] - coords_z[v2]) * (coords_z[v1] - coords_z[v2]);

    if (!grad)
        return length;

    //return(length + 0.2 * (compact[v1] + compact[v2])
    //              + enforce * (grad[v1] + grad[v2]));
    return length + enforce * (grad[v1] + grad[v2]);
}

void ScalarDataEdgeCollapse::preprocess(int &red_tri)
{
    int i, j;
    int v1, v2;
    int num_link, num_pnt;
    pair link[MAXTRI];
    int points[MAXTRI];
    Edge e;

    remove_flat_triangles(red_tri);
    initialize_connectivity();
    if (norm_x == NULL)
        generate_normals();
    max_gradient = 0.0;
    //////////////////////////////////////////////////////////////////////
    // initialize topology and compute gradients and compactness values

    for (i = 0; i < num_points; i++)
    {
        make_link(i, link, num_link);
        stars[i].boundary = check_boundary(i, link, num_link);
        stars[i].manifold = check_manifold(i, link, num_link);

        extract_points(i, num_link, link, num_pnt, points);
        if (grad)
        {
            grad[i] = compute_gradient(i, points, num_pnt, 1);
            if (grad[i] > max_gradient)
                max_gradient = grad[i];
        }

        angle[i] = vertex_on_feature_edge(i, max_angle);
        //compact[i] = compute_compactness(i);
    }
    if (max_gradient > 1E-06)
        for (i = 0; i < num_points; i++)
            grad[i] /= max_gradient;
    //////////////////////////////////////////////////////////////////////
    // make priority queue

    for (i = 0; i < num_triangles; i++)
    {
        for (j = 0; j < 3; j++)
        {
            v1 = vertex_list[tri_list[i] + j];
            v2 = vertex_list[tri_list[i] + (j + 1) % 3];
            e.set_endpoints(v1, v2);
            e.set_length(compute_weight(v1, v2));

            heap->append(e);
            ept[i][j] = 3 * i + j + 1;
        }
    }
    heap->construct();
}

void ScalarDataEdgeCollapse::update_heap(int v1, int v2, Star star, int to_update, int *vert, int to_remove, int *tri)
{
    Edge e;
    pair link[MAXTRI];
    int num_link;
    int points[MAXTRI];
    int i, j, k, a, b;
    float weight;

    for (i = 0; i < to_update; i++)
    {
        make_link(vert[i], link, num_link);
        extract_points(num_link, link, points);
        if (grad)
            grad[vert[i]] = compute_gradient(vert[i], points, num_link, 1);
    }

    for (i = 0; i < star.num_tri; i++)
    {
        if ((star.tri[i] == tri[0]) || (star.tri[i] == tri[to_remove - 1]))
        {
            for (j = 0; j < 3; j++)
            {
                k = heap->get_index(ept[star.tri[i]][j]);
                if (k <= heap->getSize())
                    heap->remove(k);
            }
        }
        else
            for (j = 0; j < 3; j++)
            {
                e = heap->get_item(ept[star.tri[i]][j]);
                a = e.get_v1();
                b = e.get_v2();

                if ((a == v1) || (b == v1) || (a == v2) || (b == v2))
                {
                    if (a == v2)
                    {
                        e.set_v1(v1);
                        a = v1;
                    }
                    if (b == v2)
                    {
                        e.set_v2(v1);
                        b = v1;
                    }

                    weight = compute_weight(a, b);
                    e.set_length(weight);

                    k = heap->get_index(ept[star.tri[i]][j]);
                    if (k <= heap->getSize())
                        heap->change(k, e);
                    else
                    { // update edge_array and heap
                        heap->insert_again(k, ept[star.tri[i]][j], e);
                    }
                }
            }
    }
}

void ScalarDataEdgeCollapse::update(int &red_tri, int v1, int v2, Star &star, pair *link, int num_link, float x_0, float y_0, float z_0)
{ // To be invoked to complete each edge collapsing iteration.
    // Removes the vertex v2 from the vertex list, updates the coordinates of v1
    // as well as all affected structures in the program:
    // stars of vertices in link, tri_list, heap, stars[v1]
    int l;
    int tri[2];
    int vert[MAXTRI];
    int to_remove = 0;
    int to_update = 0;

    // tri = pair of triangles to remove
    find_neighbors(v1, v2, star, to_remove, tri);
    if (!((to_remove == 2) || (star.boundary && (to_remove == 1))))
        Covise::sendInfo("Edge has wrong number of neighbors!");

    // vert = array of vertices in the link (to be updated)
    extract_points(v1, v2, num_link, link, to_update, vert);

    // update stars around the link
    for (l = 0; l < to_update; l++)
        update_star(v1, v2, star, l, vert, to_remove, tri);

    // update data components, interpolate if necessary
    if (data)
    {
        if (coords_x[v2] == x_0 && coords_y[v2] == y_0 && coords_z[v2] == z_0)
            data[v1] = data[v2];
        else
        {
            if (coords_x[v1] != x_0 || coords_y[v1] != y_0 || coords_z[v1] != z_0)
                data[v1] = (data[v1] + data[v2]) / 2;
        }
    }

    // update coordinates and label v2 as removed
    coords_x[v1] = x_0;
    coords_y[v1] = y_0;
    coords_z[v1] = z_0;
    is_removed[v2] = 1;

    // update star and tri_list
    update_global_structures(v1, v2, star, to_remove, tri);

    if (to_remove == 2)
        red_tri -= 2;
    else
        red_tri--;

    // copy star to stars[v1]
    delete[] stars[v1].tri;
    stars[v1].tri = new int[star.num_tri];
    for (l = 0; l < star.num_tri; l++)
        stars[v1].tri[l] = star.tri[l];
    stars[v1].num_tri = star.num_tri;
    stars[v1].boundary = star.boundary;
    stars[v1].manifold = star.manifold;

    if (angle[v2] > angle[v1])
        angle[v1] = angle[v2];

    // update compact
    //for(l = 0; l < to_update; l++)
    //  compact[vert[l]] = compute_compactness(vert[l]);
    //compact[v1] = compute_compactness(v1);

    // update heap and edge_array
    update_heap(v1, v2, star, to_update, vert, to_remove, tri);

    delete[] star.tri;
}

void ScalarDataEdgeCollapse::Reduce(int &red_tri, int &red_pnt)
{
    Edge current;
    Star star;
    int v1, v2;
    float cur_percent;
    int okay;
    char buf[100];
    float x_0, y_0, z_0;
    clock_t starttime, endtime;
    double time;
    pair link[MAXTRI];
    int num_link = 0;

    heap = new PQ<Edge>(num_triangles * 3);

    cur_percent = percent / 100.0;
    red_tri = num_triangles;
    red_pnt = num_points;

    starttime = clock();

    preprocess(red_tri);

    while ((heap->getSize() > TERMINATE) && (red_tri / (float)num_triangles > cur_percent))
    {
        current = heap->get_next();
        v1 = current.get_v1();
        v2 = current.get_v2();
        // valence test
        if (stars[v1].num_tri + stars[v2].num_tri > 5)
        { // prevent too large stars
            if (stars[v1].num_tri + stars[v2].num_tri < MAXTRI)
            {
                merge_stars(v1, v2, star);
                make_link(v1, v2, star, link, num_link);
                if (check_link(v1, v2, star, link, num_link))
                {
                    okay = sort_link(link, num_link);
                    // topology preserving tests
                    if (okay && star.manifold && !((angle[v1] || angle[v2]) && star.boundary))
                    { // simplify edge
                        if (angle[v1] && angle[v2])
                            okay = straighten_edge(v1, v2, star, link, num_link, x_0, y_0, z_0);
                        else
                        {
                            if (stars[v1].boundary && stars[v2].boundary)
                                okay = straighten_boundary(v1, v2, star, link, num_link, x_0, y_0, z_0);
                            else
                                okay = compute_newpoint(v1, v2, star, link, num_link, x_0, y_0, z_0);
                        }
                        if (okay && volume_bound >= fabs(compute_star_volume(v1, v2, star, link, num_link, x_0, y_0, z_0)))
                        {
                            update(red_tri, v1, v2, star, link, num_link, x_0, y_0, z_0);
                            red_pnt -= 1;
                        }
                        else
                        {
                            delete[] star.tri;
                        }
                    }
                    else
                    {
                        delete[] star.tri;
                    }
                }
                else
                {
                    delete[] star.tri;
                }
            }
        }
    }
    cur_percent = 100.0 * red_tri / (float)num_triangles;
    endtime = clock();
    time = (endtime - starttime) / (double)CLOCKS_PER_SEC;

    if (heap->getSize() == TERMINATE)
        sprintf(buf, "Reduction capacity exceeded: Removed %d triangles of %d, i.e. %.2f %% are left", num_triangles - red_tri, num_triangles, cur_percent);
    else
        sprintf(buf, "Removed %d triangles of %d, i.e. %.2f %% are left\n", num_triangles - red_tri, num_triangles, percent);
    Covise::sendInfo(buf);
    sprintf(buf, "Time: %.2f seconds\n", time);
    Covise::sendInfo(buf);

    delete heap;
}

/////////////////////////////////////////////////////////////////////////////
// Functions for Vector Data Supplied Surfaces                             //
/////////////////////////////////////////////////////////////////////////////

float VectorDataEdgeCollapse::L1_gradient(int v, int *points, int num)
{
    int i;
    double max, x, norm;

    // compute the maximum over all surrounding points of
    //      || data(i) - data||_1
    //   ---------------------------
    //   ||coords(i) - coords(v)||_1

    max = 0.0;
    for (i = 0; i < num; i++)
    {
        x = fabs((double)data_u[points[i]] - (double)data_u[v]);
        x += fabs((double)data_v[points[i]] - (double)data_v[v]);
        x += fabs((double)data_w[points[i]] - (double)data_w[v]);
        norm = fabs((double)coords_x[points[i]] - (double)coords_x[v]);
        norm += fabs((double)coords_y[points[i]] - (double)coords_y[v]);
        norm += fabs((double)coords_z[points[i]] - (double)coords_z[v]);

        if (norm > 5E-02)
            x /= norm;
        else
            x = 0.0;

        if (x > max)
            max = x;
    }
    return max;
}

float VectorDataEdgeCollapse::Data_deviation(int v, int *points, int num)
{
    int i;
    double max, x;

    // compute the maximum over all surrounding points of
    //      || data(i) - data||_1

    max = 0.0;
    for (i = 0; i < num; i++)
    {
        x = fabs((double)data_u[points[i]] - (double)data_u[v]);
        x += fabs((double)data_v[points[i]] - (double)data_v[v]);
        x += fabs((double)data_w[points[i]] - (double)data_w[v]);
        if (x > max)
            max = x;
    }
    return max;
}

float VectorDataEdgeCollapse::Zalgaller_gradient(int v, int *points, int num)
{
    int i;
    double diff[MAXTRI];
    double grad[3];
    double Y[MAXTRI][MAXTRI];

    // approximate the gradient vector by a least square approximation
    //      || data(i) - data||_1
    //   ---------------------------        right hand side, i = 0,...,num-1
    //   ||coords(i) - coords(v)||_2
    // scalar products of edge direction and gradient build left hand side

    for (i = 0; i < num; i++)
    {
        diff[i] = fabs((double)data_u[points[i]] - (double)data_u[v]);
        diff[i] += fabs((double)data_v[points[i]] - (double)data_v[v]);
        diff[i] += fabs((double)data_w[points[i]] - (double)data_w[v]);
        Y[i][0] = ((double)coords_x[points[i]] - (double)coords_x[v]);
        Y[i][1] = ((double)coords_y[points[i]] - (double)coords_y[v]);
        Y[i][2] = ((double)coords_z[points[i]] - (double)coords_z[v]);
    }

    if (!Least_Square(Y, diff, grad, num, 3))
    { // system of equations is underdetermined
        // add condition: (normal(v),grad(v)) = 1
        num++;
        diff[num - 1] = 0.0;
        Y[num - 1][0] = norm_x[v];
        Y[num - 1][1] = norm_y[v];
        Y[num - 1][2] = norm_z[v];
        if (!Least_Square(Y, diff, grad, num, 3))
        {
            grad[0] = 0.0;
            grad[1] = 0.0;
            grad[2] = 0.0;
        }
    }

    return fabs(grad[0]) + fabs(grad[1]) + fabs(grad[2]);
}

float VectorDataEdgeCollapse::compute_gradient(int v, int *points, int num, float norm)
{
    float x;

    switch (which_grad)
    {
    case 1:
        x = L1_gradient(v, points, num);
        break;
    case 2:
        x = Zalgaller_gradient(v, points, num);
        break;
    case 3:
        x = Data_deviation(v, points, num);
        break;
    default:
        x = L1_gradient(v, points, num);
    };
    if (norm > 1E-06)
        x /= norm;
    return x;
}

float VectorDataEdgeCollapse::compute_weight(int v1, int v2)
{
    float length;
    //////////////////////////////////////////////////////////////////////////
    // weight is square of edge length + sum of gradients in both endpoints //
    //////////////////////////////////////////////////////////////////////////
    length = (coords_x[v1] - coords_x[v2]) * (coords_x[v1] - coords_x[v2]);
    length += (coords_y[v1] - coords_y[v2]) * (coords_y[v1] - coords_y[v2]);
    length += (coords_z[v1] - coords_z[v2]) * (coords_z[v1] - coords_z[v2]);

    //return(length + 0.2 * (compact[v1] + compact[v2])
    //              + enforce * (grad[v1] + grad[v2]));
    return length + enforce * (grad[v1] + grad[v2]);
}

void VectorDataEdgeCollapse::preprocess(int &red_tri)
{
    int i, j;
    int v1, v2;
    int num_link, num_pnt;
    pair link[MAXTRI];
    int points[MAXTRI];
    Edge e;

    remove_flat_triangles(red_tri);
    initialize_connectivity();
    if (norm_x == NULL)
        generate_normals();
    max_gradient = 0.0;
    //////////////////////////////////////////////////////////////////////
    // initialize topology and compute gradients and compactness values

    for (i = 0; i < num_points; i++)
    {
        make_link(i, link, num_link);
        stars[i].boundary = check_boundary(i, link, num_link);
        stars[i].manifold = check_manifold(i, link, num_link);

        extract_points(i, num_link, link, num_pnt, points);
        grad[i] = compute_gradient(i, points, num_pnt, 1);
        if (grad[i] > max_gradient)
            max_gradient = grad[i];

        angle[i] = vertex_on_feature_edge(i, max_angle);
        //compact[i] = compute_compactness(i);
    }
    if (max_gradient > 1E-06)
        for (i = 0; i < num_points; i++)
            grad[i] /= max_gradient;
    //////////////////////////////////////////////////////////////////////
    // make priority queue

    for (i = 0; i < num_triangles; i++)
    {
        for (j = 0; j < 3; j++)
        {
            v1 = vertex_list[tri_list[i] + j];
            v2 = vertex_list[tri_list[i] + (j + 1) % 3];
            e.set_endpoints(v1, v2);
            e.set_length(compute_weight(v1, v2));

            heap->append(e);
            ept[i][j] = 3 * i + j + 1;
        }
    }
    heap->construct();
}

void VectorDataEdgeCollapse::update_heap(int v1, int v2, Star star, int to_update, int *vert, int to_remove, int *tri)
{
    Edge e;
    pair link[MAXTRI];
    int num_link;
    int points[MAXTRI];
    int i, j, k, a, b;
    float weight;

    for (i = 0; i < to_update; i++)
    {
        make_link(vert[i], link, num_link);
        extract_points(num_link, link, points);
        grad[vert[i]] = compute_gradient(vert[i], points, num_link, 1);
    }

    for (i = 0; i < star.num_tri; i++)
    {
        if ((star.tri[i] == tri[0]) || (star.tri[i] == tri[to_remove - 1]))
        {
            for (j = 0; j < 3; j++)
            {
                k = heap->get_index(ept[star.tri[i]][j]);
                if (k <= heap->getSize())
                    heap->remove(k);
            }
        }
        else
            for (j = 0; j < 3; j++)
            {
                e = heap->get_item(ept[star.tri[i]][j]);
                a = e.get_v1();
                b = e.get_v2();

                if ((a == v1) || (b == v1) || (a == v2) || (b == v2))
                {
                    if (a == v2)
                    {
                        e.set_v1(v1);
                        a = v1;
                    }
                    if (b == v2)
                    {
                        e.set_v2(v1);
                        b = v1;
                    }

                    weight = compute_weight(a, b);
                    e.set_length(weight);

                    k = heap->get_index(ept[star.tri[i]][j]);
                    if (k <= heap->getSize())
                        heap->change(k, e);
                    else
                    { // update edge_array and heap
                        heap->insert_again(k, ept[star.tri[i]][j], e);
                    }
                }
            }
    }
}

void VectorDataEdgeCollapse::update(int &red_tri, int v1, int v2, Star &star, pair *link, int num_link, float x_0, float y_0, float z_0)
{ // To be invoked to complete each edge collapsing iteration.
    // Removes the vertex v2 from the vertex list, updates the coordinates of v1
    // as well as all affected structures in the program:
    // stars of vertices in link, tri_list, heap, stars[v1]
    int l;
    int tri[2];
    int vert[MAXTRI];
    int to_remove = 0;
    int to_update = 0;

    // tri = pair of triangles to remove
    find_neighbors(v1, v2, star, to_remove, tri);
    if (!((to_remove == 2) || (star.boundary && (to_remove == 1))))
        Covise::sendInfo("Edge has wrong number of neighbors!");

    // vert = array of vertices in the link (to be updated)
    extract_points(v1, v2, num_link, link, to_update, vert);

    // update stars around the link
    for (l = 0; l < to_update; l++)
        update_star(v1, v2, star, l, vert, to_remove, tri);

    // update data components, interpolate if necessary
    if (coords_x[v2] == x_0 && coords_y[v2] == y_0 && coords_z[v2] == z_0)
    {
        data_u[v1] = data_u[v2];
        data_v[v1] = data_v[v2];
        data_w[v1] = data_w[v2];
    }
    else
    {
        if (coords_x[v1] != x_0 || coords_y[v1] != y_0 || coords_z[v1] != z_0)
        {
            data_u[v1] = (data_u[v1] + data_u[v2]) / 2;
            data_v[v1] = (data_v[v1] + data_v[v2]) / 2;
            data_w[v1] = (data_w[v1] + data_w[v2]) / 2;
        }
    }

    // update coordinates and label v2 as removed
    coords_x[v1] = x_0;
    coords_y[v1] = y_0;
    coords_z[v1] = z_0;
    is_removed[v2] = 1;

    // update star and tri_list
    update_global_structures(v1, v2, star, to_remove, tri);

    if (to_remove == 2)
        red_tri -= 2;
    else
        red_tri--;

    // copy star to stars[v1]
    delete[] stars[v1].tri;
    stars[v1].tri = new int[star.num_tri];
    for (l = 0; l < star.num_tri; l++)
        stars[v1].tri[l] = star.tri[l];
    stars[v1].num_tri = star.num_tri;
    stars[v1].boundary = star.boundary;
    stars[v1].manifold = star.manifold;

    if (angle[v2] > angle[v1])
        angle[v1] = angle[v2];

    // update compact
    //for(l = 0; l < to_update; l++)
    //  compact[vert[l]] = compute_compactness(vert[l]);
    //compact[v1] = compute_compactness(v1);

    // update heap and edge_array
    update_heap(v1, v2, star, to_update, vert, to_remove, tri);

    delete[] star.tri;
}

void VectorDataEdgeCollapse::Reduce(int &red_tri, int &red_pnt)
{
    Edge current;
    Star star;
    int v1, v2;
    float cur_percent;
    int okay;
    char buf[100];
    float x_0, y_0, z_0;
    clock_t starttime, endtime;
    double time;
    pair link[MAXTRI];
    int num_link = 0;

    heap = new PQ<Edge>(num_triangles * 3);

    cur_percent = percent / 100.0;
    red_tri = num_triangles;
    red_pnt = num_points;

    starttime = clock();

    preprocess(red_tri);

    while ((heap->getSize() > TERMINATE) && (red_tri / (float)num_triangles > cur_percent))
    {
        current = heap->get_next();
        v1 = current.get_v1();
        v2 = current.get_v2();

        // valence test
        if (stars[v1].num_tri + stars[v2].num_tri > 5)
        { // prevent too large stars
            if (stars[v1].num_tri + stars[v2].num_tri < MAXTRI)
            {
                merge_stars(v1, v2, star);
                make_link(v1, v2, star, link, num_link);
                if (check_link(v1, v2, star, link, num_link))
                {
                    okay = sort_link(link, num_link);
                    // topology preserving tests
                    if (okay && star.manifold && !((angle[v1] || angle[v2]) && star.boundary))
                    { // simplify edge
                        if (angle[v1] && angle[v2])
                            okay = straighten_edge(v1, v2, star, link, num_link, x_0, y_0, z_0);
                        else
                        {
                            if (stars[v1].boundary && stars[v2].boundary)
                                okay = straighten_boundary(v1, v2, star, link, num_link, x_0, y_0, z_0);
                            else
                                okay = compute_newpoint(v1, v2, star, link, num_link, x_0, y_0, z_0);
                        }
                        if (okay && volume_bound >= fabs(compute_star_volume(v1, v2, star, link, num_link, x_0, y_0, z_0)))
                        {
                            update(red_tri, v1, v2, star, link, num_link, x_0, y_0, z_0);
                            red_pnt -= 1;
                        }
                        else
                            delete[] star.tri;
                    }
                    else
                        delete[] star.tri;
                }
                else
                    delete[] star.tri;
            }
        }
    }
    cur_percent = 100.0 * red_tri / (float)num_triangles;
    endtime = clock();
    time = (endtime - starttime) / (double)CLOCKS_PER_SEC;

    if (heap->getSize() == TERMINATE)
        sprintf(buf, "Reduction capacity exceeded: Removed %d triangles of %d, i.e. %.2f %% are left", num_triangles - red_tri, num_triangles, cur_percent);
    else
        sprintf(buf, "Removed %d triangles of %d, i.e. %.2f %% are left\n", num_triangles - red_tri, num_triangles, percent);
    Covise::sendInfo(buf);
    sprintf(buf, "Time: %.2f seconds\n", time);
    Covise::sendInfo(buf);

    delete heap;
}
