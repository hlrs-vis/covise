/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface vertex removal class                      **
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
#include "SurfVertexData.h"
#include <util/coviseCompat.h>

coDistributedObject **ScalarDataVertexRemoval::createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name)
{
    coDistributedObject **DO_Return = new coDistributedObject *[3];
    coDoPolygons *polygons_out;
    coDoFloat *data_out;
    coDoVec3 *normals_out;
    int *vl, *pl, count, j;
    int *index;
    float *co_x, *co_y, *co_z;
    float *dt;
    float *no_x, *no_y, *no_z;

    if (red_points == 0)
        return (NULL);

    co_x = new float[red_points];
    co_y = new float[red_points];
    co_z = new float[red_points];
    no_x = new float[red_points];
    no_y = new float[red_points];
    no_z = new float[red_points];
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
        return (NULL);
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
                return (NULL);
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

            no_x[vl[count * 3]] = norm_x[vertex_list[tri_list[j]]];
            no_x[vl[count * 3 + 1]] = norm_x[vertex_list[tri_list[j] + 1]];
            no_x[vl[count * 3 + 2]] = norm_x[vertex_list[tri_list[j] + 2]];

            no_y[vl[count * 3]] = norm_y[vertex_list[tri_list[j]]];
            no_y[vl[count * 3 + 1]] = norm_y[vertex_list[tri_list[j] + 1]];
            no_y[vl[count * 3 + 2]] = norm_y[vertex_list[tri_list[j] + 2]];

            no_z[vl[count * 3]] = norm_z[vertex_list[tri_list[j]]];
            no_z[vl[count * 3 + 1]] = norm_z[vertex_list[tri_list[j] + 1]];
            no_z[vl[count * 3 + 2]] = norm_z[vertex_list[tri_list[j] + 2]];

            dt[vl[count * 3]] = data[vertex_list[tri_list[j]]];
            dt[vl[count * 3 + 1]] = data[vertex_list[tri_list[j] + 1]];
            dt[vl[count * 3 + 2]] = data[vertex_list[tri_list[j] + 2]];

            count++;
        }
    }

    if (count != red_tri)
    {
        Covise::sendError("ERROR: Creation of output objects failed (non-consistent number of triangles)");
        return (NULL);
    }

    //  num_vertices=vertex-vertice_list;

    polygons_out = new coDoPolygons(Triangle_name, red_points, co_x, co_y, co_z, 3 * red_tri, vl, red_tri, pl);

    if (!polygons_out->objectOk())
    {
        Covise::sendError("ERROR: Creation of geometry object 'meshOut' failed");
        return (NULL);
    }

    data_out = new coDoFloat(Data_name, red_points, dt);

    if (!data_out->objectOk())
    {
        Covise::sendError("ERROR: Creation of geometry object 'dataOut' failed");
        return (NULL);
    }

    normals_out = new coDoVec3(Normals_name, red_points, no_x, no_y, no_z);

    if (!normals_out->objectOk())
    {
        Covise::sendError("ERROR: Creation of geometry object 'normalsOut' failed");
        return (NULL);
    }

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

    return (DO_Return);
}

coDistributedObject **VectorDataVertexRemoval::createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name)
{
    coDistributedObject **DO_Return = new coDistributedObject *[3];
    coDoPolygons *polygons_out;
    coDoVec3 *data_out;
    coDoVec3 *normals_out;
    int *vl, *pl, count, j;
    int *index;
    float *co_x, *co_y, *co_z;
    float *dt_x, *dt_y, *dt_z;
    float *no_x, *no_y, *no_z;

    if (red_points == 0)
        return (NULL);

    co_x = new float[red_points];
    co_y = new float[red_points];
    co_z = new float[red_points];
    no_x = new float[red_points];
    no_y = new float[red_points];
    no_z = new float[red_points];
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
        return (NULL);
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
                return (NULL);
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

            no_x[vl[count * 3]] = norm_x[vertex_list[tri_list[j]]];
            no_x[vl[count * 3 + 1]] = norm_x[vertex_list[tri_list[j] + 1]];
            no_x[vl[count * 3 + 2]] = norm_x[vertex_list[tri_list[j] + 2]];

            no_y[vl[count * 3]] = norm_y[vertex_list[tri_list[j]]];
            no_y[vl[count * 3 + 1]] = norm_y[vertex_list[tri_list[j] + 1]];
            no_y[vl[count * 3 + 2]] = norm_y[vertex_list[tri_list[j] + 2]];

            no_z[vl[count * 3]] = norm_z[vertex_list[tri_list[j]]];
            no_z[vl[count * 3 + 1]] = norm_z[vertex_list[tri_list[j] + 1]];
            no_z[vl[count * 3 + 2]] = norm_z[vertex_list[tri_list[j] + 2]];

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
        return (NULL);
    }

    //  num_vertices=vertex-vertice_list;

    polygons_out = new coDoPolygons(Triangle_name, red_points, co_x, co_y, co_z, 3 * red_tri, vl, red_tri, pl);

    if (!polygons_out->objectOk())
    {
        Covise::sendError("ERROR: creation of geometry object 'meshOut' failed");
        return (NULL);
    }

    data_out = new coDoVec3(Data_name, red_points, dt_x, dt_y, dt_z);

    if (!data_out->objectOk())
    {
        Covise::sendError("ERROR: creation of geometry object 'dataOut' failed");
        return (NULL);
    }

    normals_out = new coDoVec3(Normals_name, red_points, no_x, no_y, no_z);

    if (!normals_out->objectOk())
    {
        Covise::sendError("ERROR: creation of geometry object 'normalsOut' failed");
        return (NULL);
    }

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

    return (DO_Return);
}

/////////////////////////////////////////////////////////////////////////////
// Functions for Scalar Data Supplied Surfaces                             //
/////////////////////////////////////////////////////////////////////////////

float ScalarDataVertexRemoval::L1_gradient(int v, int *points, int num)
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
    return (max);
}

float ScalarDataVertexRemoval::Data_deviation(int v, int *points, int num)
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
    return (max);
}

float ScalarDataVertexRemoval::Zalgaller_gradient(int v, int *points, int num)
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

    return (fabs(grad[0]) + fabs(grad[1]) + fabs(grad[2]));
}

float ScalarDataVertexRemoval::compute_gradient(int v, int *points, int num, float norm)
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
    return (x);
}

float ScalarDataVertexRemoval::compute_weight(int i, int *points, int num_pnt)
{
    float curv, grad, compact;
    curv = compute_curvature(i, points, num_pnt, max_curvature);
    grad = compute_gradient(i, points, num_pnt, max_gradient);
    compact = compute_compactness(i);
    return (curv + enforce * grad + compact);
}

float ScalarDataVertexRemoval::compute_weight(float curv, float grad, float compact)
{
    return (curv + enforce * grad + compact);
}

void ScalarDataVertexRemoval::preprocess(int &red_tri)
{
    int i;
    pair link[MAXTRI];
    int points[MAXTRI];
    int num_link, num_pnt;
    Vertex v;

    float *curvature;
    float *gradient;

    heap = new PQ<Vertex>(num_points);
    curvature = new float[num_points];
    gradient = new float[num_points];

    remove_flat_triangles(red_tri);
    initialize_connectivity();
    if (norm_x == NULL)
        generate_normals();
    max_gradient = max_curvature = avg_gradient = 0.0;

    for (i = 0; i < num_points; i++)
    {
        make_link(i, link, num_link);
        stars[i].boundary = check_boundary(i, link, num_link);
        stars[i].manifold = check_manifold(i, link, num_link);

        extract_points(i, num_link, link, num_pnt, points);
        curvature[i] = compute_curvature(i, points, num_pnt, 1);
        if (curvature[i] > max_curvature)
            max_curvature = curvature[i];
        gradient[i] = compute_gradient(i, points, num_pnt, 1);
        if (gradient[i] > max_gradient)
            max_gradient = gradient[i];
        avg_gradient += gradient[i];
        angle[i] = vertex_on_feature_edge(i, max_angle);
    }
    if (max_gradient > 1E-06)
        avg_gradient /= max_gradient;
    avg_gradient /= num_points;
    if (max_curvature > 1E-06)
        for (i = 0; i < num_points; i++)
            curvature[i] /= max_curvature;
    if (max_gradient > 1E-06)
        for (i = 0; i < num_points; i++)
            gradient[i] /= max_gradient;
    for (i = 0; i < num_points; i++)
        if (gradient[i] > ridge * avg_gradient)
            data_ridge[i] = 1;

    for (i = 0; i < num_points; i++)
    {
        v.set_key(i);
        v.set_weight(compute_weight(curvature[i], gradient[i], compute_compactness(i)));
        heap->append(v);
    }
    heap->construct();

    delete[] curvature;
    delete[] gradient;
}

int ScalarDataVertexRemoval::update(int v, pair *link, int num_link, triple *retriang)
{
    int i, j, k, l;
    int v0;
    int to_update[MAXTRI];
    int num_to;
    int tmp_tri[MAXTRI][MAXTRI];
    int tmp_num_tri[MAXTRI];
    int too_large = 0;
    pair local_link[MAXTRI];
    int num_locallink;
    int points[MAXTRI];
    int count;
    fl_triple normal;
    Vertex v1;

    extract_points(num_link, link, to_update);
    num_to = num_link;
    // Update stars around the link of v.
    // Fill first temporary stars, look whether the new triangulation will
    // create stars with more than MAXTRI triangles.
    // If this is the case, leave update, do not remove the vertex v.
    // If everything is okay, copy temporary structures to stars[...].
    l = 0;
    while (l < num_to && !too_large)
    {
        v0 = to_update[l];
        tmp_num_tri[l] = stars[v0].num_tri;
        for (i = 0; i < tmp_num_tri[l]; i++)
            tmp_tri[l][i] = stars[v0].tri[i];

        for (i = 0; i < tmp_num_tri[l]; i++)
        { // remove triangles of corona.triangles from each stars[v0]
            // in the border
            j = 0;
            while ((j < stars[v].num_tri) && (tmp_tri[l][i] != stars[v].tri[j]))
                j++;
            if (j != stars[v].num_tri)
            {
                for (k = i; k < tmp_num_tri[l] - 1; k++)
                    tmp_tri[l][k] = tmp_tri[l][k + 1];
                i--;
                tmp_num_tri[l]--;
            }
        }

        for (i = 0; i < num_link - 2; i++)
        { // append triangles of retri containing v0 to stars[v0].tri
            // CAUTION: triangles in retri get the same numbers of the first
            //          num_link - 2 triangles in stars[v].triangles !!!
            if (retriang[i][0] == v0 || retriang[i][1] == v0 || retriang[i][2] == v0)
            {
                if (tmp_num_tri[l] < MAXTRI - 1)
                {
                    tmp_tri[l][tmp_num_tri[l]] = stars[v].tri[i];
                    tmp_num_tri[l]++;
                }
                else
                    too_large = 1;
            }
        }
        l++;
    }
    if (too_large)
        return (0);

    for (l = 0; l < num_to; l++)
    {
        v0 = to_update[l];
        stars[v0].num_tri = tmp_num_tri[l];
        delete[] stars[v0].tri;
        stars[v0].tri = new int[stars[v0].num_tri];
        for (i = 0; i < stars[v0].num_tri; i++)
            stars[v0].tri[i] = tmp_tri[l][i];
    }

    // label the vertex v as removed
    is_removed[v] = 1;
    // update tri_list and vertex_list
    for (i = 0; i < stars[v].num_tri; i++)
    { // remove triangles in stars[v].triangles, replace the first in_retriang
        // triangles by the triangles in retri
        if (i < num_link - 2)
        {
            vertex_list[tri_list[stars[v].tri[i]]] = retriang[i][0];
            vertex_list[tri_list[stars[v].tri[i]] + 1] = retriang[i][1];
            vertex_list[tri_list[stars[v].tri[i]] + 2] = retriang[i][2];
            //printf("Update: Triangle   %d   has new points   %d    %d    %d\n", stars[v].tri[i], retriang[i][0], retriang[i][1], retriang[i][2]);
        }
        else
        { //printf("Update: Removed triangle   %d   and marked vertices   %d   %d   %d\n", stars[v].tri[i],vertex_list[tri_list[stars[v].tri[i]]], vertex_list[tri_list[stars[v].tri[i]]+1],vertex_list[tri_list[stars[v].tri[i]]+2]);
            vertex_list[tri_list[stars[v].tri[i]]] = -1;
            vertex_list[tri_list[stars[v].tri[i]] + 1] = -1;
            vertex_list[tri_list[stars[v].tri[i]] + 2] = -1;
            tri_list[stars[v].tri[i]] = -1;
        }
    }

    // update normals and curvature of all modified vertices
    // update heap
    for (i = 0; i < num_to; i++)
    {
        v0 = to_update[i];
        compute_vertex_normal(v0, normal);
        norm_x[v0] = normal[0];
        norm_y[v0] = normal[1];
        norm_z[v0] = normal[2];

        make_link(v0, local_link, num_locallink);
        extract_points(v0, num_locallink, local_link, count, points);

        v1.set_key(v0);
        v1.set_weight(compute_weight(v0, points, count));

        k = heap->get_index(v0 + 1);
        if (k <= heap->getSize())
            heap->change(k, v1);
        else
            heap->insert_again(k, v0 + 1, v1);
    }

    stars[v].num_tri = 0;
    return (1);
}

void ScalarDataVertexRemoval::Reduce(int &red_tri, int &red_pnt)
{
    Vertex current;
    int v;
    int less;
    pair link[MAXTRI];
    int num_link;
    int do_it = 1;
    //clock_t starttime, endtime;
    //double time;
    triple retri[MAXTRI];
    char buf[100];

    red_tri = num_triangles;
    red_pnt = num_points;
    percent = percent / 100.0;
    //starttime = clock();

    preprocess(red_tri);

    while ((heap->getSize() > TERMINATE) && (red_tri >= percent * num_triangles))
    {
        do_it = 1;
        current = heap->get_next();
        v = (int)current.get_key();
        make_link(v, link, num_link);
        //print_star_information(v);
        if (sort_link(link, num_link) && stars[v].manifold && !data_ridge[v])
        {
            if (stars[v].boundary)
                do_it = close_link(v, link, num_link);

            if (do_it)
            {
                if (angle[v])
                    less = retriangulate_edge(v, link, num_link, retri);
                else
                    less = make_retriangulation(v, link, num_link, retri);
                if (less > 0 && volume_bound >= compute_star_volume(v, num_link, retri))
                    if (check_retri(v, num_link, retri, less))
                        if (update(v, link, num_link, retri))
                        {
                            red_tri -= less;
                            red_pnt -= 1;
                        }
            }
        }
    }

    percent = 100.0 * red_tri / (float)num_triangles;
    //endtime = clock();
    //time = (endtime - starttime)/ (double)CLOCKS_PER_SEC;

    if (heap->getSize() == TERMINATE)
        sprintf(buf, "Reduction capacity exceeded: Removed %d triangles of %d, i.e. %.2f %% are left", num_triangles - red_tri, num_triangles, percent);
    else
        sprintf(buf, "Removed %d triangles of %d, i.e. %.2f %% are left\n", num_triangles - red_tri, num_triangles, percent);
    Covise::sendInfo(buf);
    //sprintf(buf,"Time: %.2f seconds\n",time);
    //Covise::sendInfo(buf);

    delete heap;
}

/////////////////////////////////////////////////////////////////////////////
// Functions for Vector Data Supplied Surfaces                             //
/////////////////////////////////////////////////////////////////////////////

float VectorDataVertexRemoval::L1_gradient(int v, int *points, int num)
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
    return (max);
}

float VectorDataVertexRemoval::Data_deviation(int v, int *points, int num)
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
    return (max);
}

float VectorDataVertexRemoval::Zalgaller_gradient(int v, int *points, int num)
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

    return (fabs(grad[0]) + fabs(grad[1]) + fabs(grad[2]));
}

float VectorDataVertexRemoval::compute_gradient(int v, int *points, int num, float norm)
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
    return (x);
}

float VectorDataVertexRemoval::compute_weight(int i, int *points, int num_pnt)
{
    float curv, grad, compact;
    curv = compute_curvature(i, points, num_pnt, max_curvature);
    grad = compute_gradient(i, points, num_pnt, max_gradient);
    compact = compute_compactness(i);
    return (curv + enforce * grad + compact);
}

float VectorDataVertexRemoval::compute_weight(float curv, float grad, float compact)
{
    return (curv + enforce * grad + compact);
}

void VectorDataVertexRemoval::preprocess(int &red_tri)
{
    int i;
    pair link[MAXTRI];
    int points[MAXTRI];
    int num_link, num_pnt;
    Vertex v;

    float *curvature;
    float *gradient;

    heap = new PQ<Vertex>(num_points);
    curvature = new float[num_points];
    gradient = new float[num_points];

    remove_flat_triangles(red_tri);

    initialize_connectivity();
    if (norm_x == NULL)
        generate_normals();
    max_gradient = max_curvature = 0.0;

    for (i = 0; i < num_points; i++)
    {
        make_link(i, link, num_link);
        stars[i].boundary = check_boundary(i, link, num_link);
        stars[i].manifold = check_manifold(i, link, num_link);

        extract_points(i, num_link, link, num_pnt, points);
        curvature[i] = compute_curvature(i, points, num_pnt, 1);
        if (curvature[i] > max_curvature)
            max_curvature = curvature[i];
        gradient[i] = compute_gradient(i, points, num_pnt, 1);
        if (gradient[i] > max_gradient)
            max_gradient = gradient[i];
        angle[i] = vertex_on_feature_edge(i, max_angle);
    }
    if (max_curvature > 1E-06)
        for (i = 0; i < num_points; i++)
            curvature[i] /= max_curvature;
    if (max_gradient > 1E-06)
        for (i = 0; i < num_points; i++)
            gradient[i] /= max_gradient;

    for (i = 0; i < num_points; i++)
    {
        v.set_key(i);
        v.set_weight(compute_weight(curvature[i], gradient[i], compute_compactness(i)));
        heap->append(v);
    }
    heap->construct();

    delete[] curvature;
    delete[] gradient;
}

int VectorDataVertexRemoval::update(int v, pair *link, int num_link, triple *retriang)
{
    int i, j, k, l;
    int v0;
    int to_update[MAXTRI];
    int num_to;
    int tmp_tri[MAXTRI][MAXTRI];
    int tmp_num_tri[MAXTRI];
    int too_large = 0;
    pair local_link[MAXTRI];
    int num_locallink;
    int points[MAXTRI];
    int count;
    fl_triple normal;
    Vertex v1;

    extract_points(num_link, link, to_update);
    num_to = num_link;
    // Update stars around the link of v.
    // Fill first temporary stars, look whether the new triangulation will
    // create stars with more than MAXTRI triangles.
    // If this is the case, leave update, do not remove the vertex v.
    // If everything is okay, copy temporary structures to stars[...].
    l = 0;
    while (l < num_to && !too_large)
    {
        v0 = to_update[l];
        tmp_num_tri[l] = stars[v0].num_tri;
        for (i = 0; i < tmp_num_tri[l]; i++)
            tmp_tri[l][i] = stars[v0].tri[i];

        for (i = 0; i < tmp_num_tri[l]; i++)
        { // remove triangles of corona.triangles from each stars[v0]
            // in the border
            j = 0;
            while ((j < stars[v].num_tri) && (tmp_tri[l][i] != stars[v].tri[j]))
                j++;
            if (j != stars[v].num_tri)
            {
                for (k = i; k < tmp_num_tri[l] - 1; k++)
                    tmp_tri[l][k] = tmp_tri[l][k + 1];
                i--;
                tmp_num_tri[l]--;
            }
        }

        for (i = 0; i < num_link - 2; i++)
        { // append triangles of retri containing v0 to stars[v0].tri
            // CAUTION: triangles in retri get the same numbers of the first
            //          num_link - 2 triangles in stars[v].triangles !!!
            if (retriang[i][0] == v0 || retriang[i][1] == v0 || retriang[i][2] == v0)
            {
                if (tmp_num_tri[l] < MAXTRI - 1)
                {
                    tmp_tri[l][tmp_num_tri[l]] = stars[v].tri[i];
                    tmp_num_tri[l]++;
                }
                else
                    too_large = 1;
            }
        }
        l++;
    }
    if (too_large)
        return (0);

    for (l = 0; l < num_to; l++)
    {
        v0 = to_update[l];
        stars[v0].num_tri = tmp_num_tri[l];
        delete[] stars[v0].tri;
        stars[v0].tri = new int[stars[v0].num_tri];
        for (i = 0; i < stars[v0].num_tri; i++)
            stars[v0].tri[i] = tmp_tri[l][i];
    }

    // label the vertex v as removed
    is_removed[v] = 1;
    // update tri_list and vertex_list
    for (i = 0; i < stars[v].num_tri; i++)
    { // remove triangles in stars[v].triangles, replace the first in_retriang
        // triangles by the triangles in retri
        if (i < num_link - 2)
        {
            vertex_list[tri_list[stars[v].tri[i]]] = retriang[i][0];
            vertex_list[tri_list[stars[v].tri[i]] + 1] = retriang[i][1];
            vertex_list[tri_list[stars[v].tri[i]] + 2] = retriang[i][2];
            //printf("Update: Triangle   %d   has new points   %d    %d    %d\n", stars[v].tri[i], retriang[i][0], retriang[i][1], retriang[i][2]);
        }
        else
        { //printf("Update: Removed triangle   %d   and marked vertices   %d   %d   %d\n", stars[v].tri[i],vertex_list[tri_list[stars[v].tri[i]]], vertex_list[tri_list[stars[v].tri[i]]+1],vertex_list[tri_list[stars[v].tri[i]]+2]);
            vertex_list[tri_list[stars[v].tri[i]]] = -1;
            vertex_list[tri_list[stars[v].tri[i]] + 1] = -1;
            vertex_list[tri_list[stars[v].tri[i]] + 2] = -1;
            tri_list[stars[v].tri[i]] = -1;
        }
    }

    // update normals and curvature of all modified vertices
    // update heap
    for (i = 0; i < num_to; i++)
    {
        v0 = to_update[i];
        compute_vertex_normal(v0, normal);
        norm_x[v0] = normal[0];
        norm_y[v0] = normal[1];
        norm_z[v0] = normal[2];

        make_link(v0, local_link, num_locallink);
        extract_points(v0, num_locallink, local_link, count, points);

        v1.set_key(v0);
        v1.set_weight(compute_weight(v0, points, count));

        k = heap->get_index(v0 + 1);
        if (k <= heap->getSize())
            heap->change(k, v1);
        else
            heap->insert_again(k, v0 + 1, v1);
    }

    stars[v].num_tri = 0;
    return (1);
}

void VectorDataVertexRemoval::Reduce(int &red_tri, int &red_pnt)
{
    Vertex current;
    int v;
    int less;
    pair link[MAXTRI];
    int num_link;
    int do_it = 1;
    //clock_t starttime, endtime;
    //double time;
    triple retri[MAXTRI];
    char buf[100];

    red_tri = num_triangles;
    red_pnt = num_points;
    percent = percent / 100.0;
    //starttime = clock();

    preprocess(red_tri);

    while ((heap->getSize() > TERMINATE) && (red_tri >= percent * num_triangles))
    {
        do_it = 1;
        current = heap->get_next();
        v = (int)current.get_key();
        make_link(v, link, num_link);
        //print_star_information(v);
        if (sort_link(link, num_link) && stars[v].manifold)
        {
            if (stars[v].boundary)
                do_it = close_link(v, link, num_link);
            if (do_it)
            {
                if (angle[v])
                    less = retriangulate_edge(v, link, num_link, retri);
                else
                    less = make_retriangulation(v, link, num_link, retri);
                if (less > 0 && volume_bound >= compute_star_volume(v, num_link, retri))
                    if (check_retri(v, num_link, retri, less) && update(v, link, num_link, retri))
                    {
                        red_tri -= less;
                        red_pnt -= 1;
                    }
            }
        }
    }

    percent = 100.0 * red_tri / (float)num_triangles;
    //endtime = clock();
    //time = (endtime - starttime)/ (double)CLOCKS_PER_SEC;

    if (heap->getSize() == TERMINATE)
        sprintf(buf, "Reduction capacity exceeded: Removed %d triangles of %d, i.e. %.2f %% are left", num_triangles - red_tri, num_triangles, percent);
    else
        sprintf(buf, "Removed %d triangles of %d, i.e. %.2f %% are left\n", num_triangles - red_tri, num_triangles, percent);
    Covise::sendInfo(buf);
    //sprintf(buf,"Time: %.2f seconds\n",time);
    //Covise::sendInfo(buf);

    delete heap;
}
