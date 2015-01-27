/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface vertex removal class                      **
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
 ** Date:  June 1998   V1.0                                                **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "SurfVertex.h"
#include <util/coviseCompat.h>
#include "My_Struct.h"
#include "Tri_Polygon.h"

//////////////////////////////////////////////////////////////////////////
// Geometric Reduction Procedures for Vertex Removal Strategy           //
//////////////////////////////////////////////////////////////////////////

float SurfaceVertexRemoval::compute_curvature(int v, int *points, int num)
{
    float x;
    float curv_1, curv_2;
    fl_triple normal_old, normal;
    int pnt_1[MAXTRI], pnt_2[MAXTRI];
    int num_1, num_2;

    if (angle[v] == 1)
    {
        normal_old[0] = norm_x[v];
        normal_old[1] = norm_y[v];
        normal_old[2] = norm_z[v];
        split_link(v, points, num, pnt_1, num_1, pnt_2, num_2);

        average_vertex_normal(v, pnt_1, num_1, normal);
        norm_x[v] = normal[0];
        norm_y[v] = normal[1];
        norm_z[v] = normal[2];
        switch (which_curv)
        {
        case 1:
            curv_1 = L1_curvature(v, pnt_1, num_1);
            break;
        case 2:
            curv_1 = discrete_curvature(v);
            break;
        case 3:
            curv_1 = Taubin_curvature(v, pnt_1, num_1);
            break;
        case 4:
            curv_1 = Hamann_curvature(v, pnt_1, num_1);
            break;
        default:
            curv_1 = L1_curvature(v, pnt_1, num_1);
        };
        average_vertex_normal(v, pnt_2, num_2, normal);
        norm_x[v] = normal[0];
        norm_y[v] = normal[1];
        norm_z[v] = normal[2];
        switch (which_curv)
        {
        case 1:
            curv_2 = L1_curvature(v, pnt_2, num_2);
            break;
        case 2:
            curv_2 = discrete_curvature(v);
            break;
        case 3:
            curv_2 = Taubin_curvature(v, pnt_2, num_2);
            break;
        case 4:
            curv_2 = Hamann_curvature(v, pnt_2, num_2);
            break;
        default:
            curv_2 = L1_curvature(v, pnt_2, num_2);
        };

        if (curv_2 > curv_1)
            x = curv_2;
        else
            x = curv_1;
        norm_x[v] = normal_old[0];
        norm_y[v] = normal_old[1];
        norm_z[v] = normal_old[2];
    }
    else
        switch (which_curv)
        {
        case 1:
            x = L1_curvature(v, points, num);
            break;
        case 2:
            x = discrete_curvature(v);
            break;
        case 3:
            x = Taubin_curvature(v, points, num);
            break;
        case 4:
            x = Hamann_curvature(v, points, num);
            break;
        default:
            x = L1_curvature(v, points, num);
        };

    return (x);
}

float SurfaceVertexRemoval::compute_curvature(int v, int *points, int num, float norm)
{
    float x;
    x = compute_curvature(v, points, num);
    if (norm > 1E-06)
        x /= norm;
    return (x);
}

float SurfaceVertexRemoval::compute_weight(int v, int *points, int count)
{
    float weight;

    weight = compute_curvature(v, points, count) + compute_compactness(v);
    return (weight);
}

void SurfaceVertexRemoval::average_vertex_normal(int v, int *pnt, int count, fl_triple normal)
{ // computes the normal in the vertex v by averaging over triangles
    // normals of triangles build by the points in the array points
    int i, j;
    fl_triple tri_normal, avg_normal;

    for (j = 0; j < 3; j++)
        avg_normal[j] = 0.0;
    for (i = 0; i < count - 1; i++)
    {
        compute_triangle_normal(pnt[i], pnt[(i + 1) % count], coords_x[v], coords_y[v], coords_z[v], tri_normal);
        for (j = 0; j < 3; j++)
            avg_normal[j] += tri_normal[j];
    }
    for (j = 0; j < 3; j++)
        normal[j] = avg_normal[j] / count;
}

void SurfaceVertexRemoval::preprocess(int &red_tri)
{
    int i;
    pair link[MAXTRI];
    int points[MAXTRI];
    int num_link, num_pnt;
    Vertex v;

    heap = new PQ<Vertex>(num_points);
    remove_flat_triangles(red_tri);

    initialize_connectivity();
    if (norm_x == NULL)
        generate_normals();

    for (i = 0; i < num_points; i++)
    {
        make_link(i, link, num_link);
        stars[i].boundary = check_boundary(i, link, num_link);
        stars[i].manifold = check_manifold(i, link, num_link);

        extract_points(i, num_link, link, num_pnt, points);
        v.set_key(i);
        v.set_weight(compute_weight(i, points, num_pnt));
        heap->append(v);

        angle[i] = vertex_on_feature_edge(i, max_angle);
    }
    heap->construct();
}

int SurfaceVertexRemoval::check_polygon(int num_border, float (*u)[2], int &num_exclude, int *exclude, float (*v)[2], int *uTOv)
{ // removes collinear edges in the boundary polygon
    // corresponding triangles will be inserted after retriangulation procedure
    // by function adjust_triangulation

    float x, norm;
    int counter, i;

    counter = 0;
    num_exclude = 0;

    for (i = 0; i < num_border; i++)
    {
        x = ((u[i][0] - u[(i - 1 + num_border) % num_border][0]) * (u[(i + 1) % num_border][0] - u[i][0]));
        x += ((u[i][1] - u[(i - 1 + num_border) % num_border][1]) * (u[(i + 1) % num_border][1] - u[i][1]));
        norm = (u[(i + 1) % num_border][0] - u[i][0]) * (u[(i + 1) % num_border][0] - u[i][0]);
        norm += (u[(i + 1) % num_border][1] - u[i][1]) * (u[(i + 1) % num_border][1] - u[i][1]);
        norm = sqrt(norm);
        x /= norm;
        norm = (u[i][0] - u[(i - 1 + num_border) % num_border][0]) * (u[i][0] - u[(i - 1 + num_border) % num_border][0]);
        norm += (u[i][1] - u[(i - 1 + num_border) % num_border][1]) * (u[i][1] - u[(i - 1 + num_border) % num_border][1]);
        norm = sqrt(norm);
        x /= norm;

        if (fabs(x - 1.0) < 1E-05)
            exclude[num_exclude++] = i;
        else
        {
            v[counter][0] = u[i][0];
            v[counter][1] = u[i][1];
            uTOv[counter++] = i;
            if (fabs(x + 1.0) < 1E-04)
                return (0);
        }
    }
    return (1);
}

void SurfaceVertexRemoval::adjust_triangulation(int num_border, int num_exclude, int *exclude, int *uTOv, int (*tri)[3])
{ // inserts triangles corresponding to collinear edges
    int counter, i;
    int v1, v2, v3;
    int j, to_split;
    int adjusted;
    int first, last;

    for (i = 0; i < num_border - num_exclude - 2; i++)
    {
        tri[i][0] = uTOv[tri[i][0]];
        tri[i][1] = uTOv[tri[i][1]];
        tri[i][2] = uTOv[tri[i][2]];
    }

    counter = num_border - num_exclude - 2;

    // in case the 0-th vertex is excluded together with the last vertex
    first = 0;
    adjusted = 0;
    if (exclude[0] == 0 && exclude[num_exclude - 1] == num_border - 1)
    {
        last = 0;
        while ((last + 1 < num_exclude) && (exclude[last + 1] == exclude[last] + 1))
            last++;
        first = last + 1;
    }

    // start inserting triangles
    while (adjusted < num_exclude)
    {
        v1 = (exclude[first] - 1 + num_border) % num_border;

        // in case that multiple collinear edges occur directly after each other
        last = first;
        while (exclude[(last + 1) % num_exclude] == (exclude[last % num_exclude] + 1) % num_border)
            last++;

        v2 = (exclude[last % num_exclude] + 1) % num_border;
        to_split = 0;
        while (to_split < counter && (!((tri[to_split][0] == v1 && tri[to_split][1] == v2) || (tri[to_split][1] == v1 && tri[to_split][2] == v2) || (tri[to_split][2] == v1 && tri[to_split][0] == v2))))
            to_split++;

        if (to_split == counter)
        {
            v3 = 0; // just to prevent crashes (uwe)
            printf(" Triangle with edge (%d %d) not found!!!\n", v1, v2);
        }
        else
        {
            if (tri[to_split][0] == v1 && tri[to_split][1] == v2)
                v3 = tri[to_split][2];
            else if (tri[to_split][1] == v1 && tri[to_split][2] == v2)
                v3 = tri[to_split][0];
            else
                v3 = tri[to_split][1];
        }

        tri[to_split][0] = v1;
        tri[to_split][1] = exclude[first];
        tri[to_split][2] = v3;

        for (j = 0; j <= last - first; j++)
        {
            if (j < last - first)
            {
                tri[counter][0] = exclude[(first + j) % num_exclude];
                tri[counter][1] = exclude[(first + j + 1) % num_exclude];
                tri[counter++][2] = v3;
            }
            else
            {
                tri[counter][0] = exclude[last % num_exclude];
                tri[counter][1] = v2;
                tri[counter++][2] = v3;
            }
        }
        adjusted += last + 1 - first;
        first = (last + 1) % num_exclude;
    }
}

int SurfaceVertexRemoval::check_selfintersections(int num, float (*co)[2])
{
    int i, j;
    int intersection = 0;
    float norm, x1, x2;
    float lines[MAXTRI][3];

    for (i = 0; i < num; i++)
    {
        lines[i][0] = co[(i + 1) % num][1] - co[i][1];
        lines[i][1] = co[i][0] - co[(i + 1) % num][0];
        lines[i][2] = -co[i][0] * co[(i + 1) % num][1] + co[i][1] * co[(i + 1) % num][0];
        norm = sqrt(lines[i][0] * lines[i][0] + lines[i][1] * lines[i][1]);
        if (norm > 1E-03)
        {
            lines[i][0] /= norm;
            lines[i][1] /= norm;
            lines[i][2] /= norm;
        }
    }
    i = 0;
    while (!intersection && i < num)
    {
        j = i + 2;
        while (!intersection && j < num && (j + 1) < i + num)
        { //compute_intersection point

            x1 = lines[i][0] * co[j][0] + lines[i][1] * co[j][1] + lines[i][2];
            x1 *= (lines[i][0] * co[(j + 1) % num][0] + lines[i][1] * co[(j + 1) % num][1] + lines[i][2]);
            x2 = lines[j][0] * co[i][0] + lines[j][1] * co[i][1] + lines[j][2];
            x2 *= (lines[j][0] * co[(i + 1) % num][0] + lines[j][1] * co[(i + 1) % num][1] + lines[j][2]);

            if ((x1 <= 0) && (x2 <= 0))
                intersection = 1;
            j++;
        }
        i++;
    }
    return (intersection);
}

void SurfaceVertexRemoval::split_link(int v, int *points, int num_p, int *pnt_1, int &num_1, int *pnt_2, int &num_2)
{
    (void)v;
    // if angle[v] == 1, then splits point into 2 sublinks (along a feature edge):
    // v1 .... v2
    // v2 .... v1

    int start, end, i;
    int l_end, l_start;
    int v1, v2;

    v1 = -1;
    v2 = -1;
    for (i = 0; i < num_p; i++)
        if (angle[points[i]])
        {
            if (v1 == -1)
                v1 = points[i];
            else
                v2 = points[i];
        }

    start = 0;
    while (start < num_p && points[start] != v1)
        start++;
    end = 0;
    while (end < num_p && points[end] != v2)
        end++;

    if (start > end)
    {
        l_end = end + num_p;
        l_start = start;
    }
    else
    {
        l_end = end;
        l_start = start + num_p;
    }

    for (i = start; i <= l_end; i++)
        pnt_1[i - start] = points[i % num_p];
    num_1 = l_end - start + 1;

    for (i = end; i <= l_start; i++)
        pnt_2[i - end] = points[i % num_p];
    num_2 = l_start - end + 1;
}

void SurfaceVertexRemoval::split_link(int v, int v1, int v2, pair *link, int num_link, pair *link_1, int &num_1, pair *link_2, int &num_2)
{
    (void)v;
    // splits link into 2 sublinks:
    // v1 .... v2 v1
    // v2 .... v1 v2

    int start, end, i;
    int l_end, l_start;
    start = 0;
    while (start < num_link && link[start][0] != v1)
        start++;
    end = 0;
    while (end < num_link && link[end][0] != v2)
        end++;

    if (start > end)
    {
        l_end = end + num_link;
        l_start = start;
    }
    else
    {
        l_end = end;
        l_start = start + num_link;
    }

    for (i = start; i < l_end; i++)
    {
        link_1[i - start][0] = link[i % num_link][0];
        link_1[i - start][1] = link[i % num_link][1];
    }
    link_1[l_end - start][0] = v2;
    link_1[l_end - start][1] = v1;
    num_1 = l_end - start + 1;

    for (i = end; i < l_start; i++)
    {
        link_2[i - end][0] = link[i % num_link][0];
        link_2[i - end][1] = link[i % num_link][1];
    }
    link_2[l_start - end][0] = v1;
    link_2[l_start - end][1] = v2;
    num_2 = l_start - end + 1;
}

int SurfaceVertexRemoval::retriangulate_edge(int v, pair *link, int num_link, triple *retri)
{ // If v lies on a feature edge, but is not a complex (corner) vertex, do:
    // Check whether there are two edges outgoing from v which build together
    // an (almost) straight line.
    // If there are such, then split link into two sublinks on the feature edge,
    // and retriangulate both using make_retriangulation.
    // Else do not remove the vertex.

    int i, okay;
    int v1, v2;
    float dir_1[3];
    float dir_2[3];
    pair link_1[MAXTRI], link_2[MAXTRI];
    int num_1, num_2;
    float norm, scalar;
    int n1 = 0, n2 = 0;
    triple tmp_tri[MAXTRI];

    if ((angle[v] > 1) || (angle[v] && stars[v].boundary))
        return 0;

    v1 = -1;
    v2 = -1;
    for (i = 0; i < num_link; i++)
        if (angle[link[i][0]])
        {
            if (v1 == -1)
            {
                v1 = link[i][0];
                n1 = link[i][1];
                dir_1[0] = coords_x[v1] - coords_x[v];
                dir_1[1] = coords_y[v1] - coords_y[v];
                dir_1[2] = coords_z[v1] - coords_z[v];
                norm = sqrt(dir_1[0] * dir_1[0] + dir_1[1] * dir_1[1] + dir_1[2] * dir_1[2]);
                dir_1[0] /= norm;
                dir_1[1] /= norm;
                dir_1[2] /= norm;
            }
            else
            {
                v2 = link[i][0];
                n2 = link[i][1];
                dir_2[0] = coords_x[v] - coords_x[v2];
                dir_2[1] = coords_y[v] - coords_y[v2];
                dir_2[2] = coords_z[v] - coords_z[v2];
                norm = sqrt(dir_2[0] * dir_2[0] + dir_2[1] * dir_2[1] + dir_2[2] * dir_2[2]);
                dir_2[0] /= norm;
                dir_2[1] /= norm;
                dir_2[2] /= norm;
            }
        }
    scalar = dir_1[0] * dir_2[0] + dir_1[1] * dir_2[1] + dir_1[2] * dir_2[2];
    if ((fabs(scalar - 1.0) < 1E-03) || (fabs(scalar + 1.0) < 1E-03))
    {
        split_link(v, v1, v2, link, num_link, link_1, num_1, link_2, num_2);
        okay = make_retriangulation(n1, link_1, num_1, retri);
        if (okay)
            okay = make_retriangulation(n2, link_2, num_2, tmp_tri);
        if (okay)
        {
            for (i = 0; i < num_2 - 2; i++)
            {
                retri[i + num_1 - 2][0] = tmp_tri[i][0];
                retri[i + num_1 - 2][1] = tmp_tri[i][1];
                retri[i + num_1 - 2][2] = tmp_tri[i][2];
            }
            return 2;
        }
        else
            return 0;
    }
    else
        return 0;
}

int SurfaceVertexRemoval::make_retriangulation(int v, pair *link, int num_link, triple *retri)
{
    int i;
    int count;
    double norm;
    double e1[3], e2[3];
    float u[MAXTRI][2];
    int tri[MAXTRI][3];
    int points[MAXTRI];
    int num_exclude;
    int exclude[MAXTRI + 1];
    float u_checked[MAXTRI][2];
    int uTOv[MAXTRI];
    TriPolygon *poly;

    // rebuild triangles, according to strategy
    // store new triangles in retri;

    // Retriangulate Boundary Polygon, using deBerg Algorithm:
    // 1. Project Boundary Polygon onto plane determined by t
    // 2. Retriangulate this two-dimensional polygon
    // Caution: Triangulation Algorithm needs countour of the polygon
    //          in counter-clockwise orientation!!!
    //          Vertices in contour have to be stored beginning
    //          by index 0!!!

    for (i = 0; i < num_link; i++)
        points[i] = link[i][0];

    //printf("Points in the link:\n");
    //for (i = 0; i < num_link; i++)
    //    printf("%d  ", points[i]);
    //printf("\n\n");

    if (norm_x[v] > 1E-06 || norm_x[v] < -1E-06)
    {
        e1[0] = -(norm_y[v] + norm_z[v]) / norm_x[v];
        e1[1] = 1.0;
        e1[2] = 1.0;
    }
    else if (norm_y[v] > 1E-06 || norm_y[v] < -1E-06)
    {
        e1[0] = 1.0;
        e1[1] = -(norm_x[v] + norm_z[v]) / norm_y[v];
        e1[2] = 1.0;
    }
    else
    {
        e1[0] = 1.0;
        e1[1] = 1.0;
        e1[2] = -(norm_x[v] + norm_y[v]) / norm_z[v];
    }

    norm = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    e1[0] /= norm;
    e1[1] /= norm;
    e1[2] /= norm;

    e2[0] = norm_y[v] * e1[2] - norm_z[v] * e1[1];
    e2[1] = norm_z[v] * e1[0] - norm_x[v] * e1[2];
    e2[2] = norm_x[v] * e1[1] - norm_y[v] * e1[0];

    for (i = 0; i < num_link; i++)
    {
        u[i][0] = (coords_x[points[i]] - coords_x[v]) * e1[0];
        u[i][0] += (coords_y[points[i]] - coords_y[v]) * e1[1];
        u[i][0] += (coords_z[points[i]] - coords_z[v]) * e1[2];
        u[i][1] = (coords_x[points[i]] - coords_x[v]) * e2[0];
        u[i][1] += (coords_y[points[i]] - coords_y[v]) * e2[1];
        u[i][1] += (coords_z[points[i]] - coords_z[v]) * e2[2];
    }

    if (check_polygon(num_link, u, num_exclude, exclude, u_checked, uTOv))
    {
        if (!check_selfintersections(num_link - num_exclude, u_checked))
        {
            poly = new TriPolygon(num_link - num_exclude, u_checked);
            count = poly->TriangulatePolygon(tri, 1);
            delete poly;
        }
        else
            count = 0;
    }
    else
        count = 0;

    if (count == 0)
        return (0);

    if (num_exclude)
    {
        adjust_triangulation(num_link, num_exclude, exclude, uTOv, tri);
        count += num_exclude;
    }

    if (count != num_link - 2)
    {
        cout << "Too few triangles in Retriangulation: Vertex " << v << " not removed" << endl;
        return (0);
    }

    for (i = 0; i < num_link - 2; i++)
    {
        retri[i][0] = points[tri[i][0]];
        retri[i][1] = points[tri[i][1]];
        retri[i][2] = points[tri[i][2]];
    }

    if (stars[v].boundary)
        return (1);
    else
        return (2);
}

int SurfaceVertexRemoval::check_retri(int v, int num_link, triple *retri, int less)
{
    int okay = 1;
    if (less != stars[v].num_tri - num_link + 2)
        return (0);
    int i = 0;
    while ((i < num_link - 2) && okay)
    {
        if ((retri[i][0] == retri[i][1]) || (retri[i][1] == retri[i][2]) || (retri[i][2] == retri[i][0]))
        {
            okay = 0;
            printf("Detected triangle with points %d %d %d\n", retri[i][0], retri[i][1], retri[i][2]);
        }
        i++;
    }
    return (okay);
}

int SurfaceVertexRemoval::update(int v, pair *link, int num_link, triple *retriang)
{
    int i, j, k, l;
    int v0;
    int to_update[MAXTRI];
    int num_to;
    int too_large = 0;
    int tmp_tri[MAXTRI][MAXTRI];
    int tmp_num_tri[MAXTRI];
    pair local_link[MAXTRI];
    int num_locallink;
    int points[MAXTRI];
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
        extract_points(num_locallink, local_link, points);

        v1.set_key(v0);
        v1.set_weight(compute_weight(v0, points, num_locallink));

        k = heap->get_index(v0 + 1);
        if (k <= heap->getSize())
            heap->change(k, v1);
        else
            heap->insert_again(k, v0 + 1, v1);
    }

    stars[v].num_tri = 0;
    return (1);
}

void SurfaceVertexRemoval::Reduce(int &red_tri, int &red_pnt)
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

coDistributedObject **SurfaceVertexRemoval::createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, char const *Normals_name)
{
    (void)Data_name;

    coDistributedObject **DO_Return = new coDistributedObject *[3];
    coDoPolygons *polygons_out;
    coDoVec3 *normals_out;
    int *vl, *pl, count, j;
    int *index;
    float *co_x, *co_y, *co_z;
    float *no_x, *no_y, *no_z;

    if (red_points == 0)
        return (NULL);

    co_x = new float[red_points];
    co_y = new float[red_points];
    co_z = new float[red_points];
    no_x = new float[red_points];
    no_y = new float[red_points];
    no_z = new float[red_points];

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

    normals_out = new coDoVec3(Normals_name, red_points, no_x, no_y, no_z);

    if (!normals_out->objectOk())
    {
        Covise::sendError("ERROR: creation of geometry object 'normalsOut' failed");
        return (NULL);
    }

    DO_Return[0] = polygons_out;
    DO_Return[1] = NULL;
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

    return (DO_Return);
}
