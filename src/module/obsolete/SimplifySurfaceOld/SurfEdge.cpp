/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface edge collapse class                       **
 **                      for Surface Reduction Methods                     **
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
#include "SurfEdge.h"
#include <util/coviseCompat.h>

/////////////////////////////////////////////////////////////////////////////
// Make COVISE output objects in Shared memory                             //
/////////////////////////////////////////////////////////////////////////////

coDistributedObject **SurfaceEdgeCollapse::createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name)
{
    (void)Data_name;
    (void)Normals_name;

    coDistributedObject **DO_Return = new coDistributedObject *[3];
    coDoPolygons *polygons_out;
    int *vl, *pl, count, j;
    int *index;
    float *co_x, *co_y, *co_z;

    if (red_points == 0)
        return (NULL);

    co_x = new float[red_points];
    co_y = new float[red_points];
    co_z = new float[red_points];

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

            count++;
        }
    }

    if (count != red_tri)
    {
        Covise::sendError("ERROR: non-consistent number of triangles!");
        return (NULL);
    }

    polygons_out = new coDoPolygons(Triangle_name, red_points, co_x, co_y, co_z, 3 * red_tri, vl, red_tri, pl);

    if (!polygons_out->objectOk())
    {
        Covise::sendError("ERROR: creation of geometry object 'meshOut' failed");
        return (NULL);
    }

    DO_Return[0] = polygons_out;
    DO_Return[1] = NULL;
    DO_Return[2] = NULL;

    delete[] vl;
    delete[] pl;
    delete[] index;
    delete[] co_x;
    delete[] co_y;
    delete[] co_z;

    return (DO_Return);
}

//////////////////////////////////////////////////////////////////////////
// Geometric Reduction Procedures for Edge Collapse Strategy            //
//////////////////////////////////////////////////////////////////////////

void SurfaceEdgeCollapse::make_priority_queue()
{ // Initializes the priority queue of edges we need to define the
    // order of edges to be collapsed. weight = simple edge length
    int i, j;
    int v1, v2;
    Edge e;
    float length;

    //for(i = 0; i < num_points; i++)
    //  compact[i] = compute_compactness(i);

    for (i = 0; i < num_triangles; i++)
    {
        for (j = 0; j < 3; j++)
        {

            // Surface::remove_flat_triangles(..) may have removed already triangles
            // we work only on those which are still present. R.M. 27.03.2001
            int triIndex = tri_list[i];
            if (triIndex > 0)
            {
                v1 = vertex_list[triIndex + j];
                v2 = vertex_list[triIndex + (j + 1) % 3];

                e.set_endpoints(v1, v2);

                length = 0.0;
                length += (coords_x[v1] - coords_x[v2]) * (coords_x[v1] - coords_x[v2]);
                length += (coords_y[v1] - coords_y[v2]) * (coords_y[v1] - coords_y[v2]);
                length += (coords_z[v1] - coords_z[v2]) * (coords_z[v1] - coords_z[v2]);

                //e.set_length(length + 0.05 *(compact[v1] + compact[v2]));
                e.set_length(length);

                heap->append(e);
                ept[i][j] = 3 * i + j + 1;
            }
        }
    }
    heap->construct();
}

void SurfaceEdgeCollapse::preprocess(int &red_tri)
{
    int i;
    remove_flat_triangles(red_tri);
    initialize_connectivity();

    for (i = 0; i < num_points; i++)
    {
        initialize_topology(i);
        angle[i] = vertex_on_feature_edge(i, max_angle);
    }
    make_priority_queue();
}

void SurfaceEdgeCollapse::merge_stars(int v1, int v2, Star &star)
{ //merges the stars of the vertices v1 and v2, deleting doubled triangles
    int i, j, num;
    int tri[MAXTRI];

    num = 0;
    for (i = 0; i < stars[v2].num_tri; i++)
    {
        j = 0;
        while ((j < stars[v1].num_tri) && (stars[v1].tri[j] != stars[v2].tri[i]))
            j++;
        if (j == stars[v1].num_tri)
        {
            tri[num] = stars[v2].tri[i];
            num++;
        }
    }
    star.num_tri = stars[v1].num_tri + num;
    star.tri = new int[stars[v1].num_tri + num];

    for (i = 0; i < star.num_tri; i++)
        if (i < stars[v1].num_tri)
            star.tri[i] = stars[v1].tri[i];
        else
            star.tri[i] = tri[i - stars[v1].num_tri];

    star.manifold = stars[v1].manifold && stars[v2].manifold;
    star.boundary = stars[v1].boundary || stars[v2].boundary;
    if (!((stars[v2].num_tri - num == 2) || (star.boundary && (stars[v2].num_tri - num == 1))))
        star.manifold = 0;
}

int SurfaceEdgeCollapse::straighten_edge(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0)
{
    (void)star;

    // if both v1 and v2 lie on a feature edge of the surface, we check whether
    // the edge continues in an (almost) straight line in at least one
    // direction and choose the new position accordingly
    int i;
    int count = 0;
    int which_1 = -1;
    float alpha;

    if (angle[v1] == 2 && angle[v2] == 2)
        return (0);

    if (angle[v1] < 2 && angle[v2] < 2)
    {
        for (i = 0; i < num_link; i++)
        {
            alpha = compute_inner_angle(link[i][0], v1, v2);
            if (fabs(alpha) < 1E-04)
                count++;
            else if (fabs(alpha - M_PI) < 1E-04)
            {
                count++;
                which_1 = i;
            }
        }
        if (!count)
            return (0);
        else
        {
            if (count == 1)
            {
                if (which_1 != -1)
                {
                    x_0 = coords_x[v1];
                    y_0 = coords_y[v1];
                    z_0 = coords_z[v1];
                }
                else
                {
                    x_0 = coords_x[v2];
                    y_0 = coords_y[v2];
                    z_0 = coords_z[v2];
                }
            }
            if (count == 2)
            {
                x_0 = (coords_x[v1] + coords_x[v2]) / 2.0;
                y_0 = (coords_y[v1] + coords_y[v2]) / 2.0;
                z_0 = (coords_z[v1] + coords_z[v2]) / 2.0;
            }
            else
                return (0);
        }
    }
    else
    {
        if (angle[v1] == 2)
        {
            x_0 = coords_x[v1];
            y_0 = coords_y[v1];
            z_0 = coords_z[v1];
        }
        else
        {
            x_0 = coords_x[v2];
            y_0 = coords_y[v2];
            z_0 = coords_z[v2];
        }
    }
    if (check_new_triangles(x_0, y_0, z_0, link, num_link))
        return (1);
    else
        return (0);
}

int SurfaceEdgeCollapse::straighten_boundary(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0)
{
    (void)star;

    // if both v1 and v2 lie on the boundary of the surface, we check whether
    // the boundary continues in an (almost) straight line in at least one
    // direction and choose the new position accordingly
    int count = 0;
    int which_2 = 0;
    float alpha_1, alpha_2;

    alpha_1 = compute_inner_angle(link[0][0], v2, v1);
    if (fabs(alpha_1 - M_PI) < 1E-04)
    {
        count++;
        which_2 = 1;
    }

    alpha_2 = compute_inner_angle(link[num_link - 1][1], v1, v2);
    if (fabs(alpha_2 - M_PI) < 1E-04)
        count++;

    if (!count)
        return (0);
    else
    {
        if (count == 1)
        {
            if (!which_2)
            {
                x_0 = coords_x[v2];
                y_0 = coords_y[v2];
                z_0 = coords_z[v2];
            }
            else
            {
                x_0 = coords_x[v1];
                y_0 = coords_y[v1];
                z_0 = coords_z[v1];
            }
        }
        else
        {
            x_0 = (coords_x[v1] + coords_x[v2]) / 2.0;
            y_0 = (coords_y[v1] + coords_y[v2]) / 2.0;
            z_0 = (coords_z[v1] + coords_z[v2]) / 2.0;
        }
    }
    if (check_new_triangles(x_0, y_0, z_0, link, num_link))
        return (1);
    else
        return (0);
}

int SurfaceEdgeCollapse::link_convex(int num_link, pair *link, int transform, int orient)
{ // check of convexity:
    // we drop the coordinate transform and check whether the 2D polygon (thus
    // obtained in) link is a convex one.
    int convex = 1;
    int i;
    float a, b;
    float a1, b1;
    // For each edge of the polygon, the next vertex has to lie on the right side
    // of the line defined by the edge
    i = 0;
    while (i < num_link - 1 && convex)
    {
        switch (transform)
        {
        case 1:
            a1 = coords_y[link[i][1]] - coords_y[link[i][0]];
            a = coords_y[link[i + 1][1]] - coords_y[link[i][0]];
            b = coords_z[link[i + 1][1]] - coords_z[link[i][0]];
            b1 = coords_z[link[i][1]] - coords_z[link[i][0]];
            break;
        case 2:
            a = coords_x[link[i + 1][1]] - coords_x[link[i][0]];
            a1 = coords_x[link[i][1]] - coords_x[link[i][0]];
            b = coords_z[link[i + 1][1]] - coords_z[link[i][0]];
            b1 = coords_z[link[i][1]] - coords_z[link[i][0]];
            break;
        case 3:
            a = coords_x[link[i + 1][1]] - coords_x[link[i][0]];
            a1 = coords_x[link[i][1]] - coords_x[link[i][0]];
            b = coords_y[link[i + 1][1]] - coords_y[link[i][0]];
            b1 = coords_y[link[i][1]] - coords_y[link[i][0]];
            break;
        default:
            return 0;
            break;
        };
        if (orient * ((-b1 * a) + (a1 * b)) < 0)
            convex = 0;
        i++;
    }
    return (convex);
}

int SurfaceEdgeCollapse::check_new_triangles(float x0, float y0, float z0, pair *link, int num_link)
{ // checks whether the newly computed position of the point the edge collapses
    // into guarantees a correct triangulation (no overlappings of triangles)
    int convex = 1;
    int i;
    fl_triple n1, n2;

    compute_triangle_normal(link[num_link - 1][0], link[num_link - 1][1], x0, y0, z0, n1);
    for (i = 0; i < num_link; i++)
    {
        compute_triangle_normal(link[i][0], link[i][1], x0, y0, z0, n2);
        if (!angle[link[i][0]])
            if (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2] < 0)
                convex = 0;
        n1[0] = n2[0];
        n1[1] = n2[1];
        n1[2] = n2[2];
    }
    return (convex);
}

int SurfaceEdgeCollapse::check_link(int v1, int v2, Star star, pair *link, int num_link)
{
    (void)star;

    // checks the link for exceptions when its not useful to collapse the
    // current edge
    int i, j;
    int flag = 1;

    // checks whether the two edges start from the same vertex in the link
    // thus detects bifurcations
    i = 0;
    while (i < num_link && flag)
    {
        j = i + 1;
        while (j < num_link && flag)
            if (link[j++][0] == link[i][0])
                flag = 0;
        i++;
    }

    // if v1 or v2 is boundary vertex, check whether a whole edge in the
    // link lies on the boundary
    if (stars[v1].boundary || stars[v2].boundary)
    {
        i = 0;
        while (i < num_link && flag)
        {
            if (stars[link[i][0]].boundary && stars[link[i][1]].boundary)
                flag = 0;
            i++;
        }
    }
    return (flag);
}

int SurfaceEdgeCollapse::compute_newpoint(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0)
{ // This procedure choses the strategy according to which the position of
    // the new point is to be computed.
    int okay;
    switch (new_point)
    {
    case 1:
        okay = compute_midpoint(v1, v2, star, link, num_link, x_0, y_0, z_0);
        break;
    case 2:
        okay = compute_position(v1, v2, star, link, num_link, x_0, y_0, z_0);
        break;
    case 3:
        okay = compute_endpoint(v1, v2, star, link, num_link, x_0, y_0, z_0);
        break;
    default:
        okay = compute_midpoint(v1, v2, star, link, num_link, x_0, y_0, z_0);
        break;
    };
    return (okay);
}

int SurfaceEdgeCollapse::compute_midpoint(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0)
{
    (void)star;
    // The position of the new point the edge collapses into is chosen to
    // be simply the midpoint of the edge. If v1 or v2 lies on the boundary or
    // on a feature edge, this point is chosen according to the boundary and
    // feature edge preservation tests.
    if (stars[v1].boundary || (angle[v1]))
    {
        x_0 = coords_x[v1];
        y_0 = coords_y[v1];
        z_0 = coords_z[v1];
        return (1);
    }

    if (stars[v2].boundary || (angle[v2]))
    {
        x_0 = coords_x[v2];
        y_0 = coords_y[v2];
        z_0 = coords_z[v2];
        return (1);
    }
    x_0 = (coords_x[v1] + coords_x[v2]) / 2.0;
    y_0 = (coords_y[v1] + coords_y[v2]) / 2.0;
    z_0 = (coords_z[v1] + coords_z[v2]) / 2.0;

    if (check_new_triangles(x_0, y_0, z_0, link, num_link))
        return (1);
    else
        return (0);
}

int SurfaceEdgeCollapse::compute_endpoint(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0)
{
    (void)star;

    // The position of the new point the edge collapses into is chosen to
    // be simply one endpoint of the edge. If v2 lies on the boundary or
    // on a feature edge, v2 is chosen, otherwise we take v1. This decision
    // is rather arbitrary.
    if (stars[v2].boundary || (angle[v2] > 0))
    {
        x_0 = coords_x[v2];
        y_0 = coords_y[v2];
        z_0 = coords_z[v2];
        return (1);
    }
    x_0 = coords_x[v1];
    y_0 = coords_y[v1];
    z_0 = coords_z[v1];
    if (check_new_triangles(x_0, y_0, z_0, link, num_link))
        return (1);
    else
        return (0);
}

int SurfaceEdgeCollapse::compute_position(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0)
{ // The position of the new point the edge collapses into is computed
    // according to the volume preserving strategy proposed by A.Gueziec
    // (see his paper IBM Technical Report RC 20440, 1997).
    // Do not touch ;)
    double x_c, y_c, z_c;
    double x, y, z;
    double A, B, C, D;
    double A1, A2, B1, B2, C1, C2;
    double n, N;
    double *co_x;
    double *co_y;
    double *co_z;
    //int orient;
    double plane[MAXTRI][4];
    int nonco = 0;
    int i, j, k;
    int value = 0;
    int transform;
    //int convex;
    int flag_x, flag_y, flag_z;

    flag_x = flag_y = flag_z = 0;

    if (stars[v1].boundary || (angle[v1]))
    {
        x_0 = coords_x[v1];
        y_0 = coords_y[v1];
        z_0 = coords_z[v1];
        return (1);
    }

    if (stars[v2].boundary || (angle[v2]))
    {
        x_0 = coords_x[v2];
        y_0 = coords_y[v2];
        z_0 = coords_z[v2];
        return (1);
    }
    //////////////////////////////////////////////////////////////////////////
    // Complete, connected star -> compute volume preserving position of v0 //
    //////////////////////////////////////////////////////////////////////////

    //coordinates of the centroid
    x_c = 0;
    y_c = 0;
    z_c = 0;

    int starNum = 0;
    for (i = 0; i < star.num_tri; i++)
    {
        int triIndex = tri_list[star.tri[i]];
        if (triIndex > 0)
        {
            x_c += coords_x[vertex_list[triIndex]];
            y_c += coords_y[vertex_list[triIndex]];
            z_c += coords_z[vertex_list[triIndex]];
            x_c += coords_x[vertex_list[triIndex + 1]];
            y_c += coords_y[vertex_list[triIndex + 1]];
            z_c += coords_z[vertex_list[triIndex + 1]];
            x_c += coords_x[vertex_list[triIndex + 2]];
            y_c += coords_y[vertex_list[triIndex + 2]];
            z_c += coords_z[vertex_list[triIndex + 2]];
            starNum++;
        }
        else
        {
            fprintf(stderr, "SurfaceEdgeCollapse::compute_position(..) found flat triangle\n");
        }
    }

    x_c /= (double)(3 * starNum);
    y_c /= (double)(3 * starNum);
    z_c /= (double)(3 * starNum);

    //relative coordinates of the vertices to the new origin c
    co_x = new double[star.num_tri * 3];
    co_y = new double[star.num_tri * 3];
    co_z = new double[star.num_tri * 3];

    for (i = 0; i < star.num_tri; i++)
    {
        k = 3 * i;
        int triIndex = tri_list[star.tri[i]];
        co_x[k] = coords_x[vertex_list[triIndex]] - x_c;
        co_y[k] = coords_y[vertex_list[triIndex]] - y_c;
        co_z[k] = coords_z[vertex_list[triIndex]] - z_c;
        co_x[k + 1] = coords_x[vertex_list[triIndex + 1]] - x_c;
        co_y[k + 1] = coords_y[vertex_list[triIndex + 1]] - y_c;
        co_z[k + 1] = coords_z[vertex_list[triIndex + 1]] - z_c;
        co_x[k + 2] = coords_x[vertex_list[triIndex + 2]] - x_c;
        co_y[k + 2] = coords_y[vertex_list[triIndex + 2]] - y_c;
        co_z[k + 2] = coords_z[vertex_list[triIndex + 2]] - z_c;
    }

    //equation of the plane the new vertex has to lie in (volume preservation)
    A = 0;
    B = 0;
    C = 0;
    D = 0;
    for (i = 0; i < star.num_tri; i++)
    {
        k = 3 * i;
        D += co_x[k] * co_y[k + 1] * co_z[k + 2] + co_x[k + 2] * co_y[k] * co_z[k + 1] + co_x[k + 1] * co_y[k + 2] * co_z[k];
        D -= (co_x[k] * co_y[k + 2] * co_z[k + 1] + co_x[k + 2] * co_y[k + 1] * co_z[k] + co_x[k + 1] * co_y[k] * co_z[k + 2]);
    }
    D = -D;

    for (i = 0; i < num_link; i++)
    {
        A += (coords_y[link[i][0]] - y_c) * (coords_z[link[i][1]] - z_c);
        A -= (coords_y[link[i][1]] - y_c) * (coords_z[link[i][0]] - z_c);
        B += (coords_z[link[i][0]] - z_c) * (coords_x[link[i][1]] - x_c);
        B -= (coords_z[link[i][1]] - z_c) * (coords_x[link[i][0]] - x_c);
        C += (coords_x[link[i][0]] - x_c) * (coords_y[link[i][1]] - y_c);
        C -= (coords_x[link[i][1]] - x_c) * (coords_y[link[i][0]] - y_c);
    }

    //Householder transformation
    N = sqrt(A * A + B * B + C * C);
    if (fabs(N) < 1E-06)
        transform = 0;
    else
    {
        transform = 1;
        if ((fabs(B) > fabs(A)) && (fabs(B) > fabs(C)))
            transform = 2;
        else if (fabs(C) > fabs(A))
            transform = 3;
    }
    A = A / N;
    B = B / N;
    C = C / N;
    D = D / N;

    // Test: if the link-polygon is non-convex, take the mid-point
    //       and v2 (set transform = 0)
    /*
   switch(transform)
   {  case 1: if (A < 0) orient = 1;
              else orient = -1;
              break;
      case 2: if (B < 0) orient = 1;
              else orient = -1;
              break;
      case 3: if (C < 0) orient = 1;
              else orient = -1;
              break;
   };
   */
    //if ((convex = link_convex(num_link, link, transform, orient)) == 0)
    //	transform = 0;
    //else
    //   if (convex == -1)
    //      return(0);

    if (transform != 0)
    {
        switch (transform)
        {
        case 1:
            for (i = 0; i < star.num_tri; i++)
            {
                k = 3 * i;
                co_x[k] = co_x[k] / A;
                co_y[k] = co_y[k] - co_x[k] * B / A;
                co_z[k] = co_z[k] - co_x[k] * C / A;
                co_x[k + 1] = co_x[k + 1] / A;
                co_y[k + 1] = co_y[k + 1] - co_x[k + 1] * B / A;
                co_z[k + 1] = co_z[k + 1] - co_x[k + 1] * C / A;
                co_x[k + 2] = co_x[k + 2] / A;
                co_y[k + 2] = co_y[k + 2] - co_x[k + 2] * B / A;
                co_z[k + 2] = co_z[k + 2] - co_x[k + 2] * C / A;
            }
            break;

        case 2:
            for (i = 0; i < star.num_tri; i++)
            {
                k = 3 * i;
                co_x[k] = co_x[k] - co_y[k] * A / B;
                co_y[k] = co_y[k] / B;
                co_z[k] = co_z[k] - co_y[k] * C / B;
                co_x[k + 1] = co_x[k + 1] - co_y[k + 1] * A / B;
                co_y[k + 1] = co_y[k + 1] / B;
                co_z[k + 1] = co_z[k + 1] - co_y[k + 1] * C / B;
                co_x[k + 2] = co_x[k + 2] - co_y[k + 2] * A / B;
                co_y[k + 2] = co_y[k + 2] / B;
                co_z[k + 2] = co_z[k + 2] - co_y[k + 2] * C / B;
            }
            break;

        case 3:
            for (i = 0; i < star.num_tri; i++)
            {
                k = 3 * i;
                co_x[k] = co_x[k] - co_z[k] * A / C;
                co_y[k] = co_y[k] - co_z[k] * B / C;
                co_z[k] = co_z[k] / C;
                co_x[k + 1] = co_x[k + 1] - co_z[k + 1] * A / C;
                co_y[k + 1] = co_y[k + 1] - co_z[k + 1] * B / C;
                co_z[k + 1] = co_z[k + 1] / C;
                co_x[k + 2] = co_x[k + 2] - co_z[k + 2] * A / C;
                co_y[k + 2] = co_y[k + 2] - co_z[k + 2] * B / C;
                co_z[k + 2] = co_z[k + 2] / C;
            }
        };

        //plane constants of the triangles in the star
        nonco = 0;
        for (i = 0; i < star.num_tri; i++)
        {
            k = 3 * nonco;
            plane[nonco][0] = -(co_y[k + 1] * co_z[k + 2] - co_y[k + 2] * co_z[k + 1]);
            plane[nonco][0] -= (co_y[k + 2] * co_z[k] - co_y[k] * co_z[k + 2]);
            plane[nonco][0] -= (co_y[k] * co_z[k + 1] - co_y[k + 1] * co_z[k]);

            plane[nonco][1] = co_x[k + 1] * co_z[k + 2] - co_x[k + 2] * co_z[k + 1];
            plane[nonco][1] += co_x[k + 2] * co_z[k] - co_x[k] * co_z[k + 2];
            plane[nonco][1] += co_x[k] * co_z[k + 1] - co_x[k + 1] * co_z[k];

            plane[nonco][2] = -(co_x[k + 1] * co_y[k + 2] - co_x[k + 2] * co_y[k + 1]);
            plane[nonco][2] -= (co_x[k + 2] * co_y[k] - co_x[k] * co_y[k + 2]);
            plane[nonco][2] -= (co_x[k] * co_y[k + 1] - co_x[k + 1] * co_y[k]);

            plane[nonco][3] = 0;
            plane[nonco][3] += co_x[k] * co_y[k + 1] * co_z[k + 2] + co_x[k + 2] * co_y[k] * co_z[k + 1] + co_x[k + 1] * co_y[k + 2] * co_z[k];
            plane[nonco][3] -= (co_x[k] * co_y[k + 2] * co_z[k + 1] + co_x[k + 2] * co_y[k + 1] * co_z[k] + co_x[k + 1] * co_y[k] * co_z[k + 2]);

            //normalizing the plane equation
            n = sqrt(plane[nonco][0] * plane[nonco][0] + plane[nonco][1] * plane[nonco][1] + plane[nonco][2] * plane[nonco][2]);
            plane[nonco][0] /= n;
            plane[nonco][1] /= n;
            plane[nonco][2] /= n;
            plane[nonco][3] /= n;

            //test: are there coplanar triangles?
            flag_x = 0;
            flag_y = 0;
            flag_z = 0;
            j = 0;
            while ((!(flag_x && flag_y && flag_z)) && (j < nonco))
            {
                flag_x = 0;
                flag_y = 0;
                flag_z = 0;
                if (((plane[j][0] - plane[nonco][0]) < 1E-04) && ((plane[nonco][0] - plane[j][0]) < 1E-04))
                    flag_x = 1;
                if (((plane[j][1] - plane[nonco][1]) < 1E-04) && ((plane[nonco][1] - plane[j][1]) < 1E-04))
                    flag_y = 1;
                if (((plane[j][2] - plane[nonco][2]) < 1E-04) && ((plane[nonco][2] - plane[j][2]) < 1E-04))
                    flag_z = 1;
                j++;
            }
            // Test: is triangle coplanar to the plane the new point has to lie in?
            if (!(flag_x && flag_y && flag_z))
            {
                flag_x = 0;
                flag_y = 0;
                flag_z = 0;
                switch (transform)
                {
                case 1:
                    if (((plane[j][0] - 1.0) < 1E-04) && ((1.0 - plane[j][0]) < 1E-04))
                        flag_x = 1;
                    if (((plane[j][1]) < 1E-04) && ((-plane[j][1]) < 1E-04))
                        flag_y = 1;
                    if (((plane[j][2]) < 1E-04) && ((-plane[j][2]) < 1E-04))
                        flag_z = 1;
                    break;
                case 2:
                    if (((plane[j][0]) < 1E-04) && ((plane[j][0]) < 1E-04))
                        flag_x = 1;
                    if (((plane[j][1] - 1.0) < 1E-04) && ((1.0 - plane[j][1]) < 1E-04))
                        flag_y = 1;
                    if (((plane[j][2]) < 1E-04) && ((-plane[j][2]) < 1E-04))
                        flag_z = 1;
                    break;
                case 3:
                    if (((plane[j][0]) < 1E-04) && ((-plane[j][0]) < 1E-04))
                        flag_x = 1;
                    if (((plane[j][1]) < 1E-04) && ((-plane[j][1]) < 1E-04))
                        flag_y = 1;
                    if (((plane[j][2] - 1.0) < 1E-04) && ((1.0 - plane[j][2]) < 1E-04))
                        flag_z = 1;
                    break;
                };
            }
            // Test: is triangle coplanar to the plane the new point has to lie in,
            // but with the wrong orientation - occurs with StarCD!
            if (!(flag_x && flag_y && flag_z))
            {
                flag_x = 0;
                flag_y = 0;
                flag_z = 0;
                switch (transform)
                {
                case 1:
                    if (((plane[j][0] + 1.0) < 1E-04) && ((-1.0 - plane[j][0]) < 1E-04))
                        flag_x = 1;
                    if (((plane[j][1]) < 1E-04) && ((-plane[j][1]) < 1E-04))
                        flag_y = 1;
                    if (((plane[j][2]) < 1E-04) && ((-plane[j][2]) < 1E-04))
                        flag_z = 1;
                    break;
                case 2:
                    if (((plane[j][0]) < 1E-04) && ((plane[j][0]) < 1E-04))
                        flag_x = 1;
                    if (((plane[j][1] + 1.0) < 1E-04) && ((-1.0 - plane[j][1]) < 1E-04))
                        flag_y = 1;
                    if (((plane[j][2]) < 1E-04) && ((-plane[j][2]) < 1E-04))
                        flag_z = 1;
                    break;
                case 3:
                    if (((plane[j][0]) < 1E-04) && ((-plane[j][0]) < 1E-04))
                        flag_x = 1;
                    if (((plane[j][1]) < 1E-04) && ((-plane[j][1]) < 1E-04))
                        flag_y = 1;
                    if (((plane[j][2] + 1.0) < 1E-04) && ((-1.0 - plane[j][2]) < 1E-04))
                        flag_z = 1;
                    break;
                };
            }
            if (!(flag_x && flag_y && flag_z))
                nonco++;
        }
    }
    if ((nonco > 1) && (transform != 0))
    {
        flag_x = 0;
        flag_y = 0;
        flag_z = 0;
        switch (transform)
        {
        case 1:
            //least-square solution of the resulting system
            A1 = 0;
            A2 = 0;
            B1 = 0;
            B2 = 0;
            C1 = 0;
            C2 = 0;
            for (i = 0; i < nonco; i++)
            {
                A1 += plane[i][1] * plane[i][1];
                A2 += plane[i][1] * plane[i][2];
                B2 += plane[i][2] * plane[i][2];
                C1 += plane[i][1] * (plane[i][3] - plane[i][0] * D);
                C2 += plane[i][2] * (plane[i][3] - plane[i][0] * D);
            }
            C1 = -C1;
            C2 = -C2;
            B1 = A2;
            x = -D;
            if (fabs(A1 * B2 - A2 * B1) > 1E-02)
            {
                y = (B2 * C1 - B1 * C2) / (A1 * B2 - A2 * B1);
                z = (C2 - A2 * y) / B2;
                y_0 = y + x * B;
                z_0 = z + x * C;
            }
            else
            {
                y_0 = ((coords_y[v1] - y_c) + (coords_y[v2] - y_c)) / 2.0;
                z_0 = ((coords_z[v1] - z_c) + (coords_z[v2] - z_c)) / 2.0;
            }
            //backtransform Householder
            x_0 = x * A;

            break;

        case 2:
            //least-square solution of the resulting system
            A1 = 0;
            A2 = 0;
            B1 = 0;
            B2 = 0;
            C1 = 0;
            C2 = 0;
            for (i = 0; i < nonco; i++)
            {
                A1 += plane[i][0] * plane[i][0];
                A2 += plane[i][0] * plane[i][2];
                B2 += plane[i][2] * plane[i][2];
                C1 += plane[i][0] * (plane[i][3] - plane[i][1] * D);
                C2 += plane[i][2] * (plane[i][3] - plane[i][1] * D);
            }
            C1 = -C1;
            C2 = -C2;
            B1 = A2;
            y = -D;
            if (fabs(A1 * B2 - A2 * B1) > 1E-02)
            {
                x = (B2 * C1 - B1 * C2) / (A1 * B2 - A2 * B1);
                z = (C2 - A2 * x) / B2;
                x_0 = x + y * A;
                z_0 = z + y * C;
            }
            else
            {
                x_0 = ((coords_x[v1] - x_c) + (coords_x[v2] - x_c)) / 2.0;
                z_0 = ((coords_z[v1] - z_c) + (coords_z[v2] - z_c)) / 2.0;
            }
            y_0 = y * B;
            break;

        case 3:
            //least-square solution of the resulting system
            A1 = 0;
            A2 = 0;
            B1 = 0;
            B2 = 0;
            C1 = 0;
            C2 = 0;
            for (i = 0; i < nonco; i++)
            {
                A1 += plane[i][0] * plane[i][0];
                A2 += plane[i][0] * plane[i][1];
                B2 += plane[i][1] * plane[i][1];
                C1 += plane[i][0] * (plane[i][3] - plane[i][2] * D);
                C2 += plane[i][1] * (plane[i][3] - plane[i][2] * D);
            }
            C1 = -C1;
            C2 = -C2;
            B1 = A2;
            z = -D;
            if (fabs(A1 * B2 - A2 * B1) > 1E-02)
            {
                x = (B2 * C1 - B1 * C2) / (A1 * B2 - A2 * B1);
                y = (C2 - A2 * x) / B2;
                x_0 = x + z * A;
                y_0 = y + z * B;
            }
            else
            {
                x_0 = ((coords_x[v1] - x_c) + (coords_x[v2] - x_c)) / 2.0;
                y_0 = ((coords_y[v1] - y_c) + (coords_y[v2] - y_c)) / 2.0;
            }
            //backtransform Householder
            z_0 = z * C;
            break;
        };

        //backtransform origin c
        x_0 += x_c;
        y_0 += y_c;
        z_0 += z_c;
    }
    else
    {
        x_0 = (coords_x[v1] + coords_x[v2]) / 2.0;
        y_0 = (coords_y[v1] + coords_y[v2]) / 2.0;
        z_0 = (coords_z[v1] + coords_z[v2]) / 2.0;
    }

    value = check_new_triangles(x_0, y_0, z_0, link, num_link);

    delete[] co_x;
    delete[] co_y;
    delete[] co_z;

    return (value);
}

void SurfaceEdgeCollapse::find_neighbors(int v1, int v2, Star star, int &to_remove, int *tri)
{
    int i, j, flag;
    for (i = 0; i < star.num_tri; i++)
    {
        flag = 0;
        for (j = 0; j < 3; j++)
            if ((vertex_list[tri_list[star.tri[i]] + j] == v1) || (vertex_list[tri_list[star.tri[i]] + j] == v2))
                flag++;
        if (flag == 2)
            tri[to_remove++] = star.tri[i];
    }
}

void SurfaceEdgeCollapse::update_star(int v1, int v2, Star &star, int l, int *vert, int to_remove, int *tri)
{
    (void)star;
    int i, k;
    // update triangles with vertices v1 and v2
    for (i = 0; i < stars[vert[l]].num_tri; i++)
    { // remove triangles with vertices v1 and v2
        if ((stars[vert[l]].tri[i] == tri[0]) || (stars[vert[l]].tri[i] == tri[to_remove - 1]))
        {
            for (k = i; k < stars[vert[l]].num_tri - 1; k++)
                stars[vert[l]].tri[k] = stars[vert[l]].tri[k + 1];
            i--;
            stars[vert[l]].num_tri--;
        }
        else
        { // set v1 instead of v2, if only v2 is a vertex of the triangle
            for (k = 0; k < 3; k++)
                if (vertex_list[tri_list[stars[vert[l]].tri[i]] + k] == v2)
                    vertex_list[tri_list[stars[vert[l]].tri[i]] + k] = v1;
        }
    }
}

void SurfaceEdgeCollapse::update_heap(int v1, int v2, Star star, int to_remove, int *tri)
{
    Edge e;
    int a, b;
    int i, j, k;
    float length;

    for (i = 0; i < star.num_tri; i++)
    {
        if (tri_list[star.tri[i]] > 0)
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

                    if ((a == v1 || b == v1 || a == v2 || b == v2) && a < num_points && b < num_points)
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

                        length = 0.0;
                        length += (coords_x[a] - coords_x[b]) * (coords_x[a] - coords_x[b]);
                        length += (coords_y[a] - coords_y[b]) * (coords_y[a] - coords_y[b]);
                        length += (coords_z[a] - coords_z[b]) * (coords_z[a] - coords_z[b]);

                        //e.length(length + 0.05 *(compact[a] + compact[b]));
                        e.set_length(length);

                        k = heap->get_index(ept[star.tri[i]][j]);
                        if (0 <= k && k <= heap->getSize())
                            heap->change(k, e);
                        //else  heap->insert_again(k, ept[star.tri[i]][j], e);
                    }
                }
        }
    }
}

void SurfaceEdgeCollapse::update_global_structures(int v1, int v2, Star &star, int to_remove, int *tri)
{
    int i, k, j;
    for (i = 0; i < star.num_tri; i++)
    {
        if ((star.tri[i] == tri[0]) || (star.tri[i] == tri[to_remove - 1]))
        { // remove tri with vertices v1 and v2
            vertex_list[tri_list[star.tri[i]]] = -1;
            vertex_list[tri_list[star.tri[i]] + 1] = -1;
            vertex_list[tri_list[star.tri[i]] + 2] = -1;
            tri_list[star.tri[i]] = -1;

            for (k = i; k < star.num_tri - 1; k++)
                star.tri[k] = star.tri[k + 1];
            i--;
            star.num_tri--;
        }
        else
            for (j = 0; j < 3; j++)
            {
                if (vertex_list[tri_list[star.tri[i]] + j] == v2)
                    vertex_list[tri_list[star.tri[i]] + j] = v1;
            }
    }
    stars[v2].num_tri = 0;
}

void SurfaceEdgeCollapse::update(int &red_tri, int v1, int v2, Star &star, pair *link, int num_link, float x_0, float y_0, float z_0)
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
    //angle[v1] = (angle[v1] || angle[v2]);

    // update heap and edge_array
    update_heap(v1, v2, star, to_remove, tri);

    delete[] star.tri;
}

///////////////////////////////////////////////////////////////////////////
// The main procedure for the reduction process. The algorithm bases     //
// on the iterative edge collapse algorithm proposed by A.Gueziec (1997) //
// but was extended by some stabilizing features. It is described in the //
// internal documentation available on the webpage                       //
// http://.../covise/support/documentation/modules/SimplifySurface.html  //
///////////////////////////////////////////////////////////////////////////

void SurfaceEdgeCollapse::Reduce(int &red_tri, int &red_pnt)
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
        if (stars[v1].num_tri + stars[v2].num_tri > 5 && v1 >= 0 && v1 < num_points && v2 >= 0 && v2 < num_points)
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
                            {

                                okay = straighten_boundary(v1, v2, star, link, num_link, x_0, y_0, z_0);
                            }
                            else
                            {
                                okay = compute_newpoint(v1, v2, star, link, num_link, x_0, y_0, z_0);
                            }
                        }
                        if (okay && (volume_bound >= fabs(compute_star_volume(v1, v2, star, link, num_link, x_0, y_0, z_0))))
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
