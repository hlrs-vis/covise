/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface class for Surface Reduction Methods       **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  December 1997  V1.0                                             **
\**************************************************************************/

#include "Surface.h"
#include <util/coviseCompat.h>
#include <do/coDoData.h>

//////////////////////////////////////////////////////////////////////////////
// Base Class Constructor and Triangulation Procedures                      //
//////////////////////////////////////////////////////////////////////////////

Surface::Surface(int n_points, int n_vert, int n_poly, const char *Type, int *pl_in, int *vl_in, float *x_in, float *y_in, float *z_in, float *nu_in, float *nv_in, float *nw_in)
{
    int i;
    float norm;

    num_points = n_points;
    num_vertices = n_vert;
    num_triangles = n_poly;

    stars = new Star[num_points];

    coords_x = new float[num_points];
    coords_y = new float[num_points];
    coords_z = new float[num_points];

    for (i = 0; i < n_points; i++)
    {
        coords_x[i] = x_in[i];
        coords_y[i] = y_in[i];
        coords_z[i] = z_in[i];
    }

    if (num_vertices == 3 * num_triangles)
    {
        vertex_list = new int[num_vertices];
        for (i = 0; i < num_vertices; i++)
            vertex_list[i] = vl_in[i];
        tri_list = new int[num_triangles];
        for (i = 0; i < num_triangles; i++)
            tri_list[i] = pl_in[i];
    }
    else
    {
        if (strcmp(Type, "TRIANG") == 0)
            TriStripsToTriangulation(pl_in, vl_in);
        else if (strcmp(Type, "POLYGN") == 0)
            PolyToTriangulation(pl_in, vl_in);
        else
        {
            printf("ERROR: Other mesh types than triangle strips or polygons are not supported\n");
            return;
        }
    }

    if (nu_in != NULL)
    {
        norm_x = new float[num_points];
        norm_y = new float[num_points];
        norm_z = new float[num_points];

        for (i = 0; i < num_points; i++)
        {
            norm_x[i] = nu_in[i];
            norm_y[i] = nv_in[i];
            norm_z[i] = nw_in[i];
            norm = nu_in[i] * nu_in[i] + nv_in[i] * nv_in[i] + nw_in[i] * nw_in[i];
            if (norm != 1.0)
            {
                norm = sqrt(norm);
                norm_x[i] /= norm;
                norm_y[i] /= norm;
                norm_z[i] /= norm;
            }
        }
    }
    else
    {
        norm_x = NULL;
        norm_y = NULL;
        norm_z = NULL;
    }
}

void Surface::TriStripsToTriangulation(int *poly_list, int *vertice_list)
{ // converts triangle strips back to a simple triangulation of the surface
    int i, j;
    int vert;
    int count;
    int num_strips;

    num_strips = num_triangles;
    for (i = 0; i < num_strips - 1; i++)
        if ((vert = poly_list[i + 1] - poly_list[i]) > 3)
            num_triangles += vert - 3;

    j = 0;
    while (poly_list[i] + j < num_vertices)
        j++;
    num_triangles += j - 3;
    count = 0;
    tri_list = new int[num_triangles];
    vertex_list = new int[num_triangles * 3];

    for (i = 0; i < num_strips - 1; i++)
    {
        if ((vert = poly_list[i + 1] - poly_list[i]) == 3)
        {
            tri_list[count] = count * 3;
            vertex_list[tri_list[count]] = vertice_list[poly_list[i]];
            vertex_list[tri_list[count] + 1] = vertice_list[poly_list[i] + 1];
            vertex_list[tri_list[count] + 2] = vertice_list[poly_list[i] + 2];

            count++;
        }
        else
        {
            for (j = 0; j < vert - 2; j++)
            { //decompose triangle strip
                //assume: first triangle has correct orientation
                tri_list[count] = count * 3;
                if (!(j % 2))
                {
                    vertex_list[tri_list[count]] = vertice_list[poly_list[i] + j];
                    vertex_list[tri_list[count] + 1] = vertice_list[poly_list[i] + j + 1];
                    vertex_list[tri_list[count] + 2] = vertice_list[poly_list[i] + j + 2];
                }
                else
                {
                    vertex_list[tri_list[count]] = vertice_list[poly_list[i] + j];
                    vertex_list[tri_list[count] + 1] = vertice_list[poly_list[i] + j + 2];
                    vertex_list[tri_list[count] + 2] = vertice_list[poly_list[i] + j + 1];
                }
                count++;
            }
        }
    }
    // i = num_strips - 1
    j = 0;
    while (poly_list[num_strips - 1] + j + 2 < num_vertices)
    { //decompose triangle strip
        //assume: first triangle has correct orientation
        tri_list[count] = count * 3;
        if (!(j % 2))
        {
            vertex_list[tri_list[count]] = vertice_list[poly_list[i] + j];
            vertex_list[tri_list[count] + 1] = vertice_list[poly_list[i] + j + 1];
            vertex_list[tri_list[count] + 2] = vertice_list[poly_list[i] + j + 2];
        }
        else
        {
            vertex_list[tri_list[count]] = vertice_list[poly_list[i] + j];
            vertex_list[tri_list[count] + 1] = vertice_list[poly_list[i] + j + 2];
            vertex_list[tri_list[count] + 2] = vertice_list[poly_list[i] + j + 1];
        }
        count++;
        j++;
    }
    num_vertices = 3 * num_triangles;
}

void Surface::PolyToTriangulation(int *poly_list, int *vertice_list)
{ // converts arbitrary polygons to a triangulation of the surface by
    // partitioning every polygon with n vertices into n-2 triangles
    // the triangulation is fan-like and assumes well-shaped polygons
    // -> usually it works, since usually we have rectangles or other quads
    int i, j;
    int vert;
    int count;
    int num_poly;

    // works for convex polygons only!!!

    num_poly = num_triangles;
    for (i = 0; i < num_poly - 1; i++)
        if ((vert = poly_list[i + 1] - poly_list[i]) > 3)
            num_triangles += vert - 3;
    j = 0;
    while (poly_list[i] + j < num_vertices)
        j++;

    num_triangles += j - 3;
    count = 0;
    tri_list = new int[num_triangles];
    vertex_list = new int[num_triangles * 3];

    for (i = 0; i < num_poly - 1; i++)
    {
        if ((vert = poly_list[i + 1] - poly_list[i]) == 3)
        {
            tri_list[count] = 3 * count;
            vertex_list[tri_list[count]] = vertice_list[poly_list[i]];
            vertex_list[tri_list[count] + 1] = vertice_list[poly_list[i] + 1];
            vertex_list[tri_list[count] + 2] = vertice_list[poly_list[i] + 2];

            count++;
        }
        else
        {
            for (j = 1; j < vert - 1; j++)
            {
                tri_list[count] = count * 3;
                vertex_list[tri_list[count]] = vertice_list[poly_list[i]];
                vertex_list[tri_list[count] + 1] = vertice_list[poly_list[i] + j];
                vertex_list[tri_list[count] + 2] = vertice_list[poly_list[i] + j + 1];

                count++;
            }
        }
    }
    // i = num_poly - 1
    j = 1;
    while (poly_list[i] + j + 1 < num_vertices)
    { //new triangle 0, j, j+1
        tri_list[count] = count * 3;
        vertex_list[tri_list[count]] = vertice_list[poly_list[i]];
        vertex_list[tri_list[count] + 1] = vertice_list[poly_list[i] + j];
        vertex_list[tri_list[count] + 2] = vertice_list[poly_list[i] + j + 1];
        count++;
        j++;
    }
    num_vertices = 3 * num_triangles;
}

//////////////////////////////////////////////////////////////////////////////
//  COVISE Distributed Objects - Output Objects Definition for all classes  //
//////////////////////////////////////////////////////////////////////////////
coDistributedObject **Surface::createDistributedObjects(int red_tri, int red_points, coObjInfo Triangle_name, coObjInfo Data_name, coObjInfo Normals_name)
{
    (void)red_tri;
    (void)red_points;
    (void)Data_name;

    coDistributedObject **DO_Return = new coDistributedObject *[3];
    coDoPolygons *polygons_out;
    coDoVec3 *normals_out;

    polygons_out = new coDoPolygons(Triangle_name, num_points, coords_x, coords_y, coords_z, num_vertices, vertex_list, num_triangles, tri_list);

    if (!polygons_out->objectOk())
    {
        return (NULL);
    }

    normals_out = new coDoVec3(Normals_name, num_points, norm_x, norm_y, norm_z);

    if (!normals_out->objectOk())
    {
        return (NULL);
    }

    DO_Return[0] = polygons_out;
    DO_Return[1] = NULL;
    DO_Return[2] = normals_out;

    return (DO_Return);
}

/////////////////////////////////////////////////////////////////////////////
// General Functions in Base Class Surface - to be used by all Simplifiers //
/////////////////////////////////////////////////////////////////////////////

void Surface::initialize_connectivity()
{ // makes the connectivity list stars[0..num_points-1]
    // containing for each point a list of all adjacent triangles and
    // the number of those (num_tri)

    int i, v;
    int *sl;

    sl = new int[num_points];

    // initialization: in sl will be number of triangles containing the vertex
    for (i = 0; i < num_points; i++)
        sl[i] = 0;

    for (i = 0; i < num_triangles; i++)
        if (tri_list[i] != -1)
        {
            sl[vertex_list[tri_list[i]]]++;
            sl[vertex_list[tri_list[i] + 1]]++;
            sl[vertex_list[tri_list[i] + 2]]++;
        }

    for (i = 0; i < num_points; i++)
    {
        stars[i].tri = new int[sl[i]];
        stars[i].num_tri = 0;
    }

    for (i = 0; i < num_triangles; i++)
        if (tri_list[i] != -1)
        {
            v = vertex_list[tri_list[i]];
            stars[v].tri[stars[v].num_tri] = i;
            stars[v].num_tri++;

            v = vertex_list[tri_list[i] + 1];
            stars[v].tri[stars[v].num_tri] = i;
            stars[v].num_tri++;

            v = vertex_list[tri_list[i] + 2];
            stars[v].tri[stars[v].num_tri] = i;
            stars[v].num_tri++;
        }
    delete[] sl;
}

void Surface::initialize_topology(int i)
{ // makes the topology relevant information for the vertex i
    // boundary - is the vertex a boundary vertex
    // manifold - is the 2-manifold property fulfilled (locally)

    pair link[MAXTRI];
    int num_link;

    // check boundary and manifold condition
    make_link(i, link, num_link);
    stars[i].boundary = check_boundary(i, link, num_link);
    stars[i].manifold = check_manifold(i, link, num_link);
}

int Surface::check_boundary(int v, pair *link, int num_link)
{
    (void)v;

    // test: if all start vertices of edges in the link occur also as
    //       end vertices of edges in the link, the link is connected,
    //       and boundary is set to zero, otherwise it is 1
    int j, k, flag;

    for (j = 0; j < num_link; j++)
    {
        k = 0;
        flag = 0;
        while ((k < num_link) && (!flag))
            if (link[k++][1] == link[j][0])
                flag = 1;
        if (!flag)
        {
            return 1;
            break;
        }
    }
    return (0);
}

int Surface::check_manifold(int v, pair *link, int num_link)
{
    (void)v;

    // test: if no vertex in the link is the start vertex of two edges
    //       or the end vertex of two edges simultaneously,
    //       the manifold property is fulfilled
    // bug: if two parts of the surface join nothing but the vertex
    //      i, the surface is also non-manifold
    int j, k, flag;
    for (j = 0; j < num_link; j++)
    {
        k = 0;
        flag = 0;
        while ((!flag) && (k < num_link))
        {
            if ((link[k][0] == link[j][0]) && (j != k))
                flag = 1;
            if ((link[k][1] == link[j][1]) && (j != k))
                flag = 1;
            k++;
        }
        if (flag)
            return 0;
    }
    return (1);
}

int Surface::remove_flat_triangles(int &red_tri)
{
    int t;
    // Checks for triangles with identical knots (e.g. a b a) and removes them
    // from the triangle list.
    for (t = 0; t < num_triangles; t++)
    {
        if ((vertex_list[tri_list[t]] == vertex_list[tri_list[t] + 1])
            || (vertex_list[tri_list[t] + 1] == vertex_list[tri_list[t] + 2])
            || (vertex_list[tri_list[t] + 2] == vertex_list[tri_list[t]]))
        {
            printf("Removing triangle #%d with vertices %d %d %d \n", t, vertex_list[tri_list[t]], vertex_list[tri_list[t] + 1], vertex_list[tri_list[t] + 2]);
            vertex_list[tri_list[t]] = -1;
            vertex_list[tri_list[t] + 1] = -1;
            vertex_list[tri_list[t] + 2] = -1;
            tri_list[t] = -1;
            red_tri -= 1;
        }
    }
    return (1);
}

int Surface::make_link(int v, pair *link, int &num_link)
{ // simply extracts the edges of the boundary polygon from the
    // triangles contained in stars[v].tri
    // 0 - if there are less than 3
    // 1 - otherwise
    int i, j;
    int count, which;

    num_link = 0;
    for (i = 0; i < stars[v].num_tri; i++)
    {
        count = 0;
        which = 0;
        for (j = 0; j < 3; j++)
            if (vertex_list[tri_list[stars[v].tri[i]] + j] == v)
            {
                count++;
                which = j;
            }

        if (count == 1)
        {
            link[num_link][0] = vertex_list[tri_list[stars[v].tri[i]] + ((which + 1) % 3)];
            link[num_link][1] = vertex_list[tri_list[stars[v].tri[i]] + ((which + 2) % 3)];
            num_link++;
        }
        else
            printf("Warning: v = %d, triangle #%d has vertices %d %d %d\n", v, stars[v].tri[i], vertex_list[tri_list[stars[v].tri[i]]], vertex_list[tri_list[stars[v].tri[i]] + 1], vertex_list[tri_list[stars[v].tri[i]] + 2]);
    }

    if (num_link < 3)
        return 0;
    else
        return 1;
}

void Surface::make_link(int v1, int v2, Star star, pair *link, int &num_link)
{
    int i, j;
    int count, which, flag;
    int tri[2];
    // extract boundary polygon of the star of two vertices v1, v2
    // first extract two neighbor triangles of edge (v1,v2)
    // tri = pair of triangles to remove

    count = 0;
    for (i = 0; i < star.num_tri; i++)
    {
        flag = 0;
        for (j = 0; j < 3; j++)
            if ((vertex_list[tri_list[star.tri[i]] + j] == v1) || (vertex_list[tri_list[star.tri[i]] + j] == v2))
                flag++;
        if (flag == 2)
            tri[count++] = star.tri[i];
    }
    if (count == 1)
        tri[1] = -1;

    num_link = 0;

    for (i = 0; i < star.num_tri; i++)
    {

        if (star.tri[i] != tri[0] && star.tri[i] != tri[1])
        {
            which = 0;
            for (j = 0; j < 3; j++)
                if (vertex_list[tri_list[star.tri[i]] + j] == v1 || vertex_list[tri_list[star.tri[i]] + j] == v2)
                    which = j;

            link[num_link][0] = vertex_list[tri_list[star.tri[i]] + ((which + 1) % 3)];
            link[num_link][1] = vertex_list[tri_list[star.tri[i]] + ((which + 2) % 3)];
            num_link++;
        }
    }
    return;
}

int Surface::sort_link(pair *link, int &num_link)
{ // returns 0 if there is more than one break in the link
    // otherwise the link will be sorted counterclockwise, beginning by
    // - some vertex, if there is no break (inner vertex)
    // - the break, if there is one (boundary vertex)
    // than it returns 1

    int i, j;
    int w1, w2;
    int count, which, flag;
    int index, end_chain, chain;
    int no_loop;

    which = num_link - 1;
    count = 0;
    for (i = 0; i < num_link - 1; i++)
    {
        j = i + 1;
        while ((j < num_link) && (link[j][0] != link[i][1]))
            j++;

        if (j != i + 1)
        {
            if (j == num_link)
            {
                if (which == num_link - 1)
                    end_chain = link[0][0];
                else
                    end_chain = link[which][0];
                which = i;
                count++;
                // a break occured
                // now look for the start edge of the last chain, switch places
                // with the (i+1)t edge and start sorting again
                flag = 1;
                chain = i + 1;
                no_loop = 0;
                while (flag && no_loop < num_link)
                {
                    index = i + 1;
                    while (index < num_link && link[index][1] != end_chain)
                        index++;
                    if (index < num_link)
                    {
                        end_chain = link[index][0];
                        chain = index;
                        no_loop++;
                    }
                    else
                        flag = 0;
                }
                if (chain != i + 1)
                {
                    w1 = link[chain][0];
                    w2 = link[chain][1];
                    link[chain][0] = link[i + 1][0];
                    link[chain][1] = link[i + 1][1];
                    link[i + 1][0] = w1;
                    link[i + 1][1] = w2;
                }
            }
            else
            {
                w1 = link[j][0];
                w2 = link[j][1];
                link[j][0] = link[i + 1][0];
                link[j][1] = link[i + 1][1];
                link[i + 1][0] = w1;
                link[i + 1][1] = w2;
            }
        }
    }
    if (link[0][0] != link[num_link - 1][1])
        count++;

    // if more than one break in the link do not simplify
    if (count > 1)
        return (0);

    while (which != num_link - 1)
    {
        // switch Link until break in the link is between last and zero-th edge
        w1 = link[num_link - 1][0];
        w2 = link[num_link - 1][1];
        for (i = num_link - 1; i > 0; i--)
        {
            link[i][0] = link[i - 1][0];
            link[i][1] = link[i - 1][1];
        }
        link[0][0] = w1;
        link[0][1] = w2;
        which++;
    }

    return (1);
}

int Surface::close_link(int v, pair *link, int &num_link)
{ // closes the link if there is a break in the link(v is boundary vertex)
    // and if the boundary is straight
    // by adding the edge link[num_link-1][1] <-> link[0][0]
    // sets num_link = num_link + 1
    int v1, v2;
    float x1, x2, y1, y2, z1, z2, norm1, norm2;

    link[num_link][0] = link[num_link - 1][1];
    link[num_link++][1] = link[0][0];

    v1 = link[0][0];
    v2 = link[num_link - 1][0];
    x1 = coords_x[v1] - coords_x[v];
    x2 = coords_x[v] - coords_x[v2];
    y1 = coords_y[v1] - coords_y[v];
    y2 = coords_y[v] - coords_y[v2];
    z1 = coords_z[v1] - coords_z[v];
    z2 = coords_z[v] - coords_z[v2];

    // simplify only if boundary is straight
    norm1 = sqrt(x1 * x1 + y1 * y1 + z1 * z1);
    norm2 = sqrt(x2 * x2 + y2 * y2 + z2 * z2);
    x1 /= norm1;
    y1 /= norm1;
    z1 /= norm1;
    x2 /= norm2;
    y2 /= norm2;
    z2 /= norm2;
    if ((x1 * x2 + y1 * y2 + z1 * z2 - 1.0) < 1E-04 && (x1 * x2 + y1 * y2 + z1 * z2 - 1.0) > -1E-04)
    { //printf("Link of current Vertex  % d:\n", v);
        //for (i = 0; i < num_link; i++)
        //   printf("%d  %d \n", link[i][0], link[i][1]);
        //printf("\n");
        return (1);
    }
    else
        return (0);
}

int Surface::extract_points(int num_link, pair *link, int *points)
{ // assuming that link is a closed boundary polygon, this procedure stores
    // in points all start points of edges in the boundary polygon
    int j;

    for (j = 0; j < num_link; j++)
        points[j] = link[j][0];
    return (1);
}

int Surface::extract_points(int v, int num_link, pair *link, int &num_pnt, int *points)
{ // stores all different points in the boundary polygon link to the array
    // points (also endpoints of edges who are no start points of other edges
    // in case that v lies on the boundary) -> for single vertices
    int j, k, flag;
    num_pnt = 0;
    for (j = 0; j < num_link; j++)
        points[num_pnt++] = link[j][0];
    if (stars[v].boundary)
        for (j = 0; j < num_link; j++)
        {
            flag = 0;
            k = 0;
            while (!flag && k < num_link)
                if (points[k++] == link[j][1])
                    flag = 1;
            if (!flag)
                points[num_pnt++] = link[j][1];
        }
    return (1);
}

int Surface::extract_points(int v1, int v2, int num_link, pair *link, int &num_pnt, int *points)
{ // stores all different points in the boundary polygon link to the array
    // points (also endpoints of edges who are no start points of other edges
    // in case that v lies on the boundary) -> for edges (v1, v2)
    int j, k, flag;
    num_pnt = 0;
    for (j = 0; j < num_link; j++)
        points[num_pnt++] = link[j][0];
    if (stars[v1].boundary || stars[v2].boundary)
        for (j = 0; j < num_link; j++)
        {
            flag = 0;
            k = 0;
            while (!flag && k < num_link)
                if (points[k++] == link[j][1])
                    flag = 1;
            if (!flag)
                points[num_pnt++] = link[j][1];
        }
    return (1);
}

float Surface::compute_inner_angle(int v1, int v, int v2)
{ // computes the inner angle between the edges (v1,v) and (v,v2)
    double n[3];
    double m[3];
    double norm[2];
    float gamma;
    float cosi;

    n[0] = (double)coords_x[v] - (double)coords_x[v1];
    n[1] = (double)coords_y[v] - (double)coords_y[v1];
    n[2] = (double)coords_z[v] - (double)coords_z[v1];

    m[0] = (double)coords_x[v] - (double)coords_x[v2];
    m[1] = (double)coords_y[v] - (double)coords_y[v2];
    m[2] = (double)coords_z[v] - (double)coords_z[v2];

    norm[0] = sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
    norm[1] = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (int i = 0; i < 3; i++)
    {
        m[i] /= norm[0];
        n[i] /= norm[1];
    }

    cosi = (float)(n[0] * m[0] + n[1] * m[1] + n[2] * m[2]);
#ifdef __sgi
    gamma = acosf(cosi);
#else
    gamma = acos(cosi);
#endif

    return (gamma);
}

int Surface::check_feature_angle(int v1, int v2, Star &star, float max_angle)
{ // check whether the angle enclosed by neighbour triangles of
    // the edge (v1,v2) increases max_angle

    int count, value;
    int i, j, flag;
    int tri[5];
    float cosi;
    fl_triple a0, a1;

    value = 1;
    count = 0;
    // check whether current edge has exactly two neighbour triangles
    for (i = 0; i < star.num_tri; i++)
    {
        flag = 0;
        for (j = 0; j < 3; j++)
            if ((vertex_list[tri_list[star.tri[i]] + j] == v1) || (vertex_list[tri_list[star.tri[i]] + j] == v2))
                flag++;
        if (flag == 2)
            tri[count++] = star.tri[i];
    }

    if (count != 2)
        value = 0;

    // check whether the angle enclosed by neighbour triangles increases alpha
    if (value)
    {
        compute_triangle_normal(tri[0], a0);
        compute_triangle_normal(tri[1], a1);
        cosi = (float)(a0[0] * a1[0] + a0[1] * a1[1] + a0[2] * a1[2]);
        if (-cosi >= cos(M_PI * max_angle / 180.0))
            value = 0;
    }
    return (value);
}

int Surface::vertex_on_feature_edge(int v, float max_angle)
{ // detect feature edge
    int j, k, l, flag;
    int v1;
    int value = 0;
    float cosi, cosi_max;
    int orientation = 0;
    int count;
    fl_triple tri1, tri2;

    cosi_max = (float)cos(M_PI * max_angle / 180.0);

    count = 0;
    for (j = 0; j < stars[v].num_tri; j++)
    {
        if (vertex_list[tri_list[stars[v].tri[j]]] == v)
            v1 = vertex_list[tri_list[stars[v].tri[j]] + 1];
        else
        {
            if (vertex_list[tri_list[stars[v].tri[j]] + 1] == v)
                v1 = vertex_list[tri_list[stars[v].tri[j]] + 2];
            else
                v1 = vertex_list[tri_list[stars[v].tri[j]]];
        }
        // search  right neighbour  triangle
        k = 0;
        flag = 1;
        while ((k < stars[v].num_tri) && flag)
            if (j != k)
            {
                if ((vertex_list[tri_list[stars[v].tri[k]]] != v1) && (vertex_list[tri_list[stars[v].tri[k]] + 1] != v1) && (vertex_list[tri_list[stars[v].tri[k]] + 2] != v1))
                    k++;
                else
                {
                    flag = 0;
                    for (l = 0; l < 3; l++)
                        if (vertex_list[tri_list[stars[v].tri[k]] + l] == v1)
                        {
                            if (vertex_list[tri_list[stars[v].tri[k]] + ((l + 1) % 3)] == v)
                                orientation = 1;
                            else
                                orientation = 0;
                        }
                }
            }
            else
                k++;

        if (k != stars[v].num_tri)
        { // compute angle between these neighbour triangles
            compute_triangle_normal(stars[v].tri[j], tri1);
            compute_triangle_normal(stars[v].tri[k], tri2);
            cosi = (float)(tri1[0] * tri2[0] + tri1[1] * tri2[1] + tri1[2] * tri2[2]);
            if (!orientation)
                cosi = -cosi;

            if (-cosi > cosi_max)
                count++;
        }
    }
    if (count > 2)
        value = 2;
    else if (count)
        value = 1;

    return (value);
}

float Surface::L1_curvature(int v, int *points, int num)
{ // compute the maximum over all surrounding points of
    //     || normal(i) - normal(v)||_1
    //   ------------------------------
    //     ||coords(i) - coords(v)||_1
    int i;
    double max, x, norm;
    max = 0.0;
    for (i = 0; i < num; i++)
    {
        x = fabs((double)norm_x[points[i]] - (double)norm_x[v]);
        x += fabs((double)norm_y[points[i]] - (double)norm_y[v]);
        x += fabs((double)norm_z[points[i]] - (double)norm_z[v]);

        norm = fabs((double)coords_x[points[i]] - (double)coords_x[v]);
        norm += fabs((double)coords_y[points[i]] - (double)coords_y[v]);
        norm += fabs((double)coords_z[points[i]] - (double)coords_z[v]);

        if (norm > 1E-02)
            x /= norm;
        else
            x = 0.0;
        if (x > max)
            max = x;
    }

    return ((float)max);
}

float Surface::discrete_curvature(int v)
{ // compute the sum of the inner angles of all triangles incident to v
    //     curv = 2 pi - sum(angles)
    // for boundary vertices:
    //     curv = pi - sum(angles)
    int i;
    double sum;
    int v1, v2, which;
    sum = 0.0;
    for (i = 0; i < stars[v].num_tri; i++)
    {
        if (vertex_list[tri_list[stars[v].tri[i]]] == v)
            which = 0;
        else if (vertex_list[tri_list[stars[v].tri[i]] + 1] == v)
            which = 1;
        else
            which = 2;
        v1 = vertex_list[tri_list[stars[v].tri[i]] + (which + 1) % 3];
        v2 = vertex_list[tri_list[stars[v].tri[i]] + (which + 2) % 3];
        sum += compute_inner_angle(v2, v, v1);
    }
    if (stars[v].boundary)
        sum = M_PI - sum;
    else
        sum = 2 * M_PI - sum;
    return ((float)sum);
}

float Surface::Taubin_curvature(int v, int *link, int num)
{ //Directional curvature in direction (v_i, v) can be approximated by
    //  k(v_i, v) ~ (2*Normale (transponiert) * (v_i - v)) / ||v_i - v||_2
    //Take maximum over all directions (v_i, v), v_i in the link of v

    int i;
    double norm;
    double max = 0.0; //max is initialized later
    double curv;
    double a, b, c;
    for (i = 0; i < num; i++)
    {
        a = ((double)coords_x[link[i]] - (double)coords_x[v]);
        b = ((double)coords_y[link[i]] - (double)coords_y[v]);
        c = ((double)coords_z[link[i]] - (double)coords_z[v]);
        norm = sqrt(a * a + b * b + c * c);
        curv = fabs(2.0 * (norm_x[v] * a + norm_y[v] * b + norm_z[v] * c) / norm);
        if (curv > max || i == 0)
            max = curv;
    }
    return float(max);
}

int Surface::Least_Square(double (*A)[Surface::MAXTRI], double *d, double *x, int m, int n)
{ // Solves an overdetermined system of m linear equations for n variables
    // minimizing the least square error
    // A is an m x n matrix
    // d a vector of m components - the right-hand side
    // x a vector of n components - the solution

    int i, j, k, which;
    double U[MAXTRI][MAXTRI];
    double R[MAXTRI];
    double temp[MAXTRI];
    double max;

    if (m >= MAXTRI)
    {
        printf("dimension of matrix out of range in least square solver\n");
        return (0);
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
            U[i][j] = 0.0;
        R[i] = 0.0;
    }

    for (k = 0; k < n; k++)
        for (j = 0; j < m; j++)
        {
            R[k] += A[j][k] * d[j];
            for (i = 0; i < n; i++)
                U[i][k] += A[j][i] * A[j][k];
        }
    // Dreiecksform von U
    for (i = 0; i < n; i++)
    { // find maximal absolute value in column i, label row which
        j = i;
        max = fabs(U[i][i]);
        which = i;
        while (j < n)
        {
            if (fabs(U[j][i]) > max)
            {
                which = j;
                max = fabs(U[j][i]);
            }
            j++;
        }
        // system is underdetermined
        if (max < 1E-6)
            return (0);
        if (which != i)
        { // use temp[] as temporal variables
            // switch rows #i and #which
            for (k = 0; k < n; k++)
            {
                temp[k] = U[which][k];
                U[which][k] = U[i][k];
                U[i][k] = temp[k];
            }
            temp[0] = R[which];
            R[which] = R[i];
            R[i] = temp[0];
        }
        for (j = i + 1; j < n; j++)
            U[i][j] /= U[i][i];
        R[i] /= U[i][i];
        U[i][i] = 1.0;

        for (j = i + 1; j < n; j++)
        {
            for (k = i + 1; k < n; k++)
                U[j][k] = U[j][k] - U[j][i] * U[i][k];
            R[j] = R[j] - U[j][i] * R[i];
            //U[j][i] = 0;
        }
    }

    if (U[n - 1][n - 1] > 1E-05 || U[n - 1][n - 1] < -1E-05)
        R[n - 1] /= U[n - 1][n - 1];
    else
        return (0);

    for (i = n - 1; i >= 0; i--)
    {
        x[i] = R[i];
        for (j = n - 1; j > i; j--)
            x[i] -= x[j] * U[i][j];
    }
    return (1);
}

float Surface::Hamann_curvature(int v, int *link, int num)
{ // computes an estimate for the surface curvature following an approach by
    // Hamann (1993). He approximates he triangulated surface locally by a
    // biquadratic, smooth surface, whose curvature he calculates exactly.
    double A, B, C, D;
    double x0, y0, z0;
    int i;
    double norm;
    double dist[MAXTRI];
    double e1[3], e2[3];
    double u[MAXTRI], w[MAXTRI];
    double S[3];
    //double *Y[max_tri];
    double Y[MAXTRI][MAXTRI];
    double p, q;
    double x1, x2;
    float value;

    // for (i = 0; i < num; i++)
    // Y[i] = new double[3];

    x0 = (double)coords_x[v];
    y0 = (double)coords_y[v];
    z0 = (double)coords_z[v];

    A = (double)norm_x[v];
    B = (double)norm_y[v];
    C = (double)norm_z[v];
    D = -(A * x0 + B * y0 + C * z0);

    if (A > 1E-02 || A < -1E-02)
    {
        e1[0] = -(B + C) / A;
        e1[1] = 1.0;
        e1[2] = 1.0;
    }
    else if (B > 1E-02 || B < -1E-02)
    {
        e1[0] = 1.0;
        e1[1] = -(A + C) / B;
        e1[2] = 1.0;
    }
    else
    {
        e1[0] = 1.0;
        e1[1] = 1.0;
        e1[2] = -(A + B) / C;
    }
    norm = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    e1[0] /= norm;
    e1[1] /= norm;
    e1[2] /= norm;

    e2[0] = B * e1[2] - C * e1[1];
    e2[1] = C * e1[0] - A * e1[2];
    e2[2] = A * e1[1] - B * e1[0];

    for (i = 0; i < num; i++)
    {
        dist[i] = A * (double)coords_x[link[i]] + B * (double)coords_y[link[i]] + C * (double)coords_z[link[i]] + D;
        u[i] = ((double)coords_x[link[i]] - x0) * e1[0];
        u[i] += ((double)coords_y[link[i]] - y0) * e1[1];
        u[i] += ((double)coords_z[link[i]] - z0) * e1[2];
        w[i] = ((double)coords_x[link[i]] - x0) * e2[0];
        w[i] += ((double)coords_y[link[i]] - y0) * e2[1];
        w[i] += ((double)coords_z[link[i]] - z0) * e2[2];
    }

    for (i = 0; i < num; i++)
    {
        Y[i][0] = 0.5 * u[i] * u[i];
        Y[i][1] = u[i] * w[i];
        Y[i][2] = 0.5 * w[i] * w[i];
    }

    // Least square solution of the resulting system
    if (!Least_Square(Y, dist, S, num, 3))
        value = 0.0;
    else
    { // Computing eigenvalues
        p = -(S[0] + S[2]);
        q = S[0] * S[2] - S[1] * S[1];
        if (q > p * p / 4.0)
        { // complex roots
            value = (float)sqrt(q);
        }
        else
        { // real roots
            x1 = fabs(-p / 2.0 + sqrt(p * p / 4.0 - q));
            x2 = fabs(-p / 2.0 - sqrt(p * p / 4.0 - q));
            value = (float)(0.5f * (x1 + x2));
        }
    }
    return (value);
}

void Surface::generate_normals()
{ // computes averaged vertex normals for any vertex in the surface
    int i;
    fl_triple normal;

    norm_x = new float[num_points];
    norm_y = new float[num_points];
    norm_z = new float[num_points];

    for (i = 0; i < num_points; i++)
    {
        compute_vertex_normal(i, normal);
        norm_x[i] = (float)normal[0];
        norm_y[i] = (float)normal[1];
        norm_z[i] = (float)normal[2];
    }
    return;
}

void Surface::compute_vertex_normal(int v, fl_triple normal)
{ // computes the normal in the vertex v by averaging over all adjacent
    // triangle normals
    int i, j;
    fl_triple tri_normal, avg_normal;

    for (j = 0; j < 3; j++)
        avg_normal[j] = 0.0;
    for (i = 0; i < stars[v].num_tri; i++)
    {
        compute_triangle_normal(stars[v].tri[i], tri_normal);
        for (j = 0; j < 3; j++)
            avg_normal[j] += tri_normal[j];
    }
    for (j = 0; j < 3; j++)
        normal[j] = avg_normal[j] / stars[v].num_tri;
}

void Surface::compute_triangle_normal(int t, fl_triple normal)
{ // computes the normal of the triangle t

    float A, B, C;
    float ax, ay, az, bx, by, bz;
    float norm;

    ax = coords_x[vertex_list[tri_list[t]]] - coords_x[vertex_list[tri_list[t] + 1]];
    ay = coords_y[vertex_list[tri_list[t]]] - coords_y[vertex_list[tri_list[t] + 1]];
    az = coords_z[vertex_list[tri_list[t]]] - coords_z[vertex_list[tri_list[t] + 1]];
    bx = coords_x[vertex_list[tri_list[t]]] - coords_x[vertex_list[tri_list[t] + 2]];
    by = coords_y[vertex_list[tri_list[t]]] - coords_y[vertex_list[tri_list[t] + 2]];
    bz = coords_z[vertex_list[tri_list[t]]] - coords_z[vertex_list[tri_list[t] + 2]];
    A = (ay * bz) - (az * by);
    B = -(ax * bz) + (az * bx);
    C = (ax * by) - (ay * bx);
    norm = sqrt(A * A + B * B + C * C);

    normal[0] = A / norm;
    normal[1] = B / norm;
    normal[2] = C / norm;
}

void Surface::compute_triangle_normal(int v1, int v2, float x0, float y0, float z0, fl_triple normal)
{ // computes the normal of the triangle v1, v2, new point

    float A, B, C;
    float ax, ay, az, bx, by, bz;
    float norm;

    ax = coords_x[v1] - coords_x[v2];
    ay = coords_y[v1] - coords_y[v2];
    az = coords_z[v1] - coords_z[v2];
    bx = coords_x[v1] - x0;
    by = coords_y[v1] - y0;
    bz = coords_z[v1] - z0;
    A = (ay * bz) - (az * by);
    B = -(ax * bz) + (az * bx);
    C = (ax * by) - (ay * bx);
    norm = sqrt(A * A + B * B + C * C);

    normal[0] = A / norm;
    normal[1] = B / norm;
    normal[2] = C / norm;
}

float Surface::compute_star_volume(int v, int num_link, triple *retri)
{ // computes the volume of the polyhedron build by the vertex v to be
    // removed and the retriangulation of the resulting hole, thus giving
    // a measure on the volume deficit achieved by removing that vertex
    int i;
    float volume = 0.0;
    float vol, det;
    int v1, v2, v3;
    float x0, y0, z0;

    x0 = coords_x[v];
    y0 = coords_y[v];
    z0 = coords_z[v];

    for (i = 0; i < num_link - 2; i++)
    {
        v1 = retri[i][0];
        v2 = retri[i][1];
        v3 = retri[i][2];
        det = (y0 - coords_y[v2]) * (z0 - coords_z[v3]) - (y0 - coords_y[v3]) * (z0 - coords_z[v2]);
        vol = (x0 - coords_x[v1]) * det;
        det = (x0 - coords_x[v2]) * (z0 - coords_z[v3]) - (x0 - coords_x[v3]) * (z0 - coords_z[v2]);
        vol -= (y0 - coords_y[v1]) * det;
        det = (x0 - coords_x[v2]) * (y0 - coords_y[v3]) - (x0 - coords_x[v3]) * (y0 - coords_y[v2]);
        vol += (z0 - coords_z[v1]) * det;
        volume += vol;
    }
    volume /= 6.0;
    return (volume);
}

//////////////////////////////////////////////////////////////////////////
// Debugging functions                                                  //
//////////////////////////////////////////////////////////////////////////

void Surface::print_edge_information(int v1, int v2, Star star, pair *link, int num_link)
{
    int i;

    printf("####################################\n\n");
    printf("Edge %d %d information:\n", v1, v2);
    printf("Triangles in the star:\n");
    for (i = 0; i < star.num_tri; i++)
        printf("  %d", star.tri[i]);
    printf("\n");
    printf("Vertices of these triangles:\n");
    for (i = 0; i < star.num_tri; i++)
        printf("  %d : %d  %d  %d\n", star.tri[i], vertex_list[tri_list[star.tri[i]]], vertex_list[tri_list[star.tri[i]] + 1], vertex_list[tri_list[star.tri[i]] + 2]);
    printf("Link of the star:\n");
    for (i = 0; i < num_link; i++)
        printf("  %d  %d\n", link[i][0], link[i][1]);
    printf("\n");
    printf("star.manifold = %d\n", star.manifold);
    printf("star.boundary = %d\n", star.boundary);
}

void Surface::print_star_information(int v)
{
    int i;

    printf("-------------\n\n");
    printf("Star %d information:\n", v);
    printf("Triangles in the star:\n");
    for (i = 0; i < stars[v].num_tri; i++)
        printf("  %d", stars[v].tri[i]);
    printf("\n");
    printf("Vertices of these triangles:\n");
    for (i = 0; i < stars[v].num_tri; i++)
        printf("  %d : %d  %d  %d\n", stars[v].tri[i], vertex_list[tri_list[stars[v].tri[i]]], vertex_list[tri_list[stars[v].tri[i]] + 1], vertex_list[tri_list[stars[v].tri[i]] + 2]);
    printf("manifold = %d\n", stars[v].manifold);
    printf("boundary = %d\n", stars[v].boundary);
    //printf("angle    = %d\n", angle[v]);
}

void Surface::print_retri_information(int v, int num_link, triple *retri)
{
    int i;

    printf("-------------\n\n");
    printf("Retriangulation information:\n");
    printf("Triangles in the new star:\n");
    for (i = 0; i < num_link - 2; i++)
        printf("  %d", stars[v].tri[i]);
    printf("\n");
    printf("Vertices of these triangles:\n");
    for (i = 0; i < num_link - 2; i++)
        printf("  %d : %d  %d  %d\n", stars[v].tri[i], retri[i][0], retri[i][1], retri[i][2]);
}
