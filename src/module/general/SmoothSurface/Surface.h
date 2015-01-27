/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SURFACE_H
#define _SURFACE_H
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
#include <util/coviseCompat.h>
#include <do/coDistributedObject.h>
#include <do/coDoData.h>
#include <do/coDoPolygons.h>

using namespace covise;

class Surface
{
public:
    enum
    {
        MAXTRI = 50
    };

protected:
    typedef int pair[2];
    typedef int triple[3];
    typedef double fl_triple[3];

    typedef struct
    {
        int num_tri;
        int *tri;
        int boundary;
        int manifold;
    } Star;

    int num_points;
    int num_vertices;
    int num_triangles;

    int *vertex_list;
    int *tri_list;

    Star *stars;

    float *coords_x;
    float *coords_y;
    float *coords_z;

    float *norm_x;
    float *norm_y;
    float *norm_z;

    virtual void initialize_connectivity();
    virtual void initialize_topology(int i);
    int check_boundary(int v, pair *link, int num_link);
    int check_feature_angle(int v1, int v2, Star &star, float max_angle);
    int vertex_on_feature_edge(int v, float max_angle);
    int check_manifold(int v, pair *link, int num_link);
    int remove_flat_triangles(int &red_tri);

    int make_link(int v, pair *link, int &num_link);
    void make_link(int v1, int v2, Star star, pair *link, int &num_link);
    int sort_link(pair *link, int &num_link);
    int close_link(int v, pair *link, int &num_link);
    int extract_points(int num_link, pair *link, int *points);
    int extract_points(int v, int num_link, pair *link, int &num_pnt, int *points);
    int extract_points(int v1, int v2, int num_link, pair *link, int &num_pnt, int *points);

    float L1_curvature(int v, int *points, int num);
    float discrete_curvature(int v);
    float Taubin_curvature(int v, int *points, int num);
    float Hamann_curvature(int v, int *points, int num);

    int Least_Square(double (*A)[MAXTRI], double *d, double *x, int m, int n);

    void generate_normals();
    void compute_vertex_normal(int v, fl_triple normal);
    void compute_triangle_normal(int t, fl_triple normal);
    void compute_triangle_normal(int v1, int v2, float x0, float y0, float z0, fl_triple normal);
    float compute_inner_angle(int v1, int v, int v2);
    float compute_star_volume(int v, int num_link, triple *retri);

    void print_edge_information(int v1, int v2, Star star, pair *link, int num_link);
    void print_retri_information(int v, int num_link, triple *retri);
    void print_star_information(int v);

public:
    Surface(){};
    Surface(int n_points, int n_vert, int n_poly, const char *mesh_type, int *pl, int *vl, float *x_in, float *y_in, float *z_in, float *nu_in, float *nv_in, float *nw_in);
    virtual ~Surface()
    {
        int i;

        //free(node_table);
        delete[] coords_x;
        coords_x = NULL;
        delete[] coords_y;
        coords_y = NULL;
        delete[] coords_z;
        coords_z = NULL;
        if (norm_x)
            delete[] norm_x;
        norm_x = NULL;
        if (norm_y)
            delete[] norm_y;
        norm_y = NULL;
        if (norm_z)
            delete[] norm_z;
        norm_z = NULL;
        delete[] tri_list;
        tri_list = NULL;
        delete[] vertex_list;
        vertex_list = NULL;

        // free stars;
        for (i = 0; i < num_points; i++)
        {
            delete[] stars[i].tri;
            stars[i].tri = NULL;
        }
        delete[] stars;
        stars = NULL;
    }

    void TriStripsToTriangulation(int *poly_list, int *vertice_list);
    void PolyToTriangulation(int *poly_list, int *vertice_list);
    coDistributedObject **createDistributedObjects(int red_tri, int red_points, coObjInfo Triangle_name, coObjInfo Data_name, coObjInfo Normals_name);
};
#endif // _SURFACE_H
