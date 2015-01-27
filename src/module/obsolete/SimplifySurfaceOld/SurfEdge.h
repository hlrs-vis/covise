/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SURFEDGE_H
#define _SURFEDGE_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface edge collapse class                       **
 **                     for Surface Reduction Methods                      **
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
using namespace covise;
#include "PQ.h"
#include "Surface.h"

class SurfaceEdgeCollapse : public Surface
{
protected:
    PQ<Edge> *heap; // priority queue
    int *is_removed; // vertex removed yes/no
    int *angle; // vertex on feature edge yes/no
    triple *ept; // Edge_per_Triangle: needed for updating the heap
    int new_point; // Parameter: strategy to compute new point
    float max_angle; // Parameter: feature angle to be preserved
    float percent; // Parameter: percentage of triangles to be left after reduction
    float volume_bound; // Parameter: upper bound for the volume to be removed per iteration
    float *compact; // compactness of riangles around a vertex

    void make_priority_queue();
    virtual void preprocess(int &red_tri);
    void merge_stars(int v1, int v2, Star &star);

    int link_convex(int num, pair *link, int transform, int orient);
    int check_new_triangles(float x0, float y0, float z0, pair *link, int num_link);
    int check_link(int v1, int v2, Star star, pair *link, int num_link);

    int straighten_boundary(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0);
    int straighten_edge(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0);
    int compute_position(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0);
    int compute_midpoint(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0);
    int compute_endpoint(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0);
    int compute_newpoint(int v1, int v2, Star &star, pair *link, int num_link, float &x_0, float &y_0, float &z_0);
    void find_neighbors(int v1, int v2, Star star, int &to_remove, int *tri);
    void update_heap(int v1, int v2, Star star, int to_remove, int *tri);
    void update_star(int v1, int v2, Star &star, int l, int *vert, int to_remove, int *tri);
    void update_global_structures(int v1, int v2, Star &star, int to_remove, int *tri);
    virtual void update(int &red_tri, int v1, int v2, Star &star, pair *link, int num_link, float x_0, float y_0, float z_0);

public:
    SurfaceEdgeCollapse(){};
    SurfaceEdgeCollapse(int n_points, int n_vert, int n_poly, char *mesh_type, int *pl, int *vl, float *x_in, float *y_in, float *z_in, float *nu_in, float *nv_in, float *nw_in)
        : Surface(n_points, n_vert, n_poly, mesh_type, pl, vl, x_in, y_in, z_in, nu_in, nv_in, nw_in)
    {
        is_removed = new int[n_points];
        for (int i = 0; i < n_points; i++)
            is_removed[i] = 0;
        angle = new int[n_points];
        ept = new triple[num_triangles];
        //compact = new float[num_points];
        new_point = 1;
        max_angle = 40.0;
        percent = 40.0;
        volume_bound = 0.001;
    }
    virtual ~SurfaceEdgeCollapse()
    {
        delete[] is_removed;
        delete[] angle;
        delete[] ept;
        //delete [] compact;
    }
    void Set_FeatureAngle(float feature_angle)
    {
        max_angle = feature_angle;
    }
    void Set_Percent(float perc)
    {
        if (perc >= 0.0 && perc <= 100.0)
            percent = perc;
    }
    void Set_NewPoint(int which_p)
    {
        if (which_p > 0 && which_p < 4)
            new_point = which_p;
    }
    void Set_VolumeBound(float volume)
    {
        if (volume > 0)
            volume_bound = volume;
    }

    virtual void Reduce(int &red_tri, int &red_points);
    virtual coDistributedObject **createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name);
};
#endif // _SURFEDGE_H
