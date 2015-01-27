/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SURFVERTEX_H
#define _SURFVERTEX_H
/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface vertex removal class                      **
 **                      for Surface Reduction Methods                     **
 **                                                                        **
 **                                                                        **
 **                             (C) 1998                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  June 1998  V1.0                                                 **
\**************************************************************************/

#include "PQ.h"
#include "Surface.h"

class SurfaceVertexRemoval : public Surface
{
protected:
    PQ<Vertex> *heap; // priority queue
    int *is_removed; // vertex removed yes/no
    int *angle; // vertex on feature edge yes/no
    float max_angle; // Parameter: feature angle to be preserved
    float percent; // Parameter: percentage of triangles to be left after reduction
    float volume_bound; // Parameter: upper bound for the volume to be removed per iteration
    int which_curv; // Parameter: which curvature estimation scheme to be used

    virtual void preprocess(int &red_tri);
    //virtual void vertex_on_feature_edge(int i, float max_angle);
    virtual int update(int v, pair *link, int num_link, triple *retriang);
    float compute_curvature(int v0, int *points, int count);
    float compute_curvature(int v, int *points, int num, float norm);
    float compute_weight(int v0, int *points, int count);

    int check_polygon(int num_border, float (*u)[2], int &num_exclude, int *exclude, float (*v)[2], int *uTOv);
    int check_selfintersections(int num, float (*co)[2]);
    void adjust_triangulation(int num_border, int num_exclude, int *exclude, int *uTOv, int (*tri)[3]);
    int make_retriangulation(int v, pair *link, int num_link, triple *retri);
    void split_link(int v, int *points, int num_p, int *pnt_1, int &num_1, int *pnt_2, int &num_2);
    void average_vertex_normal(int v, int *pnt, int count, fl_triple normal);
    void split_link(int v, int v1, int v2, pair *link, int num_link, pair *link_1, int &num_1, pair *link_2, int &num_2);
    int retriangulate_edge(int v, pair *link, int num_link, triple *retri);
    int check_retri(int v, int num_link, triple *retri, int less);

public:
    SurfaceVertexRemoval(){};
    SurfaceVertexRemoval(int n_points, int n_vert, int n_poly, char *mesh_type, int *pl, int *vl, float *x_in, float *y_in, float *z_in, float *nu_in, float *nv_in, float *nw_in)
        : Surface(n_points, n_vert, n_poly, mesh_type, pl, vl, x_in, y_in, z_in, nu_in, nv_in, nw_in)
    {
        is_removed = new int[n_points];
        for (int i = 0; i < n_points; i++)
            is_removed[i] = 0;
        angle = new int[n_points];
        // Default-Einstellungen
        max_angle = 40.0;
        percent = 40.0;
        volume_bound = 0.001;
        which_curv = 3;
    };
    virtual ~SurfaceVertexRemoval()
    {
        delete[] is_removed;
        delete[] angle;
    }
    void Set_FeatureAngle(float feature_angle)
    {
        max_angle = feature_angle;
    }
    void Set_Percent(float perc)
    {
        if (perc >= 0.0 && perc <= 100.0)
            percent = perc;
        else
            percent = 50.0;
    }
    void Set_Curvature(int which_c)
    {
        if (which_c > 0 && which_c < 5)
            which_curv = which_c;
        else
            which_curv = 1;
    }
    void Set_VolumeBound(float volume)
    {
        if (volume > 0)
            volume_bound = volume;
        else
            volume_bound = 0.001;
    }
    virtual void Reduce(int &red_tri, int &red_points);
    virtual coDistributedObject **createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name);
};
#endif // _SURFVERTEX_H
