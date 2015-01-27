/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SURFEDGEDATA_H
#define _SURFEDGEDATA_H
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
using namespace covise;
#include "PQ.h"
#include "Surface.h"
#include "SurfEdge.h"

class ScalarDataEdgeCollapse : public SurfaceEdgeCollapse
{
private:
    float *data;
    float *grad;
    float max_gradient;
    int which_grad;
    float enforce;

    void preprocess(int &red_tri);
    float L1_gradient(int v, int *points, int num);
    float Data_deviation(int v, int *points, int num);
    float Zalgaller_gradient(int v, int *points, int num);
    float compute_gradient(int v, int *points, int num, float norm);
    float compute_weight(int i, int *points, int num_pnt);
    float compute_weight(int v1, int v2);
    float compute_weight(float curv, float grad, float compact);
    void update_heap(int v1, int v2, Star star, int to_update, int *vert, int to_remove, int *tri);
    void update(int &red_tri, int v1, int v2, Star &star, pair *link, int num_link, float x_0, float y_0, float z_0);

public:
    ScalarDataEdgeCollapse(){};
    ScalarDataEdgeCollapse(int n_points, int n_vert, int n_poly, char *mesh_type, int *pl, int *vl, float *x_in, float *y_in, float *z_in, float *ds_in, float *nu_in, float *nv_in, float *nw_in)
        : SurfaceEdgeCollapse(n_points, n_vert, n_poly, mesh_type, pl, vl, x_in, y_in, z_in, nu_in, nv_in, nw_in)
    {
        if (ds_in)
        {
            data = new float[num_points];
            for (int i = 0; i < num_points; i++)
                data[i] = ds_in[i];
            grad = new float[num_points];
        }
        else
        {
            data = 0;
            grad = 0;
        }
        which_grad = 1;
        enforce = 1.0;
    }
    virtual ~ScalarDataEdgeCollapse()
    {
        delete[] data;
        delete[] grad;
    }
    void Set_Enforce(float enfo)
    {
        if (enfo > 0)
            enforce = enfo;
        else
            enforce = 1.0;
    }
    void Set_Gradient(int which_g)
    {
        if (which_g > 0 && which_g < 4)
            which_grad = which_g;
    }

    coDistributedObject **createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name);
    void Reduce(int &red_tri, int &red_points);
};

class VectorDataEdgeCollapse : public SurfaceEdgeCollapse
{
private:
    float *data_u;
    float *data_v;
    float *data_w;
    float *grad;
    float max_gradient;
    int which_grad;
    float enforce;

    void preprocess(int &red_tri);
    float L1_gradient(int v, int *points, int num);
    float Data_deviation(int v, int *points, int num);
    float Zalgaller_gradient(int v, int *points, int num);
    float compute_gradient(int v, int *points, int num, float norm);
    float compute_weight(int i, int *points, int num_pnt);
    float compute_weight(int v1, int v2);
    float compute_weight(float curv, float grad, float compact);
    void update_heap(int v1, int v2, Star star, int to_update, int *vert, int to_remove, int *tri);
    void update(int &red_tri, int v1, int v2, Star &star, pair *link, int num_link, float x_0, float y_0, float z_0);

public:
    VectorDataEdgeCollapse(){};
    VectorDataEdgeCollapse(int n_points, int n_vert, int n_poly, char *mesh_type, int *pl, int *vl, float *x_in, float *y_in, float *z_in, float *du_in, float *dv_in, float *dw_in, float *nu_in, float *nv_in, float *nw_in)
        : SurfaceEdgeCollapse(n_points, n_vert, n_poly, mesh_type, pl, vl, x_in, y_in, z_in, nu_in, nv_in, nw_in)
    {
        data_u = new float[num_points];
        data_v = new float[num_points];
        data_w = new float[num_points];

        for (int i = 0; i < num_points; i++)
        {
            data_u[i] = du_in[i];
            data_v[i] = dv_in[i];
            data_w[i] = dw_in[i];
        }
        grad = new float[num_points];
        which_grad = 1;
        enforce = 1.0;
    }
    void Set_Gradient(int which_g)
    {
        if (which_g > 0 && which_g < 4)
            which_grad = which_g;
    }
    void Set_Enforce(float enfo)
    {
        if (enfo > 0)
            enforce = enfo;
        else
            enforce = 1.0;
    }

    virtual ~VectorDataEdgeCollapse()
    {
        delete[] data_u;
        delete[] data_v;
        delete[] data_w;
        delete[] grad;
    }
    virtual coDistributedObject **createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name);
    void Reduce(int &red_tri, int &red_points);
};
#endif // _SURFEDGEDATA_H
