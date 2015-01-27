/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SURFVERTEXDATA_H
#define _SURFVERTEXDATA_H
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
 ** Date:  June 1998  V1.0                                                 **
\**************************************************************************/
#include <appl/ApplInterface.h>
using namespace covise;
#include "PQ.h"
#include "Surface.h"
#include "SurfVertex.h"

class ScalarDataVertexRemoval : public SurfaceVertexRemoval
{
protected:
    float *data;
    int *data_ridge;
    float max_curvature;
    float max_gradient;
    float avg_gradient;
    int which_grad;
    float enforce;
    float ridge;

    void preprocess(int &red_tri);
    int update(int v, pair *link, int num_link, triple *retriang);
    float L1_gradient(int v, int *points, int num);
    float Data_deviation(int v, int *points, int num);
    float Zalgaller_gradient(int v, int *points, int num);
    float compute_gradient(int v, int *points, int num, float norm);
    float compute_weight(int i, int *points, int num_pnt);
    float compute_weight(float curv, float grad, float compact);

public:
    ScalarDataVertexRemoval(){};
    ScalarDataVertexRemoval(int n_points, int n_vert, int n_poly, char *mesh_type, int *pl, int *vl, float *x_in, float *y_in, float *z_in, float *ds_in, float *nu_in, float *nv_in, float *nw_in)
        : SurfaceVertexRemoval(n_points, n_vert, n_poly, mesh_type, pl, vl, x_in, y_in, z_in, nu_in, nv_in, nw_in)
    {
        data = new float[num_points];
        data_ridge = new int[num_points];
        for (int i = 0; i < num_points; i++)
        {
            data[i] = ds_in[i];
            data_ridge[i] = 0;
        }

        which_grad = 1;
        enforce = 1.0;
        ridge = 1.0;
    }
    virtual ~ScalarDataVertexRemoval()
    {
        delete[] data;
        delete[] data_ridge;
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
        else
            which_grad = 1;
    }

    void Set_Ridge(int dridge)
    {
        if (dridge > 0)
            ridge = dridge;
        else
            ridge = 2.0;
    }

    void Reduce(int &red_tri, int &red_points);
    coDistributedObject **createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name);
};

class VectorDataVertexRemoval : public SurfaceVertexRemoval
{
protected:
    float *data_u;
    float *data_v;
    float *data_w;
    float max_curvature;
    float max_gradient;
    int which_grad;
    float enforce;

    void preprocess(int &red_tri);
    int update(int v, pair *link, int num_link, triple *retriang);
    float L1_gradient(int v, int *points, int num);
    float Data_deviation(int v, int *points, int num);
    float Zalgaller_gradient(int v, int *points, int num);
    float compute_gradient(int v, int *points, int num, float norm);
    float compute_weight(int i, int *points, int num_pnt);
    float compute_weight(float curv, float grad, float compact);

public:
    VectorDataVertexRemoval(){};
    VectorDataVertexRemoval(int n_points, int n_vert, int n_poly, char *mesh_type, int *pl, int *vl, float *x_in, float *y_in, float *z_in, float *du_in, float *dv_in, float *dw_in, float *nu_in, float *nv_in, float *nw_in)
        : SurfaceVertexRemoval(n_points, n_vert, n_poly, mesh_type, pl, vl, x_in, y_in, z_in, nu_in, nv_in, nw_in)
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
        which_grad = 1;
        enforce = 1.0;
    }
    void Set_Gradient(int which_g)
    {
        if (which_g > 0 && which_g < 4)
            which_grad = which_g;
        else
            which_grad = 1;
    }
    void Set_Enforce(float enfo)
    {
        if (enfo > 0)
            enforce = enfo;
        else
            enforce = 1.0;
    }

    virtual ~VectorDataVertexRemoval()
    {
        delete[] data_u;
        delete[] data_v;
        delete[] data_w;
    }
    void Reduce(int &red_tri, int &red_pnt);
    coDistributedObject **createcoDistributedObjects(int red_tri, int red_points, const char *Triangle_name, const char *Data_name, const char *Normals_name);
};
#endif // _SURFVERTEXDATA_H
