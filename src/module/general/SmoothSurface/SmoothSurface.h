/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SMOOTH_H
#define _SMOOTH_H
/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface smoothing application module              **
 **                                                                        **
 **                                                                        **
 **                             (C) 1998                                   **
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 ** Date:  August 1998  V1.0                                               **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include "Surface.h"
#include <util/coviseCompat.h>

extern float *x_in, *y_in, *z_in;

class SmoothSurface : public Surface
{
private:
    int iterations;
    float scale_1;
    float scale_2;

    typedef struct
    {
        int num;
        int *pnt;
    } Neighbor;

    Neighbor *link;
    void initialize_neighborlist();
    void Gaussian(float lambda);
    void Uwe(float lambda);
    void compute_parameters();

public:
    SmoothSurface(){};
    SmoothSurface(int n_points, int n_vert, int n_poly, const char *mesh_type, int *pl, int *vl, float *x_in, float *y_in, float *z_in, float *nu_in, float *nv_in, float *nw_in)
        : Surface(n_points, n_vert, n_poly, mesh_type, pl, vl, x_in, y_in, z_in, nu_in, nv_in, nw_in){};
    virtual ~SmoothSurface()
    {
    }
    coDistributedObject *createDistributedObjects(coObjInfo Triangle_name);

    void Set_Iterations(int N)
    {
        iterations = N;
    }
    void Set_Scale_1(float x)
    {
        scale_1 = x;
    }
    void Set_Scale_2(float x)
    {
        scale_2 = x;
    }
    void Smooth_Gaussian();
    void Smooth_Taubin();
    void Smooth_Uwe();
};

class SmoothSurfaceModule : public coSimpleModule
{
    COMODULE

private:
    coInputPort *pMeshIn;
    coOutputPort *pMeshOut;
    coChoiceParam *pMethod;
    coIntScalarParam *pIterations;
    coFloatParam *pScale1, *pScale2;

    // private member functions
    //
    int compute(const char *port);

public:
    coDistributedObject **ComputeObject(coDistributedObject **, char **, int);
    coDistributedObject **HandleObjects(coDistributedObject *mesh_object, char *Mesh_out_name);
    SmoothSurfaceModule(int argc, char **argv);
    virtual ~SmoothSurfaceModule()
    {
    }
};
#endif // _SMOOTH_H
