/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _OPTIMIZERREDUCER_H
#define _OPTIMIZERREDUCER_H
/**************************************************************************\ 
 **                                                           (C)1997 SGI  **
 **                                                                        **
 ** Description:  COVISE Surface reduction application module  using       **
 **                               Optimizer                                **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Author:  Monika Wierse                                                 **
 **                                                                        **
 ** Date:  October 1997  V1.0                                              **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include "HandleSet.h"

#include <stdio.h>

#include <GL/gl.h>
#include <Cosmo3D/csFields.h>
#include <Cosmo3D/csIndexSet.h>
#include <Cosmo3D/csCoordSet.h>
#include <Cosmo3D/csNormalSet.h>
#include <Cosmo3D/csGeoSet.h>
#include <Cosmo3D/csTriSet.h>
#include <Cosmo3D/csWindow.h>
#include <Optimizer/opSRASimplify.h>

const opReal SRA_EVAL_FAR = 1.e+20;

//extern float  *x_in,*y_in,*z_in,*s_in,*u_in,*v_in,*w_in, *nu_in,*nv_in,*nw_in;
static csVec3f *coords_stat;

static int *vl_in, *pl_in;

static float percent;
static float weight0, weight1, weight2;
static float alpha;
static float alpha_1;
static int no_data;
static float *x_in;
static float *y_in;
static float *z_in;
static float *s_in = NULL;
static float *u_in = NULL;
static float *v_in = NULL;
static float *w_in = NULL;

static char *dtype, *gtype;
static char *colorn;

typedef struct
{
    int num_tri;
    int *tri;
} Star;

class opSRASimplify_scalar_data : public opSRASimplify
{

public:
    // I want to define my own evaluation of the vertices to bring in
    // scientific data

    opReal calculateVtxEval(int index);
};

class opSRASimplify_vector_data : public opSRASimplify
{

public:
    // I want to define my own evaluation of the vertices to bring in
    // scientific data

    opReal calculateVtxEval(int index);
};

class Surface
{
private:
    int num_points;
    int num_vertices;
    int num_triangles;
    int num_poly;

    int *vertex_list;

public:
    Surface(){};
    Surface(int n_points, int n_vert, int n_poly);
    Surface(int n_poly);
    ~Surface();
    csGeoSet *gset;

    void ToTriangulationForTriStrips();
    void ToTriangulationForPolygons();
    void PrintInMesh();
    void PrintOutMesh();
    coDistributedObject **createcoDistributedObjects(int red_tri, int red_points, char *Triangle_name, char *Data_name);
    void determine_neighbours(int *neighbours, int *num_of_neighbours, int *number_of_edges);
};

class Application : public Covise_Set_Handler
{

private:
    // callback stub functions
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    // private member functions
    void compute(void *callbackData);
    void quit(void *callbackData);

public:
    coDistributedObject **Application::ComputeObject(coDistributedObject **, char **, int);

    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Data-dependent reduction of a triangulated surface");

        Covise::add_port(INPUT_PORT, "meshIn", "Set_Polygons", "Geometry");
        Covise::add_port(INPUT_PORT, "dataIn", "Set_Float|Set_Vec3", "Vertice-attached data");
        Covise::add_port(OUTPUT_PORT, "meshOut", "Set_Polygons", "The reduced geometry");
        Covise::add_port(OUTPUT_PORT, "dataOut", "Set_Float|Set_Vec3", "The interpolated scalar data");
        Covise::add_port(PARIN, "percent", "Scalar", "Percentage of triangles to be left after simplification");
        Covise::set_port_default("percent", "30.0");
        Covise::add_port(PARIN, "weight0", "Scalar", "Weight0: Hausdorff distance");
        Covise::set_port_default("weight0", "0.");
        Covise::add_port(PARIN, "weight1", "Scalar", "Weight1: first derivative");
        Covise::set_port_default("weight1", "1.");
        Covise::add_port(PARIN, "weight2", "Scalar", "Weight2: curvature");
        Covise::set_port_default("weight2", "0.");
        Covise::add_port(PARIN, "alpha", "Scalar", "alpha: for linear combination");
        Covise::set_port_default("alpha", "0.");

        Covise::set_port_required("dataIn", 0);
        Covise::set_port_dependency("dataOut", "dep dataIn");

        Covise::init(argc, argv);
        Covise::set_start_callback(Application::computeCallback, this);
        Covise::set_quit_callback(Application::quitCallback, this);
    }

    coDistributedObject **HandleObjects(coDistributedObject *mesh_object, coDistributedObject *data_object, char *Mesh_out_name, char *Data_out_name);

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _OPTIMIZERREDUCER_H
