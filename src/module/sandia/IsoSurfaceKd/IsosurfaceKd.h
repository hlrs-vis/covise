/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISOS_H
#define _ISOS_H
/**************************************************************************\
**                                                           (C)1995 RUS  **
**                                                                        **
** Description:  COVISE Isosurface application module                     **
**                                                                        **
**                                                                        **
**                             (C) 1995                                   **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **
**                                                                        **
**                                                                        **
** Author:  Uwe Woessner                                                  **
**                                                                        **
**                                                                        **
** Date:  23.02.95  V1.0                                                  **
\**************************************************************************/

#include "ApplInterface.h"
#include "covise_kd_tree.h"
#include <stdlib.h>

extern float planei, planej, planek;
extern float *x_in, *y_in, *z_in, *s_in, *u_in, *v_in, *w_in, *i_in;
extern float distance;
extern int numiso;
extern int save_memory;

typedef struct NodeInfo_s
{
    int targets[12];
    int vertice_list[12];
} NodeInfo;

typedef struct cutting_info_s
{
    int node_pairs[12];
    int nvert;
} cutting_info;

struct hash_table_entry
{
    int key;
    int targets[12];
    int vertice_list[12];
    //struct hash_table_entry *next;
};

struct small_hash_table_entry
{
    int key;
    int targets[4];
    int vertice_list[4];
    //struct hash_table_entry *next;
};

#define FIND_HASH_INDEX(key) \
    ((key) % hash_table_size)

extern cutting_info Tet_Table[];
extern cutting_info Pry_Table[];
extern cutting_info Psm_Table[];
extern cutting_info Hex_Table[];
extern cutting_info *Cutting_Info[];

class Plane
{
    friend class STR_Plane;
    friend class UNI_Plane;
    friend class RECT_Plane;

private:
    int num_nodes;
    int num_elem;
    int num_triangles;
    int num_vertices;
    int num_strips;
    int *vertice_list;
    int *vertex;
    int num_coords;
    int *ts_vertice_list;
    int *ts_line_list;
    int *neighbors;
    float *coords_x;
    float *coords_y;
    float *coords_z;
    float *coord_x;
    float *coord_y;
    float *coord_z;
    float *V_Data_U;
    float *V_Data_V;
    float *V_Data_W;
    float *S_Data;
    float *V_Data_U_p;
    float *V_Data_V_p;
    float *V_Data_W_p;
    float *S_Data_p;
    float *Normals_U;
    float *Normals_V;
    float *Normals_W;
    DO_KdTree *kdtree;
    small_hash_table_entry *small_hash_table;
    hash_table_entry *hash_table;
    int hash_table_size;

    NodeInfo *node_table;
    int Datatype;

public:
    Plane(){};
    Plane(int n_elem, int n_nodes, int Type, DO_KdTree *kd);
    ~Plane()
    {
        free(node_table);
        delete[] coords_x;
        delete[] coords_y;
        delete[] coords_z;
        delete[] vertice_list;
        if (S_Data)
        {
            delete[] S_Data;
        }
        if (V_Data_U)
        {
            delete[] V_Data_U;
            delete[] V_Data_V;
            delete[] V_Data_W;
        }
    }
    void createNormals();
    void createStrips();
    void add_vertex(int n1, int n2);
    void add_vertex(int n1, int n2, int x, int y, int z, int u, int v, int w);
    void createcoDistributedObjects(char *Data_name, char *Normal_name,
                                    char *Triangle_name, coDoSet *Data_set = NULL,
                                    coDoSet *Normal_set = NULL, coDoSet *Triangle_set = NULL);
    void createPlane();
};

class STR_Plane : public Plane
{

public:
    STR_Plane(int n_elem, int n_nodes, int Type, DO_KdTree *kd)
        : Plane(n_elem, n_nodes, Type, kd){};
    void createPlane();
};

class UNI_Plane : public Plane
{

public:
    UNI_Plane(int n_elem, int n_nodes, int Type, DO_KdTree *kd);
    void createPlane();
};

class RECT_Plane : public Plane
{

public:
    RECT_Plane(int n_elem, int n_nodes, int Type, DO_KdTree *kd);
    void createPlane();
};

class Application
{

private:
    // callback stub functions
    //
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    // private member functions
    //
    void compute(void *callbackData);
    void quit(void *callbackData);
    DO_KdTree *kdtree;

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Generate an isosurface from a data set");
        Covise::add_port(INPUT_PORT, "meshIn", "coDoUnstructuredGrid|coDoUniformGrid|coDoStructuredGrid|coDoRectilinearGrid", "Grid");
        Covise::add_port(INPUT_PORT, "isoDataIn", "coDoFloat|coDoFloat", "Data for isosurface generation");
        Covise::add_port(INPUT_PORT, "kdTreeIn", "DO_KdTree", "KdTree for fast isosurface generation");
        Covise::add_port(INPUT_PORT, "dataIn", "coDoFloat|coDoVec3|coDoFloat|coDoVec3", "Data to be maped onto the isosurface");
        Covise::add_port(OUTPUT_PORT, "meshOut", "coDoPolygons|coDoTriangleStrips", "The Isosurface");
        Covise::add_port(OUTPUT_PORT, "dataOut", "coDoFloat|coDoVec3", "interpolated data");
        Covise::add_port(OUTPUT_PORT, "normalsOut", "DO_Unstructured_V3D_Normals", "Surface normals");
        Covise::add_port(PARIN, "gennormals", "Boolean", "Supply normals");
        Covise::add_port(PARIN, "genstrips", "Boolean", "convert triangles to strips");
        Covise::add_port(PARIN, "save_memory", "Boolean", "use slower but less memory consuming algorythm");
        Covise::add_port(PARIN, "isovalue", "Scalar", "Trashold for isosurface");
        Covise::set_port_default("isovalue", "1.0");
        Covise::set_port_default("genstrips", "TRUE");
        Covise::set_port_default("gennormals", "TRUE");
        Covise::set_port_default("save_memory", "TRUE");
        Covise::set_port_required("dataIn", 0);
        Covise::set_port_required("kdTreeIn", 0);
        Covise::init(argc, argv);
        Covise::set_start_callback(Application::computeCallback, this);
        Covise::set_quit_callback(Application::quitCallback, this);
    }
    void HandleObjects(coDistributedObject *grid_object,
                       coDistributedObject *data_object,
                       coDistributedObject *i_data_object,
                       coDistributedObject *kd_data_object,
                       char *Triangle_out_name,
                       char *Normal_out_name,
                       char *Data_out_name,
                       coDoSet *Triangle_set_object = NULL,
                       coDoSet *Normal_set_object = NULL,
                       coDoSet *Data_set_object = NULL);

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};

#endif // _ISOS_H
