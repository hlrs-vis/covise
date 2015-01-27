/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AVS_Triangles_H
#define _AVS_Triangles_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface reduction application module              **
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
 ** Date:  January 1998  V1.0                                              **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

class RW_TriangleMesh
{
private:
    int Data;
    int num_tri, num_vert, num_points;
    int *tri_list;
    int *vertex_list;
    float *coords_x;
    float *coords_y;
    float *coords_z;

    float *data;

public:
    RW_TriangleMesh()
    {
    }
    RW_TriangleMesh(int n_t, int n_v, int n_c, int *tl, int *vl, float *cx, float *cy, float *cz, float *dt)
    {
        int i;
        tri_list = new int[n_t];
        vertex_list = new int[n_v];
        coords_x = new float[n_c];
        coords_y = new float[n_c];
        coords_z = new float[n_c];
        if (dt)
        {
            data = new float[n_c];
            Data = 1;
        }
        else
            Data = 0;
        num_tri = n_t;
        num_vert = n_v;
        num_points = n_c;
        for (i = 0; i < n_t; i++)
            tri_list[i] = tl[i];
        for (i = 0; i < n_v; i++)
            vertex_list[i] = vl[i];
        for (i = 0; i < n_c; i++)
        {
            coords_x[i] = cx[i];
            coords_y[i] = cy[i];
            coords_z[i] = cz[i];
            if (dt)
                data[i] = dt[i];
        }
    }

    int Write_TriangleMesh(char *filename);
    int Read_TriangleMesh(char *filename);
    void CreatecoDistributedObjects(char *Mesh_out_name, char *Data_out_name);
    ~RW_TriangleMesh()
    {
        delete[] tri_list;
        delete[] vertex_list;
        delete[] coords_x;
        delete[] coords_y;
        delete[] coords_z;
        if (Data)
            delete[] data;
    }
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

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Reads or writes triangle meshes in the INDEX exchange format");
        Covise::add_port(INPUT_PORT, "meshIn", "Polygons", "Geometry to write");
        Covise::add_port(INPUT_PORT, "dataIn", "Float|Vec3", "Vertice-attached data");
        Covise::add_port(OUTPUT_PORT, "meshOut", "Polygons", "The read geometry");
        Covise::add_port(OUTPUT_PORT, "dataOut", "Float|Vec3", "The interpolated scalar data");
        Covise::add_port(PARIN, "filename", "Browser", "File name, idx suffix assumes idx file format, otherise ucd file format");

        Covise::set_port_default("filename", "data/index/ear.idx *.idx");
        Covise::set_port_required("meshIn", 0);
        Covise::set_port_required("dataIn", 0);
        Covise::init(argc, argv);
        Covise::set_start_callback(Application::computeCallback, this);
        Covise::set_quit_callback(Application::quitCallback, this);
    }

    void HandleObjects(const coDistributedObject *mesh_object, const coDistributedObject *data_object, char *Mesh_out_name, char *Data_out_name, char *filename);

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _AVS_Triangles_H
