/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RWUSG_H
#define _RWUSG_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for COVISE USG data        	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.11.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

typedef struct
{
    int n_elem;
    int n_conn;
    int n_coord;
} USG_HEADER;

typedef struct
{
    int xs;
    int ys;
    int zs;
} STR_HEADER;

class Application
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);
    coDistributedObject *readData(char *Name);
    void readattrib(coDistributedObject *tmp_Object);
    void writeattrib(coDistributedObject *tmp_Object);
    void writeobj(coDistributedObject *tmp_Object);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    //  Parameter names
    char *grid_Path;
    char *Mesh;
    char *Mesh_in;

    //  Local data
    int n_coord, n_elem, n_conn;
    int *el, *vl, *tl;
    float *x_coord;
    float *y_coord;
    float *z_coord;

    //  Shared memory data
    coDoUniformGrid *ugrid;
    coDoPolygons *pol;
    coDoTriangleStrips *tri;
    coDoRectilinearGrid *rgrid;
    coDoStructuredGrid *sgrid;
    coDoUnstructuredGrid *mesh;
    coDoFloat *us3d;
    coDoVec3 *us3dv;
    coDoFloat *s3d;
    coDoVec3 *s3dv;

public:
    Application(int argc, char *argv[])
    {
        Mesh = NULL;
        Covise::set_module_description("Read OR Write COVISE Data");
        Covise::add_port(INPUT_PORT, "mesh_in", "UnstructuredGrid|RectilinearGrid|StructuredGrid|Float|Vec3|Float|Vec3|Polygons|TriangleStrips|Geometry|Lines", "mesh_in");
        Covise::set_port_required("mesh_in", 0);
        Covise::add_port(OUTPUT_PORT, "mesh", "UnstructuredGrid|RectilinearGrid|StructuredGrid|Float|Vec3|Float|Vec3|Polygons|TriangleStrips|Geometry|Lines", "mesh");
        Covise::add_port(PARIN, "grid_path", "Browser", "File path");
        Covise::set_port_default("grid_path", "../ *.covise");
        Covise::add_port(PARIN, "timestep", "IntSlider", "blabla");
        Covise::set_port_default("timestep", "0 0 0");

        Covise::init(argc, argv);
        Covise::set_quit_callback(Application::quitCallback, this);
        Covise::set_start_callback(Application::computeCallback, this);
    }

    void parseStartStep(char *, int *, int *);

    void run()
    {
        Covise::main_loop();
    }
    ~Application()
    {
    }
};
#endif // _RWUSG_H
