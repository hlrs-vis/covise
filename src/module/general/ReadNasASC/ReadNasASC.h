/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_DASA_H
#define _READ_DASA_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description: Read module for DASA Nastran data                         **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Andreas Wierse                             **
 **                     VirCinity IT-Consulting GmbH                       **
 **                            Schulstrsse 15                              **
 **                            71229 Warmbronn                             **
 **                                                                        **
 ** Date:  17.06.99  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

class Application
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);
    int read_length();
    int read_vertices(float *, float *, float *);
    int read_vertices2(float *, float *, float *);
    int read_tetras(int *, int *);
    int read_tris(int *, int *);
    int read_scalar(float *);
    int read_vector(float *, float *, float *);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    //  Local data
    FILE *fp;
    int numVertices, numTetras, numTris;

public:
    Application(int argc, char *argv[])

    {
        Covise::set_module_description("Read TECPLOT data");
        Covise::add_port(OUTPUT_PORT, "mesh", "UnstructuredGrid", "Gitter");
        Covise::add_port(OUTPUT_PORT, "velocity", "Vec3", "Geschwindigkeits Daten");
        Covise::add_port(OUTPUT_PORT, "pressure", "Float", "Druck Daten");
        Covise::add_port(OUTPUT_PORT, "temp", "Float", "Temperatur Daten");
        Covise::add_port(OUTPUT_PORT, "mach", "Float", "Mach-Zal");
        Covise::add_port(OUTPUT_PORT, "poly", "Polygons", "Triangle  Mesh");
        Covise::add_port(PARIN, "path", "Browser", "Filename");
        Covise::set_port_default("path", "data/DASA *");

        Covise::init(argc, argv);
        Covise::set_quit_callback(Application::quitCallback, this);
        Covise::set_start_callback(Application::computeCallback, this);
    }

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _READ_DASA_H
