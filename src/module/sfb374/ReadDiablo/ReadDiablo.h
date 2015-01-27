/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READDIABLO_H
#define _READDIABLO_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Diablo data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                              Christoph Kunz                            **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.03.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

class Application
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    //  Parameter names
    char *haerte_path;
    char *mesh_path;
    char *Mesh;
    char *Data;

    //  Local data
    int n_coord, n_elem;
    int *el, *vl, *tl;
    float *x_coord;
    float *y_coord;
    float *z_coord;
    float *dat;

    //  Shared memory data
    coDoUnstructuredGrid *mesh;
    coDoFloat *data;

public:
    Application(int argc, char *argv[])

    {

        Mesh = 0L;
        Data = 0L;

        Covise::set_module_description("Simulierter Haertekoerper enlesen");

        Covise::add_port(OUTPUT_PORT, "mesh", "coDoUnstructuredGrid", "grid");
        Covise::add_port(OUTPUT_PORT, "data", "coDoFloat", "data");

        Covise::add_port(PARIN, "mesh_path", "Browser", "Mesh file path");
        Covise::add_port(PARIN, "haerte_path", "Browser", "Data file path");

        Covise::set_port_default("mesh_path",
                                 "covise/data/FDNEUT.chrissi *");
        Covise::set_port_default("haerte_path",
                                 "covise/data/HAERTE.chrissi *");

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
#endif // _READDiablo_H
