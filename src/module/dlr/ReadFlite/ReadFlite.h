/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READFLITE_H
#define _READFLITE_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Ihs data         	                  **
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
    char *grid_Path;
    char *Mesh;
    char *Surface;
    char *Rho;
    char *Rhou;
    char *Rhov;
    char *Rhow;
    char *Rhoe;

    //  Local data
    int n_coord, n_elem, n_triang;
    int *el, *vl, *tl;
    float *x_coord;
    float *y_coord;
    float *z_coord;
    float *x_coord2;
    float *y_coord2;
    float *z_coord2;
    float *ru, *rv, *rw, *re, *r;

    //  Shared memory data
    coDoUnstructuredGrid *mesh;
    coDoFloat *rho;
    coDoFloat *rhou;
    coDoFloat *rhov;
    coDoFloat *rhow;
    coDoFloat *rhoe;
    coDoPolygons *surface;

public:
    Application(int argc, char *argv[])

    {

        Mesh = 0L;
        Surface = 0L;
        Rho = 0L;
        Rhou = 0L;
        Rhov = 0L;
        Rhow = 0L;
        Rhoe = 0L;
        Covise::set_module_description("Read FLITE data");
        Covise::add_port(OUTPUT_PORT, "mesh", "coDoUnstructuredGrid", "Grid");
        Covise::add_port(OUTPUT_PORT, "surface", "coDoPolygons", "Surface");
        Covise::add_port(OUTPUT_PORT, "rho", "coDoFloat", "rho");
        Covise::add_port(OUTPUT_PORT, "rhou", "coDoFloat", "rhou");
        Covise::add_port(OUTPUT_PORT, "rhov", "coDoFloat", "rhov");
        Covise::add_port(OUTPUT_PORT, "rhow", "coDoFloat", "rhow");
        Covise::add_port(OUTPUT_PORT, "rhoe", "coDoFloat", "rhoe");
        Covise::add_port(PARIN, "grid_path", "Browser", "Grid file path");
        Covise::set_port_default("grid_path", "data/F16/test.flite *.flite");
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
#endif // _READFLITE_H
