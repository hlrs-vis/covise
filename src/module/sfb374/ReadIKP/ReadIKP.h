/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READIKP_H
#define _READIKP_H
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

#define MAXT 100

class Application
{

private:
    static const int maxtimesteps;

    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);
    void init_module();
    int get_meshheader();
    int get_nodes();
    void get_mesh();
    int get_timestep();

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    //  Parameter names
    char *data_path;
    char *Mesh;
    char *Data;
    int min, max, timestep;

    //  Local data
    char lastname[300];
    int n_coord, n_elem, n_dimension, n_conn, n_groups, n_timesteps;
    int *el, *vl, *tl, *elsav, *vlsav, *tlsav;
    int timestepped, Forces_choice;
    float *x_coord_start;
    float *y_coord_start;
    float *z_coord_start;
    float *dat_start;
    FILE *data_fp;

    //  Displacement memory data
    long timepos[MAXT];
    float *x_nodes, *y_nodes, *z_nodes;

    //  Marc File Format variables
    int NDISTL, NSET, NSPRNG, NDIE, NSTRES, INUM, NNODMX;

    //  Shared memory data
    coDoUnstructuredGrid *mesh;
    coDoFloat *data;

public:
    Application(int argc, char *argv[])

    {

        Mesh = 0L;
        Data = 0L;
        timestepped = 0;
        lastname[0] = '0';
        x_nodes = NULL;
        y_nodes = NULL;
        z_nodes = NULL;

        Covise::set_module_description("IKP 2D Daten einlesen");

        Covise::add_port(OUTPUT_PORT, "mesh", "coDoUnstructuredGrid", "grid");
        Covise::add_port(OUTPUT_PORT, "data", "coDoFloat", "data");

        Covise::add_port(PARIN, "data_path", "Browser", "Data file path");
        Covise::add_port(PARIN, "TimeStep", "Slider", "Time Stepper");
        Covise::add_port(PARIN, "Forces", "Choice", "Forces choice");

        Covise::set_port_default("data_path",
                                 "covise/data/sfb374/ikp/t10_2.t19 *");
        Covise::set_port_default("TimeStep", "0 100 0");
        Covise::set_port_default("Forces", "1 external reaction");

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
#endif // _READIKP_H
