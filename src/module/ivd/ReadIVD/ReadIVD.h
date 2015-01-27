/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READIVD_H
#define _READIVD_H
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
#include <do/coDoRectilinearGrid.h>
#include <do/coDoData.h>
#include <stdlib.h>
#include <stdio.h>

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
    char *data_Path;
    char *Mesh;
    char *Velocity;
    char *Temperature;

    //  Local data
    int n_u, n_v, n_w;
    float *x_coord;
    float *y_coord;
    float *z_coord;
    float *vx;
    float *vy;
    float *vz;
    float *tem;

    //  Shared memory data
    coDoRectilinearGrid *mesh;
    coDoVec3 *veloc;
    coDoFloat *temp;

public:
    Application(int argc, char *argv[])

    {

        Mesh = 0L;
        Velocity = 0L;
        Temperature = 0L;
        Covise::set_module_description("Read FLITE data");
        Covise::add_port(OUTPUT_PORT, "mesh", "coDoRectilinearGrid", "Grid");
        Covise::add_port(OUTPUT_PORT, "velocity", "coDoVec3", "velocity");
        Covise::add_port(OUTPUT_PORT, "temperature", "coDoFloat", "temperature");
        Covise::add_port(PARIN, "grid_path", "Browser", "Grid file path");
        Covise::set_port_default("grid_path", "data/ivd/Geometry *");
        Covise::add_port(PARIN, "data_path", "Browser", "Data file path");
        Covise::set_port_default("data_path", "data/ivd/Timestep01 *");
        Covise::add_port(PARIN, "numt", "FloatScalar", "Number of Timesteps");
        Covise::set_port_default("numt", "100");
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
