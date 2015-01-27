/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READS_H
#define _READS_H

/**************************************************************************\
**                                                           (C)1994 RUS  **
**                                                                        **
** Description: Read module for Hermes model data                         **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author:                                                                **
**                                                                        **
**                             Dirk Rantzau                               **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **
**                                                                        **
** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <do/coDoSet.h>
#include <do/coDoPoints.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoData.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif

class Application
{

private:
    // member functions
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    //  Parameter names

    //  Local data

    //  Shared memory data
    coDoSet *p_set, *v_set, *r_set, *g_set;
    coDoPoints *points;
    coDoStructuredGrid *s_grid;
    coDoVec3 *velocity;
    coDoFloat *radius;

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Read data from ica1");
        Covise::add_port(OUTPUT_PORT, "points", "Points", "points");
        Covise::add_port(OUTPUT_PORT, "grid", "StructuredGrid", "grid");
        Covise::add_port(OUTPUT_PORT, "velocity", "Vec3", "velocity");
        Covise::add_port(OUTPUT_PORT, "radius", "Float", "radius");
        Covise::add_port(PARIN, "path", "Browser", "Data file path");
        Covise::set_port_default("path", "data/ica1/3D.dat *dat*");
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

#endif // _READP_H
