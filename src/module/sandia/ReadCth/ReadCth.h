/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READCTH_H
#define _READCTH_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for PCTH data         	                  **
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
 ** Date:  17.07.97  V1.0                                                  **
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
    char *data_Path;
    char *Data;
    char *Mesh;

    //  Local data
    int xdim, ydim, zdim;
    float *s_data;

    //  Shared memory data
    coDoRectilinearGrid *mesh;
    coDoFloat *data;

public:
    Application(int argc, char *argv[])

    {

        Data = 0L;
        data = 0L;
        Mesh = 0L;
        mesh = 0L;
        Covise::set_module_description("Read CTH data from Sandia");
        Covise::add_port(OUTPUT_PORT, "mesh", "coDoUniformGrid", "Grid");
        Covise::add_port(OUTPUT_PORT, "data", "coDoFloat", "Data");
        Covise::add_port(PARIN, "data_path", "Browser", "Data file path");
        Covise::add_port(PARIN, "numt", "Scalar", "Nuber of Timesteps");
        Covise::add_port(PARIN, "numb", "Scalar", "Nuber of Blocks");
        Covise::add_port(PARIN, "currb", "Scalar", "Current Block");
        Covise::add_port(PARIN, "varnum", "Scalar", "variable to extract");
        Covise::set_port_default("data_path", "data/sandia/viz_out.0 *");
        Covise::set_port_default("currb", "0");
        Covise::set_port_default("numt", "2");
        Covise::set_port_default("numb", "1");
        Covise::set_port_default("varnum", "2");
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
#endif // _READIHS_H
