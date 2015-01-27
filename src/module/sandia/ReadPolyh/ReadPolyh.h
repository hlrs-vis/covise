/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READPOLYH_H
#define _READPOLYH_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for POLYH data         	                  **
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
 ** Date:  26.07.97  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define GAMMA_FLAG 1
#define PARAGON_FLAG 2
#define NCUBE_FLAG 4
#define PCTH_FLAG 8
#define PAGOSA_FLAG 16
#define DECIMATE_FLAG 32
#define DISAMBIGUATE_FLAG 64
#define THRESH_RANGE_FLAG 128
#define GRAINS_FLAG 256
#define HYDROBLOCK_FLAG 512
#define AVG_NORM_FLAG 1024

#define NONEXISTANT -1
#define FILENAME_SIZE 150

#define FAILURE 0
#define SUCCESS 1

#define XAXIS 0
#define YAXIS 1
#define ZAXIS 2

#define EPSILON .1e-10

typedef struct Point
{
    float x;
    float y;
    float z;
} Point;

typedef struct he
{
    int num_processors;
    int num_thresholds;
    int vol_cnt;
    int volume_code;
    int flags;
} Header;

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
    float *vx_data;
    float *vy_data;
    float *vz_data;

    //  Shared memory data
    coDoPolygons *mesh;
    coDoVec3 *data;

public:
    Application(int argc, char *argv[])

    {

        Data = 0L;
        data = 0L;
        Mesh = 0L;
        mesh = 0L;
        Covise::set_module_description("Read POLYH data from Sandia");
        Covise::add_port(OUTPUT_PORT, "mesh", "coDoPolygons", "Grid");
        Covise::add_port(OUTPUT_PORT, "data", "DO_Unstructured_V3D_Normals", "Normals");
        Covise::add_port(PARIN, "data_path", "Browser", "Data file path");
        Covise::add_port(PARIN, "numt", "Scalar", "Nuber of Timesteps");
        Covise::add_port(PARIN, "isonum", "Scalar", "Id of Isosurface to read");
        Covise::set_port_default("data_path", "data/sandia/viz_out.0 *");
        Covise::set_port_default("numt", "2");
        Covise::set_port_default("isonum", "0");
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
