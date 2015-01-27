/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READIHS_H
#define _READIHS_H
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
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>

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
    char *Veloc;
    char *Press;
    char *K_name;
    char *EPS_name;
    char *B_U_name;
    char *STR_name;

    //  Local data
    int n_coord, n_elem;
    int *el, *vl, *tl;
    float *x_coord;
    float *y_coord;
    float *z_coord;
    float *str;
    float *eps;
    float *b_u;
    float *k;
    float *u, *v, *w;
    float *p;

    //  Shared memory data
    coDoUnstructuredGrid *mesh;
    coDoVec3 *veloc;
    coDoFloat *press;
    coDoFloat *K;
    coDoFloat *EPS;
    coDoFloat *B_U;
    coDoFloat *STR;

public:
    Application(int argc, char *argv[])

    {

        Mesh = 0L;
        Veloc = 0L;
        Press = 0L;
        K_name = 0L;
        EPS_name = 0L;
        B_U_name = 0L;
        STR_name = 0L;
        Covise::set_module_description("Read data from Ihs FENFLOSS");
        Covise::add_port(OUTPUT_PORT, "mesh", "UnstructuredGrid", "Grid");
        Covise::add_port(OUTPUT_PORT, "velocity", "Vec3", "velocity");
        Covise::add_port(OUTPUT_PORT, "pressure", "Float", "pressure");
        Covise::add_port(OUTPUT_PORT, "K", "Float", "K");
        Covise::add_port(OUTPUT_PORT, "EPS", "Float", "EPS");
        Covise::add_port(OUTPUT_PORT, "B_U", "Float", "B_U");
        Covise::add_port(OUTPUT_PORT, "NUt", "Float", "Nut");
        Covise::add_port(PARIN, "grid_path", "Browser", "Fidap Neutral file path");
        //Covise::add_port(PARIN,"data_path","Browser","Data file path");
        Covise::set_port_default("grid_path", "data/fidap/bosch/FDNEUT *NEU*");
        //Covise::set_port_default("data_path","data/saug3.sim.1 *.sim*");
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
