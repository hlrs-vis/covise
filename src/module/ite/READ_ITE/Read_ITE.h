/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READITE_H
#define _READITE_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for ITE data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                                                                        **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  Tue May 27 1997                                                 **
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
    char *data_Path;
    char *gridelem_Path; //Fuer File.ELE/mnt/stu/pr/rus00390/saeed/ite/mnt/stu/pr/rus00390/saeed/ite
    char *Mesh;
    char *Veloc;
    char *Press;

    //  Local data
    int n_coord, n_elem; //Anzahl in File
    int n_co, n_el; //Anzahl der benoetigten Elemente/Koordinaten
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

public:
    Application(int argc, char *argv[])
    {

        Mesh = 0L;
        Veloc = 0L;
        Press = 0L;

        Covise::set_module_description("Read data from ITE");
        Covise::add_port(OUTPUT_PORT, "mesh", "coDoUnstructuredGrid", "Grid");
        Covise::add_port(OUTPUT_PORT, "velocity", "Vec3", "velocity");
        Covise::add_port(OUTPUT_PORT, "pressure", "coDoFloat", "pressure");

        Covise::add_port(PARIN, "grid_path", "Browser", "Grid file path");
        Covise::add_port(PARIN, "data_path", "Browser", "Data file path");
        Covise::add_port(PARIN, "gridelem_path", "Browser", "GridElem file path");

        //Covise::add_port(PARIN,"numt","Scalar","Number of Timesteps");

        Covise::set_port_default("grid_path", "~/ite/team24.cor  *cor");
        Covise::set_port_default("data_path", "~/ite/team24.ini  *ini");
        Covise::set_port_default("gridelem_path", "~/ite/team24.ele  *ele");

        //Covise::set_port_default("numt","1");

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
#endif // _READITE_H
