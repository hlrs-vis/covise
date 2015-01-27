/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPLICATION_H
#define _APPLICATION_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE CuttingPlane application module                   **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau                                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

class Application
{

private:
    // callback stub functions
    //
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    // private member functions
    //
    void compute(void *callbackData);
    void quit(void *callbackData);
    int get_color_rgb(float *r, float *g, float *b, const char *color);
    void genpolygons(char *GeometryN);

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Show each element of an unstructured Grid");
        Covise::add_port(INPUT_PORT, "meshIn", "UnstructuredGrid", "input mesh");
        Covise::add_port(INPUT_PORT, "colors", "Vec3|RGBA", "colors");
        //Covise::add_port(INPUT_PORT,"group","Byte_Array","Group index");
        Covise::add_port(OUTPUT_PORT, "geometry", "Geometry", "Geometry");
        //		     Covise::add_port(PARIN,"gennormals","Boolean","Supply normals");
        //		     Covise::set_port_default("gennormals","FALSE");
        Covise::set_port_required("colors", 0);
        //Covise::set_port_required("group",0);
        Covise::init(argc, argv);
        Covise::set_start_callback(Application::computeCallback, this);
        Covise::set_quit_callback(Application::quitCallback, this);
    }

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _APPLICATION_H
