/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GEOMREDUCE_H
#define _GEOMREDUCE_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface reduction application module              **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  April 1997  V1.0                                                **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include "HandleSet.h"
#include <stdlib.h>

extern float *x_in, *y_in, *z_in, *s_in, *u_in, *v_in, *w_in;

class Application : public Covise_Set_Handler
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

public:
    // for sets
    coDistributedObject **ComputeObject(coDistributedObject **, char **, int);

    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Reduces the number of polygons in a mesh");
        Covise::add_port(INPUT_PORT, "meshIn", "Set_Polygons|Set_TriangleStrips", "Geometry");
        Covise::add_port(INPUT_PORT, "normalsIn", "Set_Unstructured_V3D_Normals", "Vertice-attached normals");
        Covise::add_port(OUTPUT_PORT, "meshOut", "Set_Polygons|Set_TriangleStrips", "The reduced geometry");
        Covise::add_port(OUTPUT_PORT, "normalsOut", "Set_Unstructured_V3D_Normals", "The interpolated normals");
        Covise::add_port(PARIN, "percent", "Scalar", "Percentage of triangles to be removed");
        Covise::set_port_default("percent", "30");
        Covise::add_port(PARIN, "feature_angle", "Scalar", "Feature angle to be preserved");
        Covise::set_port_default("feature_angle", "40.0");
        Covise::add_port(PARIN, "new_point", "Choice", "New point calculation strategy");
        Covise::set_port_default("new_point", "2 midpoint volume_preserving endpoint");
        Covise::init(argc, argv);
        Covise::set_start_callback(Application::computeCallback, this);
        Covise::set_quit_callback(Application::quitCallback, this);
    }

    coDistributedObject **HandleObjects(coDistributedObject *mesh_object, coDistributedObject *normal_object, char *Mesh_out_name, char *Normal_out_name);

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _GEOMREDUCE_H
