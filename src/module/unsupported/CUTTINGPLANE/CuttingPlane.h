/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPLICATION_H
#define _APPLICATION_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE ColorMap application module                       **
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
#include <stdlib.h>

class Application
{

private:
    // callback stub functions
    //
    static void computeCallback(void *userData, void *callbackData);
    static void paramCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    //  Routines
    void create_strgrid_plane();
    void create_rectgrid_plane();
    void create_unigrid_plane();
    void create_scalar_plane();
    void create_vector_plane();
    // private member functions
    //
    void compute(void *callbackData);
    void param(void *callbackData);
    void quit(void *callbackData);
    void handle_objects(coDistributedObject *mesh, coDistributedObject *data, char *Outgridname, char *Outdataname, coDistributedObject **grid_set_out = NULL, coDistributedObject **data_set_out = NULL);
    int calctype; //choice value (which output to generate)
public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Extract a plane from a structured data set");
        Covise::add_port(INPUT_PORT, "meshIn", "Set_StructuredGrid|Set_RectilinearGrid|Set_UniformGrid", "input mesh");
        Covise::add_port(INPUT_PORT, "dataIn", "Set_Float|Set_Vec3", "input data");
        Covise::add_port(OUTPUT_PORT, "meshOut", "Set_StructuredGrid|Set_RectilinearGrid|Set_UniformGrid", "Cuttingplane");
        Covise::add_port(OUTPUT_PORT, "dataOut", "Set_Float|Set_Vec3", "interpolated data");
        Covise::add_port(PARIN, "plane", "Choice", "cutting direction");
        Covise::add_port(PARIN, "i_index", "Slider", "value of i-index");
        Covise::add_port(PARIN, "j_index", "Slider", "value of j-index");
        Covise::add_port(PARIN, "k_index", "Slider", "value of k-index");
        Covise::set_port_default("plane", "1 cut_along_i cut_along_j cut_along_k");
        Covise::set_port_default("i_index", "1 70  1");
        Covise::set_port_default("j_index", "1 44  1");
        Covise::set_port_default("k_index", "1 113 1");
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
