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

    //  Routines
    void create_strgrid_plane();
    void create_rectgrid_plane();
    void create_unigrid_plane();
    void create_scalar_plane();
    void create_vector_plane();
    //

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Generate Histogram from a data set");
        Covise::add_port(INPUT_PORT, "Data", "Set_Float|Set_Float", "scalar data");
        Covise::add_port(OUTPUT_PORT, "2dplot", "Set_Vec2", "plotdata");
        Covise::add_port(OUTPUT_PORT, "minmax", "MinMax_Data", "minmax");
        Covise::add_port(PARIN, "NumBuck", "IntSlider", "Number of Buckets");
        Covise::set_port_default("NumBuck", "5 50 5");
        //Covise::add_port(PAROUT,"Min","Scalar","Min");
        //Covise::set_port_default("Min","0.0");
        //Covise::add_port(PAROUT,"Max","Scalar","Max");
        //Covise::set_port_default("Max","0.0");
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
