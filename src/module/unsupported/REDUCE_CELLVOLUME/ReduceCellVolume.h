/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPLICATION_H
#define _APPLICATION_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description:  COVISE Cell_Reduce application module                    **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Oliver Heck                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  16.05.95  V1.0                                                  **
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
    static void quitCallback(void *userData, void *callbackData);

    // private member functions
    //
    void compute(void *callbackData);
    void quit(void *callbackData);

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Reduces the Volume of a Cell");
        Covise::add_port(INPUT_PORT, "meshIn", "coDoUnstructuredGrid", "input mesh");
        Covise::add_port(OUTPUT_PORT, "meshOut", "coDoPolygons", "Reduced_Cells");
        Covise::add_port(PARIN, "factor", "Slider", "Reduction Factor");
        Covise::set_port_default("factor", "0.0 1.0 0.5");
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
