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
    static void quitCallback(void *userData, void *callbackData);

    // private member functions
    //
    void compute(void *callbackData);
    void quit(void *callbackData);

    // private data

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Trace particles in flow fields");
        Covise::add_port(INPUT_PORT, "mesh", "coDoRectilinearGrid|coDoStructuredGrid", "input mesh");
        Covise::add_port(INPUT_PORT, "velocity", "coDoVec3", "velocity");
        Covise::add_port(OUTPUT_PORT, "lines", "coDoLines", "particle lines");
        Covise::add_port(PARIN, "istart", "Choice", "starting mode");
        Covise::add_port(PARIN, "istep", "Choice", "step type");
        Covise::add_port(PARIN, "dl", "Scalar", "step value (it depends on istep)");
        Covise::add_port(PARIN, "delta", "Scalar", "angular tolerance (used if > 0)");
        Covise::add_port(PARIN, "npmax", "Scalar", "max no. of points to be generated");
        Covise::add_port(PARIN, "step", "Scalar", "step between starting positions");
        Covise::add_port(PARIN, "start1", "Vector", "starting position 1");
        Covise::add_port(PARIN, "start2", "Vector", "starting position 2");
        Covise::add_port(PARIN, "color", "String", "named color for lines");
        Covise::set_port_default("istart", "1 IJK XYZ");
        Covise::set_port_default("istep", "2 space time");
        Covise::set_port_default("dl", "0.02");
        Covise::set_port_default("delta", "0.1");
        Covise::set_port_default("npmax", "1000");
        Covise::set_port_default("step", "1");
        Covise::set_port_default("start1", "3 8 11");
        Covise::set_port_default("start2", "4 25 12");
        Covise::set_port_default("color", "red");
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
