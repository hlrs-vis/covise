/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CELLTRACER_H
#define _CELLTRACER_H
/**************************************************************************\ 
 **                                                           (C)1996 RUS  **
 **                                                                        **
 ** Description:  COVISE Tracer for Cells application module	              **
 **                                                                        **
 **                                                                        **
 **                             (C) 1996                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner, Oliver Heck, Andreas Wierse                     **
 **                                                                        **
 **                                                                        **
 ** Date:  12.02.96  V1.0                                                  **
 ** Date:  13.10.98  V2.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <appl/CoviseAppModule.h>
#include <stdlib.h>

class Application : public CoviseAppModule
{

private:
    // private member functions
    //
    // called whenever the module is executed
    coDistributedObject **compute(coDistributedObject **, char **);
    float *xStart, *yStart, *zStart;

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Generate streamlines from an unstr. data set");
        Covise::add_port(INPUT_PORT, "meshIn", "DO_CellGrid", "input mesh");
        Covise::add_port(INPUT_PORT, "dataIn", "coDoVec3|coDoVec3", "input data");
        Covise::add_port(OUTPUT_PORT, "lines", "coDoLines", "Streamline");
        Covise::add_port(OUTPUT_PORT, "dataOut", "coDoFloat", "Output data");
        Covise::add_port(OUTPUT_PORT, "velocOut", "coDoVec3", "Output velocity");
        Covise::add_port(PARIN, "nbrs_comp", "Choice", "compute neighbourlist as");
        Covise::add_port(PARIN, "no_startp", "Slider", "number of startpoints");
        Covise::add_port(PARIN, "startpoint1", "Vector", "Startpoint1 for Streamline");
        Covise::add_port(PARIN, "startpoint2", "Vector", "Startpoint2 for Streamline");

        Covise::add_port(PARIN, "normal", "Vector", "...");
        Covise::set_port_default("normal", "0.0 0.0 1.0");

        Covise::add_port(PARIN, "direction", "Vector", "...");
        Covise::set_port_default("direction", "1.0 0.0 0.0");
        Covise::add_port(PARIN, "option", "Choice", "Method of interpolation");
        Covise::add_port(PARIN, "stp_control", "Choice", "stepsize control");
        Covise::add_port(PARIN, "tdirection", "Choice", "direction of interpolation");
        Covise::add_port(PARIN, "reduce", "Choice", "Point reduction on output geometry");
        Covise::add_port(PARIN, "whatout", "Choice", "Component of output data");
        Covise::add_port(PARIN, "trace_eps", "Scalar", "Epsilon, be careful with this");
        Covise::add_port(PARIN, "trace_len", "Scalar", "Maximum length of trace");
        Covise::add_port(PARIN, "startStyle", "Choice", "how to compute starting-points");
        Covise::set_port_default("startStyle", "1 line plane");
        Covise::set_port_default("nbrs_comp", "2 pre_process on_the_fly");
        Covise::set_port_default("no_startp", "1 10 2");
        Covise::set_port_default("startpoint1", "1.0 1.0 1.0");
        Covise::set_port_default("startpoint2", "1.0 2.0 1.0");
        Covise::set_port_default("option", "2 rk2 rk4");
        Covise::set_port_default("stp_control", "2 position 5<ang<12 2<ang<6");
        Covise::set_port_default("tdirection", "3 forward backward both xyplane cube");
        Covise::set_port_default("reduce", "1 off 3deg 5deg");
        Covise::set_port_default("whatout", "1 mag v_x v_y v_z number");
        Covise::set_port_default("trace_eps", "0.00001");
        Covise::set_port_default("trace_len", "1.0");
        Covise::init(argc, argv);
        char *in_names[] = { "meshIn", "dataIn", NULL };
        char *out_names[] = { "lines", "dataOut", "velocOut", NULL };
        setPortNames(in_names, out_names);

        setCallbacks();
        //Covise::set_start_callback(Application::computeCallback,this);
        //Covise::set_quit_callback(Application::quitCallback,this);
    }

    void run()
    {
        Covise::main_loop();
    }
    void computeStartPoints();

    ~Application()
    {
    }
};
#endif // _CELLTRACER_H
