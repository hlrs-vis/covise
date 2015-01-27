/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _S_TRACER_H
#define _S_TRACER_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Tracer for structured grids	         	          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Andreas Werner                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  03.01.97  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

#if defined(__sgi) || defined(__linux__)
#define _USE_EXCEPTIONS_
#endif

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

    //  Local data
    int runCnt_;
    float crit_angle;
    //  Shared memory data

public:
    enum
    {
        V_MAG = 1,
        V_X = 2,
        V_Y = 3,
        V_Z = 4,
        NUMBER = 5,
        TIME = 6
    };

    Application(int argc, char *argv[])
        : runCnt_(0)
    {
        Covise::set_module_description("Particle Tracer for block-structured grids");

        // two outputs: Mesh and Surface
        Covise::add_port(INPUT_PORT, "meshIn", "StructuredGrid|UniformGrid|RectilinearGrid", "The cartesian mesh");
        Covise::add_port(INPUT_PORT, "dataIn", "Vec3", "The velocity");
        Covise::add_port(OUTPUT_PORT, "lines", "Lines", "Streamline");
        Covise::add_port(OUTPUT_PORT, "dataOut", "Float", "Output data");
        Covise::add_port(OUTPUT_PORT, "velocOut", "Vec3", "Output velocity");
        Covise::add_port(PARIN, "no_startp", "IntSlider", "number of startpoints");
        Covise::add_port(PARIN, "startpoint1", "FloatVector", "Startpoint1 for Streamline");
        Covise::add_port(PARIN, "startpoint2", "FloatVector", "Startpoint2 for Streamline");
        Covise::add_port(PARIN, "direction", "Choice", "direction of interpolation");
        Covise::add_port(PARIN, "whatout", "Choice", "Component of output data");
        Covise::add_port(PARIN, "cellfr", "FloatScalar", "Runge-Kutta-Konstante");
        Covise::add_port(PARIN, "crit_angle", "FloatScalar", "Critical angle (in degrees) for wall detection");
        Covise::set_port_default("no_startp", "1 10 2");
        Covise::set_port_default("startpoint1", "1.0 1.0 1.0");
        Covise::set_port_default("startpoint2", "1.0 2.0 1.0");
        Covise::set_port_default("direction", "3 forward backward both");
        Covise::set_port_default("whatout", "1 mag v_x v_y v_z number time");
        Covise::set_port_default("cellfr", "0.5");
        Covise::set_port_default("crit_angle", "9.9");

        // Do the setup
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

class WristWatch
{
private:
    timeval myClock;

public:
    WristWatch();
    ~WristWatch();

    void start();
    void stop(char *s);
};
#endif // _S_TRACER_H
