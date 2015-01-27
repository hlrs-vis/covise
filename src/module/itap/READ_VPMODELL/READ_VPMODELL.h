/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#ifndef _READ_VPMODELL2
#define _READ_VPMODELL2

// includes

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#include <iostream.h>
#include <string.h>
#include <fstream.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

class Application
{
private:
    // callback-stuff
    static void computeCallback(void *userData, void *callbackData);

    // main
    void compute();

    // parameters fuer die Eingabe
    char *datapath;
    int n1_n2_n3[3];
    float o1_o2_o3[3];
    float d1_d2_d3[3];
    int n1, n2, n3, n_bytes; //n4,time,
    float o1, o2, o3; //,o4;
    float d1, d2, d3; //,d4;

public:
    // Hier werden Die Modul definiert und bezeichnet
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("READ_VPMODELL,reads binary data ( int or float ) and builds a 3D uniform grid");

        // Die hier kommen unten heraus
        Covise::add_port(OUTPUT_PORT, "grid", "Set_UniformGrid", "grid out");

        Covise::add_port(OUTPUT_PORT, "data", "Set_Float", "data out");

        // Die sollen oben herein
        // file einlesen
        Covise::add_port(PARIN, "datapath", "Browser", "file Datapath");
        Covise::set_port_default("datapath", "/mnt/mit/ext/pr/lka11112/mkarb/VIS/_vpmodel.H@");

        Covise::add_port(PARIN, "n1_n2_n3", "Vector", "...Grid size");
        Covise::set_port_default("n1_n2_n3", "215 432 459");

        Covise::add_port(PARIN, "o1_o2_o3", "Vector", "origin");
        Covise::set_port_default("o1_o2_o3", "000 -244 -186");

        Covise::add_port(PARIN, "d1_d2_d3", "Vector", "deltas");
        Covise::set_port_default("d1_d2_d3", "1 1 1");

        /*			//Covise::add_port(PARIN, "n4", "Scalar","...number of time steps");
                                 //Covise::set_port_default("n4","4");

                                 Covise::add_port(PARIN, "o4", "Scalar", "time 0");
                                 Covise::set_port_default("o4","10.8");

                                 Covise::add_port(PARIN, "d4", "Scalar", "delta time");
                                 Covise::set_port_default("d4","10.8");

                                 //Covise::add_port(PARIN, "time", "Scalar", "time step");
                                 //Covise::set_port_default("time","0");
         Covise::add_port(PARIN, "timestep", "Slider", "time step");
         Covise::set_port_default("timestep", "0 4 0");
         */

        Covise::add_port(PARIN, "n_bytes", "Scalar",
                         "1 byte for int or 4 for float data type");
        Covise::set_port_default("n_bytes", "4");

        Covise::init(argc, argv);
        Covise::set_start_callback(Application::computeCallback, this);
    }

    void run()
    {
        Covise::main_loop();
    }
};
#endif // _READ_VPMODELL2
