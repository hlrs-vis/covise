/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_FIRE_H
#define _READ_FIRE_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for PATRAN Neutral and Results Files          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Reiner Beller                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  18.07.97  V0.1                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;

class Application
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    //  Local data
    char File_name[512];
    char quantity[512];
    int timestep;

    //  Shared memory data

public:
    /*********************************
       *                               *
       *     C O N S T R U C T O R     *
       *                               *
       *********************************/

    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Read FIRE data");

        // Fileset Name
        Covise::add_port(PARIN, "file_name", "Browser", "Fileset path");
        Covise::set_port_default("file_name", "~/bosch/mb3v *.*");

        // Parameter inputs
        Covise::add_port(PARIN, "timestep", "Scalar", "Timestep");
        Covise::set_port_default("timestep", "1");
        Covise::add_port(PARIN, "quantity", "Choice", "Quantity");
        Covise::set_port_default("quantity", "1 vel pressure temp dens vis");

        // Output
        Covise::add_port(OUTPUT_PORT, "mesh", "coDoUnstructuredGrid", "Mesh output");
        Covise::add_port(OUTPUT_PORT, "data1", "coDoVec3 | coDoFloat", "Data Field 1 output");

        // Do the setup
        Covise::init(argc, argv);
        Covise::set_quit_callback(Application::quitCallback, this);
        Covise::set_start_callback(Application::computeCallback, this);

        // Set internal object pointers to Files and Filenames
        File_name[0] = '\0';
    }

    inline void run()
    {
        Covise::main_loop();
    }
    ~Application();
};
#endif // _READ_FIRE_H
