/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_IV_H
#define _READ_IV_H

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description: Read module for Inventor data                             **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                                (C) 1995                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Author: D. Rantzau,Uwe Woessner                                        **
 ** Date:   14.02.95  V1.0                                                 **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

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
    int err_status;
    char *Path;
    char *IvDescr;

    //  Shared memory data
    coDoText *descr;

    //  Local data
    int count;

    int openFile();
    char *filename;
    FILE *fp;

public:
    Application(int argc, char *argv[])

    {
        err_status = 0;
        count = 1;

        Path = 0L;
        Covise::set_module_description("Read data from Inventor file");
        Covise::add_port(OUTPUT_PORT, "grid", "coDoUnstructuredGrid", "Grid");
        //			 Covise::set_port_default("path","/usr/share/data/models/chair.iv *.iv");
        Covise::add_port(OUTPUT_PORT, "data1", "coDoVec3", "Vector data");
        Covise::add_port(OUTPUT_PORT, "data2", "coDoFloat", "Scalar data");
        Covise::add_port(PARIN, "path", "Browser", "file path");
        Covise::set_port_default("path", "~ *");
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
#endif // _READ_IV_H
