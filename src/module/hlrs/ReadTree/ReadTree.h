/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_TREE_H
#define _READ_TREE_H

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description: Read module for Tree data                                 **
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
#include <util/coviseCompat.h>
#include <do/coDoText.h>
extern char *readTree(const char *fileName, int ignore_lengths, bool useColoring);
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

public:
    Application(int argc, char *argv[])

    {
        err_status = 0;
        count = 1;

        Path = 0L;
        if (strcmp(argv[0], "ReadTree") == 0)
        {
            Covise::set_module_description("Read data from tree file and convert it to iv");
            Covise::add_port(OUTPUT_PORT, "descr", "Text_Iv", "Iv Description");
            Covise::add_port(PARIN, "path", "Browser", "file path");
            Covise::set_port_default("path", "/usr/local/data/test.tree *.tree");
            Covise::add_port(PARIN, "colors", "Boolean", "Use depth coloring");
            Covise::set_port_default("colors", "TRUE");
            Covise::add_port(PARIN, "length", "Boolean", "Ingore Length");
            Covise::set_port_default("length", "FALSE");
        }
        else
        {
            Covise::set_module_description("Read Cassified Volumes data");
            Covise::add_port(OUTPUT_PORT, "descr", "coDoText_Cv", "Cv Description");
            Covise::add_port(PARIN, "path", "Browser", "file path");
            Covise::set_port_default("path", "data/brainsmall.cv *.cv");
        }
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
