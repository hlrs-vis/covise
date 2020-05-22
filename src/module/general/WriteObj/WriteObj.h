/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WRITE_OBJ_H
#define _WRITE_OBJ_H

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

class Application
{

private:
    //  field names
    //static const char *fieldNames[];

    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    //  Private class functions

    void writeObject(const coDistributedObject *geomObj,
                     const coDistributedObject *colorObj,
                     FILE *file,
                     bool singleLines,
                     int &vertexOffset);

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Generate Wavefront obj file from Covise Object");

        Covise::add_port(PARIN, "path", "Browser", "Output file path");
        Covise::set_port_default("path", "/var/tmp/ *.obj");
        Covise::add_port(PARIN, "new", "Boolean", "New File");
        Covise::set_port_default("new", "TRUE");
        Covise::add_port(PARIN, "singleLines", "Boolean", "Write each single connection as a separate line segment");
        Covise::set_port_default("singleLines", "FALSE");
        Covise::add_port(INPUT_PORT, "dataIn0", "Polygons|Lines", "Data Input");
        Covise::set_port_required("dataIn0", 1);
        Covise::add_port(INPUT_PORT, "dataIn1", "RGBA", "Vertex Colors");
        Covise::set_port_required("dataIn1", 0);

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
#endif // _WRITE_OBJ_H
