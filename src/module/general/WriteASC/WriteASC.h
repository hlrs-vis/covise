/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WRITE_ASC_H
#define _WRITE_ASC_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Generate ASCII file from Covise Object                    **
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
 ** Date:  05.01.97  V0.1                                                  **
\**************************************************************************/

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

    void writeObject(const coDistributedObject *tmp_Object, FILE *file, int verbose, int indent);

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Generate ASCII file from Covise Object");

        Covise::add_port(PARIN, "path", "Browser", "Output file path");
        Covise::set_port_default("path", "/var/tmp/ *.asc");
        Covise::add_port(PARIN, "new", "Boolean", "New File");
        Covise::set_port_default("new", "TRUE");
        Covise::add_port(PARIN, "verbose", "Boolean", "Full Representation");
        Covise::set_port_default("verbose", "TRUE");
        Covise::add_port(INPUT_PORT, "dataIn", "UnstructuredGrid|UniformGrid|Vec2|Points|Float|Float|Vec3|Vec3|Tensor|Polygons|Lines|IntArr|RGBA|Text|ColorMap", "Data Input");
        Covise::set_port_required("dataIn", 1);

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
#endif // _READSTAR_H
