/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__TETRAHEDRONIZE_H)
#define __TETRAHEDRONIZE_H

#include <appl/ApplInterface.h>
using namespace covise;
#include <appl/CoviseAppModule.h>

class Tetrahedronize : public CoviseAppModule
{
public:
    // initialize covise
    Tetrahedronize(int argc, char *argv[])
    {
        Covise::set_module_description("transform any unstructured grid into tetrahedrons");

        Covise::add_port(INPUT_PORT, "gridIn", "UnstructuredGrid", "grid to be transformed");
        Covise::add_port(INPUT_PORT, "dataIn", "Vec3|Float|Float|Vec3", "data on grid to be transformed");

        Covise::add_port(OUTPUT_PORT, "gridOut", "UnstructuredGrid", "grid consisting of tetrehedrons only");
        Covise::add_port(OUTPUT_PORT, "dataOut", "Vec3|Float", "data mapped to tetrahedron grid");

        Covise::add_port(PARIN, "regulate", "Boolean", "try to generate a regular grid ?");
        Covise::set_port_default("regulate", "FALSE");

        Covise::init(argc, argv);

        const char *in_names[] = { "gridIn", "dataIn", NULL };
        const char *out_names[] = { "gridOut", "dataOut", NULL };

        setPortNames(in_names, out_names);

        setCallbacks();

        return;
    };

    // nothing to clean up
    Tetrahedronize(){};

    // called whenever the module is executed
    coDistributedObject **compute(const coDistributedObject **, char **);

    // tell covise that we're up and running
    void run()
    {
        Covise::main_loop();
    }
};
#endif // __TETRAHEDRONIZE_H
