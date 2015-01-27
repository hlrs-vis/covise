/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__TESTBED_H)
#define __TESTBED_H

// includes
#include <appl/ApplInterface.h>
using namespace covise;
#include <appl/CoviseReadModule.h>

class Testbed : public CoviseReadModule
{
protected:
    // the parameters
    int outputMesh, outputData;
    long int u, v, w;
    float vertex[3];

    // build the output
    void buildOutput(char **);

public:
    Testbed(int argc, char *argv[])
    {
        Covise::set_module_description("generate testbed data-sets");

        Covise::add_port(OUTPUT_PORT, "mesh", "UnstructuredGrid|Points|Lines|Polygons|TriangleStrips", "input grid");
        Covise::add_port(OUTPUT_PORT, "vectorData_pe", "Vec3", "vectordata per element");
        Covise::add_port(OUTPUT_PORT, "vectorData_pp", "Vec3", "vectordata per primitve");
        Covise::add_port(OUTPUT_PORT, "scalarData_pe", "Float", "scalardata per element");
        Covise::add_port(OUTPUT_PORT, "scalarData_pp", "Float", "scalardata per primitive");

        Covise::add_port(PARIN, "outputMesh", "Choice", "see console output for description");
        Covise::set_port_default("outputMesh", "1 cuboid points lines");

        Covise::add_port(PARIN, "outputData", "Choice", "see console output for description");
        Covise::set_port_default("outputData", "1 circular");

        Covise::add_port(PARIN, "u", "IntScalar", "see console output for description");
        Covise::set_port_default("u", "10");

        Covise::add_port(PARIN, "v", "IntScalar", "see console output for description");
        Covise::set_port_default("v", "10");

        Covise::add_port(PARIN, "w", "IntScalar", "see console output for description");
        Covise::set_port_default("w", "10");

        Covise::add_port(PARIN, "vertex", "FloatVector", "see console output for description");
        Covise::set_port_default("vertex", "0.0 0.0 0.0");

        Covise::init(argc, argv);

        setCallbacks();

        return;
    };

    Testbed(){};

    void compute(const char *port);

    void run()
    {
        Covise::main_loop();
    }
};
#endif // __TESTBED_H
