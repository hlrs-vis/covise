/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__TETRAVOLUME_H)
#define __TETRAVOLUME_H

// includes
#include <appl/ApplInterface.h>
using namespace covise;
#include <appl/CoviseAppModule.h>

class TetraVolume : public CoviseAppModule
{

public:
    TetraVolume(int argc, char *argv[])
    {
        Covise::set_module_description("compute volumes of tetrahedrons");

        Covise::add_port(INPUT_PORT, "gridIn", "UnstructuredGrid", "should only contain tetrahedrons");

        Covise::add_port(OUTPUT_PORT, "volOut", "Float", "...");

        Covise::init(argc, argv);

        const char *in_names[] = { "gridIn", NULL };
        const char *out_names[] = { "volOut", NULL };

        setPortNames(in_names, out_names);

        setCallbacks();

        return;
    };

    TetraVolume(){};

    coDistributedObject **compute(const coDistributedObject **, char **);

    float calculateVolume(float p0[3], float p1[3], float p2[3], float p3[3]);

    void run()
    {
        Covise::main_loop();
    }
};
#endif // __TETRAVOLUME_H
