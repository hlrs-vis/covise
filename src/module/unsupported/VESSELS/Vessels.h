/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__VESSELS_H)
#define __VESSELS_H

// includes
#include <appl/ApplInterface.h>
using namespace covise;
#include <appl/CoviseAppModule.h>

class Vessels : public CoviseAppModule
{

protected:
    // parameters
    float pointInside[3];
    float value;
    int shrinkTimes, blowTimes;
    int filter1;

    // objectInfo
    int numX, numY, numZ;
    int point[3];
    float *dataIn, *dataOut;

    // data
    float *dataBfr;
    void copyBuffer(float *tgt, const float *src);

    // main work
    void buildOutput();

    // some tools

    /////
    ///// FILTERs that are used by buildOutput
    /////
    void blackOrWhite_filter(float *tgt, const float *src, const float t);
    void shrink_filter(float *tgt, const float *src);
    void blow_filter(float *tgt, const float *src);

public:
    Vessels(int argc, char *argv[])
    {
        Covise::set_module_description("extract blood-vessels from ct-scan data");

        Covise::add_port(INPUT_PORT, "grid_in", "Set_UniformGrid", "...");
        Covise::add_port(INPUT_PORT, "data_in", "Set_Float", "...");

        Covise::add_port(OUTPUT_PORT, "data_out", "Set_Float", "...");

        Covise::add_port(PARIN, "pointinside", "Vector", "...");
        Covise::set_port_default("pointinside", "0.2 0.2 0.2");

        Covise::add_port(PARIN, "value", "Scalar", "...");
        Covise::set_port_default("value", "100.0");

        Covise::add_port(PARIN, "shrink_times", "Scalar", "...");
        Covise::set_port_default("shrink_times", "0");

        Covise::add_port(PARIN, "blow_times", "Scalar", "...");
        Covise::set_port_default("blow_times", "0");

        Covise::add_port(PARIN, "filter1", "Choice", "...");
        Covise::set_port_default("filter1", "1 shrink'n'blow blow'n'shrink");

        Covise::init(argc, argv);

        char *in_names[] = { "grid_in", "data_in", NULL };
        char *out_names[] = { "data_out", NULL };

        setPortNames(in_names, out_names);

        setCopyAttributes(1);

        setCallbacks();

        return;
    };

    Vessels(){};

    coDistributedObject **compute(coDistributedObject **, char **);

    void run()
    {
        Covise::main_loop();
    }
};
#endif // __VESSELS_H
