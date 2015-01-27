/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__READ_HEAT3D_H)
#define __READ_HEAT3D_H

// Read module for .h3d files
// Lars Frenzel, 08/12/1998

#include <appl/ApplInterface.h>
using namespace covise;

class ReadHEAT3D
{
private:
    void compute(void *);
    void quit(void *);

    // COVISE-callbacks
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    // parameters

    // local data
    int numSeries;
    struct
    {
        int xDim, yDim, zDim;
        float x0, y0, z0;
        float dx, dy, dz;
    } header;
    float *tempData;

    // own stuff
    void readGlobalHeader(FILE *h3d);
    void readHeader(FILE *h3d);
    void readData(FILE *h3d);

    coDoUnstructuredGrid *buildUSG(char *name)
    {
        return (NULL);
    };
    coDoFloat *buildUS3D(char *name)
    {
        return (NULL);
    };
    coDoFloat *buildS3D(char *name);
    coDoStructuredGrid *buildSGrid(char *name);

public:
    ReadHEAT3D(int argc, char *argv[])
    {
        Covise::set_module_description("read Heat3D-files");

        Covise::add_port(OUTPUT_PORT, "grid", "coDoUnstructuredGrid|coDoStructuredGrid", "the grid");
        Covise::add_port(OUTPUT_PORT, "temperature", "coDoFloat|coDoFloat", "temperature-field");

        Covise::add_port(PARIN, "filepath", "Browser", ".h3d file path");
        Covise::add_port(PARIN, "output", "Choice", "structured or unstructured output");

        Covise::set_port_default("filepath", "./ *.h3d");
        Covise::set_port_default("output", "1 structured unstructured");

        Covise::init(argc, argv);
        Covise::set_quit_callback(ReadHEAT3D::quitCallback, this);
        Covise::set_start_callback(ReadHEAT3D::computeCallback, this);
    };

    ~ReadHEAT3D(){};

    void run()
    {
        Covise::main_loop();
        return;
    }
};
#endif // __READ_HEAT3D_H
