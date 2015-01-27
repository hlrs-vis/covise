/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SAMPLE_H
#define __SAMPLE_H

// includes
#include <appl/ApplInterface.h>
using namespace covise;
#include <appl/CoviseAppModule.h>
#include <covise/covise_volumedata.h>
#include <math.h>

class Sample : public CoviseAppModule
{
protected:
    char *Density;
    int fillDistance, createVolume;
    int num_scalars;
    int num_vectors;
    float *vx, *vy, *vz;
    float *vxo, *vyo, *vzo, *so;
    float p_min[3];
    float p_max[3];
    int sentWarning;

    void SampleMean(int n, int nv,
                    float *px,
                    float *py,
                    float *pz,
                    float *buffer,
                    int size_i,
                    int size_j,
                    int size_k);
    void SampleMeanv(int n, int nv, float *px, float *py, float *pz,
                     float *scalars, int size_i, int size_j, int size_k);

    void SampleDensity(int n, int nv,
                       float *px,
                       float *py,
                       float *pz,
                       int size_i,
                       int size_j,
                       int size_k);

    void WriteDensity(char *filename, char *density, int size);
    float getAverage(int i, int j, int k, int size_i, int size_j, int size_k, int *numofparticles, float *xs);
    void getAverage(int i, int j, int k, int size_i, int size_j, int size_k, int *numofparticles, float *xs, float *ys, float *zs);

    DO_Volume_Data *getVolume(coDistributedObject *scalar_obj,
                              int num_volume,
                              int np,
                              float *px,
                              float *py,
                              float *pz,
                              int size_x,
                              int size_y,
                              int size_z);

public:
    Sample(int argc, char *argv[])
    {

        Covise::set_module_description("sampling module");

        Covise::add_port(INPUT_PORT, "Points", "coDoPoints|coDoStructuredGrid|coDoUnstructuredGrid", "points");
        Covise::add_port(INPUT_PORT,
                         "SData",
                         "coDoFloat|coDoFloat|coDoVec3|coDoVec3",
                         "scattered data");
        Covise::add_port(OUTPUT_PORT, "Grid", "coDoUniformGrid", "volume data");
        Covise::add_port(OUTPUT_PORT, "OData", "coDoFloat|coDoVec3", "output data");
        Covise::add_port(OUTPUT_PORT, "Volume", "DO_Volume_Data", "volume data");

        Covise::add_port(PARIN, "SizeX", "Scalar", "Gridsize X");
        Covise::add_port(PARIN, "SizeY", "Scalar", "Gridsize Y");
        Covise::add_port(PARIN, "SizeZ", "Scalar", "Gridsize Z");
        Covise::add_port(PARIN, "Mode", "Choice", "sampling mode");
        Covise::add_port(PARIN, "fillDistance", "Scalar", "Maximum distance in Zells to look for values");
        Covise::add_port(PARIN, "createVolume", "Boolean", "Create Volume Object");

        Covise::set_port_default("SizeX", "64");
        Covise::set_port_default("SizeY", "64");
        Covise::set_port_default("SizeZ", "64");
        Covise::set_port_default("Mode",
                                 "1 mean-value density");
        Covise::set_port_default("createVolume",
                                 "FALSE");
        Covise::set_port_default("fillDistance", "0");
        Covise::init(argc, argv);

        char *in_names[] = { "Points", "SData", NULL };
        char *out_names[] = { "Grid", "OData", "Volume", NULL };

        setPortNames(in_names, out_names);

        setCopyAttributes(1);
        setComputeTimesteps(1);
        setComputeMultiblock(1);

        setCallbacks();

        return;
    };

    Sample(){};

    coDistributedObject **compute(coDistributedObject **, char **);

    void run()
    {
        Covise::main_loop();
    }
};
#endif
