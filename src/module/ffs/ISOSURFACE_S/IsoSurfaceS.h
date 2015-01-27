/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__ISOSURFACE_S)
#define __ISOSURFACE_C

class IsoSurfaceS;

#include <appl/ApplInterface.h>
using namespace covise;
#include <appl/CoviseAppModule.h>

class IsoSurfaceS : public CoviseAppModule
{
protected:
    // parameters of the module
    int algorithm;
    float isoValue;
    float isoPoint[3];

    void fillRecursive(char *newNodes, char *oldNodes, int numI, int numJ, int numK, int i, int j, int k);

    // algorithms
    // voxels (this should be the fastest, lowest memory consumtion etc.)
    coDistributedObject *issVoxels(coDoUniformGrid *grid, coDoFloat *data, char *outName);

    // marching cubes - high quality
    coDistributedObject *issMCubes(coDoUniformGrid *grid, coDoFloat *data, char *outName);
    void computeMCubesTbl();

    // skeleton climbing....hype
    coDistributedObject *issSkeletonClimbing(coDoUniformGrid *grid, coDoFloat *data, char *outName);

public:
    IsoSurfaceS(int argc, char *argv[])
    {
        Covise::set_module_description("bla");

        Covise::add_port(INPUT_PORT, "gridIn", "Set_UniformGrid", "grid");
        Covise::add_port(INPUT_PORT, "dataIn", "Set_Float", "data");

        Covise::add_port(OUTPUT_PORT, "surfOut", "Set_TriangleStrips|Set_Polygons", "isosurface");
        Covise::add_port(OUTPUT_PORT, "normOut", "Set_Unstructured_V3D_Normals", "normals for isosurface");

        Covise::add_port(PARIN, "isoValue", "Scalar", "threshold for isosurface");
        Covise::set_port_default("isoValue", "1.0");

        Covise::add_port(PARIN, "algorithm", "Choice", "which algorithm should be used");
        Covise::set_port_default("algorithm", "3 voxels marchcubes skelclimb");

        Covise::add_port(PARIN, "isopoint", "Vector", "for filtering");
        Covise::set_port_default("isopoint", "0.0 0.0 0.0");

        Covise::init(argc, argv);

        char *in_names[] = { "gridIn", "dataIn", NULL };
        char *out_names[] = { "surfOut", "normOut", NULL };

        setPortNames(in_names, out_names);
        setCallbacks();

        return;
    };

    ~IsoSurfaceS();

    // called whenever the module is executed
    coDistributedObject **compute(coDistributedObject **, char **);

    // tell covise that we're up and running
    void run()
    {
        Covise::main_loop();
    }
};
#endif // __ISOSURFACE_S
