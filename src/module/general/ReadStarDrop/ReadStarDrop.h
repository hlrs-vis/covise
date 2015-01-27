/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __READ_STAR_DROP_H_
#define __READ_STAR_DROP_H_

// 15.03.00

#include <api/coModule.h>
using namespace covise;
class ReadStarDrop : public coModule
{
public:
    typedef int int32;
    typedef float float32;

    // droplet record data type
    struct DropRec
    {
        int32 origIxd; // FORTRAN header later overwritten by index
        int32 num;
        float32 x, y, z, u, v, w;
        int32 itrak;
        float32 t;
        int32 icel;
        float32 temp, diam, mass, coun;
        int32 length2; // FORTRAN trailer
    };

    // droplet info field
    struct DropInfo
    {
        float startTime, endTime, endFly;
        int numTraces, firstStuck;
        DropRec *drop;
    };

private:
    enum
    {
        NUM_DATA_PORTS = 10
    };

    // input: the timesteps from the ReadSTAR module: replicate if necessary
    coInputPort *p_grid_in, *p_dataIn[NUM_DATA_PORTS];

    // output ports: loop-through + dropled data
    coOutputPort *p_grid_out, *p_dataOut[NUM_DATA_PORTS], *p_dropOut[6], *p_mapping;

    // enum for port numbers
    enum
    {
        LOCA = 0,
        VELO = 1,
        TEMP = 2,
        DIAM = 3,
        MASS = 4,
        COUN = 5
    };

    // enum for stuckHandling
    enum
    {
        NO_STUCK = 0,
        WITH_STUCK = 1,
        ONLY_STUCK = 2
    };

    static const char *s_species[6];

    // parameters
    coFileBrowserParam *p_filename;
    coIntScalarParam *p_numSteps; // =0 means auotmatic
    coIntScalarParam *p_maxDrops; // =0 means all
    coChoiceParam *p_stickHandling;

    // the last filename we read in correctly (to avoid double reads)
    char *d_lastFileName;

    // The droplet data: one struct per droplet, each containing one
    // droplet trace
    DropInfo *d_dropArr;

    // StarCD scaling factor: versions <3000 internal, >=3000 : used
    float d_scale;

    // number of Droplets currently stored
    int d_numDrops;

    // read the meshes from the input object: return 1 on error, 0 if ok
    //
    //                        NULL on error and set number of meshes
    int retrieveMeshes(const coDistributedObject *const *&meshSet,
                       int &numGridSteps, float *&gridTime);

    // read the droplet data: return 0 on success, 1 on error
    int readFile(const char *filename);

    // droplet sorting function for qsort
    static int dropCompare(const void *d1, const void *d2);

    // create time array from given grid return number of steps if auto-mode
    int timeArray(int numGridSteps, const float *gridTime,
                  int &numSteps, float *&timeFrame, int *&meshIdx);

    // create the mesh output and the other loop-through objects
    int createLoopThrough(const coDistributedObject *const *meshSet, int numGridSteps,
                          int numSteps, int *meshIdx);

    // create the droplet-dependent objects
    int createDropletObj(int numSteps, float *timeFrame);

    // read the set at a given port, return NULL if not a set or #elem wrong
    const coDistributedObject *const *readSet(coInputPort *port, int numElem);

    // count number of droplets in a given Interval
    int countDroplets(float tMin, float tMax, int stickHandling);

    // get one fitting droplet in a given Interval
    DropRec *getDroplet(int dropNo, float tMin, float tMax, int stickHandling);

    // find the first and last step of all droplets
    void getSteps(int numSteps, float *timeFrame,
                  int *dropStart, int *dropEnd, int stickHandling);

public:
    ReadStarDrop(int argc, char *argv[]);

    virtual ~ReadStarDrop();

    /// compute call-back
    virtual int compute(const char *port);
};
#endif
