/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__TETRA_NEIGHBOR_H)
#define __TETRA_NEIGHBOR_H

#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include <appl/ApplInterface.h>
using namespace covise;
#include <appl/CoviseAppModule.h>

class TetraNeighbor : public CoviseAppModule
{
private:
#if defined(__sgi)
    int m_numNodes; // number of nodes to use for multiprocessing
#endif
    float getNeighbor(int el, int c0, int c1, int c2, int numNeighbors, int *neighborList, int *neighborIndexList, int numPoints);
    int localNeighbors(const coDoUnstructuredGrid *grid, float *newNeighborList, float tval = -1);
    void timeNeighbors(const coDoUnstructuredGrid *grid, float *neighborList,
                       coDoUnstructuredGrid *nextGrid, float *nextNeighborList);

    float tetraVolume(float p0[3], float p1[3], float p2[3], float p3[3]);
    int isInCell(float p0[3], float p1[3], float p2[3], float p3[3], float px[3]);

public:
    // initialize covise
    TetraNeighbor(int argc, char *argv[])
    {
        Covise::set_module_description("compute special neighborlist for tetrahedra-grids");

        Covise::add_port(INPUT_PORT, "gridIn", "UnstructuredGrid", "...");

        Covise::add_port(OUTPUT_PORT, "neighborOut", "Float", "...");

        Covise::init(argc, argv);

#if defined(__sgi)
        m_numNodes = 1;
        coCoviseConfig::getEntry("HostInfo.NumProcessors", &m_numNodes);
#endif

        const char *in_names[] = { "gridIn", NULL };
        const char *out_names[] = { "neighborOut", NULL };

        setPortNames(in_names, out_names);
        setComputeTimesteps(0); // we do it

        setCallbacks();

        return;
    };

    // nothing to clean up
    TetraNeighbor(){};

    // called whenever the module is executed
    coDistributedObject **compute(const coDistributedObject **, char **);

    // tell covise that we're up and running
    void run()
    {
        Covise::main_loop();
    }

    // for SMP
    void goSMP(const coDistributedObject *const *setIn,
               coDoFloat **setOut, int numStart, int numNodes);
    void runSMP(void *p);
};
#endif // __TETRA_NEIGHBOR_H
