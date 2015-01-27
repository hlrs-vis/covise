/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Filter program in COVISE API                           ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

#include "BoundingBox.h"
#include <alg/coBoundingBox.h>
#include <float.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
BoundingBox::BoundingBox(int argc, char *argv[])
    : coModule(argc, argv, "Find Bounding Box of a grid", true)
{

    // Ports
    p_mesh = addInputPort("GridIn0", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid|Polygons|Points|Lines|TriangleStrips", "mesh");
    p_box = addOutputPort("GridOut0", "Lines", "bounding box");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int BoundingBox::compute(const char *)
{
    coBoundingBox BBox;
    const coDistributedObject *mesh = p_mesh->getCurrentObject();

    float min[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

    const char *ts = mesh->getAttribute("TIMESTEP");
    int numSteps = 1;
    std::stringstream timesteps;
    int minSteps, maxSteps;
    if (ts)
    {
        timesteps << ts;
        timesteps >> minSteps >> maxSteps;
        numSteps = maxSteps + 1 - minSteps;
    }
    std::stringstream buffer;
    buffer << "TimeSteps = " << numSteps;
    sendInfo("%s", buffer.str().c_str());

    BBox.calcBB((coDistributedObject *)mesh, min, max);
    if (min[0] == FLT_MAX)
    {
        sendInfo("No elements found");
        p_box->setCurrentObject(new coDoLines(p_box->getObjName(), 0, 0, 0));
    }
    else
    {
        char buffer[256];
        int i;
        sprintf(buffer, "Bounding Box of %s:", mesh->getName());
        for (i = 0; i < 3; i++)
        {
            sendInfo(" %c : [ %10g  %10g ] , center = %10g",
                     'x' + i, min[i], max[i], 0.5 * (min[i] + max[i]));
        }

        // create bouding box with lines
        coDoLines *l = BBox.createBox(p_box->getObjName(), min[0], min[1], min[2],
                                      max[0] - min[0], max[1] - min[1], max[2] - min[2]);
        p_box->setCurrentObject(l);
    }

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Tools, BoundingBox)
