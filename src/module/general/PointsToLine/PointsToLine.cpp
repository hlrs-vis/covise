/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "Hello, world!" in COVISE API                          ++
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

// this includes our own class's headers
#include "PointsToLine.h"

#include <do/coDoPoints.h>
#include <do/coDoLines.h>

using namespace covise;

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
PointsToLine::PointsToLine(int argc, char *argv[])
    : coSimpleModule(argc, argv, "connect points to a line")
{
    p_in = addInputPort("geoIn", "Points", "input points");
    p_out = addOutputPort("geoOut", "Lines", "output line");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once every time the module is executed
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int PointsToLine::compute(const char *port)
{
    (void)port;

    const coDistributedObject *obj = p_in->getCurrentObject();
    if (const coDoPoints *points = dynamic_cast<const coDoPoints *>(obj))
    {
        int n = points->getNumPoints();
        float *px, *py, *pz;
        points->getAddresses(&px, &py, &pz);
        coDoLines *lines = new coDoLines(p_out->getObjName(), n, n, 1);
        float *lx, *ly, *lz;
        int *vl, *ll;
        lines->getAddresses(&lx, &ly, &lz, &vl, &ll);
        for (int i=0; i<n; ++i)
        {
            lx[i] = px[i];
            ly[i] = py[i];
            lz[i] = pz[i];
            vl[i] = i;
        }
        ll[0] = 0;
        lines->addAttribute("COLOR", "magenta");
        p_out->setCurrentObject(lines);
        return SUCCESS;
    }

    return STOP_PIPELINE;
}

// instantiate an object of class PointsToLine and register with COVISE
MODULE_MAIN(Module, PointsToLine)
