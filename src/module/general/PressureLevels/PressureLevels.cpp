/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PressureLevels.h"
#include <do/covise_gridmethods.h>
#include <do/coDoData.h>
#include <do/coDoStructuredGrid.h>

// Designed to calculate pressure levels from hybrid sigma pressure levels in
// weather data (Harmonie model) on structured grids,

enum
{
    P0 = 0,
    A = 1,
    B = 2,
    Ps = 3
};
PressureLevels::PressureLevels(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Hybrid sigma pressure level converter")
{
    p_inGrid = addInputPort("GridIn", "StructuredGrid", "input grid");

    p_inData[P0] = addInputPort("p0", "Float", "p0");
    p_inData[A] = addInputPort("a", "Float", "a");
    p_inData[B] = addInputPort("b", "Float", "b");
    p_inData[Ps] = addInputPort("ps", "Float", "ps");
    for (int i = 4; i < numParams; i++)
    {
        char namebuf[50];
        sprintf(namebuf, "InputData%d", i);
        p_inData[i] = addInputPort(namebuf, "Float", namebuf);
    }

    p_outGrid = addOutputPort("GridOut", "StructuredGrid", "output grid");
}

PressureLevels::~PressureLevels()
{
}

int PressureLevels::compute(const char *)
{
    // open object
    const coDoStructuredGrid *in_grid = dynamic_cast<const coDoStructuredGrid *>(p_inGrid->getCurrentObject());
    if (in_grid == 0)
    {
        Covise::sendError("Oh no! Wrong data type to grid input");
        return STOP_PIPELINE;
    }
    int nx, ny, nz;
    in_grid->getGridSize(&nx, &ny, &nz);
    float *xcIn, *ycIn, *zcIn;
    in_grid->getAddresses(&xcIn, &ycIn, &zcIn);

    const coDoFloat *in_data[numParams];
    for (int i = 0; i < numParams; i++)
    {
        in_data[i] = dynamic_cast<const coDoFloat *>(p_inData[i]->getCurrentObject());
        if (in_data[i] == 0)
        {
            Covise::sendError("Oh no! Wrong data type to port %d", i);
            return STOP_PIPELINE;
        }
    }

    coDoStructuredGrid *out_grid = new coDoStructuredGrid(
        p_outGrid->getObjName(), nx, ny, nz);
    float *xcOut, *ycOut, *zcOut;
    out_grid->getAddresses(&xcOut, &ycOut, &zcOut);

    // Now fill the output grid.
    // FIXME: assumes levels to be last dimension
    float *p0, *a, *b, *ps;
    in_data[P0]->getAddress(&p0);
    in_data[A]->getAddress(&a);
    in_data[B]->getAddress(&b);
    in_data[Ps]->getAddress(&ps);

    int m = 0, n = 0;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++, m++)
            for (int k = 0; k < nz; k++, n++)
            {
                xcOut[n] = xcIn[n];
                ycOut[n] = ycIn[n];
                zcOut[n] = p0[0] * a[k] + ps[m] * b[k];
            }

    return SUCCESS;
}

MODULE_MAIN(Tools, PressureLevels)
