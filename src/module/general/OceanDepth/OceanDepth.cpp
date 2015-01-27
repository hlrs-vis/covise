/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OceanDepth.h"
#include <do/covise_gridmethods.h>
#include <do/coDoData.h>
#include <do/coDoStructuredGrid.h>

// Designed to calculate depth levels from ROMS data

enum
{
    Sr = 0,
    Ze = 1,
    H = 2
};
OceanDepth::OceanDepth(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Ocean depth level converter")
{
    p_inGrid = addInputPort("GridIn", "StructuredGrid", "input grid");

    p_inData[Sr] = addInputPort("sr", "Float", "sr");
    p_inData[Ze] = addInputPort("ze", "Float", "ze");
    p_inData[H] = addInputPort("h", "Float", "h");
    /*for(int i=5; i<numParams; i++){
		char namebuf[50];
		sprintf(namebuf, "InputData%d", i);
		p_inData[i] = addInputPort(namebuf,"Float", namebuf);
	}*/

    p_outGrid = addOutputPort("GridOut", "StructuredGrid", "output grid");
}

OceanDepth::~OceanDepth()
{
}

int OceanDepth::compute(const char *)
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
    float *sr, *ze, *h;
    in_data[Sr]->getAddress(&sr);
    in_data[Ze]->getAddress(&ze);
    in_data[H]->getAddress(&h);

    int m = 0, n = 0;
    for (int i = 0; i < nx; i++)
    {
        //m = 0;
        for (int j = 0; j < ny; j++, m++)
            for (int k = 0; k < nz; k++, n++)
            {
                xcOut[n] = xcIn[n];
                ycOut[n] = ycIn[n];
                zcOut[n] = sr[k] + ze[m] * (1. + sr[k] / h[m]);
                if (n % 100 == 0)
                    fprintf(stderr, "%.2e = %.2e + %.2e * ( 1. + %.2e / %.2e)\n",
                            zcOut[n],
                            sr[k], ze[m], sr[k], h[m]);
            }
    }

    return SUCCESS;
}

MODULE_MAIN(Tools, OceanDepth)
