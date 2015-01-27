/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/**************************************************************************\ 
 **                                                   	      (C)2002 RUS **
 **                                                                        **
 ** Description: READ Dasim result files             	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **                             **
 **                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <do/coDoData.h>
#include <do/coDoStructuredGrid.h>
#include "ReadDasim.h"

ReadDasim::ReadDasim(int argc, char *argv[])
    : coModule(argc, argv, "Dasim Reader")
{

    // the output ports
    p_mesh = addOutputPort("mesh", "StructuredGrid", "mesh");
    p_data = addOutputPort("data", "Float|Vec3",
                           "data");

    p_fileParam = addFileBrowserParam("filename", "dasim file");
    p_fileParam->setValue("data/ipt/filename", "*xzk");
}

ReadDasim::~ReadDasim()
{
}

// =======================================================

int ReadDasim::compute(const char *)
{
    // open the file
    char buf[1000];
    int nu = 0, nv = 0, nw = 0;
    int i, j, k;
    float *xCoord, *yCoord, *zCoord, *scalar;
    float x, y, z, s;
    coDoStructuredGrid *str_grid = NULL;
    coDoFloat *ustr_s3d_out = NULL;
    FILE *fp = fopen(p_fileParam->getValue(), "r");
    if (fp)
    {
        if (fgets(buf, 1000, fp) == NULL)
        {
            sendError("Premature End of file");
            return FAIL;
        }
        if (sscanf(buf, "%d %d %d", &nu, &nv, &nw) != 3)
        {
            cerr << "ReadDasim::compute: sscanf1 failed" << endl;
        }
        str_grid = new coDoStructuredGrid(p_mesh->getObjName(), nu, nv, nw);
        str_grid->getAddresses(&xCoord, &yCoord, &zCoord);
        ustr_s3d_out = new coDoFloat(p_data->getObjName(), nu * nv * nw);
        ustr_s3d_out->getAddress(&scalar);
        for (k = 0; k < nw; k++)
        {
            for (j = 0; j < nv; j++)
            {
                for (i = 0; i < nu; i++)
                {

                    if (fgets(buf, 1000, fp) == NULL)
                    {
                        sendError("Premature End of file");
                        return FAIL;
                    }
                    if (sscanf(buf, "%f %f %f %f", &x, &y, &z, &s) != 4)
                    {
                        cerr << "ReadDasim::compute: sscanf2 failed" << endl;
                    }
                    xCoord[i * nw * nv + j * nw + k] = x;
                    yCoord[i * nw * nv + j * nw + k] = y;
                    zCoord[i * nw * nv + j * nw + k] = z;
                    scalar[i * nw * nv + j * nw + k] = s;
                }
            }
        }
        p_mesh->setCurrentObject(str_grid);
        p_data->setCurrentObject(ustr_s3d_out);
    }
    else
    {
        sendError("could not open file: %s", p_fileParam->getValue());
        return FAIL;
    }

    return SUCCESS;
}

MODULE_MAIN(IO, ReadDasim)
