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
#include <do/coDoStructuredGrid.h>
#include <do/coDoData.h>
#include "ReadDasim.h"

int main(int argc, char *argv[])
{
    ReadDasim *application = new ReadDasim(argc, argv);
    application->start(argc, argv);
    return 0;
}

ReadDasim::ReadDasim(int argc, char *argv[])
    : coModule(argc, argv, "Dasim Reader")
{

    // the output ports
    p_mesh = addOutputPort("mesh", "StructuredGrid", "mesh");
    p_data = addOutputPort("data", "Float|Vec3",
                           "data");
    p_fileParam = addFileBrowserParam("laerm_ist.xzt", "a File browser");
    p_fileParam->setValue("~/vipar/wum_min/laerm_ist.xzt", "*.xz*");

    //    p_fileParam = addFileBrowserParam("file_path","dasim file");
    //    p_fileParam->setValue("file_path","~/vipar/wum_min/filename *xzk");
}

ReadDasim::~ReadDasim()
{
}

// =======================================================

int ReadDasim::compute(const char * /*port*/)
{
    // open the file
    char buf[1000];
    int nu = 0, nv = 0, nw = 0;
    int i, j, k;
    float *xCoord, *yCoord, *zCoord, *scalar;
    float x, y, z, s;
    coDoStructuredGrid *str_grid = NULL;
    coDoFloat *str_s3d_out = NULL;
    FILE *fp = fopen(p_fileParam->getValue(), "r");
    if (fp)
    {
        if (fgets(buf, 1000, fp) == NULL)
        {
            sendError("Premature End of file");
            return FAIL;
        }
        sscanf(buf, "%d %d %d", &nu, &nv, &nw);
        str_grid = new coDoStructuredGrid(p_mesh->getObjName(), nu, nv, nw);
        str_grid->getAddresses(&xCoord, &yCoord, &zCoord);
        str_s3d_out = new coDoFloat(p_data->getObjName(), nu * nv * nw);
        str_s3d_out->getAddress(&scalar);
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
                    sscanf(buf, "%f %f %f %f", &x, &y, &z, &s);
                    xCoord[i * nw * nv + j * nw + k] = x;
                    yCoord[i * nw * nv + j * nw + k] = y;
                    zCoord[i * nw * nv + j * nw + k] = z;
                    scalar[i * nw * nv + j * nw + k] = s;
                }
            }
        }
        p_mesh->setCurrentObject(str_grid);
        p_data->setCurrentObject(str_s3d_out);
    }
    else
    {
        sendError("could not open file: %s", p_fileParam->getValue());
        return FAIL;
    }

    return SUCCESS;
}
