/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	                  **
 **                                                                        **
 ** Description: READ SoundVol result files             	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Jens Wiesner                                                   **
 **                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoData.h>
#include "ReadSoundVol.h"

ReadSoundVol::ReadSoundVol(int argc, char *argv[])
    : coModule(argc, argv, "SoundVol Reader")
{

    // the output ports
    p_mesh = addOutputPort("mesh", "UniformGrid", "mesh");
    p_data = addOutputPort("data", "Float", "data");
    p_fileParam = addFileBrowserParam("Filename", "File browser (laerm_ist.vol)");
    p_fileParam->setValue("~/covise/data/karte/RRLK4004.vol", "*.vol*");
}

ReadSoundVol::~ReadSoundVol()
{
}

// =======================================================

int ReadSoundVol::compute(const char *)
{
    // open the file
    char buf[1000];
    int xcount, ycount, zcount, xmin, xmax, ymin, ymax, zmin, zmax;
    int i, j, k;
    float *scalar;
    float x, y, z, s;
    coDoUniformGrid *str_grid = NULL;
    coDoFloat *ustr_s3d_out = NULL;
    FILE *fp = fopen(p_fileParam->getValue(), "r");
    if (fp)
    {
        if (fgets(buf, 1000, fp) == NULL)
        {
            sendError("Premature End of file");
            return FAIL;
        }
        sscanf(buf, "%d %d %d %d %d %d %d %d %d", &xcount, &ycount, &zcount, &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);
        //printf("%d %d %d %d %d %d %d %d %d %d",xcount,ycount,zcount,xmin,xmax,ymin,ymax,zmin,zmax);
        str_grid = new coDoUniformGrid(p_mesh->getObjName(), xcount, ycount, zcount, (float)xmin, (float)xmax, (float)ymin, (float)ymax, (float)zmin, (float)zmax);

        //str_grid->getAddresses( &xCoord, &yCoord, &zCoord );
        ustr_s3d_out = new coDoFloat(p_data->getObjName(), xcount * ycount * zcount);
        ustr_s3d_out->getAddress(&scalar);

        for (k = 0; k < zcount; k++)
        {
            for (j = ycount - 1; j >= 0; j--)
            {
                for (i = 0; i < xcount; i++)
                {

                    if (fgets(buf, 1000, fp) == NULL)
                    {
                        sendError("Premature End of file");
                        return FAIL;
                    }
                    sscanf(buf, "%f %f %f %f", &x, &y, &z, &s);
                    /*xCoord[i*xcount*ycount+j*xcount+k]=x;
               yCoord[i*xcount*ycount+j*xcount+k]=y;
               zCoord[i*xcount*ycount+j*xcount+k]=z;
               scalar[i*xcount*ycount+j*xcount+k]=s;*/
                    //		  scalar[k*xcount*ycount+j*xcount+i]=s; //	1.Variante
                    //	2.Variante
                    scalar[i * zcount * ycount + j * zcount + k] = s;
                    //		  scalar[k*zcount+ycount+j*xcount+i]=s;
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

MODULE_MAIN(IO, ReadSoundVol)
