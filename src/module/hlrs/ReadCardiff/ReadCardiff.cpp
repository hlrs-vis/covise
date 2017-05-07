/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	                  **
 **                                                                        **
 ** Description: READ Cardiff result files             	                  **
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
#include "ReadCardiff.h"

ReadCardiff::ReadCardiff(int argc, char *argv[])
    : coModule(argc, argv, "Cardiff Reader")
{

    // the output ports
    p_mesh = addOutputPort("mesh", "UniformGrid", "mesh");
	p_temperature = addOutputPort("temperature", "Float", "temperature");
	p_pressure = addOutputPort("pressure", "Float", "pressure");
	p_con1 = addOutputPort("con1", "Float", "con1");
	p_con2 = addOutputPort("con2", "Float", "con2");
	p_con3 = addOutputPort("con3", "Float", "con3");
	p_velo = addOutputPort("velocity", "Vec3", "velocity");
    p_fileParam = addFileBrowserParam("Filename", "File browser (laerm_ist.vol)");
    p_fileParam->setValue("/data/workshop/test.txt", "*.*");
}

ReadCardiff::~ReadCardiff()
{
}

// =======================================================

int ReadCardiff::compute(const char *)
{
    // open the file
    char buf[1000];
    int xcount, ycount, zcount, xmin, xmax, ymin, ymax, zmin, zmax;
    int i, j, k;
	float *scalar;
	float *vx,*vy,*vz;
    float x, y, z, s;
    coDoUniformGrid *str_grid = NULL;
    coDoFloat *ustr_s3d_out = NULL;
	coDoVec3 *ustr_v3d_out=NULL;
    FILE *fp = fopen(p_fileParam->getValue(), "r");
    if (fp)
    {
        if (fgets(buf, 1000, fp) == NULL)
        {
            sendError("Premature End of file");
            return FAIL;
        }
        sscanf(buf, "%d %d %d", &xcount, &ycount, &zcount);
        //printf("%d %d %d %d %d %d %d %d %d %d",xcount,ycount,zcount,xmin,xmax,ymin,ymax,zmin,zmax);
		xmin = 0;
		ymin = 0;
		zmin = 0;
		xmax = xcount;
		ymax = ycount;
		zmax = zcount;
        str_grid = new coDoUniformGrid(p_mesh->getObjName(), xcount, ycount, zcount, (float)xmin, (float)xmax, (float)ymin, (float)ymax, (float)zmin, (float)zmax);

		p_mesh->setCurrentObject(str_grid);
        //str_grid->getAddresses( &xCoord, &yCoord, &zCoord );

		ustr_v3d_out = new coDoVec3(p_velo->getObjName(), xcount * ycount * zcount);
		ustr_v3d_out->getAddresses(&vx, &vy, &vz);

		for (i = 0; i < xcount; i++)
		{
			for (j = ycount - 1; j >= 0; j--)
			{
				for (k = 0; k < zcount; k++)
				{
					float s;
					fscanf(fp, "%13f", &s);
					vx[i * zcount * ycount + j * zcount + k] = s;
				}
			}
		}
		for (i = 0; i < xcount; i++)
		{
			for (j = ycount - 1; j >= 0; j--)
			{
				for (k = 0; k < zcount; k++)
				{
					float s;
					fscanf(fp, "%13f", &s);
					vy[i * zcount * ycount + j * zcount + k] = s;
				}
			}
		}
		for (i = 0; i < xcount; i++)
		{
			for (j = ycount - 1; j >= 0; j--)
			{
				for (k = 0; k < zcount; k++)
				{
					float s;
					fscanf(fp, "%13f", &s);
					vz[i * zcount * ycount + j * zcount + k] = s;
				}
			}
		}
		p_velo->setCurrentObject(ustr_v3d_out);

		ustr_s3d_out = new coDoFloat(p_pressure->getObjName(), xcount * ycount * zcount);
		ustr_s3d_out->getAddress(&scalar);
		p_pressure->setCurrentObject(ustr_s3d_out);
		float *temp;
		ustr_s3d_out = new coDoFloat(p_temperature->getObjName(), xcount * ycount * zcount);
		ustr_s3d_out->getAddress(&temp);

		p_temperature->setCurrentObject(ustr_s3d_out);
		for (i = 0; i < xcount; i++)
		{
			for (j = ycount - 1; j >= 0; j--)
			{
				for (k = 0; k < zcount; k++)
				{
					float s;
					fscanf(fp, "%13f", &s);
					scalar[i * zcount * ycount + j * zcount + k] = s;
					fscanf(fp, "%13f", &s);
					temp[i * zcount * ycount + j * zcount + k] = s;
				}
			}
		}





		ustr_s3d_out = new coDoFloat(p_con1->getObjName(), xcount * ycount * zcount);
		ustr_s3d_out->getAddress(&scalar);

		for (i = 0; i < xcount; i++)
		{
			for (j = ycount - 1; j >= 0; j--)
			{
				for (k = 0; k < zcount; k++)
				{
					float s;
					fscanf(fp, "%13f", &s);
					scalar[i * zcount * ycount + j * zcount + k] = s;
				}
			}
		}
		p_con1->setCurrentObject(ustr_s3d_out);

		ustr_s3d_out = new coDoFloat(p_con2->getObjName(), xcount * ycount * zcount);
		ustr_s3d_out->getAddress(&scalar);

		for (i = 0; i < xcount; i++)
		{
			for (j = ycount - 1; j >= 0; j--)
			{
				for (k = 0; k < zcount; k++)
				{
					float s;
					fscanf(fp, "%13f", &s);
					scalar[i * zcount * ycount + j * zcount + k] = s;
				}
			}
		}
		p_con2->setCurrentObject(ustr_s3d_out);

		ustr_s3d_out = new coDoFloat(p_con3->getObjName(), xcount * ycount * zcount);
		ustr_s3d_out->getAddress(&scalar);

		for (i = 0; i < xcount; i++)
		{
			for (j = ycount - 1; j >= 0; j--)
			{
				for (k = 0; k < zcount; k++)
				{
					float s;
					fscanf(fp, "%13f", &s);
					scalar[i * zcount * ycount + j * zcount + k] = s;
				}
			}
		}
		p_con3->setCurrentObject(ustr_s3d_out);

		ustr_v3d_out = new coDoVec3(p_velo->getObjName(), xcount * ycount * zcount);
		ustr_v3d_out->getAddresses(&vx, &vy, &vz);

		for (i = 0; i < xcount; i++)
		{
			for (j = ycount - 1; j >= 0; j--)
			{
				for (k = 0; k < zcount; k++)
				{
					float s;
					fscanf(fp, "%13f", &s);
					vx[i * zcount * ycount + j * zcount + k] = s;
				}
			}
		}
		for (i = 0; i < xcount; i++)
		{
			for (j = ycount - 1; j >= 0; j--)
			{
				for (k = 0; k < zcount; k++)
				{
					float s;
					fscanf(fp, "%13f", &s);
					vy[i * zcount * ycount + j * zcount + k] = s;
				}
			}
		}
		for (i = 0; i < xcount; i++)
		{
			for (j = ycount - 1; j >= 0; j--)
			{
				for (k = 0; k < zcount; k++)
				{
					float s;
					fscanf(fp, "%13f", &s);
					vz[i * zcount * ycount + j * zcount + k] = s;
				}
			}
		}
		p_velo->setCurrentObject(ustr_v3d_out);
    }
    else
    {
        sendError("could not open file: %s", p_fileParam->getValue());
        return FAIL;
    }

    return SUCCESS;
}

MODULE_MAIN(IO, ReadCardiff)
