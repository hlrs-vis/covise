/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                   	    (C)2007 HLRS  **
**                                                                        **
** Description: READ IFU Measurements and simulation results              **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Uwe Woessner                                                   **
**                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include "ReadIFUtxt.h"

ReadIFUtxt::ReadIFUtxt(int argc, char *argv[])
    : coModule(argc, argv, "READ IFU Measurements and simulation results")
{
    // the output ports
    p_mesh = addOutputPort("mesh", "Polygons", "mesh");
    p_velo = addOutputPort("data", "Vec3", "strain");
    p_mesh2 = addOutputPort("mesh2", "Polygons", "mesh");
    p_velo2 = addOutputPort("data2", "Vec3", "strain");
    p_dx_12 = addOutputPort("vector", "Vec3", "vec fro coordinates grid1 to grid2");

    //files
    p_GridPath = addFileBrowserParam("file1", "File1 (eg Experiment)");
    p_GridPath->setValue("/data/ifu", "*.txt");

    p_read_GridPath2 = addBooleanParam("read_file2", "read second file as well?");
    p_read_GridPath2->setValue(0);

    p_GridPath2 = addFileBrowserParam("file2", "File2 (eg Simulation)");
    p_GridPath2->setValue("/data/ifu", "*.txt");
    /*
   p_xGridSize = addInt32Param("DimX", "X Dimension");
   p_xGridSize->setValue(120);
   p_yGridSize = addInt32Param("DimY", "Y Dimension");
   p_yGridSize->setValue(12);
*/
    p_offset = addFloatVectorParam("Offset", "offset of mesh2");
    p_offset->setValue(0., 0., 0.);
}

ReadIFUtxt::~ReadIFUtxt()
{
}

// =======================================================

int ReadIFUtxt::compute(const char *)
{
    // open the file
    bool sim = false;
    int i, j;
    float *xCoord, *yCoord, *zCoord;
    float *xCoord2, *yCoord2, *zCoord2;
    coDoPolygons *grid = NULL;
    coDoPolygons *grid2 = NULL;
    coDoVec3 *velo_out = NULL;
    coDoVec3 *velo_out2 = NULL;
    coDoVec3 *diff_12 = NULL;
    int xs = 10, ys = 10;
    //xs = p_xGridSize->getValue();
    //ys = p_yGridSize->getValue();
    float *vx, *vy, *vz;
    float *vx2, *vy2, *vz2;
    float *dx, *dy, *dz;
    int nc;
    int np;
    int nv;
    int *pol, *vert;

#define BUFSIZE 1000
    char line[BUFSIZE];
    char identifier[BUFSIZE];
    int number;
    FILE *fp = fopen(p_GridPath->getValue(), "rb");
    if (fp)
    {
        fgets(line, BUFSIZE, fp); // header
        if (line[0] == '#')
        {
            sim = false;
        }
        else
        {
            sim = true;
        }

        fgets(line, BUFSIZE, fp);

        while (line[0] == '#')
        {
            sscanf(line, "# %s %d\n", identifier, &number);
            if (!strncmp(identifier, "DimX", 4))
            {
                fprintf(stderr, "setting %s to %d\n", identifier, number);
                xs = number;
            }
            if (!strncmp(identifier, "DimY", 4))
            {
                fprintf(stderr, "setting %s to %d\n", identifier, number);
                ys = number;
            }
            np = (xs - 1) * (ys - 1);
            nv = np * 4;
            fgets(line, BUFSIZE, fp);
        }

        int n = 0;
        int p = 0;

        nc = xs * ys;
        np = (xs - 1) * (ys - 1);
        nv = np * 4;

        grid = new coDoPolygons(p_mesh->getObjName(), nc, nv, np);
        grid->getAddresses(&xCoord, &yCoord, &zCoord, &vert, &pol);
        velo_out = new coDoVec3(p_velo->getObjName(), nc);
        velo_out->getAddresses(&vx, &vy, &vz);

        for (i = 0; i < ys - 1; i++)
        {
            for (j = 0; j < xs - 1; j++)
            {
                pol[p] = p * 4;
                p++;
                vert[n] = i * xs + j;
                n++;
                vert[n] = i * xs + j + 1;
                n++;
                vert[n] = (i + 1) * xs + j + 1;
                n++;
                vert[n] = (i + 1) * xs + j;
                n++;
            }
        }

        n = 0;

        int xi, yi;
        for (i = 0; i < ys; i++)
        {
            for (j = 0; j < xs; j++)
            {
                //n=i*ys+j;
                //fgets(line,BUFSIZE,fp);
                char *c = line;
                while (*c)
                {
                    if (*c == ',')
                        *c = '.';
                    c++;
                }
                if (sim)
                    sscanf(line, "%d %f %f %f %f %f %f", &xi, xCoord + n, yCoord + n, zCoord + n, vx + n, vy + n, vz + n);
                else
                    sscanf(line, "%d %d %f %f %f %f %f %f", &xi, &yi, xCoord + n, yCoord + n, zCoord + n, vx + n, vy + n, vz + n);

                n++;

                fgets(line, BUFSIZE, fp);
            }
        }

        fclose(fp);
    }

    p_mesh->setCurrentObject(grid);
    p_velo->setCurrentObject(velo_out);

    if (p_read_GridPath2->getValue())
    {
        fprintf(stderr, "reading file 2\n");

        int *pol2, *vert2;

        fp = fopen(p_GridPath2->getValue(), "rb");
        if (fp)
        {
            fgets(line, BUFSIZE, fp); // header
            if (line[0] == '#')
            {
                sim = false;
            }
            else
            {
                sim = true;
            }

            fgets(line, BUFSIZE, fp);

            while (line[0] == '#')
            {
                sscanf(line, "# %s %d\n", identifier, &number);
                if (!strncmp(identifier, "DimX", 4))
                {
                    fprintf(stderr, "setting %s to %d\n", identifier, number);
                    xs = number;
                }
                if (!strncmp(identifier, "DimY", 4))
                {
                    fprintf(stderr, "setting %s to %d\n", identifier, number);
                    ys = number;
                }
                np = (xs - 1) * (ys - 1);
                nv = np * 4;
                fgets(line, BUFSIZE, fp);
            }

            nc = xs * ys;
            np = (xs - 1) * (ys - 1);
            nv = np * 4;

            grid2 = new coDoPolygons(p_mesh2->getObjName(), nc, nv, np);
            grid2->getAddresses(&xCoord2, &yCoord2, &zCoord2, &vert2, &pol2);
            velo_out2 = new coDoVec3(p_velo2->getObjName(), nc);
            velo_out2->getAddresses(&vx2, &vy2, &vz2);
            diff_12 = new coDoVec3(p_dx_12->getObjName(), nc);
            diff_12->getAddresses(&dx, &dy, &dz);

            int n = 0;
            int p = 0;

            for (i = 0; i < ys - 1; i++)
            {
                for (j = 0; j < xs - 1; j++)
                {
                    pol2[p] = p * 4;
                    p++;
                    vert2[n] = i * xs + j;
                    n++;
                    vert2[n] = i * xs + j + 1;
                    n++;
                    vert2[n] = (i + 1) * xs + j + 1;
                    n++;
                    vert2[n] = (i + 1) * xs + j;
                    n++;
                }
            }

            n = 0;

            float offset[3];
            p_offset->getValue(offset[0], offset[1], offset[2]);

            int xi, yi;
            for (i = 0; i < ys; i++)
            {
                for (j = 0; j < xs; j++)
                {
                    //n=i*ys+j;
                    //fgets(line,BUFSIZE,fp);
                    char *c = line;
                    while (*c)
                    {
                        if (*c == ',')
                            *c = '.';
                        c++;
                    }
                    if (sim)
                        sscanf(line, "%d %f %f %f %f %f %f", &xi, xCoord2 + n, yCoord2 + n, zCoord2 + n, vx2 + n, vy2 + n, vz2 + n);
                    else
                        sscanf(line, "%d %d %f %f %f %f %f %f", &xi, &yi, xCoord2 + n, yCoord2 + n, zCoord2 + n, vx2 + n, vy2 + n, vz2 + n);

                    xCoord2[n] += offset[0];
                    yCoord2[n] += offset[1];
                    zCoord2[n] += offset[2];

                    dx[n] = xCoord2[n] - xCoord[n];
                    dy[n] = yCoord2[n] - yCoord[n];
                    dz[n] = zCoord2[n] - zCoord[n];

                    n++;

                    fgets(line, BUFSIZE, fp);
                }
            }

            fclose(fp);
        }

        p_mesh2->setCurrentObject(grid2);
        p_velo2->setCurrentObject(velo_out2);
        p_dx_12->setCurrentObject(diff_12);
    }

    return SUCCESS;
}

MODULE_MAIN(IO, ReadIFUtxt)
