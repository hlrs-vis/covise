/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
 **                                                   	                  **
 **                                                                        **
 ** Description: READ imk KA maps             	                  **
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
#include <do/coDoPolygons.h>
#include <do/coDoData.h>
#include "ReadMap.h"

ReadMap::ReadMap(int argc, char *argv[])
    : coModule(argc, argv, "Map Reader")
{
    // the output ports
    p_mesh = addOutputPort("mesh", "Polygons", "mesh");
    p_velo = addOutputPort("data", "Vec3", "velo");

    //files
    p_GridPath = addFileBrowserParam("topo", "topo File browser");
    p_GridPath->setValue("/data/imk_uka/topo", "*topo*");
    p_dirPath = addFileBrowserParam("wdir", "wdir File browser");
    p_dirPath->setValue("/data/imk_uka/wdir", "*wdir*");
    p_vPath = addFileBrowserParam("vv", "vv File browser");
    p_vPath->setValue("/data/imk_uka/vv", "*vv*");

    p_xGridSize = addInt32Param("DimX", "X Dimension");
    p_xGridSize->setValue(60);
    p_yGridSize = addInt32Param("DimY", "Y Dimension");
    p_yGridSize->setValue(50);

    p_xSize = addFloatParam("XSize", "X Size");
    p_xSize->setValue(60);
    p_ySize = addFloatParam("YSize", "Y Size");
    p_ySize->setValue(50);

    p_zScale = addFloatParam("ZScale", "Z Scale");
    p_zScale->setValue(1);
}

ReadMap::~ReadMap()
{
}

// =======================================================

int ReadMap::compute(const char *)
{
    // open the file
    int i, j;
    float *xCoord, *yCoord, *zCoord;
    ;
    coDoPolygons *grid = NULL;
    coDoVec3 *velo_out = NULL;
    int xs, ys;
    xs = p_xGridSize->getValue();
    ys = p_yGridSize->getValue();
    float dx = p_xSize->getValue() / xs;
    float dy = p_ySize->getValue() / ys;
    float *vx, *vy, *vz;
    int nc = xs * ys;
    int np = (xs - 1) * (ys - 1);
    int nv = np * 4;
    int *pol, *vert;

    grid = new coDoPolygons(p_mesh->getObjName(), nc, nv, np);
    grid->getAddresses(&xCoord, &yCoord, &zCoord, &vert, &pol);
    velo_out = new coDoVec3(p_velo->getObjName(), nc);
    velo_out->getAddresses(&vx, &vy, &vz);
    int n = 0;
    for (i = 0; i < xs; i++)
    {
        for (j = 0; j < ys; j++)
        {
            n = i * ys + j;
            xCoord[n] = i * dx;
            yCoord[n] = j * dy;
            zCoord[n] = 0;
        }
    }
    n = 0;
    int p = 0;
    for (i = 0; i < xs - 1; i++)
    {
        for (j = 0; j < ys - 1; j++)
        {
            pol[p] = p * 4;
            p++;
            vert[n] = i * ys + j;
            n++;
            vert[n] = i * ys + j + 1;
            n++;
            vert[n] = (i + 1) * ys + j + 1;
            n++;
            vert[n] = (i + 1) * ys + j;
            n++;
        }
    }
    FILE *fp = fopen(p_GridPath->getValue(), "rb");
    if (fp)
    {
        int dummy, dummy2;
        fread(&dummy, 4, 1, fp);

        int ret = fread(zCoord, sizeof(int), nc, fp);
        if (ret != nc)
            fprintf(stderr, "ReadMap: fread1 failed\n");
        if (dummy != nc * 4)
            byteSwap(zCoord, nc);
        float zs = p_zScale->getValue();
        for (i = 0; i < nc; i++)
            zCoord[i] *= zs;

        fread(&dummy2, 4, 1, fp);
        fclose(fp);
    }
    float *velos = new float[nc];
    fp = fopen(p_vPath->getValue(), "rb");
    if (fp)
    {
        int dummy, dummy2;
        fread(&dummy, 4, 1, fp);

        int ret = fread(velos, sizeof(int), nc, fp);
        if (ret != nc)
            fprintf(stderr, "ReadMap: fread2 failed\n");
        if (dummy != nc * 4)
            byteSwap(velos, nc);
        fread(&dummy2, 4, 1, fp);
        fclose(fp);
    }
    float *dirs = new float[nc];
    fp = fopen(p_dirPath->getValue(), "rb");
    if (fp)
    {
        int dummy, dummy2;
        fread(&dummy, 4, 1, fp);

        int ret = fread(dirs, sizeof(int), nc, fp);
        if (ret != nc)
            fprintf(stderr, "ReadMap: fread3 failed\n");
        if (dummy != nc * 4)
            byteSwap(dirs, nc);
        fread(&dummy2, 4, 1, fp);
        fclose(fp);
    }
    for (i = 0; i < nc; i++)
    {
        vx[i] = (float)sin(dirs[i] / 180.0 * M_PI) * velos[i];
        vy[i] = (float)cos(dirs[i] / 180.0 * M_PI) * velos[i];
        vz[i] = 0;
    }

    delete[] velos;
    delete[] dirs;
    p_mesh->setCurrentObject(grid);
    p_velo->setCurrentObject(velo_out);

    return SUCCESS;
}

MODULE_MAIN(IO, ReadMap)
