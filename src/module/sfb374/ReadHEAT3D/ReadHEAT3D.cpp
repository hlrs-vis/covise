/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Read module for .h3d files
// Lars Frenzel, 08/12/1998

#include <appl/ApplInterface.h>
#include "ReadHEAT3D.h"

int main(int argc, char *argv[])
{
    ReadHEAT3D *app;

    app = new ReadHEAT3D(argc, argv);
    app->run();

    return 0;
}

////////////////////////////////////////////////////////////////////////
// COVISE-callbacks, don't change anything !
void ReadHEAT3D::quitCallback(void *userData, void *callbackData)
{
    ReadHEAT3D *app;

    app = (ReadHEAT3D *)userData;
    app->quit(callbackData);

    return;
}

void ReadHEAT3D::computeCallback(void *userData, void *callbackData)
{
    ReadHEAT3D *app;

    app = (ReadHEAT3D *)userData;
    app->compute(callbackData);

    return;
}

////////////////////////////////////////////////////////////////////////

void ReadHEAT3D::quit(void *)
{
    return;
}

void ReadHEAT3D::compute(void *)
{
    char *gridName, *tempName;
    char *filepath;

    int output;

    int i;

    FILE *h3d;

    coDoFloat *us3d = NULL;
    coDoUnstructuredGrid *usg = NULL;
    coDoFloat *s3d = NULL;
    coDoStructuredGrid *sgrid = NULL;

    // get parameters
    Covise::get_browser_param("filepath", &filepath);
    Covise::get_choice_param("output", &output);

    // get object-names
    gridName = Covise::get_object_name("grid");
    tempName = Covise::get_object_name("temperature");

    // open file
    if ((h3d = Covise::fopen(filepath, "r")) == NULL)
    {
        char bfr[1024];
        sprintf(bfr, "ERROR: can't open file %s", filepath);
        Covise::sendError(bfr);
    }

    // here we go
    readGlobalHeader(h3d);

    // only stationary data/grid supported at the moment

    readHeader(h3d);
    readData(h3d);
    if (output == 1 || 1)
    {
        if (gridName)
        {
            sgrid = buildSGrid(gridName);
            delete sgrid;
        }
        if (tempName)
        {
            s3d = buildS3D(tempName);
            delete s3d;
        }
    }
    else
    {
        if (gridName)
        {
            usg = buildUSG(gridName);
            delete usg;
        }
        if (tempName)
        {
            us3d = buildUS3D(tempName);
            delete us3d;
        }
    }
    delete[] tempData;

    // done
    return;
}

void ReadHEAT3D::readGlobalHeader(FILE *h3d)
{
    fscanf(h3d, "%i\n", &numSeries);
    return;
}

void ReadHEAT3D::readHeader(FILE *h3d)
{
    fscanf(h3d, "%i %i %i\n", &(header.xDim), &(header.yDim), &(header.zDim));
    fscanf(h3d, "%e %e %e\n", &(header.x0), &(header.y0), &(header.z0));
    fscanf(h3d, "%e %e %e\n", &(header.dx), &(header.dy), &(header.dz));
    return;
}

void ReadHEAT3D::readData(FILE *h3d)
{
    int i, l;

    l = header.xDim * header.yDim * header.zDim;

    tempData = new float[l];

    for (i = 0; i < l; i++)
        fscanf(h3d, "%e\n", &(tempData[i]));

    return;
}

coDoFloat *ReadHEAT3D::buildS3D(char *name)
{
    coDoFloat *r = NULL;

    r = new coDoFloat(name, header.xDim, header.yDim, header.zDim, tempData);

    return (r);
}

coDoStructuredGrid *ReadHEAT3D::buildSGrid(char *name)
{
    coDoStructuredGrid *r = NULL;
    float *xCoord, *yCoord, *zCoord;
    int l;
    int i, j, k;

    l = header.xDim * header.yDim * header.zDim;

    xCoord = new float[l];
    yCoord = new float[l];
    zCoord = new float[l];

    l = 0;
    for (i = 0; i < header.xDim; i++)
        for (j = 0; j < header.yDim; j++)
            for (k = 0; k < header.zDim; k++)
            {
                xCoord[l] = header.x0 + ((float)i) * header.dx;
                yCoord[l] = header.y0 + ((float)j) * header.dy;
                zCoord[l] = header.z0 + ((float)k) * header.dz;
                l++;
            }

    r = new coDoStructuredGrid(name, header.xDim, header.yDim, header.zDim, xCoord, yCoord, zCoord);

    delete[] xCoord;
    delete[] yCoord;
    delete[] zCoord;

    return (r);
}
