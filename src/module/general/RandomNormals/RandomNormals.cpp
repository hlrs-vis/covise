/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RandomNormals.h"
#include <do/coDoData.h>
#include <do/coDoSpheres.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>

/// Constructor
RandomNormals::RandomNormals(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Generate random normals for input grid")
{

    // Create ports:
    piGrid = addInputPort("GridIn0", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid|Polygons|Points", "grid or points");
    poData = addOutputPort("DataOut0", "Vec3", "data");
}

/// Compute routine: load checkpoint file
int RandomNormals::compute(const char *)
{
    const coDistributedObject *obj = piGrid->getCurrentObject();
    if (!obj)
    {
        Covise::sendError("did not receive input data");
        return STOP_PIPELINE;
    }

    int npoints = 0;
    if (const coDoAbstractStructuredGrid *grid = dynamic_cast<const coDoAbstractStructuredGrid *>(obj))
    {
        int dimX = 0, dimY = 0, dimZ = 0;
        grid->getGridSize(&dimX, &dimY, &dimZ);
        npoints = dimX * dimY * dimZ;
    }
    else if (const coDoPoints *points = dynamic_cast<const coDoPoints *>(obj))
    {
        npoints = points->getNumPoints();
    }
    else if (const coDoSpheres *sph = dynamic_cast<const coDoSpheres *>(obj))
    {
        npoints = sph->getNumSpheres();
    }
    else if (const coDoPolygons *poly = dynamic_cast<const coDoPolygons *>(obj))
    {
        npoints = poly->getNumPoints();
    }
    else if (const coDoUnstructuredGrid *grid = dynamic_cast<const coDoUnstructuredGrid *>(obj))
    {
        int no_elem, no_conn;
        grid->getGridSize(&no_elem, &no_conn, &npoints);
    }
    else
    {
        Covise::sendError("received incompatible data");
        return STOP_PIPELINE;
    }

    coDoVec3 *data = new coDoVec3(poData->getObjName(), npoints);
    float *dx, *dy, *dz;
    data->getAddresses(&dx, &dy, &dz);
    for (int i = 0; i < npoints; ++i)
    {
#ifndef WIN32
        dx[i] = drand48() * 2. - 1.;
        dy[i] = drand48() * 2. - 1.;
        dz[i] = drand48() * 2. - 1.;
#else

        dx[i] = rand() / (double)RAND_MAX * 2. - 1.;
        dy[i] = rand() / (double)RAND_MAX * 2. - 1.;
        dz[i] = rand() / (double)RAND_MAX * 2. - 1.;
#endif
        double l = sqrt(dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i]);
        dx[i] /= l;
        dy[i] /= l;
        dz[i] /= l;
    }

    // Assign sets to output ports:
    if (data)
    {
        poData->setCurrentObject(data);
    }
    else
    {
        Covise::sendError("did not produce reasonable output");
        return STOP_PIPELINE;
    }

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Converter, RandomNormals)
