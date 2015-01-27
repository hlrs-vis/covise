/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GridToData.h"
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSpheres.h>
#include <do/coDoTriangleStrips.h>

/// Constructor
GridToData::GridToData(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Transform structured grids to structured data or points and unstructured grids to unstructured data")
{

    // Create ports:
    piGrid = addInputPort("GridIn0", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid|TriangleStrips|Polygons|Lines|Points", "grid or points");
    poData = addOutputPort("DataOut0", "Vec3", "data");
}

/// Compute routine: load checkpoint file
int GridToData::compute(const char *)
{
    const coDistributedObject *obj = piGrid->getCurrentObject();
    if (!obj)
    {
        Covise::sendError("did not receive input data");
        return STOP_PIPELINE;
    }

    coDoVec3 *data = NULL;
    int dimX = 0, dimY = 0, dimZ = 0;
    if (const coDoUniformGrid *grid = dynamic_cast<const coDoUniformGrid *>(obj))
    {
        grid->getGridSize(&dimX, &dimY, &dimZ);
        float minX, maxX;
        float minY, maxY;
        float minZ, maxZ;
        grid->getMinMax(&minX, &maxX, &minY, &maxY, &minZ, &maxZ);

        data = new coDoVec3(poData->getObjName(), dimX * dimY * dimZ);
        float *dx, *dy, *dz;
        data->getAddresses(&dx, &dy, &dz);
        for (int i = 0; i < dimX; i++)
        {
            for (int j = 0; j < dimY; j++)
            {
                for (int k = 0; k < dimZ; k++)
                {
                    int ind = coIndex(i, j, k, dimX, dimY, dimZ);
                    dx[ind] = minX + i * ((maxX - minX) / dimX);
                    dy[ind] = minY + j * ((maxY - minY) / dimY);
                    dz[ind] = minZ + k * ((maxZ - minZ) / dimZ);
                }
            }
        }
    }
    else if (const coDoRectilinearGrid *grid = dynamic_cast<const coDoRectilinearGrid *>(obj))
    {
        grid->getGridSize(&dimX, &dimY, &dimZ);
        float *gx, *gy, *gz;
        grid->getAddresses(&gx, &gy, &gz);

        data = new coDoVec3(poData->getObjName(), dimX * dimY * dimZ);
        float *dx, *dy, *dz;
        data->getAddresses(&dx, &dy, &dz);
        for (int i = 0; i < dimX; i++)
        {
            for (int j = 0; j < dimY; j++)
            {
                for (int k = 0; k < dimZ; k++)
                {
                    int ind = coIndex(i, j, k, dimX, dimY, dimZ);
                    dx[ind] = gx[i];
                    dy[ind] = gy[j];
                    dz[ind] = gz[k];
                }
            }
        }
    }
    else if (const coDoStructuredGrid *grid = dynamic_cast<const coDoStructuredGrid *>(obj))
    {
        grid->getGridSize(&dimX, &dimY, &dimZ);
        float *gx, *gy, *gz;
        grid->getAddresses(&gx, &gy, &gz);

        data = new coDoVec3(poData->getObjName(), dimX * dimY * dimZ);
        float *dx, *dy, *dz;
        data->getAddresses(&dx, &dy, &dz);
        for (int i = 0; i < dimX * dimY * dimZ; i++)
        {
            dx[i] = gx[i];
            dy[i] = gy[i];
            dz[i] = gz[i];
        }
    }
    else if (const coDoPoints *points = dynamic_cast<const coDoPoints *>(obj))
    {
        int no_points = points->getNumPoints();
        float *gx, *gy, *gz;
        points->getAddresses(&gx, &gy, &gz);

        data = new coDoVec3(poData->getObjName(), no_points);
        float *dx, *dy, *dz;
        data->getAddresses(&dx, &dy, &dz);
        for (int i = 0; i < no_points; i++)
        {
            dx[i] = gx[i];
            dy[i] = gy[i];
            dz[i] = gz[i];
        }
    }
    else if (const coDoSpheres *spheres = dynamic_cast<const coDoSpheres *>(obj))
    {
        int no_points = spheres->getNumSpheres();
        float *gx, *gy, *gz, *gr;
        spheres->getAddresses(&gx, &gy, &gz, &gr);

        data = new coDoVec3(poData->getObjName(), no_points);
        float *dx, *dy, *dz;
        data->getAddresses(&dx, &dy, &dz);
        for (int i = 0; i < no_points; i++)
        {
            dx[i] = gx[i];
            dy[i] = gy[i];
            dz[i] = gz[i];
        }
    }
    else if (const coDoUnstructuredGrid *grid = dynamic_cast<const coDoUnstructuredGrid *>(obj))
    {
        int no_el, no_conn, no_coord;
        grid->getGridSize(&no_el, &no_conn, &no_coord);
        int *elem, *conn;
        float *gx, *gy, *gz;
        grid->getAddresses(&elem, &conn, &gx, &gy, &gz);

        data = new coDoVec3(poData->getObjName(), no_coord);
        float *dx, *dy, *dz;
        data->getAddresses(&dx, &dy, &dz);
        for (int i = 0; i < no_coord; i++)
        {
            dx[i] = gx[i];
            dy[i] = gy[i];
            dz[i] = gz[i];
        }
    }
    else if (const coDoTriangleStrips *tri = dynamic_cast<const coDoTriangleStrips *>(obj))
    {
        int *c_l, *p_l;
        float *gx, *gy, *gz;
        tri->getAddresses(&gx, &gy, &gz, &c_l, &p_l);
        int no_coord = tri->getNumPoints();

        data = new coDoVec3(poData->getObjName(), no_coord);
        float *dx, *dy, *dz;
        data->getAddresses(&dx, &dy, &dz);
        for (int i = 0; i < no_coord; i++)
        {
            dx[i] = gx[i];
            dy[i] = gy[i];
            dz[i] = gz[i];
        }
    }
    else if (const coDoPolygons *poly = dynamic_cast<const coDoPolygons *>(obj))
    {
        int *c_l, *p_l;
        float *gx, *gy, *gz;
        poly->getAddresses(&gx, &gy, &gz, &c_l, &p_l);

        int no_coord = poly->getNumPoints();
        data = new coDoVec3(poData->getObjName(), no_coord);
        float *dx, *dy, *dz;
        data->getAddresses(&dx, &dy, &dz);
        for (int i = 0; i < no_coord; i++)
        {
            dx[i] = gx[i];
            dy[i] = gy[i];
            dz[i] = gz[i];
        }
    }
    else if (const coDoLines *lines = dynamic_cast<const coDoLines *>(obj))
    {
        int *c_l, *l_l;
        float *gx, *gy, *gz;
        lines->getAddresses(&gx, &gy, &gz, &c_l, &l_l);

        int no_coord = lines->getNumPoints();
        data = new coDoVec3(poData->getObjName(), no_coord);
        float *dx, *dy, *dz;
        data->getAddresses(&dx, &dy, &dz);
        for (int i = 0; i < no_coord; i++)
        {
            dx[i] = gx[i];
            dy[i] = gy[i];
            dz[i] = gz[i];
        }
    }
    else
    {
        Covise::sendError("received incompatible data");
        return STOP_PIPELINE;
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

MODULE_MAIN(Converter, GridToData)
