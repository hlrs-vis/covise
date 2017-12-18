/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReducePoints.h"
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSpheres.h>
#include <do/coDoTriangleStrips.h>

#ifdef _WIN32
inline double drand48()
{
    return (double(rand()) / RAND_MAX);
}
#endif

/// Constructor
ReducePoints::ReducePoints(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Remove a fraction of the input points")
{

    // Create ports:
    piGrid = addInputPort("GridIn0", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid|TriangleStrips|Polygons|Lines|Points", "grid or points");
    poPoints = addOutputPort("GridOut0", "Points", "points");

    // Parameters
    pSkip = addIntSliderParam("skip", "skip these many points after keeping one");
    pSkip->setMin(0);
    pSkip->setMax(99);
    pSkip->setValue(0);
    pProbability = addFloatSliderParam("probability", "probability to keep a point");
    pProbability->setMin(0.0);
    pProbability->setMax(1.0);
    pProbability->setValue(1.0);
}

void ReducePoints::addPoint(float x, float y, float z)
{
    if (skip == 0 || counter % (skip + 1) == 0)
    {
        if (probability >= 1. || drand48() < probability)
        {
            coordX.push_back(x);
            coordY.push_back(y);
            coordZ.push_back(z);
        }
    }
    ++counter;
}

/// Compute routine: load checkpoint file
int ReducePoints::compute(const char *)
{
    const coDistributedObject *obj = piGrid->getCurrentObject();
    if (!obj)
    {
        Covise::sendError("did not receive input data");
        return STOP_PIPELINE;
    }

    counter = 0;
    coordX.clear();
    coordY.clear();
    coordZ.clear();
    skip = pSkip->getValue();
    probability = pProbability->getValue();
    int dimX = 0, dimY = 0, dimZ = 0;
    if (const coDoUniformGrid *grid = dynamic_cast<const coDoUniformGrid *>(obj))
    {
        grid->getGridSize(&dimX, &dimY, &dimZ);
        float minX, maxX;
        float minY, maxY;
        float minZ, maxZ;
        grid->getMinMax(&minX, &maxX, &minY, &maxY, &minZ, &maxZ);

        for (int i = 0; i < dimX; i++)
        {
            for (int j = 0; j < dimY; j++)
            {
                for (int k = 0; k < dimZ; k++)
                {
                    addPoint(minX + i * ((maxX - minX) / dimX),
                             minY + j * ((maxY - minY) / dimY),
                             minZ + k * ((maxZ - minZ) / dimZ));
                }
            }
        }
    }
    else if (const coDoRectilinearGrid *grid = dynamic_cast<const coDoRectilinearGrid *>(obj))
    {
        grid->getGridSize(&dimX, &dimY, &dimZ);
        float *gx, *gy, *gz;
        grid->getAddresses(&gx, &gy, &gz);

        for (int i = 0; i < dimX; i++)
        {
            for (int j = 0; j < dimY; j++)
            {
                for (int k = 0; k < dimZ; k++)
                {
                    addPoint(gx[i], gy[j], gz[k]);
                }
            }
        }
    }
    else if (const coDoStructuredGrid *grid = dynamic_cast<const coDoStructuredGrid *>(obj))
    {
        grid->getGridSize(&dimX, &dimY, &dimZ);
        float *gx, *gy, *gz;
        grid->getAddresses(&gx, &gy, &gz);

        for (int i = 0; i < dimX * dimY * dimZ; i++)
        {
            addPoint(gx[i], gy[i], gz[i]);
        }
    }
    else if (const coDoPoints *points = dynamic_cast<const coDoPoints *>(obj))
    {
        int no_points = points->getNumPoints();
        float *gx, *gy, *gz;
        points->getAddresses(&gx, &gy, &gz);

        for (int i = 0; i < no_points; i++)
        {
            addPoint(gx[i], gy[i], gz[i]);
        }
    }
    else if (const coDoSpheres *spheres = dynamic_cast<const coDoSpheres *>(obj))
    {
        int no_points = spheres->getNumSpheres();
        float *gx, *gy, *gz, *gr;
        spheres->getAddresses(&gx, &gy, &gz, &gr);

        for (int i = 0; i < no_points; i++)
        {
            addPoint(gx[i], gy[i], gz[i]);
        }
    }
    else if (const coDoUnstructuredGrid *grid = dynamic_cast<const coDoUnstructuredGrid *>(obj))
    {
        int no_el, no_conn, no_coord;
        grid->getGridSize(&no_el, &no_conn, &no_coord);
        int *elem, *conn;
        float *gx, *gy, *gz;
        grid->getAddresses(&elem, &conn, &gx, &gy, &gz);

        for (int i = 0; i < no_coord; i++)
        {
            addPoint(gx[i], gy[i], gz[i]);
        }
    }
    else if (const coDoTriangleStrips *tri = dynamic_cast<const coDoTriangleStrips *>(obj))
    {
        int *c_l, *p_l;
        float *gx, *gy, *gz;
        tri->getAddresses(&gx, &gy, &gz, &c_l, &p_l);
        int no_coord = tri->getNumPoints();

        for (int i = 0; i < no_coord; i++)
        {
            addPoint(gx[i], gy[i], gz[i]);
        }
    }
    else if (const coDoPolygons *poly = dynamic_cast<const coDoPolygons *>(obj))
    {
        int *c_l, *p_l;
        float *gx, *gy, *gz;
        poly->getAddresses(&gx, &gy, &gz, &c_l, &p_l);
        int no_coord = poly->getNumPoints();

        for (int i = 0; i < no_coord; i++)
        {
            addPoint(gx[i], gy[i], gz[i]);
        }
    }
    else if (const coDoLines *lines = dynamic_cast<const coDoLines *>(obj))
    {
        int *c_l, *l_l;
        float *gx, *gy, *gz;
        lines->getAddresses(&gx, &gy, &gz, &c_l, &l_l);
        int no_coord = lines->getNumPoints();

        for (int i = 0; i < no_coord; i++)
        {
            addPoint(gx[i], gy[i], gz[i]);
        }
    }
    else
    {
        Covise::sendError("received incompatible data");
        return STOP_PIPELINE;
    }

    coDoPoints *points = new coDoPoints(poPoints->getNewObjectInfo(), (int)coordX.size(),
                                        &coordX[0], &coordY[0], &coordZ[0]);
    poPoints->setCurrentObject(points);
    coordX.clear();
    coordY.clear();
    coordZ.clear();

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Converter, ReducePoints)
