/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoPoints.h>
#include "DataToGrid.h"

/// Constructor
DataToGrid::DataToGrid(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Transform structured data to a structured grid or unstructured data to points")
{

    // Create ports:
    piData = addInputPort("DataIn0", "Vec3", "data describing grid or points");
    piTopGrid = addInputPort("GridIn0", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid",
                             "grid for copying topology");
    piTopGrid->setRequired(0);
    poGrid = addOutputPort("GridOut0", "StructuredGrid|Points|UnstructuredGrid", "generated structured grid or points");

    pcDataDirection = addChoiceParam("DataDirection", "axis along which scalar data describes the grid");
    const char *axes[] = { "X-axis", "Y-axis", "Z-axis" };
    pcDataDirection->setValue(3, axes, 0);

    pbStrGrid = addBooleanParam("GenerateGrid", "generate structured grid even when no topology object is available");
    pbStrGrid->setValue(1);

    psDimX = addInt32Param("DimX", "x-dimension of structured output grid");
    psDimY = addInt32Param("DimY", "x-dimension of structured output grid");
    psDimZ = addInt32Param("DimZ", "x-dimension of structured output grid");
    psDimX->setValue(0);
    psDimY->setValue(0);
    psDimZ->setValue(0);
}

/// Compute routine: load checkpoint file
int DataToGrid::compute(const char *)
{
    const coDistributedObject *obj = piData->getCurrentObject();
    if (!obj)
    {
        sendError("did not receive input data");
        return STOP_PIPELINE;
    }

    int no_data = 0;
    if (const coDoAbstractData *data = dynamic_cast<const coDoAbstractData *>(obj))
    {
        no_data = data->getNumPoints();
    }
    else
    {
        sendError("did not receive compatible input data");
        return STOP_PIPELINE;
    }

    const coDistributedObject *topObj = piTopGrid->getCurrentObject();
    float *px = NULL, *py = NULL, *pz = NULL;
    int dimX = 0, dimY = 0, dimZ = 0;
    coDoUnstructuredGrid *unstr = NULL;
    const coDoUnstructuredGrid *top = dynamic_cast<const coDoUnstructuredGrid *>(topObj);
    if (top)
    {
        // copy connectivity, but change cell coords
        int no_elem, no_conn, no_coord;
        top->getGridSize(&no_elem, &no_conn, &no_coord);
        if (no_data != no_coord)
        {
            sendInfo("data not compatible with topology grid, ignoring topology");
            return STOP_PIPELINE;
        }
        else
        {
            unstr = new coDoUnstructuredGrid(poGrid->getObjName(), no_elem, no_conn, no_coord, top->hasTypeList());
            int *top_elem, *top_conn;
            float *dummy;
            top->getAddresses(&top_elem, &top_conn, &dummy, &dummy, &dummy);
            int *elem, *conn;
            unstr->getAddresses(&elem, &conn, &px, &py, &pz);
            for (int i = 0; i < no_elem; i++)
            {
                elem[i] = top_elem[i];
            }
            for (int i = 0; i < no_conn; i++)
            {
                conn[i] = top_conn[i];
            }
            if (top->hasTypeList())
            {
                int *tlist, *top_tlist;
                top->getTypeList(&top_tlist);
                unstr->getTypeList(&tlist);
                for (int i = 0; i < no_elem; i++)
                {
                    tlist[i] = top_tlist[i];
                }
            }
        }
    }
    else if (const coDoAbstractStructuredGrid *str = dynamic_cast<const coDoAbstractStructuredGrid *>(topObj))
    {
        // create a structured grid, but use dimensions of topology input grid
        str->getGridSize(&dimX, &dimY, &dimZ);
    }
    else
    {
        // create a structured grid of user-specified size
        if (topObj)
        {
            sendInfo("topology grid of unknown type, ignoring topology");
        }

        dimX = psDimX->getValue();
        dimY = psDimY->getValue();
        dimZ = psDimZ->getValue();
    }

    coDoStructuredGrid *grid = NULL;
    coDoPoints *points = NULL;
    if (const coDoVec3 *data = dynamic_cast<const coDoVec3 *>(obj))
    {
        float *dx, *dy, *dz;
        data->getAddresses(&dx, &dy, &dz);

        if (top || !pbStrGrid->getValue())
        {
            if (!unstr)
            {
                points = new coDoPoints(poGrid->getObjName(), no_data);
                points->getAddresses(&px, &py, &pz);
            }

            for (int i = 0; i < no_data; i++)
            {
                px[i] = dx[i];
                py[i] = dy[i];
                pz[i] = dz[i];
            }
        }
        else
        {
            if (dimX * dimY * dimZ != no_data)
            {
                sendError("incompatible data size");
                return STOP_PIPELINE;
            }
            grid = new coDoStructuredGrid(poGrid->getObjName(), dimX, dimY, dimZ);
            float *gx, *gy, *gz;
            grid->getAddresses(&gx, &gy, &gz);
            for (int i = 0; i < dimX * dimY * dimZ; i++)
            {
                gx[i] = dx[i];
                gy[i] = dy[i];
                gz[i] = dz[i];
            }
        }
    }
    else if (const coDoFloat *data = dynamic_cast<const coDoFloat *>(obj))
    {
        float *d;
        data->getAddress(&d);

        grid = new coDoStructuredGrid(poGrid->getObjName(), dimX, dimY, dimZ);
        float *gx, *gy, *gz;
        grid->getAddresses(&gx, &gy, &gz);
        switch (pcDataDirection->getValue())
        {
        case 0:
            for (int i = 0; i < dimX; i++)
            {
                for (int j = 0; j < dimY; j++)
                {
                    for (int k = 0; k < dimZ; k++)
                    {
                        int ind = coIndex(i, j, k, dimX, dimY, dimZ);
                        gx[ind] = d[ind];
                        gy[ind] = float(j);
                        gz[ind] = float(k);
                    }
                }
            }
            break;
        case 1:
            for (int i = 0; i < dimX; i++)
            {
                for (int j = 0; j < dimY; j++)
                {
                    for (int k = 0; k < dimZ; k++)
                    {
                        int ind = coIndex(i, j, k, dimX, dimY, dimZ);
                        gx[ind] = float(i);
                        gy[ind] = d[ind];
                        gz[ind] = float(k);
                    }
                }
            }
            break;
        case 2:
            for (int i = 0; i < dimX; i++)
            {
                for (int j = 0; j < dimY; j++)
                {
                    for (int k = 0; k < dimZ; k++)
                    {
                        int ind = coIndex(i, j, k, dimX, dimY, dimZ);
                        gx[ind] = float(i);
                        gy[ind] = float(j);
                        gz[ind] = d[ind];
                    }
                }
            }
            break;
        default:
            fprintf(stderr, "unhandled choice for data direction\n");
            break;
        }
    }
    else
    {
        sendError("received incompatible data");
        return STOP_PIPELINE;
    }

    // Assign sets to output ports:
    if (grid)
    {
        poGrid->setCurrentObject(grid);
    }
    else if (unstr)
    {
        poGrid->setCurrentObject(unstr);
    }
    else if (points)
    {
        poGrid->setCurrentObject(points);
    }
    else
    {
        sendError("did not produce reasonable output");
        return STOP_PIPELINE;
    }

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Converter, DataToGrid)
