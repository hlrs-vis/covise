/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "StackSlices.h"
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoSet.h>

StackSlices::StackSlices(int argc, char *argv[])
    : coModule(argc, argv, "Stack several uniform grids on top of each other")
{
    const char *DirChoice[] = { "x-axis", "y-axis", "z-axis" };

    //parameters
    p_direction = addChoiceParam("Direction", "stacking direction");
    p_direction->setValue(3, DirChoice, 0);
    p_sliceDistance = addFloatParam("SliceDistance", "distance of slices");
    p_sliceDistance->setValue(-1.0);

    //ports
    p_gridIn = addInputPort("GridIn0", "UniformGrid", "input grid");
    p_dataIn = addInputPort("DataIn0", "Float|RGBA|Byte", "input data");
    p_gridOut = addOutputPort("GridOut0", "UniformGrid", "output grid");
    p_dataOut = addOutputPort("DataOut0", "Float|RGBA|Byte", "output data");
}

int StackSlices::compute(const char *)
{
    const float Eps = 1e-10;
    const int axis = p_direction->getValue();
    const float dist = p_sliceDistance->getValue();

    const coDistributedObject *inGrid = p_gridIn->getCurrentObject();
    const coDistributedObject *inData = p_dataIn->getCurrentObject();
    if (!inGrid || !inData)
    {
        sendInfo("need both grid and data");
        return STOP_PIPELINE;
    }

    std::vector<const coDoUniformGrid *> allGrids;
    std::vector<const coDoAbstractData *> allData;

    const coDoSet *gridSet = dynamic_cast<const coDoSet *>(inGrid);
    const coDoSet *dataSet = dynamic_cast<const coDoSet *>(inData);
    if (!gridSet && !dataSet)
    {
        allGrids.push_back(dynamic_cast<const coDoUniformGrid *>(inGrid));
        allData.push_back(dynamic_cast<const coDoAbstractData *>(inData));
    }
    else if (gridSet && dataSet)
    {
        int noGrid=0, noData=0;
        const coDistributedObject *const *gridObjs = gridSet->getAllElements(&noGrid);
        const coDistributedObject *const *dataObjs = dataSet->getAllElements(&noData);
        if (noGrid != noData)
        {
            sendInfo("cardinality mismatch between grid and data");
            return STOP_PIPELINE;
        }

        for (int i=0; i<noGrid; ++i)
        {
            allGrids.push_back(dynamic_cast<const coDoUniformGrid *>(gridObjs[i]));
            allData.push_back(dynamic_cast<const coDoAbstractData *>(dataObjs[i]));
        }
    }
    else
    {
        sendInfo("mismatch between grid and data");
        return STOP_PIPELINE;
    }

    int numSlices=0;
    int sx=-1, sy=-1, sz=-1;
    float xmin=0., xmax=0., ymin=0., ymax=0., zmin=0., zmax=0.;
    float dx=0., dy=0., dz=0.;
    for (int i=0; i<allGrids.size(); ++i)
    {
        const coDoUniformGrid *g = allGrids[i];
        const coDoAbstractData *d = allData[i];
        if (g == NULL || d == NULL)
        {
            sendInfo("received invalid input data types");
            return STOP_PIPELINE;
        }
        int nx, ny, nz;
        g->getGridSize(&nx, &ny, &nz);
        if (allData[i]->getNumPoints() != nx*ny*nz)
        {
            sendInfo("grid size and data size mismatch for timestep/block %d", i);
            return STOP_PIPELINE;
        }
        if (sx == -1 && sy == -1 && sz == -1)
        {
            sx = nx;
            sy = ny;
            sz = nz;
            g->getMinMax(&xmin, &xmax, &ymin, &ymax, &zmin, &zmax);
            g->getDelta(&dx, &dy, &dz);
        }
        else
        {
            if (axis != 0 && nx != sx)
            {
                sendInfo("grid size of %d for timestep/block %d does not match expected x-dim %d", nx, i, sx);
                return STOP_PIPELINE;
            }
            if (axis != 1 && ny != sy)
            {
                sendInfo("grid size of %d for timestep/block %d does not match expected y-dim %d", nx, i, sx);
                return STOP_PIPELINE;
            }
            if (axis != 2 && nz != sz)
            {
                sendInfo("grid size of %d for timestep/block %d does not match expected z-dim %d", nx, i, sx);
                return STOP_PIPELINE;
            }
        }
        if (axis == 0)
            numSlices += nx;
        else if (axis == 1)
            numSlices += ny;
        else if (axis == 2)
            numSlices += nz;
    }
    if (axis == 0)
    {
        if (dist >= 0.f)
            dx = dist;
        else if (dx <= Eps)
            dx = (dy+dz)/2.;
        sx = numSlices;
        xmax = xmin+sx*dx;
    }
    else if (axis == 1)
    {
        if (dist >= 0.f)
            dy = dist;
        else if (dy <= Eps)
            dy = (dx+dz)/2.;
        sy = numSlices;
        ymax = zmin+sy*dz;
    }
    else if (axis == 2)
    {
        if (dist >= 0.f)
            dz = dist;
        else if (dz <= Eps)
            dz = (dy+dz)/2.;
        sz = numSlices;
        zmax = zmin+sz*dz;
    }

    coDoUniformGrid *outGrid = new coDoUniformGrid(p_gridOut->getObjName(), sx, sy, sz,
                                                   xmin, xmax, ymin, ymax, zmin, zmax);
    outGrid->copyAllAttributes(allGrids[0]);

    coDoAbstractData *outData = allData[0]->cloneType(p_dataOut->getObjName(), sx*sy*sz);
    outData->copyAllAttributes(allData[0]);

    int ox=0, oy=0, oz=0;
    for (int i=0; i<allGrids.size(); ++i)
    {
        const coDoUniformGrid *g = allGrids[i];
        const coDoAbstractData *d = allData[i];
        int nx, ny, nz;
        g->getGridSize(&nx, &ny, &nz);

        for (int ix=0; ix<nx; ++ix)
        {
            for (int iy=0; iy<ny; ++iy)
            {
                for (int iz=0; iz<nz; ++iz)
                {
                    int di = coIndex(ix+ox, iy+oy, iz+oz, sx, sy, sz);
                    int si = coIndex(ix, iy, iz, nx, ny, nz);
                    outData->cloneValue(di, allData[i], si);
                }
            }
        }

        if (axis == 0)
        {
            ox += nx;
        }
        else if (axis == 1)
        {
            oy += ny;
        }
        else if (axis == 2)
        {
            oz += nz;
        }
    }

    p_gridOut->setCurrentObject(outGrid);
    p_dataOut->setCurrentObject(outData);

    return SUCCESS;
}


MODULE_MAIN(Tools, StackSlices)
