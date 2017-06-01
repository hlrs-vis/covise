/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 RUS  **
 **                                                                        **
 ** Description: Read IMD checkpoint files from ITAP.                      **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                     Juergen Schulze-Doebold                            **
 **     High Performance Computing Center University of Stuttgart          **
 **                         Allmandring 30                                 **
 **                         70550 Stuttgart                                **
 **                                                                        **
 ** Cration Date: 03.09.2002                                               **
\**************************************************************************/

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoPoints.h>
#include <do/coDoUniformGrid.h>

#include <api/coModule.h>
#include <alg/coChemicalElement.h>
#include <limits.h>
#include <float.h>
#include <cassert>
#include "ReadCUBE.h"

/// Constructor
coReadXYZ::coReadXYZ(int argc, char *argv[])
    : coModule(argc, argv, "Read CUBE files containing atom positions and electron densities")
{

    // Create ports:
    poPoints = addOutputPort("Location", "Points", "Atom location");
    poTypes = addOutputPort("Element", "Int", "Atom element type");
    poBounds = addOutputPort("Bounds", "Points", "Boundary atoms");

    poGrid = addOutputPort("Grid", "UniformGrid", "grid for density");
    poDensity = addOutputPort("Density", "Float", "density");

    // Create parameters:
    pbrFilename = addFileBrowserParam("Filename", "CUBE file");
    pbrFilename->setValue("data/", "*.cube;*.CUBE/*");
}

/// Compute routine: load checkpoint file
int coReadXYZ::compute(const char *)
{
    // Open first checkpoint file:
    const char *path = pbrFilename->getValue();

    float minmax[2][3] = {
        { FLT_MAX, FLT_MAX, FLT_MAX },
        { -FLT_MAX, -FLT_MAX, -FLT_MAX }
    };

    int totalAtoms = 0;

    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        sendError("failed to open XYZ file %s for reading.", path);
        return STOP_PIPELINE;
    }

    for (int i=0; i<2; ++i)
    {
        char line[1024];
        if (!fgets(line, sizeof(line), fp))
        {
            sendError("failed to read header line %d", i);
            fclose(fp);
            return STOP_PIPELINE;
        }
    }

    int numAtoms;
    float originX, originY, originZ;
    if (fscanf(fp, "%d %f %f %f", &numAtoms, &originX, &originY, &originZ) != 4)
    {
        sendError("failed to read header number of atoms or origin");
        fclose(fp);
        return STOP_PIPELINE;
    }

    int gridSize[3];
    float voxelSize[3][3];
    for (int i=0; i<3; ++i)
    {
        if (fscanf(fp, "%d %f %f %f", &gridSize[i], &voxelSize[i][0], &voxelSize[i][1], &voxelSize[i][2]) != 4)
        {
            sendError("failed to read grid info line %d", i);
            fclose(fp);
            return STOP_PIPELINE;
        }
        for (int d=0; d<3; ++d) {
            if (d == i)
            {
                if (voxelSize[i][d] == 0.)
                    sendWarning("voxel size in direction %d is zero", i);
            }
            else
            {
                if (voxelSize[i][d] != 0.)
                    sendWarning("voxels not axis aligned");
            }
        }
    }

    coDoUniformGrid *grid = new coDoUniformGrid(poGrid->getObjName(),
                                                gridSize[0], gridSize[1], gridSize[2],
            originX, voxelSize[0][0]*gridSize[0], originY, voxelSize[1][1]*gridSize[1], originZ, voxelSize[2][2]*gridSize[2]);
    poGrid->setCurrentObject(grid);

    coDoInt *doTypes = new coDoInt(poTypes->getObjName(), numAtoms);
    int *types = doTypes->getAddress();
    coDoPoints *doPoints = new coDoPoints(poPoints->getObjName(), numAtoms);
    float *x, *y, *z;
    doPoints->getAddresses(&x, &y, &z);

        for (int k=0; k<numAtoms; ++k)
        {
            fpos_t pos;
            fgetpos(fp, &pos);
            float mass;
            int n = fscanf(fp, "%d %f %f %f %f\n", &types[k], &mass, &x[k], &y[k], &z[k]);
            if (n != 5)
            {
                sendError("Failed to read data for atom %d.", k);
                fclose(fp);
                return STOP_PIPELINE;
            }
        }

    poTypes->setCurrentObject(doTypes);
    poPoints->setCurrentObject(doPoints);

    int numDensity = gridSize[0]*gridSize[1]*gridSize[2];
    coDoFloat *doDensity = new coDoFloat(poDensity->getObjName(), numDensity);
    float *density = doDensity->getAddress();
    for (int k=0; k<numDensity; ++k)
    {
        int n = fscanf(fp, "%f", &density[k]);
        if (n != 1)
        {
                sendError("Failed to read density value %d.", k);
                fclose(fp);
                return STOP_PIPELINE;
        }
    }
    poDensity->setCurrentObject(doDensity);
    fclose(fp);

    // Create set objects:

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, coReadXYZ)
