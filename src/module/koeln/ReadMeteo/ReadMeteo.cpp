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
#include <do/coDoStructuredGrid.h>
#include <api/coModule.h>
#include <virvo/vvtoolshed.h>
#include <limits.h>
#include <float.h>
#include "ReadMeteo.h"

#define BUFINC 1

/// Constructor
coReadMeteo::coReadMeteo(int argc, char *argv[])
    : coModule(argc, argv, "Read files to create a list of points (stars) and scalar parameters.")
    , lineBuf(NULL)
    , lineBufSize(0)
{

    // Create ports:
    poData = addOutputPort("Data", "StructuredGrid|Float", "Height Field or data");
    poData->setInfo("Height field or data");

    // Create parameters:
    pbrFile = addFileBrowserParam("FilePath", "First file of sequence or single file");
    pbrFile->setValue("data/", "*");

    pDimX = addInt32Param("DimX", "X Dimension");
    pDimX->setValue(40);

    pDimY = addInt32Param("DimY", "Y Dimension");
    pDimY->setValue(40);

    pDimZ = addInt32Param("DimZ", "Z Dimension");
    pDimZ->setValue(16);

    pScale = addFloatParam("Scale", "Scale factor");
    pScale->setValue(0.001f);

    pboDataMode = addBooleanParam("DataMode", "Read files as data (instead of height information)");
    pboDataMode->setValue(true);

    pboWarnings = addBooleanParam("Warnings", "Display warnings when reading files");
    pboWarnings->setValue(true);
}

/// @return absolute value of a vector
float coReadMeteo::absVector(float x, float y, float z)
{
    return sqrt(x * x + y * y + z * z);
}

/// @return true if warnings are to be displayed
bool coReadMeteo::displayWarnings()
{
    return pboWarnings->getValue();
}

bool coReadMeteo::readLine(FILE *fp)
{
    int offset = 0;

    for (;;)
    {
        if (lineBuf && lineBufSize > 1)
        {
            if (!fgets(lineBuf + offset, lineBufSize - offset, fp))
            {
                return false;
            }

            int len = strlen(lineBuf);
            if (len > 0)
            {
                if (lineBuf[len - 1] == '\n')
                    return true;
            }
        }

        lineBuf = (char *)realloc(lineBuf, lineBufSize + BUFINC);
        if (lineBuf == NULL)
        {
            lineBufSize = 0;
            return false;
        }

        if (lineBufSize > 0)
        {
            offset = lineBufSize - 1;
        }
        else
        {
            offset = 0;
        }
        lineBufSize += BUFINC;
    }

    return false;
}

bool coReadMeteo::readArray(FILE *fp, float **data, int numElems)
{
    float scale = pScale->getValue();
    *data = new float[numElems];
    char *pos = NULL, *npos = NULL;
    for (int i = 0; i < numElems; i++)
    {
        while (!lineBuf || lineBuf[0] == '#' || npos == pos)
        {
            if (!readLine(fp))
            {
                return false;
            }
            pos = lineBuf;
            if (lineBuf[0] == '#')
                fprintf(stderr, "comment: %s\n", lineBuf);
        }

        if (pos)
        {
            (*data)[i] = strtod(pos, &npos);
            (*data)[i] *= scale;
            if (pos == npos)
            {
                i--;
                continue;
            }
            pos = npos;
            npos = NULL;
        }
    }
    return true;
}

/// Compute routine: load checkpoint file
int coReadMeteo::compute(const char *)
{
    FILE *fp;
    char *filename;

    bool dataMode = pboDataMode->getValue();

    // Open first checkpoint file:
    const char *path = pbrFile->getValue();

    if (!vvToolshed::isFile(path))
    {
        sendError("Checkpoint file %s not found.", path);
        return STOP_PIPELINE;
    }

    // Create temporary filename that can be modified to increase:
    filename = new char[strlen(path) + 1];
    strcpy(filename, path);

    std::vector<coDistributedObject *> grids;
    std::vector<coDistributedObject *> data;

    // Read time steps one by one:
    int timestep = 0;
    while ((fp = fopen(filename, "rb")))
    {
        //double time;

        // read the arrays
        int dimX = pDimX->getValue();
        int dimY = pDimY->getValue();
        int dimZ = pDimZ->getValue();
        float *rawData = NULL;
        if (!readArray(fp, &rawData, dimX * dimY * dimZ))
        {
            sendError("Failed to load the data.");
            delete[] rawData;
            fclose(fp);
            return STOP_PIPELINE;
        }

        if (!dataMode)
        {
            float *x = new float[dimX * dimY * dimZ];
            float *y = new float[dimX * dimY * dimZ];
            float *z = new float[dimX * dimY * dimZ];

            for (int k = 0; k < dimZ; k++)
            {
                for (int j = 0; j < dimY; j++)
                {
                    for (int i = 0; i < dimX; i++)
                    {
                        x[i * dimY * dimZ + j * dimZ + k] = i;
                        y[i * dimY * dimZ + j * dimZ + k] = j;
                        z[i * dimY * dimZ + j * dimZ + k] = rawData[k * dimY * dimX + j * dimX + i];
                    }
                }
            }

            char name[1024];
            snprintf(name, sizeof(name), "%s_%d", poData->getObjName(), timestep);
            coDoStructuredGrid *doGrid = new coDoStructuredGrid(name,
                                                                dimX, dimY, dimZ,
                                                                x, y, z);

            grids.push_back(doGrid);
        }
        else
        {
            float *values = new float[dimX * dimY * dimZ];
            for (int k = 0; k < dimZ; k++)
            {
                for (int j = 0; j < dimY; j++)
                {
                    for (int i = 0; i < dimX; i++)
                    {
                        values[i * dimY * dimZ + j * dimZ + k] = rawData[k * dimY * dimX + j * dimX + i];
                    }
                }
            }
            char name[1024];
            snprintf(name, sizeof(name), "%s_%d", poData->getObjName(), timestep);
            coDoFloat *doData = new coDoFloat(name, dimX * dimY * dimZ, values);
            data.push_back(doData);
        }

        delete[] rawData;

        fclose(fp);

        timestep++;
        // Process next time step:
        if (!vvToolshed::increaseFilename(filename))
            break;
    }
    delete[] filename;

    if (timestep == 0)
    {
        sendError("No atoms loaded.");
        return STOP_PIPELINE;
    }

    // data has been loaded and can now be converted to sets

    // Create set objects:
    coDoSet *set = NULL;
    if (dataMode)
    {
        data.push_back(NULL);
        set = new coDoSet(poData->getObjName(), &data[0]);
        data.clear();
    }
    else
    {
        grids.push_back(NULL);
        set = new coDoSet(poData->getObjName(), &grids[0]);
        grids.clear();
    }

    // Set timestep attribute:
    if (timestep > 1)
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "%d %d", 0, timestep - 1);
        set->addAttribute("TIMESTEP", buf);
    }

    // Assign sets to output ports:
    poData->setCurrentObject(set);

    sendInfo("Timesteps loaded: %d", timestep);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, coReadMeteo)
