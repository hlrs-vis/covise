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
#include <do/coDoPoints.h>
#include <do/coDoLines.h>
#include <do/coDoData.h>
#include <api/coModule.h>
#include <limits.h>
#include <float.h>
#include <util/coVector.h>
#include "ReadCellTree.h"

#define BUFINC 1

const char *initialCellTypes[] = { "Select Cell Type File" };

/// Constructor
coReadCellTree::coReadCellTree(int argc, char *argv[])
    : coModule(argc, argv, "Read files to create a list of points (stars) and scalar parameters.")
    , lineBuf(NULL)
    , lineBufSize(0)
{
    // Create ports:
    poPoints = addOutputPort("Points", "Points", "Points representing cell types");
    poPoints->setInfo("Points representing cell types");

    poTreeLines = addOutputPort("TreeLines", "Lines", "Lines representing closest relationships between cell types");
    poTreeLines->setInfo("Lines representing closest relationships between cell types");

    poLines = addOutputPort("Lines", "Lines", "Lines representing similarities between cell types");
    poLines->setInfo("Lines representing similarities between cell types");

    poDiameters = addOutputPort("Diameters", "Float",
                                "Diameters representing difference from actual similarity");
    poDiameters->setInfo("Diameters representing difference from actual similarity");

    poConnectivity = addOutputPort("Connectivity", "Float",
                                   "Connectivity information within tree");
    poConnectivity->setInfo("Connectivity information within tree");

    // Create parameters:

    // Create parameters:
    pFile = addFileBrowserParam("FilePath", "File containing adjacency matrix");
    pFile->setValue("/raid/projekte/mathedata/baum/Abstands_Mittel.txt", "*.txt/*");

    pCellDescFile = addFileBrowserParam("DescFilePath", "File containing cell type descriptions");
    pCellDescFile->setValue("/raid/projekte/mathedata/baum/Zelltypen_IDs.txt", "*.txt/*");

    pRootCell = addChoiceParam("rootCell", "Number of root cell type");
    pRootCell->setValue(1, initialCellTypes, 0);

    //pboWarnings = addBooleanParam("Warnings", "Display warnings when reading files");
    //pboWarnings->setValue(true);
}

/// @return absolute value of a vector
float coReadCellTree::absVector(float x, float y, float z)
{
    return sqrt(x * x + y * y + z * z);
}

bool coReadCellTree::readLine(FILE *fp)
{
    int offset = 0;

    for (;;)
    {
        if (lineBuf && lineBufSize > 1)
        {
            if (!fgets(lineBuf + offset, (int)lineBufSize - offset, fp))
            {
                return false;
            }

            int len = (int)strlen(lineBuf);
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
            offset = (int)lineBufSize - 1;
        }
        else
        {
            offset = 0;
        }
        lineBufSize += BUFINC;
    }

    return false;
}

bool coReadCellTree::readArray(FILE *fp, float **data, int numElems)
{
    *data = new float[numElems];
    char *pos = NULL, *npos = NULL;
    for (int i = 0; i < numElems; i++)
    {
        while (!lineBuf || lineBuf[0] == '#' || npos == pos)
        {
            if (!readLine(fp))
            {
                fprintf(stderr, "only read %d instead of %d elems\n", i, numElems);
                return false;
            }
            pos = lineBuf;
            if (lineBuf[0] == '#')
                fprintf(stderr, "comment: %s\n", lineBuf);
        }

        if (pos)
        {
            (*data)[i] = float(strtod(pos, &npos));
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

void coReadCellTree::param(const char *paramName, bool inMapLoading)
{
    if (!strcmp(paramName, "SetModuleTitle"))
    {
        // just ignore
    }
    else if (!strcmp(paramName, pCellDescFile->getName()))
    {
        const char *path = pCellDescFile->getValue();

        FILE *fp = fopen(path, "r");
        if (!fp)
        {
            if (!inMapLoading)
            {
                sendError("Failed to open file %s.", path);
            }
            return;
        }

        for (int i = 0; i < cellTypes.size(); i++)
        {
            delete[] cellTypes[i];
        }
        cellTypes.clear();

        while (!feof(fp))
        {
            char buf[10240];
            if (!fgets(buf, sizeof(buf), fp))
            {
                break;
            }
            size_t len = strlen(buf);
            while (len > 1 && isspace(buf[len - 1]))
            {
                buf[len - 1] = '\0';
                len--;
            }
            char *name = new char[len + 1];
            strcpy(name, buf);
            cellTypes.push_back(name);
        }
        fclose(fp);

        pRootCell->setValue((int)cellTypes.size(), &cellTypes[0], 0);
    }
    else
    {
        sendInfo("unhandled immediate parameter %s", paramName);
    }
}

/// Compute routine: load checkpoint file
int coReadCellTree::compute(const char *)
{
    // Open first checkpoint file:
    const char *path = pFile->getValue();

    // Create temporary filename that can be modified to increase:
    char *filename = new char[strlen(path) + 1];
    strcpy(filename, path);

    int numCells = int(cellTypes.size());
    int numDistances = numCells * numCells;

    // Read time steps one by one:
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        sendError("File %s not found.", filename);
        return STOP_PIPELINE;
    }
    float *distance = NULL;
    if (fp)
    {
        if (!readArray(fp, &distance, numDistances))
        {
            sendError("Failed to load the data.");
            delete[] distance;
            fclose(fp);
            return STOP_PIPELINE;
        }
        fclose(fp);
    }
    else
    {
        sendError("No distances loaded.");
        return STOP_PIPELINE;
    }
    delete[] filename;

    float *realDistance = new float[numDistances];
    memcpy(realDistance, distance, sizeof(float) * numDistances);

    float maxDistance = 0.f;
    for (int i = 0; i < numDistances; i++)
    {
        if (distance[i] > maxDistance)
            maxDistance = distance[i];
    }

    coVector *point = new coVector[numCells];
    point[0] = coVector(0.f, 0.f, 0.f);
    for (int i = 1; i < numCells; i++)
    {
        point[i] = coVector(rand() / float(RAND_MAX) * maxDistance * .5,
                            rand() / float(RAND_MAX) * maxDistance * .5,
                            rand() / float(RAND_MAX) * maxDistance * .5);
    }

    coVector *force = new coVector[numCells];

    fprintf(stderr, "\n");
    float finish = 1.e-5f;
    float exponent = 1.5f;
    double forces;
    float adapt = 0.01f;
    do
    {
        forces = 0.f;
        for (int i = 0; i < numCells; i++)
        {
            force[i] = coVector(0.f, 0.f, 0.f);
            for (int j = 0; j < numCells; j++)
            {
                if (j == i)
                    continue;

                coVector dist = point[j] - point[i];
                double len = dist.length();
                force[i] = force[i] + dist * (1. / len) * pow(1 - distance[i * numCells + j] / maxDistance, exponent) * (len - distance[i * numCells + j]);
            }
            forces += force[i].length();
        }

        for (int i = 0; i < numCells; i++)
        {
            point[i] = point[i] + force[i] * adapt;
        }

        fprintf(stderr, "forces=%f\r", forces);
    } while (forces >= finish);
    fprintf(stderr, "\n");

    float *x = new float[numCells];
    float *y = new float[numCells];
    float *z = new float[numCells];
    for (int i = 0; i < numCells; i++)
    {
        x[i] = float(point[i][0]);
        y[i] = float(point[i][1]);
        z[i] = float(point[i][2]);
    }

    coDoPoints **points = new coDoPoints *[numCells + 1];
    points[numCells] = NULL;
    for (int i = 0; i < numCells; i++)
    {
        char name[1024];
        sprintf(name, "%s_%d", poPoints->getObjName(), i);
        float *xx = new float[1];
        float *yy = new float[1];
        float *zz = new float[1];
        *xx = x[i];
        *yy = y[i];
        *zz = z[i];
        points[i] = new coDoPoints(name, 1, xx, yy, zz);
        points[i]->addAttribute("LABEL", cellTypes[i]);
    }

    coDoSet *setPoints = new coDoSet(poPoints->getObjName(), (coDistributedObject **)points);
    //coDoPoints *covPoints = new coDoPoints(poPoints->getObjName(), numCells, x, y, z);
    poPoints->setCurrentObject(setPoints);

    int *corners = new int[(numCells - 1) * 2];
    int *lines = new int[numCells - 1];
    for (int i = 0; i < numCells - 1; i++)
    {
        lines[i] = 2 * i;
    }

    int *usedCells = new int[numCells];
    int numUsedCells = 1;
    usedCells[0] = pRootCell->getValue();

    for (int j = 0; j < numCells; j++)
    {
        if (j != usedCells[0])
            distance[j * numCells + usedCells[0]] = -1.f;
    }

    int *conn = new int[numDistances];
    for (int i = 0; i < numDistances; i++)
    {
        conn[i] = 0;
    }

    for (int i = 0; i < numCells - 1; i++)
    {
        float minDist = FLT_MAX;
        int minBegin = 0, minEnd = 0;
        for (int j = 0; j < numCells; j++)
        {
            for (int k = 0; k < numUsedCells; k++)
            {
                if (j == usedCells[k])
                    continue;

                float d = distance[usedCells[k] * numCells + j];
                //fprintf(stderr, "(%d,%d): d=%f\n", j, usedCells[k], d);
                if (minDist > d && d > 0.f)
                {
                    minDist = d;
                    minBegin = usedCells[k];
                    minEnd = j;
                }
            }
        }

        usedCells[numUsedCells++] = minEnd;
        //fprintf(stderr, "new line: %d -> %d (%f)\n", minBegin, minEnd, minDist);
        for (int j = 0; j < numCells; j++)
        {
            if (j != minEnd)
                distance[j * numCells + minEnd] = -1.f;
        }

        corners[i * 2] = minBegin;
        corners[i * 2 + 1] = minEnd;

        conn[minBegin * numCells + minEnd] = 1;
        conn[minEnd * numCells + minBegin] = 1;
    }

    coDoLines *covTreeLines = new coDoLines(poTreeLines->getObjName(), numCells, x, y, z,
                                            (numCells - 1) * 2, corners,
                                            numCells - 1, lines);
    poTreeLines->setCurrentObject(covTreeLines);

    coDoFloat *diameters = new coDoFloat(poDiameters->getObjName(), (numCells - 1) * numCells / 2);
    float *dia = NULL;
    diameters->getAddress(&dia);

    coDoFloat *connectivity = new coDoFloat(poConnectivity->getObjName(), (numCells - 1) * numCells / 2);
    float *connect = NULL;
    connectivity->getAddress(&connect);

    int *allCorners = new int[(numCells - 1) * numCells];
    int *allLines = new int[(numCells - 1) * numCells / 2];
    int lc = 0;
    for (int i = 0; i < numCells; i++)
    {
        for (int j = i + 1; j < numCells; j++)
        {
            allCorners[lc * 2] = i;
            allCorners[lc * 2 + 1] = j;
            allLines[lc] = 2 * lc;

            coVector dist = point[j] - point[i];
            double actualDist = dist.length();
            dia[lc] = float(pow(realDistance[i * numCells + j] / actualDist, 5) * .02);

            connect[lc] = float(conn[i * numCells + j]);

            lc++;
        }
    }

    coDoLines *covLines = new coDoLines(poLines->getObjName(), numCells, x, y, z,
                                        (numCells - 1) * numCells, allCorners,
                                        (numCells - 1) * numCells / 2, allLines);
    poLines->setCurrentObject(covLines);

    poDiameters->setCurrentObject(diameters);
    poConnectivity->setCurrentObject(connectivity);

#if 0
   delete[] conn;
   delete[] connect;
   delete[] allCorners;
   delete[] allLines;
#endif

#if 0
   fprintf(stderr, "corners:");
   for(int i=0; i<(numCells-1)*2; i++)
   {
      fprintf(stderr, " %d", corners[i]);
   }
   fprintf(stderr, "\n");
   fprintf(stderr, "lines:");
   for(int i=0; i<(numCells-1); i++)
   {
      fprintf(stderr, " %d", lines[i]);
   }
   fprintf(stderr, "\n");
#endif

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, coReadCellTree)
