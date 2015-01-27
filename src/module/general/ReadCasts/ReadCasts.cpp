/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**
 * Read module for CASTS geometry and temperature data
 *
 * Initial version: 2006-09-29 fn
 * (C) 2006 by HLRS University of Stuttgart
 */

#include <qregexp.h>
#include <malloc.h>
#include <util/byteswap.h>

#include "ReadCasts.h"

ReadCasts::ReadCasts(int argc, char *argv[])
#ifndef YAC
    : coSimpleModule(argc, argv, "Read Casts")
{
#else
    : coFunctionModule(argc, argv, "Read Casts")
{
#endif

    // file browser parameter
    p_gridFile = addFileBrowserParam("grid_path", "Grid file path");
    p_dataFile = addFileBrowserParam("data_path", "Data file path");

    // Output ports
    p_gridOut = addOutputPort("mesh", "UnstructuredGrid", "unstructured grid");
    p_dataOut = addOutputPort("temperature", "Float", "temperature");
}

ReadCasts::~ReadCasts()
{

    list<struct material *>::iterator iter;
    for (iter = materials.begin(); iter != materials.end(); iter++)
        delete (*iter);

    materials.clear();
}

void ReadCasts::param(const char *paraName, bool inMapLoading)
{

    if (inMapLoading)
        return;

    if (!strcmp(paraName, p_gridFile->getName()))
    {
        struct stat buf;
        if (stat(p_gridFile->getValue(), &buf) == -1)
            sendError("Error: gridfile not found");
    }

    else if (!strcmp(paraName, p_dataFile->getName()))
    {
        struct stat buf;
        if (stat(p_dataFile->getValue(), &buf) == -1)
            sendError("Error: datafile not found");
    }
}

int ReadCasts::compute(const char *)
{

    int numTimeSteps;

    if (!p_gridFile->getValue())
    {
        sendError("Error: gridfile not set");
        return FAIL;
    }

    if (!p_dataFile->getValue())
    {
        sendError("Error: datafile not set");
        return FAIL;
    }

    materials.clear();

    bool data = readData(p_dataFile->getValue(), numTimeSteps);
    bool grid = readGrid(p_gridFile->getValue(), numTimeSteps);

    if (!grid || !data)
        return FAIL;

    return SUCCESS;
}

bool ReadCasts::readGrid(const char *fileName, int numTimeSteps)
{

    FILE *gridFile;
    int state = NUM;
    bool done = false;

    int point = 0, element = 0, c_index = 0;
    int numElements = -1, numMaterials = -1, numCorners = -1, numPoints = -1;
    char line[256];

    float *xCoord = NULL, *yCoord = NULL, *zCoord = NULL;
    int *corners = NULL, *types = NULL, *elements = NULL;

    int lineNumber = 0;
    bool errorReporting = true;

    coDoUnstructuredGrid *grid = NULL;
    coObjInfo gridInfo = p_gridOut->getNewObjectInfo();

    if (!(gridFile = fopen(fileName, "r")))
    {

        snprintf(line, 256, "could not open grid file %s", fileName);
        sendError(line);
        return false;
    }

    while (!done && fgets(line, 256, gridFile))
    {
        lineNumber++;

        switch (state)
        {

        case NUM:
        {
            if (!strcmp(line, "\n"))
            {

                if (numElements != -1 && numCorners != -1 && numPoints != -1)
                {
                    state = MAT;
#ifndef YAC
                    char buf[256];
                    snprintf(buf, 256, "%s_%d", gridInfo.getName(), 0);
#else
                    coObjInfo buf = gridInfo;
#endif
                    grid = new coDoUnstructuredGrid(buf, numElements,
                                                    numCorners, numPoints, 1);

                    grid->getAddresses(&elements, &corners, &xCoord, &yCoord, &zCoord);
                    grid->getTypeList(&types);

                    numCorners = 0;
                }
                else
                {
                    if (numElements == -1)
                        sendError("Error: number of elements not in gridfile header");
                    if (numPoints == -1)
                        sendError("Error: number of points not in gridfile header");
                    done = 1;
                }
                break;
            }

            QRegExp r("^\\s+Anzahl der (\\w+)\\s*:\\s*(\\d+)");
            if (r.indexIn(line) > -1)
            {
                QString name = r.cap(1);
                int num = r.cap(2).toInt();
                cout << name.toStdString() << " " << num << std::endl;

                if (name == "Stoffe")
                    numMaterials = num;

                else if (name == "Punkte")
                    numPoints = num;

                else if (name == "Volumenelemente")
                {
                    // for the moment, assume all elements are hexahedra, the
                    // real number of corners will be read later
                    numElements = num;
                    numCorners = num * 8;
                }
            }
            break;
        }
        case MAT:
        {
            // materials are not used for now, parse and store them anyway
            QRegExp r("^\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+$");
            if (!strcmp(line, "\n"))
            {
                state = COORD_HEADER;
                break;
            }
            if (r.indexIn(line) > -1)
            {
                struct material *mat = new material();
                materials.push_back(mat);
                for (int index = 0; index < 10; index++)
                {
                    bool ok;
                    int num = r.cap(index + 1).toInt(&ok);
                    if (!ok)
                    {
                        char buf[256];
                        snprintf(buf, 256, "Error: wrong material in gridfile line %d", lineNumber);
                        sendError(buf);
                        done = true;
                        break;
                    }
                    ((int *)mat)[index] = num;
                }
            }
            break;
        }

        case COORD_HEADER:
            if (!strcmp(line, "\n"))
                state = COORD;

            break;

        case COORD:
        {
            if (!strcmp(line, "\n"))
            {
                state = ELEM_HEADER;
                break;
            }
            QRegExp r("\\s+(-?\\d+.\\d+E.\\d\\d)\\s+(-?\\d+.\\d+E.\\d\\d)\\s+(-?\\d+.\\d+E.\\d\\d)");
            if (r.indexIn(line) > -1)
            {

                if (point < numPoints)
                {
                    xCoord[point] = r.cap(1).toFloat();
                    yCoord[point] = r.cap(2).toFloat();
                    zCoord[point] = r.cap(3).toFloat();
                    point++;
                }
                else
                {
                    if (errorReporting)
                    {
                        char buf[256];
                        snprintf(buf, 256, "Error: too many coordinates in gridfile line %d", lineNumber);
                        sendWarning(buf);
                        errorReporting = false;
                    }
                    break;
                }
            }
            break;
        }

        case ELEM_HEADER:
            if (!strcmp(line, "\n"))
                state = ELEM;

            break;

        case ELEM:
        {
            if (!strcmp(line, "\n"))
            {
                done = true;
                break;
            }

            QRegExp r("^\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+$");
            if (r.indexIn(line) > -1)
            {
                if (element < numElements)
                {
                    int index;
                    elements[element] = c_index;
                    for (index = 0; index < 8; index++)
                    {
                        int n = r.cap(index + 1).toInt();
                        if (n != 0)
                        {
                            corners[c_index] = n - 1;
                            c_index++;
                        }
                        else
                        {
                            if (index == 4)
                            {
                                types[element] = TYPE_TETRAHEDER;
                                numCorners += 4;
                            }
                            else if (index == 6)
                            {
                                types[element] = TYPE_PRISM;
                                numCorners += 6;
                            }
                            else
                            {
                                printf("wrong volume edge count: %d\n", index - 1);
                                done = true;
                            }
                            break;
                        }
                    }
                    if (index == 8)
                    {
                        types[element] = TYPE_HEXAEDER;
                        numCorners += 8;
                    }
                }
                else
                {
                    if (errorReporting)
                    {
                        char buf[256];
                        snprintf(buf, 256, "Error: too many elements in gridfile line %d", lineNumber);
                        sendWarning(buf);
                        errorReporting = false;
                    }
                }
                element++;
            }
            break;
        }

        default:
            char buf[256];
            snprintf(buf, 256, "Error: reached an unknown state in gridfile line %d", lineNumber);
            sendError(buf);
            state = UNKNOWN;
            done = true;
        }
    }

    grid->setSizes(numElements, numCorners, numPoints);

#ifndef YAC
    int index;
    coDoUnstructuredGrid **grids = new coDoUnstructuredGrid *[numTimeSteps + 1];
    for (index = 0; index < numTimeSteps; index++)
    {
        grids[index] = grid;
        if (index > 0)
            grid->incRefCount();
    }
    grids[numTimeSteps] = NULL;
    coDoSet *gridSet = new coDoSet(gridInfo, (coDistributedObject **)grids);

    snprintf(line, 256, "1 %d", numTimeSteps);
    gridSet->addAttribute("TIMESTEP", line);

    p_gridOut->setCurrentObject(gridSet);
#else
    (void)numTimeSteps;
    p_gridOut->setCurrentObject(grid);
    coOutputPort **out = new coOutputPort *[1];
    out[0] = p_gridOut;
    waitForOutportAvailability(1, out, 999.0);
    createdObjects(1, out);
    delete[] out;
#endif

    return true;
}

/*
 * determine if data needs to be byteswapped
 *
 * return -1 on error
 *         0 no byteswapping needed
 *         1 data needs to be byteswapped
 */
int ReadCasts::swapData(const char *fileName)
{

    int nread;
    FILE *dataFile;

    if (!(dataFile = fopen(fileName, "r")))
        return -1;

    int fileLength;
    struct stat buf;
    if (stat(fileName, &buf) != -1)
    {
        fileLength = buf.st_size;

        struct
        {
            int length;
            float time;
            int nodes;
        } header;

        nread = fread(&header, sizeof(header), 1, dataFile);
        if (nread != 1)
        {
            fclose(dataFile);
            return -1;
        }
        if (header.length > 0 && header.length < fileLength)
        {
            int l = header.length;
            int n = header.nodes;
            fseek(dataFile, header.nodes * sizeof(float) + sizeof(int), SEEK_CUR);
            nread = fread(&header, sizeof(header), 1, dataFile);
            if (nread == 1 && header.length == l && header.nodes == n)
            {
                fclose(dataFile);
                return 0;
            }
        }

        fseek(dataFile, 0, SEEK_SET);
        nread = fread(&header, sizeof(header), 1, dataFile);
        if (nread != 1)
        {
            fclose(dataFile);
            return -1;
        }

        byteSwap(header.length);
        byteSwap(header.nodes);

        if (header.length > 0 && header.length < fileLength)
        {
            /*
         int l = header.length;
         int n = header.nodes;
         fseek(dataFile, header.nodes * sizeof(float) + sizeof(int), SEEK_CUR);
         nread = fread(&header, sizeof(header), 1, dataFile);
         byteSwap(header.length);
         byteSwap(header.nodes);
         if (nread == 1 && header.length == l && header.nodes == n) {
            fclose(dataFile);
*/
            return 1;
            /*
         }
*/
        }
    }
    fclose(dataFile);
    return -1;
}

bool ReadCasts::readData(const char *fileName, int &numTimeSteps)
{

    FILE *dataFile;
    char name[256];
    numTimeSteps = 0;

    vector<coDoFloat *> timeSteps;
    bool done = false;
    int nread;
#ifndef YAC
    char buf[256];
#endif

    int swap = swapData(fileName);
    if (swap == -1)
        return false;

    if (!(dataFile = fopen(fileName, "r")))
    {

        snprintf(name, 256, "could not open data file %s", fileName);
        sendError(name);
        return false;
    }

    coObjInfo dataInfo = p_dataOut->getNewObjectInfo();

    while (!done)
    {

        struct
        {
            int length;
            float time;
            int nodes;
        } header;

        float *data;

        nread = fread(&header, sizeof(header), 1, dataFile);
        if (nread != 1)
            break;
        if (swap)
        {
            byteSwap(header.length);
            byteSwap(header.time);
            byteSwap(header.nodes);
        }

#ifndef YAC
        snprintf(buf, 256, "%s_%d", dataInfo.getName(), numTimeSteps);
        coDoFloat *step = new coDoFloat(buf, header.nodes);
#else
        coDoFloat *step = new coDoFloat(dataInfo, header.nodes);
#endif

        timeSteps.push_back(step);

        step->getAddress(&data);
        nread = fread(data, sizeof(float), header.nodes, dataFile);
        if (nread != header.nodes)
            break;
        if (swap)
            byteSwap(data, header.nodes);
        nread = fread(&header.length, sizeof(int), 1, dataFile);
        if (nread != 1)
            break;

#ifdef YAC
        char time[256];
        snprintf(time, 256, "%d %d", numTimeSteps, 62);
        step->addAttribute("TIMESTEP", time);
        step->setInfo(0, 1, numTimeSteps, 62, (float)numTimeSteps);

        p_dataOut->setCurrentObject(step);
        coOutputPort **out = new coOutputPort *[1];
        out[0] = p_dataOut;
        waitForOutportAvailability(1, out, 999.0);
        createdObjects(1, out);
        delete[] out;
#endif
        numTimeSteps++;
    }

#ifndef YAC
    int index;
    coDoFloat **steps = new coDoFloat *[timeSteps.size()];
    for (index = 0; index < timeSteps.size(); index++)
        steps[index] = timeSteps[index];

    steps[numTimeSteps] = NULL;
    coDoSet *dataSet = new coDoSet(dataInfo, (coDistributedObject **)steps);
    snprintf(buf, 256, "1 %d", numTimeSteps);
    dataSet->addAttribute("TIMESTEP", buf);
    p_dataOut->setCurrentObject(dataSet);
#endif

    return true;
}

MODULE_MAIN(IO, ReadCasts)
